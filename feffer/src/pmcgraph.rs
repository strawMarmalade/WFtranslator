use std::cmp::Ordering;
use std::collections::HashMap;

type AdjMtx = HashMap<usize, Vec<usize>>;

#[derive(Default, Clone, Debug, Eq, PartialEq)]
pub struct PmcGraph{
    /*edges is the concatenation over all vertices of all vertices each vertex is connected to
    so if we have the graph 1 - 2 - 3 - 1, then edges would be 2, 3, 1, 3, 1, 2 as 1 is connected to 2 and 3
    and 2 is connected to 1 and 3 and 3 is connected to 1 and 2
    */
    edges: Vec<usize>,
    /* vertices is a list that at index j is the sum of the degrees of every vertex 0,...,j-2 */
    vertices: Vec<usize>,
    /* the degree of each vertex */
    degree: Vec<usize>,
    /* the min and max degrees of the entire graph */
    min_degree: usize,
    max_degree: usize,
    //want f64 here but can't compare those in rust with the Eq trait, so instead use a fraction and to check will have to compare values but eh 
    avg_degree: (usize,usize), 
    is_gstats: bool,
    //we leave away the string fn
    //and the adjoint matrix adj for now
    max_core: usize,
    kcore: Vec<usize>,
    kcore_order: Vec<usize>,
}

impl PmcGraph{
    pub fn new(verts: Vec<usize>, edgs: Vec<(usize,usize)>) -> PmcGraph{
        let mut vertices: Vec<usize> = vec![];
        let mut edges: Vec<usize> = vec![];
        let mut vert_list: AdjMtx = AdjMtx::new();
        let mut currentStartVert: usize = edgs[0].0;
        let mut now: std::time::Instant = std::time::Instant::now();
        vert_list.insert(currentStartVert, vec![]);
        for e in edgs {
            if e.0 != currentStartVert {
                /*as we hope the edges will be somewhat ordered we 
                we might save so time by first checking against that*/
                /*now if the vertex is not already contained we add it to the hashmap */
                if !vert_list.contains_key(&e.0){
                    vert_list.insert(e.0, vec![]);
                }
                /*now update the current vector for the next round */
                currentStartVert = e.0;
            }
            /*probably however the end node will be the one that keeps on changing so we'll just have to check every time whether or not its already in the hashmap or not */
            if !vert_list.contains_key(&e.1){
                vert_list.insert(e.1, vec![]);
            }
            /*now we actually add the edge to the vertex list */
            if let Some(lst) = vert_list.get_mut(&e.0) {
                lst.push(e.1);
                //self.max_degree = cmp::max(self.max_degree, lst.len());
    
            } else {
                panic!("The node {} does not belong to this graph.", e.0);
            }
            if let Some(lst) = vert_list.get_mut(&e.1) {
                lst.push(e.0);
                //self.max_degree = cmp::max(self.max_degree, lst.len());
            } else {
                panic!("The node {} does not belong to this graph.", e.1);
            }
        }
        /* now we should have the graph in the hashmap and will convert it to the form we need it */
        vertices.push(edges.len());

        for ver in verts.iter() {
            if let Some(lst) = vert_list.get(ver) {
                /* we have found this vertex in the hashmap and add all the edges and the vertex*/
                edges.extend(lst.into_iter());
                vertices.push(edges.len());
            } else {
                panic!("The node {} does not belong to this graph.", ver);
            }
        }
        let mut elapsed_time = now.elapsed();
        println!(
            "Building the weird graph struct from input took {} milliseconds.",
            elapsed_time.as_millis()
        );
        let n = vertices.len();
        let mut g = PmcGraph { vertices: vertices, edges: edges, degree: vec![0; n-1], min_degree: 0, max_degree: 0,
            avg_degree: (0,1), is_gstats: false, max_core: 0, kcore: vec![], kcore_order: vec![]};
        now = std::time::Instant::now();
        g.vertex_degrees();
        elapsed_time = now.elapsed();
        println!(
            "Calculating degrees took {} milliseconds.",
            elapsed_time.as_millis()
        );
        now = std::time::Instant::now();
        g.compute_cores();
        elapsed_time = now.elapsed();
        println!(
            "Computing cores took {} milliseconds.",
            elapsed_time.as_millis()
        );
        g

    }
    pub fn vertex_degrees(&mut self) {
        let n = self.vertices.len() - 1;

        // initialize min and max to degree of first vertex
        let mut max_degree: usize = self.vertices[1] - self.vertices[0] ;
        let mut min_degree: usize = self.vertices[1] - self.vertices[0];

        for v in 0..n {
            self.degree[v] = self.vertices[v+1] - self.vertices[v];
            if max_degree < self.degree[v] {
                max_degree = self.degree[v];
            } 
            if self.degree[v] < min_degree {
                min_degree = self.degree[v];
            } 
        }
        self.max_degree = max_degree;
        self.min_degree = min_degree;
        self.avg_degree = (self.edges.len(), n);
        return;
    }
    pub fn compute_cores(&mut self) {
        let n: usize = self.vertices.len(); 

        let mut pos: Vec<usize> = vec![0;n];
        if self.kcore_order.len() > 0 {
            //let tmp: Vec<usize> = vec![0; n];
            self.kcore = vec![0; n];
            self.kcore_order = vec![0; n];
        }
        else {
            self.kcore_order.resize(n, 0);
            self.kcore.resize(n, 0);
        }

        let mut md: usize = 0;        

        for v in 1..n {
            self.kcore[v] = self.vertices[v] - self.vertices[v-1];
            if self.kcore[v] > md {
                md = self.kcore[v];
            }
        }

        let md_end: usize = md+1;

        let mut bin: Vec<usize> = vec![0; md_end];

        for v in 1..n {
            bin[self.kcore[v]] += 1;
        }

        let mut start: usize = 1;
        let mut num: usize = 0;

        for d in 0..md_end {
            num = bin[d];
            bin[d] = start;
            start += num;
        }

        for v in 1..n {
            pos[v] = bin[self.kcore[v]];
            self.kcore_order[pos[v]] = v;
            bin[self.kcore[v]] += 1;
        }

        for d in (2..=md).rev() {
            bin[d] = bin[d-1];
        }
        bin[0] = 1;

        let mut v: usize = 0;
        let mut u: usize = 0;
        let mut w: usize = 0;
        let mut du: usize = 0;
        let mut pu: usize = 0;
        let mut pw: usize = 0;


        // kcores
        for i in 1..n {
            v=self.kcore_order[i];
            for j in self.vertices[v-1]..self.vertices[v]{
                u = self.edges[j] + 1;
                if self.kcore[u] > self.kcore[v] {
                    du = self.kcore[u];   
                    pu = pos[u];
                    pw = bin[du];
                    w = self.kcore_order[pw];
                    if u != w {
                        pos[u] = pw;   
                        self.kcore_order[pu] = w;
                        pos[w] = pu;   
                        self.kcore_order[pw] = u;
                    }
                    bin[du] += 1;
                    self.kcore[u] -= 1;
                }
            }
        }
        for v in 0..n-1 {
            self.kcore[v] = self.kcore[v+1] + 1; // K + 1
            self.kcore_order[v] = self.kcore_order[v+1]-1;
        }
        self.max_core = self.kcore[self.kcore_order[n-2]] - 1;
    }
    pub fn search_bounds(&self) -> Vec<usize> {
        let V = &self.vertices;
        let n: usize = V.len();
        let E = &self.edges;
        let K = &self.kcore;
        let order = &self.kcore_order;
        let degree = &self.degree;
        let mut clique: Vec<usize> = vec![];
        let mut C_max: Vec<usize> = vec![];
        let mut X: Vec<usize> = vec![];
        let ub = self.max_core + 1;
        let mut P: Vec<Vertex> = vec![];
        let mut T: Vec<Vertex> = vec![];
        let mut ind: Vec<bool> = vec![false; n-1];
        let mut found_ub: bool = false;

        let mut v: usize = 0;
        let mut mc_prev: usize = 0;
        let mut mc: usize = 0;
        let mut mc_cur: usize = 0;

        for i in (0..n-1).rev() {
            if found_ub {
                continue;
            }
            v = order[i];
            mc_cur = mc;
            mc_prev = mc_cur;

            if K[v] > mc {
                for j in V[v]..V[v+1] {
                    if K[E[j]] > mc {
                        P.push(Vertex { id: E[j], b: degree[E[j]] });
                    }
                }
                /*sort_by(|a, b| a.partial_cmp(b).unwrap());
assert_eq!(floats, */
                if P.len() > mc_cur {
                    P.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    PmcGraph::branch(&self, &mut P, 1, &mut mc_cur, &mut clique, &mut ind);
                    
                    if mc_cur > mc_prev {
                        if mc < mc_cur {
                            /* Here have to make sure if multiple threads it wont kill itself */
                            mc = mc_cur;
                            clique.push(v);
                            C_max = clique.clone();
                            if mc >= ub {
                                found_ub = true;
                            }
                            //println!("{:?}", C_max);
                        }
                    }
                
                
                }
                clique = X.clone();
                P = T.clone();
            }
        }
        //println!("Heuristic: clique= {:?}", C_max);
        C_max
    }
    pub fn branch(&self, P: &mut Vec<Vertex>, sz: usize, mc: &mut usize, C: &mut Vec<usize>, ind: &mut Vec<bool>){
        if P.len() > 0{
            let u = P.pop().unwrap().id;
            let V = &self.vertices;
            let E = &self.edges;
            let K = &self.kcore;

            for j in V[u]..V[u+1] {
                ind[E[j]] = true;
            }
            let mut R: Vec<Vertex>  = vec![];
            for i in 0..P.len() {
                if ind[P[i].id] {
                    if K[P[i].id] > *mc {
                        R.push(P[i].clone());
                    }
                }
            }
            for j in V[u]..V[u+1] {
                ind[E[j]] = false;
            }
            let mc_prev = mc.clone();
            self.branch(&mut R, sz+1, mc, C, ind);

            if *mc > mc_prev {
                C.push(u);
            }
        }
        else if sz > *mc {
            *mc = sz;
        }
    }
}

#[derive(Default, Clone, Debug, Eq, PartialEq)]
pub struct Vertex {
    id: usize,
    b: usize,
}
impl Ord for Vertex {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.b).cmp(&other.b)
    }
}
impl PartialOrd for Vertex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}