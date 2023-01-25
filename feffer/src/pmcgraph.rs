//use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::sync::{Arc, Barrier, Mutex};
use threadpool::ThreadPool;


/* this is the node amount bound. If we have more than 4_294_967_295 many nodes, change this to usize/u64 */
type NAB = u32;
type AdjMtx = HashMap<NAB, Vec<NAB>>;


#[derive(Default, Clone, Debug, Eq, PartialEq)]
pub struct PmcGraph{
    /*edges is the concatenation over all vertices of all vertices each vertex is connected to
    so if we have the graph 1 - 2 - 3 - 1, then edges would be 2, 3, 1, 3, 1, 2 as 1 is connected to 2 and 3
    and 2 is connected to 1 and 3 and 3 is connected to 1 and 2
    */
    edges: Vec<NAB>,
    /* vertices is a list that at index j is the sum of the degrees of every vertex 0,...,j-2 */
    vertices: Vec<usize>,
    /* the degree of each vertex */
    degree: Vec<NAB>,
    /* the min and max degrees of the entire graph */
    min_degree: NAB,
    max_degree: NAB,
    //want f64 here but can't compare those in rust with the Eq trait, so instead use a fraction and to check will have to compare values but eh 
    //avg_degree: (NAB,NAB), 
    //is_gstats: bool,
    //we leave away the string fn
    //and the adjoint matrix adj for now
    max_core: NAB,
    kcore: Vec<NAB>,
    kcore_order: Vec<NAB>,
}

impl PmcGraph{
    pub fn new(verts: Vec<NAB>, edgs: Vec<(NAB,NAB)>) -> PmcGraph{
        let mut vertices: Vec<usize> = vec![];
        let mut edges: Vec<NAB> = vec![];
        let mut vert_list: AdjMtx = AdjMtx::new();
        let mut current_start_vert: NAB = edgs[0].0;
        let mut now: std::time::Instant = std::time::Instant::now();
        vert_list.insert(current_start_vert, vec![]);
        for e in edgs {
            if e.0 != current_start_vert {
                /*as we hope the edges will be somewhat ordered we 
                we might save so time by first checking against that*/
                /*now if the vertex is not already contained we add it to the hashmap */
                if !vert_list.contains_key(&e.0){
                    vert_list.insert(e.0, vec![]);
                }
                /*now update the current vector for the next round */
                current_start_vert = e.0;
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
            //avg_degree: (0,1), is_gstats: false, 
            max_core: 0, kcore: vec![], kcore_order: vec![]};
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
        let mut max_degree: NAB = (self.vertices[1] - self.vertices[0]) as NAB;
        let mut min_degree: NAB = (self.vertices[1] - self.vertices[0]) as NAB;

        for v in 0..n {
            self.degree[v] = (self.vertices[v+1] - self.vertices[v]) as NAB;
            if max_degree < self.degree[v] {
                max_degree = self.degree[v];
            } 
            if self.degree[v] < min_degree {
                min_degree = self.degree[v];
            } 
        }
        self.max_degree = max_degree;
        self.min_degree = min_degree;
        //self.avg_degree = (self.edges.len() as NAB, n as NAB);
        return;
    }
    pub fn compute_cores(&mut self) {
        let n: usize = self.vertices.len(); 

        let mut pos: Vec<NAB> = vec![0;n];
        if self.kcore_order.len() > 0 {
            //let tmp: Vec<usize> = vec![0; n];
            self.kcore = vec![0; n];
            self.kcore_order = vec![0; n];
        }
        else {
            self.kcore_order.resize(n, 0);
            self.kcore.resize(n, 0);
        }

        let mut md: NAB = 0;        

        for v in 1..n {
            self.kcore[v] = (self.vertices[v] - self.vertices[v-1]) as NAB;
            if self.kcore[v] > md {
                md = self.kcore[v];
            }
        }

        let md_end: NAB = md+1;

        let mut bin: Vec<NAB> = vec![0; md_end as usize];

        for v in 1..n {
            bin[self.kcore[v] as usize] += 1;
        }

        let mut start: NAB = 1;
        let mut num: NAB;

        for d in 0..md_end {
            num = bin[d as usize];
            bin[d as usize] = start;
            start += num;
        }

        for v in 1..n {
            pos[v] = bin[self.kcore[v] as usize];
            self.kcore_order[pos[v] as usize] = v as NAB;
            bin[self.kcore[v] as usize] += 1;
        }

        for d in (2..=md).rev() {
            bin[d as usize] = bin[d as usize - 1];
        }
        bin[0] = 1;

        let mut v: usize;
        let mut u: usize;
        let mut w: usize;
        let mut du: usize;
        let mut pu: usize;
        let mut pw: usize;

        // kcores
        for i in 1..n {
            v=self.kcore_order[i as usize] as usize;
            for j in self.vertices[v-1]..self.vertices[v]{
                u = self.edges[j] as usize + 1;
                if self.kcore[u] > self.kcore[v] {
                    du = self.kcore[u] as usize;   
                    pu = pos[u] as usize;
                    pw = bin[du] as usize;
                    w = self.kcore_order[pw] as usize;
                    if u != w {
                        pos[u] = pw as NAB;   
                        self.kcore_order[pu] = w as NAB;
                        pos[w] = pu as NAB;   
                        self.kcore_order[pw] = u as NAB;
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
        self.max_core = self.kcore[self.kcore_order[n-2] as usize] - 1;
    }
    pub fn search_bounds<'a>(&'a self) -> Vec<NAB> {
        let verts = Arc::new(&self.vertices);
        let n = Arc::new(verts.len());
        let edgs = Arc::new(&self.edges);
        let kcores = Arc::new(&self.kcore);
        let order = Arc::new(&self.kcore_order);
        let degree = Arc::new(&self.degree);
        let mut clique: Arc<Mutex<Vec<NAB>>> = Arc::new(Mutex::new(vec![]));
        let mut c_max: Arc<Mutex<Vec<NAB>>>  = Arc::new(Mutex::new(vec![]));
        //let mut X: Vec<usize> = vec![];
        let ub = Arc::new(self.max_core + 1);
        let mut pairs: Arc<Mutex<Vec<(NAB,NAB)>>> = Arc::new(Mutex::new(vec![]));
        //let mut T: Vec<Vertex> = vec![];
        let mut ind: Arc<Mutex<Vec<bool>>> = Arc::new(Mutex::new(vec![false; *n-1]));
        let mut found_ub = Arc::new(false);

        let mut v: Arc<Mutex<NAB>> = Arc::new(Mutex::new(0));
        let mut mc_prev: Arc<Mutex<NAB>> = Arc::new(Mutex::new(0));
        let mut mc: Arc<Mutex<NAB>> = Arc::new(Mutex::new(0));
        let mut mc_cur: Arc<Mutex<NAB>> = Arc::new(Mutex::new(0));

        let lock = Arc::new(AtomicBool::new(false)); // value answers "am I locked?"
        let n_workers = *n;
        let n_jobs = *n-1;
        let pool = ThreadPool::new(n_workers);
        let barrier = Arc::new(Barrier::new(n_jobs + 1));

        /* here start threads */
        for i in (0..*n-1).rev() {
            let barrier = barrier.clone();
            let mut clique = clique.clone();
            let mut c_max = c_max.clone();
            //let mut X: Vec<usize> = vec![];
            let mut pairs = pairs.clone();
            //let mut T: Vec<Vertex> = vec![];
            let mut ind = ind.clone();
            let mut found_ub = found_ub.clone();
    
            let mut v = v.clone();
            let mut mc_prev = mc_prev.clone();
            let mut mc = mc.clone();
            let mut mc_cur = mc_cur.clone();
            pool.execute(|| {
                if !*found_ub {
                v = Arc::new(Mutex::new(*order.get(i).unwrap()));
                mc_cur = mc;
                mc_prev = mc_cur;

                if kcores[*v.lock().unwrap() as usize] > *mc.lock().unwrap() {
                    for j in verts[*v.lock().unwrap() as usize]..verts[*v.lock().unwrap() as usize + 1] {
                        if kcores[edgs[j as usize] as usize] > *mc.lock().unwrap() {
                            pairs.lock().unwrap().push((edgs[j as usize], degree[edgs[j as usize] as usize]));
                        }
                    }
                    /*sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(floats, */
                    if pairs.lock().unwrap().len() > *mc_cur.lock().unwrap() as usize{
                        /* If I want to get exactly the same result as the actual
                        pmc code I need to do stable sort and make pmc do stable sort as well */
                        pairs.lock().unwrap().sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        // for jjj in 0..P.len(){
                        //     println!("id:{} b:{}", P[jjj].id, P[jjj].b);
                        // }
                        self.branch(&mut *pairs.lock().unwrap(), 1, &mut *mc_cur.lock().unwrap(), &mut *clique.lock().unwrap(), &mut *ind.lock().unwrap());
                        
                        if *mc_cur.lock().unwrap() > *mc_prev.lock().unwrap() {
                            if *mc.lock().unwrap() < *mc_cur.lock().unwrap() {
                                // Try to acquire the lock by setting it to true
                                while lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).unwrap() { }
                                /* Here have to make sure if multiple threads it wont kill itself */
                                if *mc.lock().unwrap() < *mc_cur.lock().unwrap() {
                                    mc = mc_cur;
                                    clique.lock().unwrap().push(*v.lock().unwrap());
                                    c_max = clique;
                                    if *mc.lock().unwrap() >= *ub {
                                        *found_ub = true;
                                    }
                                    println!("{}", c_max.lock().unwrap().len());
                                }
                                lock.store(false, Ordering::Release);
                            }
                        }
                    }
                }
                clique = Arc::new(Mutex::new(vec![]));
                pairs = Arc::new(Mutex::new(vec![]));
            }
            });
        }
        barrier.wait();
        //println!("Heuristic: clique= {:?}", C_max);
        *c_max.lock().unwrap()
    }
    pub fn branch<'a>(&'a self, pairs: &'a mut Vec<(NAB,NAB)>, sz: NAB, mc: &'a mut NAB, cliq: &'a mut Vec<NAB>, ind: &'a mut Vec<bool>){
        if pairs.len() > 0{
            let u = pairs.pop().unwrap().0;
            let verts = &self.vertices;
            let edgs = &self.edges;
            let kcores = &self.kcore;

            for j in verts[u as usize]..verts[u as usize +1] {
                ind[edgs[j as usize] as usize] = true;
            }
            let mut remain: Vec<(NAB,NAB)>  = vec![];
            for i in 0..pairs.len() {
                if ind[pairs[i as usize].0 as usize] {
                    if kcores[pairs[i as usize].0 as usize] > *mc {
                        remain.push(pairs[i as usize].clone());
                    }
                }
            }
            for j in verts[u as usize]..verts[u as usize +1] {
                ind[edgs[j as usize] as usize] = false;
            }
            let mc_prev = mc.clone();
            self.branch(&mut remain, sz+1, mc, cliq, ind);

            if *mc > mc_prev {
                cliq.push(u);
            }
        }
        else if sz > *mc {
            *mc = sz;
        }
    }
}

// #[derive(Default, Clone, Debug, Eq, PartialEq)]
// pub struct Vertex {
//     id: usize,
//     b: usize,
// }
// impl Ord for Vertex {
//     fn cmp(&self, other: &Self) -> Ordering {
//         (self.b).cmp(&other.b)
//     }
// }
// impl PartialOrd for Vertex {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         Some(self.cmp(other))
//     }
// }