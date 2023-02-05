/*
This code is based on https://github.com/ryanrossi/pmc and has just been
reimplemented in rust
*/
use parking_lot::RwLock;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::mpsc::channel;
use std::sync::mpsc::{Receiver, Sender};

/* this is the node amount bound. If we have more than 4_294_967_295 many nodes, change this to usize/u64 */
type Nab = u32;
//type AdjMtx = HashMap<NAB, Vec<NAB>>;

#[derive(Default, Clone, Debug, Eq, PartialEq)]
pub struct PmcGraph {
    /*edges is the concatenation over all vertices of all vertices each vertex is connected to
    so if we have the graph 1 - 2 - 3 - 1, then edges would be 2, 3, 1, 3, 1, 2 as 1 is connected to 2 and 3
    and 2 is connected to 1 and 3 and 3 is connected to 1 and 2
    */
    edges: Vec<Nab>,
    /* vertices is a list that at index j is the sum of the degrees of every vertex 0,...,j-2 */
    vertices: Vec<usize>,
    /* the degree of each vertex */
    degree: Vec<Nab>,
    /* the min and max degrees of the entire graph */
    pub min_degree: Nab,
    max_degree: Nab,
    //want f64 here but can't compare those in rust with the Eq trait, so instead use a fraction and to check will have to compare values but eh
    //avg_degree: (NAB,NAB),
    //is_gstats: bool,
    //we leave away the string fn
    //and the adjoint matrix adj for now
    max_core: Nab,
    kcore: Vec<Nab>,
    kcore_order: Vec<Nab>,
}

impl PmcGraph {
    pub fn new(verts: Vec<Nab>, edgs: Vec<(Nab, Nab)>) -> PmcGraph {
        let mut vertices: Vec<usize> = Vec::with_capacity(verts.len() + 1);
        let mut edges: Vec<Nab> = Vec::with_capacity(edgs.len() * 2);
        //let mut vert_list: AdjMtx = AdjMtx::new();
        //let mut current_start_vert: NAB = edgs[0].0;
        //let mut now: std::time::Instant = std::time::Instant::now();

        let vertlen = verts.len();
        let backup_verts = verts.clone();

        let mut vert_enum = verts
            .into_iter()
            .zip(0..(vertlen as Nab))
            .map(|p| (p.0, (p.1, vec![])))
            .collect::<HashMap<Nab, (Nab, Vec<Nab>)>>();

        for e in edgs.into_iter() {
            let val1 = vert_enum.get(&e.1).unwrap().0;
            let val0 = vert_enum.get_mut(&e.0).unwrap();

            val0.1.push(val1);

            let val00 = val0.0;

            let val11 = vert_enum.get_mut(&e.1).unwrap();
            val11.1.push(val00);
        }

        /* now we should have the graph in the hashmap and will convert it to the form we need it */
        vertices.push(edges.len());
        for ver in backup_verts {
            edges.extend(&vert_enum[&ver].1);
            vertices.push(edges.len());
        }
        //let mut elapsed_time = now.elapsed();
        // println!(
        //     "\tBuilding the graph struct from input took {} milliseconds.",
        //     elapsed_time.as_millis()
        // );
        let n = vertices.len();
        let mut g = PmcGraph {
            vertices,
            edges,
            degree: vec![0; n - 1],
            min_degree: 0,
            max_degree: 0,
            //avg_degree: (0,1), is_gstats: false,
            max_core: 0,
            kcore: vec![],
            kcore_order: vec![],
        };
        g.vertex_degrees();
        //now = std::time::Instant::now();
        g.compute_cores();
        //elapsed_time = now.elapsed();
        // println!(
        //     "\tComputing cores took {} milliseconds.",
        //     elapsed_time.as_millis()
        // );
        g
    }
    pub fn vertex_degrees(&mut self) {
        let n = self.vertices.len() - 1;

        // initialize min and max to degree of first vertex
        let mut max_degree: Nab = (self.vertices[1] - self.vertices[0]) as Nab;
        let mut min_degree: Nab = (self.vertices[1] - self.vertices[0]) as Nab;

        for v in 0..n {
            self.degree[v] = (self.vertices[v + 1] - self.vertices[v]) as Nab;
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
    }
    pub fn compute_cores(&mut self) {
        let n: usize = self.vertices.len();

        let mut pos: Vec<Nab> = vec![0; n];
        if !self.kcore_order.is_empty() {
            //let tmp: Vec<usize> = vec![0; n];
            self.kcore = vec![0; n];
            self.kcore_order = vec![0; n];
        } else {
            self.kcore_order.resize(n, 0);
            self.kcore.resize(n, 0);
        }

        let mut md: Nab = 0;

        for v in 1..n {
            self.kcore[v] = (self.vertices[v] - self.vertices[v - 1]) as Nab;
            if self.kcore[v] > md {
                md = self.kcore[v];
            }
        }

        let md_end: Nab = md + 1;

        let mut bin: Vec<Nab> = vec![0; md_end as usize];

        for v in 1..n {
            bin[self.kcore[v] as usize] += 1;
        }

        let mut start: Nab = 1;
        let mut num: Nab;

        for d in 0..md_end {
            num = bin[d as usize];
            bin[d as usize] = start;
            start += num;
        }

        for v in 1..n {
            pos[v] = bin[self.kcore[v] as usize];
            self.kcore_order[pos[v] as usize] = v as Nab;
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
            v = self.kcore_order[i] as usize;
            for j in self.vertices[v - 1]..self.vertices[v] {
                u = self.edges[j] as usize + 1;
                if self.kcore[u] > self.kcore[v] {
                    du = self.kcore[u] as usize;
                    pu = pos[u] as usize;
                    pw = bin[du] as usize;
                    w = self.kcore_order[pw] as usize;
                    if u != w {
                        pos[u] = pw as Nab;
                        self.kcore_order[pu] = w as Nab;
                        pos[w] = pu as Nab;
                        self.kcore_order[pw] = u as Nab;
                    }
                    bin[du] += 1;
                    self.kcore[u] -= 1;
                }
            }
        }
        for v in 0..n - 1 {
            self.kcore[v] = self.kcore[v + 1] + 1; // K + 1
            self.kcore_order[v] = self.kcore_order[v + 1] - 1;
        }
        self.max_core = self.kcore[self.kcore_order[n - 2] as usize] - 1;
    }
    pub fn search_bounds(&self) -> Vec<Nab> {
        let verts = &self.vertices;
        let edgs = &self.edges;
        let kcores = &self.kcore;
        let order = &self.kcore_order;
        let degree = &self.degree;

        let order_len = order.len();

        let ub = self.max_core + 1;
        let found_ub_glo = RwLock::new(false);

        let mc_glo: RwLock<Nab> = RwLock::new(0);

        let (sender_glo, receiver): (Sender<(Nab, Vec<Nab>)>, Receiver<(Nab, Vec<Nab>)>) =
            channel();

        /* here start threads */
        order.into_par_iter().rev().skip(1).for_each_with(sender_glo,|sender, w|{
            //let sender = sender_glo.clone();
            let found_ub = *found_ub_glo.read();
            if !found_ub {
                //drop(found_ub);
                //v = order[i];
                let v = *w;
                let mut pairs: Vec<(Nab,Nab)> = vec![];
                let mut clique: Vec<Nab> = vec![];
                let mc = *mc_glo.read();
                if kcores[v as usize] > mc {
                    for j in verts[v as usize]..verts[v as usize + 1] {
                        if kcores[edgs[j] as usize] > mc {
                            pairs.push((edgs[j], degree[edgs[j] as usize]));
                        }
                    }
                    /* Current largest clique size */
                    let mc_cur = mc;

                    if pairs.len() > mc_cur as usize{
                        /* If I want to get exactly the same result as the actual
                        pmc code I need to do stable sort and make pmc do stable sort as well */
                        pairs.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        /* I keep the old value, let the branch function modify
                        a copy of it */
                        let mut mc_cur_mut = mc_cur;
                        self.branch(pairs, 1, &mut mc_cur_mut, &mut clique);
                        /* If this branch func did better than before, send this to the receiver */
                        if mc_cur_mut > mc_cur {
                            clique.push(v);
                            sender.send((mc_cur_mut, clique)).unwrap();
                        }
                    }
                }
            }
        });
        let mut c_max: Vec<Nab> = vec![];
        for _ in 0..(order_len - 1) {
            /* We block until we get something from some thread */
            let results = receiver.recv().unwrap();

            let len = c_max.len();
            let reslen = results.0;
            /* If some thread found something better than the current
            largest clique */
            if len < reslen as usize {
                /* Edit the maxclique size*/
                let mut _mc_mut = *mc_glo.write();
                _mc_mut = reslen;

                c_max = results.1.clone();
                /* Note that we set the value of mc to be mc_cur
                and only this thread can edit, so this is okay */
                if reslen >= ub {
                    let mut _found_ub_mut = *found_ub_glo.write();
                    _found_ub_mut = true;
                }
                //println!("\tFound clique of len {}", c_max.len());
            }
            drop(results);
        }
        c_max
    }

    pub fn branch(&self, pairs: Vec<(Nab, Nab)>, sz: Nab, mc: &mut Nab, cliq: &mut Vec<Nab>) {
        if !pairs.is_empty() {
            let u = pairs[pairs.len() - 1].0;
            let verts = &self.vertices;
            let edgs = &self.edges;
            let kcores = &self.kcore;

            // for j in verts[u as usize]..verts[u as usize +1] {
            //     ind[edgs[j as usize] as usize] = true;
            // }
            // let mut remain: Vec<(NAB,NAB)>  = vec![];
            // for i in 0..pairs.len() {
            //     if ind[pairs[i as usize].0 as usize] {
            //         if kcores[pairs[i as usize].0 as usize] > *mc {
            //             remain.push(pairs[i as usize].clone());
            //         }
            //     }
            // }
            // for j in verts[u as usize]..verts[u as usize +1] {
            //     ind[edgs[j as usize] as usize] = false;
            // }

            /*
            A functional way to do the below imperative thing. It's slower,
            but I think more memory efficient.
            //let ind = (verts[u as usize]..verts[u as usize + 1]).map(|j| edgs[j]);

            let remain: Vec<(NAB, NAB)> = (0..pairs.len())
                .filter(|i| ind[pairs[*i].0 as usize] && kcores[pairs[*i].0 as usize] > *mc)
                .map(|i| pairs[i].clone())
                .collect();
            */

            let mut ind: Vec<bool> = vec![false; verts.len() - 1];
            for j in verts[u as usize]..verts[u as usize + 1] {
                ind[edgs[j] as usize] = true;
            }

            let mut remain: Vec<(Nab, Nab)> = vec![];
            for i in 0..pairs.len() {
                if ind[pairs[i].0 as usize] {
                    if kcores[pairs[i].0 as usize] > *mc {
                        remain.push(pairs[i]);
                    }
                }
            }

            drop(ind);
            drop(pairs);

            let mc_prev = *mc;
            self.branch(remain, sz + 1, mc, cliq);

            if *mc > mc_prev {
                cliq.push(u);
            }
        } else if sz > *mc {
            *mc = sz;
        }
    }
}