use ndarray::{self, Array1};
// use rand::SeedableRng;
// use rand::{self, distributions, prelude::Distribution};
// use std::f32;
// use std::fs::File;
// use std::io::Write;
//use rand_chacha::ChaCha8Rng;
//use threadpool::ThreadPool;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

//use std::time;
mod pmcgraph;
//mod graph;
//mod backtracking;
//mod branch_and_bound;

//use crate::graph::Graph;
use crate::pmcgraph::PmcGraph;


type NAB = u32;
static DIM: usize = 2;

// /// Solves the maximum clique problem by using a branch and bound.
// pub fn solve_branch_and_bound(graph: &Graph) -> Graph {
//     branch_and_bound(&graph, &graph.nodes_ord_by_degree(), Graph::default())
// }

// /// Solves the maximum clique problem by using a backtracking.
// pub fn solve_backtracking(graph: &Graph) -> Graph {
//     backtracking(&graph, &graph.nodes(), Graph::default())
// }

// fn branch_and_bound(graph: &Graph, nodes: &[usize], mut clique: Graph) -> Graph {
//     // Clone current solution
//     let mut subgraph = clique.clone();
//     // Visit all nodes
//     for (i, &n) in nodes.iter().enumerate() {
//         // Prune branch if the current `k`-clique subgraph cannot increase
//         if clique.degree() >= graph.degree_of(n) {
//             break;
//         }
//         // Add node
//         subgraph.insert_node(n);
//         // Add edges
//         for c in subgraph.nodes() {
//             if graph.adjlst_of(n).contains(&c) {
//                 subgraph.insert_edge((c, n));
//             }
//         }
//         // Create a search branch and get the branch best solution
//         let sol = branch_and_bound(graph, &nodes[i + 1..], subgraph.clone());
//         // Check if the branch best solution is better than the current one
//         if (sol.is_complete() && clique.is_empty())
//             || (sol.is_complete() && sol.degree() > clique.degree())
//         {
//             clique = sol;
//         }
//         // Remove added node
//         subgraph.remove_node(n);
//     }
//     clique
// }

// fn backtracking(graph: &Graph, nodes: &[usize], mut clique: Graph) -> Graph {
//     // Clone current solution
//     let mut subgraph = clique.clone();
//     // Visit all nodes
//     for (i, n) in nodes.iter().enumerate() {
//         // Add node
//         subgraph.insert_node(*n);
//         // Add edges
//         for c in subgraph.nodes() {
//             if graph.adjlst_of(*n).contains(&c) {
//                 subgraph.insert_edge((c, *n));
//             }
//         }
//         // Create a backtracking branch and get the branch best solution
//         let sol = backtracking(graph, &nodes[i + 1..], subgraph.clone());
//         // Check if the branch best solution is better than the current one
//         if (sol.is_complete() && clique.is_empty())
//             || (sol.is_complete() && sol.degree() >= clique.degree())
//         {
//             clique = sol;
//         }
//         // Remove added node
//         subgraph.remove_node(*n);
//     }
//     clique
// }

// /// Redirects the graph to the selected solver, run it and return a maximum
// /// clique subgraph.
// pub fn solve(graph: &Graph, solver: &Solver) -> Result<Graph, &'static str> {
//   // Check if the graph is empty
//   if graph.is_empty() { return Err("the graph is empty") }
//   // If the graph has only one node return the graph
//   if graph.nlen() == 1 { return Ok(graph.clone()) }
//   // If the graph has two nodes and only one edge return the graph
//   if graph.nlen() == 2 && graph.elen() == 1 { return Ok(graph.clone()) }
//   // If the graph degree is two return a adjacent pair of nodes
//   if graph.degree() == 2 && graph.elen() <= 2 {
//     let mut solution = Graph::default();
//     let n1 = graph.nodes()[0];
//     let n2 = graph.adjlst_of(n1)[0];
//     solution.insert_node(n1);
//     solution.insert_node(n2);
//     solution.insert_edge((n1, n2));
//     return Ok(solution)
//   }
//   // Run solver and return solution
//   Ok(match solver {
//     Solver::Backtracking => backtracking::solve(&graph),
//     Solver::BranchAndBound => branch_and_bound::solve(&graph),
//   })
// }

// fn dot_product(a: &[f32], b: &[f32]) -> f32 {
//     // Calculate the dot product of two vectors.
//     assert_eq!(a.len(), b.len());
//     let mut product: f32 = 0.0;
//     for i in 0..a.len() {
//         product += a[i] * b[i];
//     }
//     return product;
// }

// fn dist(x_point: &[f32], q_point: &[f32]) -> f32{
//     let diff = x_point - q_point;
//     let sum: f32 = dot_product(diff,diff);
//     return sum.sqrt();
// }

// fn maxDotFunc(x_point: &Array1<f32>, q_point: &Array1<f32>) -> f32 {
//     f32::abs(x_point.dot(q_point)/x_point.dot(x_point))
// }

#[allow(dead_code)]
fn mu(x_point: &Array1<f32>, q_point: &Array1<f32>) -> f32 {
    let diff: Array1<f32> = x_point - q_point;
    let dist: f32 = diff.dot(&diff);
    (1.0 / (dist - 1.0 / 3.0)).exp()
        / ((1.0 / (dist - 1.0 / 3.0)).exp() + (1.0 / (1.0 / 2.0 - dist)).exp())
}
#[allow(dead_code)]
fn calc_dist_from1(x0_point: &Array1<f32>, x_point: &Array1<f32>, y_vals: &[Array1<f32>]) -> f32 {
    if y_vals.len() != 0 {
        let max_y: f32 = y_vals
            .into_iter()
            .map(|y| f32::abs(x_point.dot(y) / x_point.dot(x_point)))
            .reduce(f32::max)
            .unwrap();
        max_y.max(f32::abs(
            1.0 - (x0_point - x_point).dot(&(x0_point - x_point)),
        ))
    } else {
        f32::abs(1.0 - (x0_point - x_point).dot(&(x0_point - x_point)))
    }
}

#[allow(dead_code)]
fn argmin(nets: &Vec<f32>) -> usize {
    *nets
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .get_or_insert(0_usize)
}

#[allow(dead_code)]
fn find_disc2(x0_point: &Array1<f32>, x_vals: &[Array1<f32>], n: u32) -> Vec<Array1<f32>> {
    let mut output: Vec<Array1<f32>> = vec![];
    for _i in 0..n {
        output.append(&mut vec![x_vals[argmin(
            &mut x_vals
                .into_iter()
                .map(|x| calc_dist_from1(x0_point, x, &output))
                .collect::<Vec<f32>>(),
        )]
        .clone()]);
    }
    output
}

#[allow(dead_code)]
fn find_disc(x0_point: &Array1<f32>, x_vals: &[Array1<f32>], n: u32) -> Vec<Array1<f32>> {
    let mut output: Vec<Array1<f32>> = vec![];
    for _i in 0..n {
        let distance_vec: Vec<f32> = x_vals
            .into_iter()
            .map(|x| calc_dist_from1(x0_point, x, &output))
            .collect::<Vec<f32>>();
        let mut current_min: f32 = distance_vec[0];
        let mut current_index: usize = 0;
        for j in 0..distance_vec.len() {
            if distance_vec[j] < current_min {
                current_index = j;
                current_min = distance_vec[j];
            }
        }
        output.append(&mut vec![x_vals[current_index].clone()]);
    }
    output
}

fn main() {
    // let range: distributions::Uniform<f32> = distributions::Uniform::from(0.0..1.0);
    // // let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
    // const AMOUNT: usize = 5000;
    // let mut rng = ChaCha8Rng::seed_from_u64(42);
    // let mut now: std::time::Instant = std::time::Instant::now();
    // let x_vals: [Array1<f32>; AMOUNT] =
    //     [(); AMOUNT].map(|_| range.sample_iter(&mut rng).take(DIM).collect());
    // let mut elapsed_time = now.elapsed();
    // println!(
    //     "Generating random points took {} milliseconds.",
    //     elapsed_time.as_millis()
    // );
    let mut now = std::time::Instant::now();
    let split_size: NAB = 5000;

    let f = BufReader::new(File::open("/home/leo/Documents/work/WFtranslator/feffer/dataMat10.txt").unwrap());

    let arr: Vec<Array1<f32>> = f.lines()
        .map(|l| l
            .unwrap()
            .split_whitespace()
            .map(|number| number.parse().unwrap())
            .collect()
        )
        .collect();
    let mut elapsed_time = now.elapsed();
    println!(
        "Reading the file of points took {} milliseconds.",
        elapsed_time.as_millis()
    );
    let len = arr.len();
    let split_yes = true;
    let divisor_r: f32 = 200.0;
    if split_yes {
        println!("We are in chunking mode");
        let amt_splits = f32::ceil((len as f32)/(split_size as f32)) as NAB;
    
        // let path = "matrix3.txt";
        // let mut output = File::create(path).unwrap();
    
        let mut collected_verts: Vec<NAB> = vec![];
        let now_glo = std::time::Instant::now();
        now = std::time::Instant::now();
        for cur_split in 0..amt_splits {
            if cur_split != amt_splits -1 {
                let mut edgs: Vec<(NAB,NAB)> = vec![];
                let verts = (0..split_size).collect();
                for j in 0..split_size {
                    for k in 0..j {
                        let v1 = &arr[(j+split_size*cur_split) as usize];
                        let v2 = &arr[(k+split_size*cur_split) as usize];
                        let diff: Array1<f32> = (v1 - v2).iter().map(|coord| *coord/divisor_r).collect();
                        let dist: f32 = diff.dot(&diff);
                        if dist >= 1.0/100.0 {
                            //write!(output, "{} {}\n", (k+1).to_string(), (j+1).to_string()).unwrap();
                            edgs.push((j,k));
                        }
                    }
                }
                let graph = PmcGraph::new(verts, edgs);
                let now2 = std::time::Instant::now();
                collected_verts.extend(graph.search_bounds().into_iter().map(|val| val+split_size*cur_split));
                elapsed_time = now2.elapsed();
                println!(
                    "\tIt took {} milliseconds to compute the clique in chunk {}",
                    elapsed_time.as_millis(), cur_split
                );
            }
            else {
                let verts: Vec<NAB> = (0..(len as NAB - split_size*(amt_splits-1))).collect();
                let mut edgs: Vec<(NAB,NAB)> = vec![];
                for j in 0..(len as NAB - split_size*cur_split) {
                    for k in 0..j {
                        let v1 = &arr[(j+split_size*cur_split) as usize];
                        let v2 = &arr[(k+split_size*cur_split) as usize];
                        let diff: Array1<f32> = (v1 - v2).iter().map(|coord| *coord/divisor_r).collect();
                        let dist: f32 = diff.dot(&diff);
                        if dist >= 1.0/100.0 {
                            //write!(output, "{} {}\n", (k+1).to_string(), (j+1).to_string()).unwrap();
                            edgs.push((j,k));
                        }
                    }
                }
                let graph = PmcGraph::new(verts, edgs);
                let now2 = std::time::Instant::now();
                collected_verts.extend(graph.search_bounds().into_iter().map(|val| val+split_size*cur_split));
                elapsed_time = now2.elapsed();
                println!(
                    "\tIt took {} milliseconds to compute the clique in chunk {}",
                    elapsed_time.as_millis(), cur_split
                );
            }
        }
        elapsed_time = now.elapsed();
        println!(
            "It took {} seconds to get all of the separate cliques in each chunk\n",
            elapsed_time.as_secs(),
        );
        now = std::time::Instant::now();
        let col_len = collected_verts.len();
        let mut col_edgs: Vec<(NAB,NAB)> = vec![];
        for j in 0..col_len {
            for k in 0..j {
                let v1 = &arr[collected_verts[j] as usize];
                let v2 = &arr[collected_verts[k] as usize];
                let diff: Array1<f32> = (v1 - v2).iter().map(|coord| *coord/divisor_r).collect();
                let dist: f32 = diff.dot(&diff);
                if dist >= 1.0/100.0 {
                    col_edgs.push((j as NAB, k as NAB));
                }
            }
        }
        let graph = PmcGraph::new((0..(col_len as NAB)).collect::<Vec<NAB>>(), col_edgs);
        elapsed_time = now.elapsed();
        println!(
            "It took {} milliseconds to build the final graph",
            elapsed_time.as_millis(),
        );
        now = std::time::Instant::now();
        if graph.min_degree == col_len as NAB - 1 {
            println!("Clique of len {} is {:?}", collected_verts.len(), collected_verts);
        }
        else {
            let clique: Vec<NAB> = graph.search_bounds().into_iter().map(|val| collected_verts[val as usize]).collect();
            println!("Clique of len {} is {:?}", clique.len(), clique);
        }
        elapsed_time = now.elapsed();
        println!(
            "It took {} milliseconds to find the final max clique",
            elapsed_time.as_millis(),
        );
        elapsed_time = now_glo.elapsed();
        println!(
            "\nIn total the entire process took {} seconds",
            elapsed_time.as_secs(),
        );
    }
    else {
        println!("We are in non-chunking mode");

        let verts: Vec<NAB> = (0..(len as NAB)).collect();
        let mut edgs: Vec<(NAB,NAB)> = vec![];
    
        //let path = "matrix2.txt";
        //let mut output = File::create(path).unwrap();
        let now_glo = std::time::Instant::now();

        let mut now = std::time::Instant::now();
        for j in 0..len {
            for k in 0..j {
                let v1 = &arr[j];
                let v2 = &arr[k];
                let diff: Array1<f32> = (v1 - v2).iter().map(|coord| *coord/divisor_r).collect();
                let dist: f32 = diff.dot(&diff);
                if dist >= 1.0/100.0 {
                    edgs.push((j as NAB,k as NAB));
                }
            }
        }    
        let graph: PmcGraph = PmcGraph::new(verts, edgs);
        let mut elapsed_time = now.elapsed();
        println!(
            "Building the graph took {} milliseconds.",
            elapsed_time.as_millis()
        );
        now = std::time::Instant::now();
        let clique = graph.search_bounds();
        println!("Clique of size {} is {:?}", clique.len(), clique);
        elapsed_time = now.elapsed();
        println!(
            "Finding max clique took {} milliseconds.",
            elapsed_time.as_millis()
        );
        elapsed_time = now_glo.elapsed();
        println!(
            "The total process took {} seconds.",
            elapsed_time.as_secs()
        );
    }
}
