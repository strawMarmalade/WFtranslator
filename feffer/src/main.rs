use na::{Vector6, Matrix6};
//use ndarray::{self, Array1, Array2};
use nalgebra as na;
// use rand::SeedableRng;
// use rand::{self, distributions, prelude::Distribution};
// use std::f32;
// use std::fs::File;
use std::io::Write;
//use rand_chacha::ChaCha8Rng;
//use threadpool::ThreadPool;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
//use std::iter::{zip, repeat};
//use ndarray_linalg::norm::Norm;

//use std::time;
mod pmcgraph;
//mod graph;
//mod backtracking;
//mod branch_and_bound;

//use crate::graph::Graph;
use crate::pmcgraph::PmcGraph;

type NAB = u32;

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

// #[allow(dead_code)]
// fn mu(x_point: &Array1<f32>, q_point: &Array1<f32>) -> f32 {
//     let diff: Array1<f32> = x_point - q_point;
//     let dist: f32 = diff.dot(&diff);
//     (1.0 / (dist - 1.0 / 3.0)).exp()
//         / ((1.0 / (dist - 1.0 / 3.0)).exp() + (1.0 / (1.0 / 2.0 - dist)).exp())
// }

fn calc_dist_from1(x0_point: &Vector6<f32>, x_point: &Vector6<f32>, y_vals: &[Vector6<f32>]) -> f32 {
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

fn argmin(nets: &Vec<f32>) -> usize {
    nets
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap()
}

fn find_disc(x0_point: &Vector6<f32>, x_vals: &[Vector6<f32>], n: u32) -> Vec<Vector6<f32>> {
    let mut output: Vec<Vector6<f32>> = vec![];
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

fn unit_ball_around_x(x_point: &Vector6<f32>, points: &[Vector6<f32>]) -> Vec<Vector6<f32>> {
    let mut output: Vec<Vector6<f32>> = vec![];
    for x in points {
        if (x_point-x).norm() <= 1.0 {
            output.push(x.clone());
        }
    }
    output
}

fn get_qs(max_clique: Vec<NAB>, points: &[Vector6<f32>]) -> (Vec<Vector6<f32>>, Vec<Matrix6<f32>>) {
    let points_q = max_clique.iter().map(|p| points[*p as usize]);
    let points_clone = points_q.clone();
    let x_ones = points_q.clone().map(|q| unit_ball_around_x(&q, points));
    let spaces_q = points_q.zip(x_ones).map(|(q,x)| na::Matrix6x4::from_columns(&find_disc(&q, &x, 4)));
    let projs = spaces_q
        .map(|mat| 
            na::linalg::QR::new(mat).q())
            .map(|q| q*q.transpose());
    (points_clone.collect(), projs.collect())
}


// #[allow(dead_code)]
// fn find_disc(x0_point: &Array1<f32>, x_vals: &[Array1<f32>], n: u32) -> Vec<Array1<f32>> {
//     let mut output: Vec<Array1<f32>> = vec![];
//     for _i in 0..n {
//         let distance_vec: Vec<f32> = x_vals
//             .into_iter()
//             .map(|x| calc_dist_from1(x0_point, x, &output))
//             .collect::<Vec<f32>>();
//         let mut current_min: f32 = distance_vec[0];
//         let mut current_index: usize = 0;
//         for j in 0..distance_vec.len() {
//             if distance_vec[j] < current_min {
//                 current_index = j;
//                 current_min = distance_vec[j];
//             }
//         }
//         output.append(&mut vec![x_vals[current_index].clone()]);
//     }
//     output
// }

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

fn chunk_clique(chunk_size: u32, verts: &Vec<NAB>, arr: &Vec<Vector6<f32>>, divisor_r: f32, increase_factor: u32) -> Vec<NAB> {
    let mut collected_verts: Vec<NAB> = vec![];
    let chunks = verts.chunks(chunk_size as usize);

    let chunk_amount = chunks.len();

    for chunk in chunks {
        let chunk_len = chunk.len();
        let mut edgs: Vec<(NAB, NAB)> = vec![];
        for j in 0..chunk_len {
            for k in 0..j {
                if (&arr[chunk[j] as usize] - &arr[chunk[k] as usize])
                    .norm()    
                    >= divisor_r / 100.0
                {
                    edgs.push((chunk[j] as NAB, chunk[k] as NAB));
                }
            }
        }
        println!(
            "\tWe have #verts= {}, #edges={}, density={}.",
            chunk_len,
            edgs.len(),
            ((2 * edgs.len()) as f32) / ((chunk_len as f32) * (chunk_len - 1) as f32)
        );

        let graph = PmcGraph::new(chunk.to_vec(), edgs);
        let now2 = std::time::Instant::now();
        collected_verts.extend(
            graph
                .search_bounds()
                .into_iter()
                .map(|val| chunk[val as usize]),
        );
        let elapsed_time = now2.elapsed();
        println!(
            "\tIt took {} milliseconds to compute the clique\n",
            elapsed_time.as_millis(),
        );
    }
    if chunk_amount == 1 {
        return collected_verts;
    }
    chunk_clique(chunk_size*increase_factor, &collected_verts, arr, divisor_r, increase_factor)
}

fn read_file_to_mat(file_path: &String) -> Vec<Vector6<f32>> {
    let f = BufReader::new(File::open(file_path).unwrap());

    f
        .lines()
        .map(|l| {
            Vector6::from_iterator(
            l.unwrap()
                .split_whitespace()
                .map(|number| number.parse::<f32>().unwrap())
            )
        })
        .collect()
}


fn main() {
    let args: Vec<String> = env::args().collect();
    let mut chunk_size: NAB = 0;

    match args.len() {
        1 => println!("Give me a file!"),
        2 => (), //corresponds to just a filename so then we don't chunk
        _ => chunk_size = args[2].parse::<usize>().unwrap() as NAB,
    }

    let file_path = &args[1];

    let divisor_r: f32 = 200.0;

    let now = std::time::Instant::now();
    let arr = read_file_to_mat(file_path);
    let mut elapsed_time = now.elapsed();
    let len = arr.len();
    println!(
        "Reading the file of points took {} milliseconds and we have {} many points",
        elapsed_time.as_millis(),
        len
    );
    if chunk_size > 0 {
        println!("We are in chunking mode with chunk size {}", chunk_size);
    }
    else {
        println!("We are in non-chunking mode");
        chunk_size = len as NAB + 1;
    }
    let max_clique = chunk_clique(chunk_size, &(0..(len as NAB)).collect::<Vec<NAB>>(), &arr, divisor_r, 2);

    println!("Found max clique of len {}: {:?}", max_clique.len(), max_clique);

    let now2 = std::time::Instant::now();
    let vals = get_qs(max_clique, &arr);
    let elapsed_time2 = now2.elapsed();
    println!("It took {} micro seconds to calc {} qr decomps", elapsed_time2.as_micros(), vals.0.len());

    let path = "QQTvals2.txt";
    let mut output = File::create(path).unwrap();
    for val in vals.1 {
        writeln!(output, "{:?}", val).unwrap();
    }

    elapsed_time = now.elapsed();
    println!("The total process took {} seconds.", elapsed_time.as_secs());
}
