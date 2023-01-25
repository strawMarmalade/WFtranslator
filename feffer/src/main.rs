use ndarray::{self, Array1};
use rand::SeedableRng;
use rand::{self, distributions, prelude::Distribution};
use std::f32;
use std::fs::File;
use std::io::Write;
use rand_chacha::ChaCha8Rng;

//use std::time;
mod pmcgraph;
//mod graph;
//mod backtracking;
//mod branch_and_bound;

//use crate::graph::Graph;
use crate::pmcgraph::PmcGraph;

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
    let range: distributions::Uniform<f32> = distributions::Uniform::from(0.0..1.0);
    // let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
    const AMOUNT: usize = 5000;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut now: std::time::Instant = std::time::Instant::now();
    let x_vals: [Array1<f32>; AMOUNT] =
        [(); AMOUNT].map(|_| range.sample_iter(&mut rng).take(DIM).collect());
    let mut elapsed_time = now.elapsed();
    println!(
        "Generating random points took {} milliseconds.",
        elapsed_time.as_millis()
    );

    let verts: Vec<usize> = (0..AMOUNT).collect();
    let mut edgs: Vec<(usize,usize)> = vec![];

    //let path = "matrix2.txt";
    //let mut output = File::create(path).unwrap();

    now = std::time::Instant::now();
    for j in 0..AMOUNT {
        for k in 0..j {
            let diff: Array1<f32> = x_vals[j].clone() - x_vals[k].clone();
            let dist: f32 = diff.dot(&diff);
            //println!("{}", dist);
            if dist >= 1.0/100.0 {
                edgs.push((j,k));
                //write!(output, "{} {}\n", (k+1).to_string(), (j+1).to_string()).unwrap();
                //graph.insert_edge((k + 1, j + 1));
                //graph.insert_edge((j + 1, k + 1));
            }
        }
    }
    elapsed_time = now.elapsed();
    println!(
        "Calculating the edges took {} milliseconds.",
        elapsed_time.as_millis()
    );

    let graph: PmcGraph = PmcGraph::new(verts, edgs);
    now = std::time::Instant::now();
    let clique = graph.search_bounds();
    println!("Clique of size {} is {:?}", clique.len(), clique);
    elapsed_time = now.elapsed();
    println!(
        "Finding max clique took {} milliseconds.",
        elapsed_time.as_millis()
    );


    elapsed_time = now.elapsed();
    println!(
        "Generating graph took {} milliseconds.",
        elapsed_time.as_millis()
    );

    // now = std::time::Instant::now();
    // //let max_clique = solve_branch_and_bound(&graph);
    // //let _nodes_of_clique = max_clique.nodes();
    // elapsed_time = now.elapsed();
    // println!(
    //     "Generating max clique took {} seconds.",
    //     elapsed_time.as_secs()
    // );

    // let xs: Array1<f32> = range.sample_iter(&mut rng).take(DIM).collect();
    // let mut now: std::time::Instant = std::time::Instant::now();
    // let res1: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>>> = find_disc(&xs, &x_vals, 60);
    // let elapsed_time = now.elapsed();
    // println!("Running slow_function() took {} seconds.", elapsed_time.as_millis());
    // now = std::time::Instant::now();
    // let res2: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>>> = find_disc2(&xs, &x_vals, 60);
    // let elapsed_time = now.elapsed();

    // println!("Running slow_function() took {} seconds.", elapsed_time.as_millis());
    // println!("First objects are {:?} and {:?}", res1[0], res2[0])
}
