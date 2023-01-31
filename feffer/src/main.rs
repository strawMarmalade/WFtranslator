use na::{Vector6, Matrix6};
//use ndarray::{self, Array1, Array2};
use nalgebra as na;
// use rand::SeedableRng;
// use rand::{self, distributions, prelude::Distribution};
// use std::f32;
// use std::fs::File;
//use std::io::Write;
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
const DELTA: f32 = 0.7; //want C_{18}n\delta < C_{18}n 1/(n*20) = 6/20 


fn mu(x_point: &Vector6<f32>, q_point: &Vector6<f32>) -> f32 {
    let dist: f32 = (x_point - q_point).norm();
    (1.0 / (dist - 1.0 / 3.0)).exp()
        / ((1.0 / (dist - 1.0 / 3.0)).exp() + (1.0 / (1.0 / 2.0 - dist)).exp())
}

fn calc_dist_from1(x0_point: &Vector6<f32>, x_point: &Vector6<f32>, y_vals: &[Vector6<f32>]) -> f32 {
    if y_vals.len() != 0 {
        y_vals
            .into_iter()
            .map(|y| f32::abs(x_point.dot(y) / x_point.norm()))
            .reduce(f32::max)
            .unwrap()
            .max(
                f32::abs(
            1.0 - (x0_point - x_point).norm(),
            ))
    } else {
        f32::abs(1.0 - (x0_point - x_point).norm())
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

fn find_disc(x0_point: &Vector6<f32>, x_vals: &Vec<Vector6<f32>>, n: usize) -> Option<Vec<Vector6<f32>>> {
    match x_vals.len() {
        4 => return Some(x_vals.clone()),
        0..=3 => {
            //println!("Error, this ball has too few points in it");
            return None    
        }
        _ =>    {
            let mut output: Vec<Vector6<f32>> = vec![];
            let mut x_vals_copy = x_vals.clone();
            for _ in 0..n {
                let min_index = argmin(
                    &x_vals_copy
                        .iter()
                        .map(|x| calc_dist_from1(x0_point, &x, &output))
                        .collect::<Vec<f32>>()
                );
                output.push(x_vals_copy[min_index]);
                x_vals_copy.remove(min_index);
            }
            Some(output)
        }
    }
}

// fn unit_ball_around_x(x_point: &Vector6<f32>, points: &Vec<Vector6<f32>>) -> Vec<Vector6<f32>> {
//     //let points_clone = ;
//     // let mut output: Vec<Vector6<f32>> = vec![];
//     // for x in points {
//     //     if (x_point-x).norm() <= 1.0 {
//     //         output.push(x.clone());
//     //     }
//     // }
//     // output
//     points.clone().into_iter().filter(|x| (x_point-*x).norm() <= 1.0).collect()
// }


fn ball_rad_r(x_point: &Vector6<f32>, points: &Vec<Vector6<f32>>, r: f32) -> Vec<Vector6<f32>> {
    points.clone()
    .into_iter()
    .filter(|x| 0.0 < (x_point-*x).norm() && (x_point-*x).norm() <= r)
    .collect()
    // let mut output: Vec<Vector6<f32>> = vec![];
    // for p in points {
    //     if 0.0 < (x_point-p).norm() && 
    //         (x_point-p).norm() <= r {
    //             output.push(p.clone());
    //     }
    // }
    // output
}

fn points_dist(points_a: &Vec<Vector6<f32>>, points_b: &Vec<Vector6<f32>>) -> f32 {
    /*
    This goes thru every point in pointsA and checks if each point in b is var close to it
    */
    // !pointsA.iter().map(|a| pointsB.iter().map(|b| (a-b).norm() < var).any(|x| x == false)).any(|x| x == false)
    let mut max: f32 = 0.0;
    for a in points_a {
        let mut min = (a-points_b[0]).norm();
        for b in points_b {
            min = min.min((a-b).norm());
        }
        max = max.max(min);
    }
    max
}

fn find_r(arr: &Vec<Vector6<f32>>) -> Option<f32>{
    let mut r: f32 = 1.0;
    let mut r_abs_min: f32 = 0.0;
    let mut r_abs_max: f32 = 200.0;
    //let mut our_arr = arr.clone();
    //our_arr = our_arr.iter().map(|x| x*r).collect();

    for _ in 0..15 {
        println!("{}", r);
        let r_balls = arr.iter().map(|point| (point, ball_rad_r(point, &arr, r)));
        for ball in r_balls {
            let disc = find_disc(ball.0, &ball.1, 4);
            match disc {
                Some(x) => {
                    let mut aff = x.clone();
                    aff.push(*ball.0);
                    if points_dist(&ball.1,&aff) >= DELTA*r {
                        println!("Distance {} was > {}, have to increase r, ball size is {}, point is {}", points_dist(&ball.1,&aff), DELTA, ball.1.len(), ball.0);
                        //r_abs_min = r_abs_min.max(r);
                        r_abs_max = r_abs_max.min(r);
                        break;
                    }
                    //discs.push(x);
                },
                _ => { 
                    //r_abs_max = r_abs_max.min(r);
                    r_abs_min = r_abs_min.max(r);
                    println!("Found point {} a too small one:{}, have to decrease r", ball.0, ball.1.len());
                    break;
                }
            }
        }
        //arr = arr.iter().map(|x| x*(r_abs_max + r_abs_min)/2.0/r).collect();
        if r == (r_abs_max + r_abs_min)/2.0 {
            return Some(r);
        }
        r = (r_abs_max + r_abs_min)/2.0;
    }
    let r_balls = arr.iter().map(|point| (point, ball_rad_r(point, &arr, r)));
    for ball in r_balls {
        let disc = find_disc(ball.0, &ball.1, 4);
        match disc {
            Some(x) => { 
                if points_dist(&ball.1,&x) >= DELTA*r {
                    println!("You have to increase DELTA for this to work.");
                    r_abs_max = r_abs_max.min(r);
                    return None;
                }
                //discs.push(x);
            },
            _ => { 
                r_abs_min = r_abs_min.max(r);
                //println!("Found a too small one:{}, have to increase r", ball.1.len());
                return None;
            }
        }
    }
    Some(r)
}

fn get_qs(max_clique: Vec<NAB>, points: &Vec<Vector6<f32>>) -> (Vec<Vector6<f32>>, Vec<Matrix6<f32>>) {
    let points_q = max_clique.iter().map(|p| points[*p as usize]);
    let points_clone = points_q.clone();
    let x_ones = points_q.clone().map(|q| ball_rad_r(&q, points,  40.0));
    let spaces_q = points_q.zip(x_ones).map(|(q,x)| na::Matrix6x4::from_columns(&find_disc(&q, &x, 4).unwrap()));
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

fn define_f<'a>(arr_at_clique: &'a Vec<Vector6<f32>>, mats: &'a Vec<Matrix6<f32>>) 
    -> impl Fn(Vector6<f32>) -> Vector6<f32> + 'a {
    /*
    The code below is the rust equivalent of this python code:
    def gettingF(points_q, Qs):
        Ps = [lambda y: Qs[val]@y+points_q[val] for val in range(len(points_q))]
        mus = [lambda y: mu(y,q) for q in points_q]
        phis = [lambda y: mus[val](y)*Ps[val](y)+(1-mus[val](y))*y for val in range(len(points_q))]
        return lambda y: fFromPhis(y, phis=phis)
 
    def fFromPhis(y0, phis):
        y = y0
        for phi in phis:
            y = phi(y)
        return y
    */
    let projs = 
            arr_at_clique.into_iter()
            .zip(mats)
            .map(|(arr,mat)| 
                {move |y: Vector6<f32>| (mat*y + arr,mu(&y, arr))});
    let phis = projs.map(|func| {move |y: Vector6<f32>| func(y).1*func(y).0 + (1.0-func(y).1)*y});
    move |y: Vector6<f32>| phis.clone().fold(y, move|acc, phi| phi(acc))
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

    let divisor_r: f32 = 4000.0;

    let now = std::time::Instant::now();
    let arr = read_file_to_mat(file_path);
    find_r(&arr);

    // let mut elapsed_time = now.elapsed();
    // let len = arr.len();
    // println!(
    //     "Reading the file of points took {} milliseconds and we have {} many points",
    //     elapsed_time.as_millis(),
    //     len
    // );
    // if chunk_size > 0 {
    //     println!("We are in chunking mode with chunk size {}", chunk_size);
    // }
    // else {
    //     println!("We are in non-chunking mode");
    //     chunk_size = len as NAB + 1;
    // }
    // let max_clique = chunk_clique(chunk_size, &(0..(len as NAB)).collect::<Vec<NAB>>(), &arr, divisor_r, 2);

    // println!("Found max clique of len {}: {:?}", max_clique.len(), max_clique);

    // let now2 = std::time::Instant::now();
    // let vals = get_qs(max_clique, &arr);
    // let elapsed_time2 = now2.elapsed();
    // println!("It took {} micro seconds to calc {} qr decomps", elapsed_time2.as_micros(), vals.0.len());

    // let func = define_f(&vals.0, &vals.1);
    // let start: Vector6<f32> = Vector6::from_vec(vec![0.04,1.0,32.0,1.0,0.0,0.345]);
    // println!("{}", func(start));

    // // let path = "QQTvals2.txt";
    // // let mut output = File::create(path).unwrap();
    // // for val in vals.1 {
    // //     writeln!(output, "{:?}", val).unwrap();
    // // }

    // elapsed_time = now.elapsed();
    // println!("The total process took {} seconds.", elapsed_time.as_secs());
}
