use argmin::core::{CostFunction, Error, Gradient, Executor, State, Hessian};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use plotters::prelude::*;
use argmin::solver::trustregion::TrustRegion;
use na::{Vector6, Vector4, Matrix4x3, Matrix4};
use finitediff::FiniteDiff;
//use ndarray::{self, Array1, Array2};
use nalgebra as na;
use ndarray::{Array1, array};
// use rand::SeedableRng;
// use rand::{self, distributions, prelude::Distribution};
// use std::f32;
// use std::fs::File;
//use std::io::Write;
//use rand_chacha::ChaCha8Rng;
//use threadpool::ThreadPool;
use std::env;
use std::f64::consts::PI;

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
type FLO = f64;
const DELTA: FLO = 2000.0*0.016; //want C_{18}n\delta < C_{18}n 1/(n*20) = 6/20 
//const DIV: FLO = 1.0;//1.0;

fn mu(x_point: &Vector4<FLO>, q_point: &Vector4<FLO>) -> FLO {
    let dist: FLO = (x_point - q_point).norm();
    (1.0 / (dist - 1.0 / 3.0)).exp()
        / ((1.0 / (dist - 1.0 / 3.0)).exp() + (1.0 / (1.0 / 2.0 - dist)).exp())
}

fn calc_dist_from1(x0_point: &Vector6<FLO>, x_point: &Vector6<FLO>, y_vals: &[Vector6<FLO>]) -> FLO {
    if y_vals.len() != 0 {
        y_vals
            .into_iter()
            .map(|y| FLO::abs(x_point.dot(y) / x_point.norm()))
            .reduce(FLO::max)
            .unwrap()
            .max(
                FLO::abs(
            1.0 - (x0_point - x_point).norm(),
            ))
    } else {
        FLO::abs(1.0 - (x0_point - x_point).norm())
    }
}

fn argmin(nets: &Vec<FLO>) -> usize {
    nets
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap()
}

fn find_disc(x0_point: &Vector6<FLO>, x_vals: &Vec<Vector6<FLO>>, n: usize) -> Option<Vec<Vector6<FLO>>> {
    match x_vals.len() {
        4 => return Some(x_vals.clone()),
        0..=3 => {
            //println!("Error, this ball has too few points in it");
            return None    
        }
        _ =>    {
            let mut output: Vec<Vector6<FLO>> = vec![];
            let mut x_vals_copy = x_vals.clone();
            for _ in 0..n {
                let min_index = argmin(
                    &x_vals_copy
                        .iter()
                        .map(|x| calc_dist_from1(x0_point, &x, &output))
                        .collect::<Vec<FLO>>()
                );
                output.push(x_vals_copy[min_index]);
                x_vals_copy.remove(min_index);
            }
            Some(output)
        }
    }
}

// fn unit_ball_around_x(x_point: &Vector6<FLO>, points: &Vec<Vector6<FLO>>) -> Vec<Vector6<FLO>> {
//     //let points_clone = ;
//     // let mut output: Vec<Vector6<FLO>> = vec![];
//     // for x in points {
//     //     if (x_point-x).norm() <= 1.0 {
//     //         output.push(x.clone());
//     //     }
//     // }
//     // output
//     points.clone().into_iter().filter(|x| (x_point-*x).norm() <= 1.0).collect()
// }

fn ball_rad_r(x_point: &Vector6<FLO>, points: &Vec<Vector6<FLO>>, r: FLO) -> Vec<Vector6<FLO>> {
    points.clone()
    .into_iter()
    .filter(|x| 0.0 < (x_point-*x).norm() && (x_point-*x).norm() <= r)
    .collect()
    // let mut output: Vec<Vector6<FLO>> = vec![];
    // for p in points {
    //     if 0.0 < (x_point-p).norm() && 
    //         (x_point-p).norm() <= r {
    //             output.push(p.clone());
    //     }
    // }
    // output
}

fn points_dist(points_a: &Vec<Vector6<FLO>>, points_b: &Vec<Vector6<FLO>>) -> FLO {
    /*
    This goes thru every point in pointsA and checks if each point in b is var close to it
    */
    // !pointsA.iter().map(|a| pointsB.iter().map(|b| (a-b).norm() < var).any(|x| x == false)).any(|x| x == false)
    let mut max: FLO = 0.0;
    for a in points_a {
        let mut min = (a-points_b[0]).norm();
        for b in points_b {
            min = min.min((a-b).norm());
        }
        max = max.max(min);
    }
    max
}

fn find_r(arr: &Vec<Vector6<FLO>>) -> Option<FLO>{
    let mut r: FLO = 1.0;
    let mut r_abs_min: FLO = 0.0;
    let mut r_abs_max: FLO = 200.0;
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

fn get_qs(max_clique: Vec<NAB>, points: &Vec<Vector6<FLO>>) -> (Vec<Vector6<FLO>>, Vec<Matrix4<FLO>>) {
    let points_q = max_clique.iter().map(|p| points[*p as usize]);
    let points_clone = points_q.clone();
    //let x_ones = points_q.clone().map(|q| ball_rad_r(&q, points,  40.0));
    //let spaces_q = points_q.zip(x_ones).map(|(q,x)| na::Matrix6x4::from_columns(&find_disc(&q, &x, 4).unwrap()));
    let projs = points_q.map(|q| proj_from_normal(q));//spaces_q
        //.map(|mat| 
        //    na::linalg::QR::new(mat).q())
        //    .map(|q| q*q.transpose());
    (points_clone.collect(), projs.collect())
}

fn proj_from_normal(point: Vector6<FLO>) -> Matrix4<FLO>{
    let normal: Vector4<FLO> = Vector4::from_vec(vec![FLO::cos(point[2]/180.0*PI),FLO::sin(point[2]/190.0*PI),-FLO::cos(point[5]/180.0*PI),-FLO::sin(point[5]/180.0*PI)]);
    println!("normal: {}", normal);
    let mat: Matrix4x3<FLO> = Matrix4x3::new(
        -normal[1],-normal[2],-normal[3],
        normal[0],normal[3],-normal[2],
        -normal[3],normal[0],normal[1],
        normal[2],-normal[1], normal[0]
    );
    let q = na::linalg::QR::new(mat).q();//.map(|q| q*q.transpose())
    q*q.transpose()
}

fn chunk_clique(chunk_size: u32, verts: &Vec<NAB>, arr: &Vec<Vector6<FLO>>, divisor_r: FLO, increase_factor: u32) -> Vec<NAB> {
    let mut collected_verts: Vec<NAB> = vec![];
    let chunks = verts.chunks(chunk_size as usize);

    let chunk_amount = chunks.len();

    for chunk in chunks {
        let chunk_len = chunk.len();
        let mut edgs: Vec<(NAB, NAB)> = vec![];
        for j in 0..chunk_len {
            for k in 0..j {
                if 100.0*(&arr[chunk[j] as usize] - &arr[chunk[k] as usize])
                    .norm()    
                    >= divisor_r 
                {
                    edgs.push((chunk[j] as NAB, chunk[k] as NAB));
                }
            }
        }
        println!(
            "\tWe have #verts= {}, #edges={}, density={}.",
            chunk_len,
            edgs.len(),
            ((2 * edgs.len()) as FLO) / ((chunk_len as FLO) * (chunk_len - 1) as FLO)
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

fn read_file_to_mat(file_path: &String) -> Vec<Vector6<FLO>> {
    let f = BufReader::new(File::open(file_path).unwrap());

    f
        .lines()
        .map(|l| {
            Vector6::from_iterator(
            l.unwrap()
                .split_whitespace()
                .map(|number| number.parse::<FLO>().unwrap())
            )
        })
        .collect()
}

fn coords (point: &Vector6<FLO>) -> Vector4<FLO> {
    Vector4::from_vec(vec![point[0],point[1],point[3],point[4]])
}

fn define_f<'a>(arr_at_clique: &'a Vec<Vector6<FLO>>, mats: &'a Vec<Matrix4<FLO>>) 
    -> Box<dyn Fn(Vector4<FLO>) -> Vector4<FLO> + 'a> {
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
            .map(|arr| coords(&arr))
            .zip(mats)
            .map(|(arr,mat)| 
                {move |y: Vector4<FLO>| (mat*y + arr,mu(&y, &arr))});
    let phis = projs.map(|func| {move |y: Vector4<FLO>| func(y).1*func(y).0 + (1.0-func(y).1)*y});
    Box::new(move |y: Vector4<FLO>| phis.clone().fold(y, move|acc, phi| phi(acc)))
}

fn check_r (max_clique: Vec<NAB>, points: &Vec<Vector6<FLO>>, r: FLO) {
    let mut counter = 0;
    let my_points: Vec<Vector6<FLO>> = points.into_iter().map(|p| p/r).collect();
    let points_q = max_clique.iter().map(|p| my_points[*p as usize]);
    let mut x_ones = points_q.clone().map(|q| ball_rad_r(&q, &my_points,  1.0));
    let mut projs = points_q.clone().map(|q| proj_from_normal(q));
    let mut max_ball = 0;
    let mut avg_ball = 0;
    for q in points_q {
        let mut max: FLO = 0.0;
        let ball = x_ones.next().unwrap();
        let proj = projs.next().unwrap();
        avg_ball += ball.len();
        max_ball = max_ball.max(ball.len());
        //print!("{} ", ball.len());
        for w in ball {
            max = max.max((coords(&(q+w))-proj*coords(&w)).norm());
        }
        //println!("dist {}", max);
        if max > DELTA*18.0 {
            //println!("Around point {} we are too far away", q);
            counter+=1;
        }
    }
    println!("Avg points in ball: {}", (avg_ball as f32)/(max_clique.len() as f32));
    println!("Max points in ball: {}", max_ball);
    println!("#Times we're too far away: {}", counter);
    println!("Clique length: {}", max_clique.len());
    //let max_dists = x_ones.zip(projs).zip(points_q).map(|(wp,q)| wp.0.iter().map(|w| coords(&w)-wp.1*coords(&w) + coords(&q)));
}

struct FuncToMin<'a> {
    point_to_find: &'a Vector4<FLO>,
    fun: &'a Box<dyn Fn(Vector4<FLO>) -> Vector4<FLO> + 'a>,
}

impl FuncToMin<'_> {
    fn norm_func(&self, p: &Vector4<FLO>) -> FLO {
        FLO::sqrt(((self.fun)(*p)[2]-self.point_to_find[2]).powi(2)+((self.fun)(*p)[3]-self.point_to_find[3]).powi(2))
        //((self.fun)(*p) - self.point_to_find).norm()
    }
    fn array_norm_func(&self, y: Array1<FLO>) -> FLO {
        self.norm_func(&Vector4::from_column_slice(y.as_slice().unwrap()))
    }
    fn array_gradient(&self, y: Array1<FLO>) -> Vector4<FLO> {
        self.gradient(&Vector4::from_column_slice(y.as_slice().unwrap())).unwrap()
    }
}

impl CostFunction for FuncToMin<'_> {
    /// Type of the parameter vector
    type Param = Vector4<FLO>;
    /// Type of the return value computed by the cost function
    type Output = FLO;
    /// Apply the cost function to a parameter `p`
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        print!("{} ", self.norm_func(p));
        Ok(self.norm_func(p))
    }
}

impl Gradient for FuncToMin<'_> {
    /// Type of the parameter vector
    type Param = Vector4<FLO>;
    /// Type of the gradient
    type Gradient = Vector4<FLO>;
    /// Compute the gradient at parameter `p`.
    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {            
        let grad = FiniteDiff::forward_diff(&Array1::from_vec(vec![p[0],p[1],p[2],p[3]]), &{|x| self.array_norm_func(x.clone()) });
        println!("{}", grad);
        Ok(Vector4::from_column_slice(grad.as_slice().unwrap()))
    }
}

impl Hessian for FuncToMin<'_> {
    type Param = Vector4<FLO>;
    /// Type of the gradient
    type Hessian = Matrix4<FLO>;

    /// Compute gradient of rosenbrock function
    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        //let func2 = {|y: &Array1<FLO>| self.norm_func(Vector4::from_column_slice(y.as_slice().unwrap()))};
        let grad = FiniteDiff::central_hessian(&Array1::from_vec(vec![p[0],p[1],p[2],p[3]]), &{|x| Array1::from_vec(vec![self.array_gradient(x.clone())[0],self.array_gradient(x.clone())[1],self.array_gradient(x.clone())[2],self.array_gradient(x.clone())[3]])});//self.array_norm_func(x.clone()) });
        //let grad = FiniteDiff::forward_hessian_nograd(&y, &func2);
        println!("{}", grad);

        Ok(Matrix4::new(
            grad[(0,0)], grad[(0,1)], grad[(0,2)],grad[(0,3)],
            grad[(1,0)], grad[(1,1)], grad[(1,2)],grad[(1,3)],
            grad[(2,0)], grad[(2,1)], grad[(2,2)],grad[(2,3)],
            grad[(3,0)], grad[(3,1)], grad[(3,2)],grad[(3,3)],
        ))
        //Ok(Matrix4::from_column_slice(grad.as_slice().unwrap()))
        //Ok(rosenbrock_2d_hessian(param, 1.0, 100.0))
    }
}

fn solve<'a>(points_at_clique: &Vec<Vector6<FLO>>, func: &'a Box<dyn Fn(Vector4<FLO>) -> Vector4<FLO> + 'a>, point_to_find: &Vector4<FLO>) {

    for start in points_at_clique.into_iter().map(|q| coords(&q)) {
        let cost = FuncToMin {point_to_find, fun: func};
    
        println!("{}", &cost.norm_func(&start));
        println!("{}", &cost.norm_func(&(start + Vector4::from_vec(vec![DELTA/2.0,DELTA/2.0,DELTA/2.0,DELTA/2.0]))));
    
        let cp = argmin::solver::trustregion::CauchyPoint::new();
        let tr = TrustRegion::new(cp).with_max_radius(DELTA).unwrap().with_radius(DELTA/10.0).unwrap();
    
        //let linesearch: MoreThuenteLineSearch<Vector4<FLO>, Vector4<FLO>, FLO> = MoreThuenteLineSearch::new()
        //    .with_bounds(DELTA/100000.0,DELTA/10000.0).expect("msg");
        //let solver = SteepestDescent::new(linesearch);
    
        let res = Executor::new(cost, tr)
        // Via `configure`, one has access to the internally used state.
        // This state can be initialized, for instance by providing an
        // initial parameter vector.
        // The maximum number of iterations is also set via this method.
        // In this particular case, the state exposed isstart
        // Population based solvers use `PopulationState` instead of 
        // `IterState`.
        .configure(|state|
            state
                // Set initial parameters (depending on the solver,
                // this may be required)
                .param(start)
                // Set maximum iterations to 10
                // (optional, set to `std::u64::MAX` if not provided)
                .max_iters(5)
                // Set target cost. The solver stops when this cost
                // function value is reached (optional)
                //.target_cost(0.0)
        )
        // run the solver on the defined problem
        .run().unwrap();
        println!("{}", res);
        println!("starting point was {}", start);
    }    
}

fn lin_solve<'a>(points_at_clique: &Vec<Vector6<FLO>>, func: &'a Box<dyn Fn(Vector4<FLO>) -> Vector4<FLO> + 'a>, point_to_find: &Vector4<FLO>) {
    let cost = FuncToMin {point_to_find, fun: func};
    let mut best_point: Vector4<FLO> = coords(&points_at_clique[0]);
    let mut best_val: FLO = 10e10;
    for start in points_at_clique.into_iter().map(|q| coords(&q)) {
        let mut cur_point = start;
        //println!("f(curr_point)={}", func(cur_point));
        for i in 0..4 {
            let mut change: Vec<FLO> = vec![];
            for j in 0..4 {
                if j == i {
                    change.push(DELTA);
                }
                else {
                    change.push(0.0);
                }
            }
            let change_vec = Vector4::from_vec(change);
            if func(change_vec) < func(cur_point) {
                cur_point += change_vec;
            }
            else {
                cur_point -= change_vec;

            }
        }
        //println!("{}", cost.norm_func(&start));
        if cost.norm_func(&cur_point) < best_val {
            best_point = cur_point;
            best_val = cost.norm_func(&cur_point);
            println!("{}", best_val);

        }
        //println!("{}", &cost.norm_func(&(start + Vector4::from_vec(vec![DELTA/2.0,DELTA/2.0,DELTA/2.0,DELTA/2.0]))));
    }   
    println!("best val {} and best point: {:?}", best_val, best_point);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut chunk_size: NAB = 0;
    let mut divisor_r: FLO = 1.0;

    match args.len() {
        1 => println!("Give me a file!"),
        2 => (), //corresponds to just a filename so then we don't chunk
        3 => chunk_size = args[2].parse::<usize>().unwrap() as NAB,
        _ => {
            chunk_size = args[2].parse::<usize>().unwrap() as NAB; 
            divisor_r = args[3].parse::<FLO>().unwrap();
            }
    }
    println!("We are choosing r to be {}", divisor_r);
    let file_path = &args[1];

    let now = std::time::Instant::now();
    let arr = read_file_to_mat(file_path);
    //find_r(&arr);

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
    let max_clique = vec![654, 622, 606, 550, 481, 293, 199, 82, 183, 90, 155, 490, 110, 994, 120, 121, 990, 128, 129, 137, 138, 144, 145, 151, 102, 156, 161, 165, 169, 170, 93, 175, 179, 92, 184, 1014, 191, 192, 195, 85, 81, 200, 1018, 875, 218, 219, 76, 222, 225, 845, 232, 841, 73, 242, 243, 72, 1037, 253, 256, 262, 263, 1049, 68, 270, 271, 276, 277, 281, 1059, 285, 290, 294, 64, 299, 1168, 1222, 1355, 658, 655, 546, 649, 648, 646, 640, 1317, 636, 1307, 1303, 623, 618, 617, 612, 611, 605, 602, 599, 598, 595, 594, 585, 581, 580, 577, 576, 573, 572, 567, 566, 560, 1377, 1373, 549, 556, 107, 1226, 422, 423, 426, 427, 542, 538, 535, 534, 438, 441, 442, 531, 446, 447, 450, 451, 454, 455, 530, 460, 461, 466, 467, 472, 473, 476, 477, 480, 54, 484, 487, 48, 527, 491, 526, 496, 497, 502, 503, 504, 505, 510, 511, 519, 520, 1359, 513, 514, 44, 1365, 39, 38, 522, 523, 495, 494, 47, 51, 53, 58, 34, 59, 1369, 435, 434, 433, 541, 432, 543, 61, 1368, 416, 413, 412, 1376, 411, 410, 547, 557, 407, 559, 406, 403, 564, 565, 402, 1227, 1381, 1382, 397, 396, 393, 392, 1230, 1234, 1240, 1242, 584, 1248, 588, 589, 590, 591, 1249, 1252, 1256, 1269, 1260, 1264, 1268, 1272, 603, 1386, 1273, 1387, 1390, 1278, 1282, 615, 1394, 1283, 1285, 1288, 1398, 1289, 1291, 1402, 629, 630, 1294, 632, 1295, 1406, 635, 1299, 1311, 639, 1313, 641, 1321, 1412, 645, 1323, 1324, 1327, 1329, 1420, 1437, 1438, 1335, 1341, 661, 662, 1349, 665, 1351, 668, 669, 670, 671, 674, 675, 676, 321, 679, 680, 320, 319, 1218, 1211, 1210, 1206, 1202, 1198, 1194, 1191, 316, 1188, 1186, 313, 1181, 1180, 1178, 1177, 312, 1174, 1172, 1167, 718, 719, 309, 308, 1163, 1162, 1160, 1159, 239, 1154, 1152, 1151, 305, 304, 1147, 1146, 1143, 1142, 1140, 1361, 1135, 1134, 751, 1128, 756, 757, 1121, 1120, 1103, 1102, 1095, 1091, 1087, 1083, 1080, 1079, 301, 1076, 1075, 1069, 1068, 300, 1065, 1063, 298, 1060, 284, 791, 282, 1055, 1054, 1053, 1052, 799, 278, 67, 265, 1048, 264, 1045, 1044, 809, 1041, 257, 813, 817, 1036, 249, 821, 822, 248, 247, 826, 827, 246, 831, 1032, 833, 1031, 69, 1029, 837, 228, 238, 237, 842, 236, 231, 846, 229, 849, 1025, 1024, 853, 856, 857, 860, 861, 864, 865, 1022, 869, 871, 872, 109, 1021, 226, 215, 214, 211, 208, 881, 884, 885, 887, 888, 207, 1017, 891, 893, 894, 206, 77, 196, 899, 186, 901, 1013, 91, 906, 907, 178, 1010, 911, 1006, 172, 916, 917, 920, 921, 1003, 923, 162, 160, 928, 929, 930, 931, 100, 101, 934, 935, 938, 939, 152, 147, 942, 943, 949, 956, 957, 960, 961, 964, 966, 146, 969, 970, 974, 976, 977, 139, 980, 981, 984, 986, 131, 989, 130, 123, 122, 995, 115, 114, 998, 999, 1000, 362, 983, 946, 985, 936, 948, 971, 937, 973, 991, 950, 993, 975, 951, 952, 963, 953, 965, 944, 945, 900, 924, 922, 1005, 913, 1007, 1008, 1009, 912, 1011, 910, 908, 902, 794, 898, 896, 890, 880, 878, 876, 874, 868, 866, 852, 850, 848, 1028, 838, 1030, 836, 834, 832, 830, 828, 819, 818, 816, 814, 1040, 812, 1042, 1043, 810, 808, 806, 805, 804, 803, 802, 801, 800, 798, 796, 795, 772, 793, 1058, 792, 790, 788, 787, 786, 1064, 785, 1066, 784, 783, 782, 781, 1071, 1072, 1073, 1074, 780, 779, 778, 777, 776, 775, 1081, 774, 773, 1084, 702, 1086, 771, 1088, 770, 1090, 769, 1092, 768, 1094, 767, 1096, 766, 1098, 1099, 1100, 1101, 765, 764, 763, 762, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 761, 760, 759, 758, 1124, 1125, 1126, 1127, 753, 1129, 752, 1131, 1132, 1133, 750, 748, 747, 746, 745, 1139, 744, 1141, 743, 742, 741, 740, 739, 738, 737, 736, 735, 734, 733, 732, 731, 1155, 730, 1157, 729, 728, 727, 726, 725, 724, 723, 722, 721, 720, 715, 714, 713, 1171, 712, 1173, 711, 1175, 710, 709, 708, 707, 706, 705, 704, 703, 647, 1185, 701, 1187, 700, 1189, 699, 698, 697, 696, 695, 1195, 1196, 1197, 694, 1199, 693, 1201, 692, 1203, 691, 1205, 690, 1207, 689, 1209, 688, 687, 686, 685, 1214, 1215, 1216, 1217, 684, 1219, 683, 1221, 682, 1223, 681, 388, 512, 417, 401, 1229, 400, 1231, 391, 1233, 390, 1235, 1236, 1237, 389, 1239, 345, 1241, 387, 1243, 386, 1245, 385, 384, 383, 382, 1250, 1251, 381, 1253, 380, 1255, 379, 1257, 378, 1259, 377, 1261, 376, 1263, 375, 1265, 1266, 1267, 374, 373, 372, 371, 370, 369, 1274, 1275, 368, 1277, 367, 1279, 366, 365, 364, 363, 1284, 0, 361, 360, 359, 358, 1290, 357, 356, 355, 354, 353, 1296, 1297, 1298, 352, 1300, 351, 1302, 350, 1304, 349, 1306, 348, 1308, 347, 1310, 346, 1312, 324, 1314, 344, 1316, 343, 1318, 342, 1320, 341, 340, 339, 338, 337, 1326, 336, 1328, 335, 1330, 334, 1332, 333, 1334, 332, 1336, 1337, 1338, 331, 1340, 330, 1342, 1343, 1344, 1345, 1346, 329, 1348, 328, 1350, 327, 1352, 326, 1354, 325, 1356, 209, 1358, 1, 1360, 45, 1362, 43, 1364, 42, 1366, 1367, 37, 35, 33, 32, 1372, 31, 30, 29, 28, 27, 1378, 26, 1380, 25, 24, 1383, 23, 22, 21, 20, 19, 1389, 18, 1391, 17, 1393, 16, 1395, 15, 1397, 14, 1399, 13, 1401, 12, 1403, 11, 1405, 10, 1407, 1408, 1409, 9, 1411, 8, 1413, 1414, 1415, 1416, 1417, 7, 1419, 6, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 5, 4, 3, 2, 289];
    
    
    //chunk_clique(chunk_size, &(0..(len as NAB)).collect::<Vec<NAB>>(), &arr, divisor_r, 2);

    println!("Found max clique of len {}: {:?}", max_clique.len(), max_clique);


    let now2 = std::time::Instant::now();
    let vals = get_qs(max_clique.clone(), &arr);
    for point in &vals.0 {
        let normal: Vector4<FLO> = Vector4::from_vec(vec![FLO::cos(point[2]/180.0*PI),FLO::sin(point[2]/190.0*PI),-FLO::cos(point[5]/180.0*PI),-FLO::sin(point[5]/180.0*PI)]);
        println!("normal: {}", normal);
    }
    let elapsed_time2 = now2.elapsed();
    println!("It took {} micro seconds to calc {} qr decomps", elapsed_time2.as_micros(), vals.0.len());
    let start = coords(&vals.0[9].clone()); //4
    let point_to_find = coords(&vals.0[21].clone());//8

    let func = define_f(&vals.0, &vals.1);

    let cost = FuncToMin {point_to_find: &point_to_find, fun: &func};
    
    println!("{}", &cost.norm_func(&start));

    let root_drawing_area = BitMapBackend::new("/home/leo/Documents/work/WFtranslator/feffer/images/0.1.png", (1024, 768))
    .into_drawing_area();

    root_drawing_area.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root_drawing_area)
        .build_cartesian_2d(-DELTA*10000.0..DELTA*10000.0, (cost.norm_func(&start)-0.1)..(cost.norm_func(&start)+0.1))
        .unwrap();

    chart.draw_series(LineSeries::new(
        (-(f64::ceil(DELTA*1000.0) as i32)..(f64::ceil(DELTA*1000.0) as i32)).map(|x| x as f64 * 1000.0).map(|x| (x, cost.norm_func(&(start + Vector4::from_vec(vec![0.0,x,0.0,0.0]))))),
        &RED
    )).unwrap();
    check_r(max_clique, &arr, divisor_r);

    //lin_solve(&vals.0, &func, &point_to_find);
    println!("point to find: {}",point_to_find);

    //solve(&vals.0, &func, &point_to_find);
    //let point_to_find: Vector4<FLO> = Vector4::from_vec(vec![0.04,1.0,32.0,1.0]);
    //println!("{}", func(start));

    



    
 
    //Best parameter vector
    // let best = res.state().get_best_param().unwrap();

    // // Cost function value associated with best parameter vector
    // let best_cost = res.state().get_best_cost();
    // println!("best param {}, best cost {}", best, best_cost);
    // let path = "QQTvals2.txt";
    // let mut output = File::create(path).unwrap();
    // for val in vals.1 {
    //     writeln!(output, "{:?}", val).unwrap();
    // }

    elapsed_time = now.elapsed();
    println!("The total process took {} seconds.", elapsed_time.as_secs());
}
