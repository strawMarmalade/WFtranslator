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
const DELTA: FLO = 0.016; //want C_{18}n\delta < C_{18}n 1/(n*20) = 6/20 
//const DIV: FLO = 1.0;//1.0;

fn mu(x_point: &Vector4<FLO>, q_point: &Vector4<FLO>) -> FLO {
    let dist: FLO = (x_point - q_point).norm();
    (1.0 / (dist - 1.0 / 3.0)).exp()
        / ((1.0 / (dist - 1.0 / 3.0)).exp() + (1.0 / (1.0 / 2.0 - dist)).exp())
}

fn calc_dist_from1(x0_point: &Vector4<FLO>, x_point: &Vector4<FLO>, y_vals: &[Vector4<FLO>]) -> FLO {
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

fn find_disc(x0_point: &Vector4<FLO>, x_vals: &Vec<Vector4<FLO>>, n: usize) -> Option<Vec<Vector4<FLO>>> {
    match x_vals.len() {
        4 => return Some(x_vals.clone()),
        0..=3 => {
            //println!("Error, this ball has too few points in it");
            return None    
        }
        _ =>    {
            let mut output: Vec<Vector4<FLO>> = vec![];
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

fn ball_rad_r(x_point: &Vector4<FLO>, points: &Vec<Vector4<FLO>>, r: FLO) -> Vec<Vector4<FLO>> {
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

fn points_dist(points_a: &Vec<Vector4<FLO>>, points_b: &Vec<Vector4<FLO>>) -> FLO {
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

fn find_r(arr: &Vec<Vector4<FLO>>) -> Option<FLO>{
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

fn get_qs(max_clique: Vec<NAB>, points: &Vec<Vector6<FLO>>) -> (Vec<Vector4<FLO>>, Vec<Matrix4<FLO>>) {
    let points_q = max_clique.iter().map(|p| points[*p as usize]);
    let points_clone = points_q.clone().map(|q| coords(&q));
    //let x_ones = points_q.clone().map(|q| ball_rad_r(&q, points,  40.0));
    //let spaces_q = points_q.zip(x_ones).map(|(q,x)| na::Matrix6x4::from_columns(&find_disc(&q, &x, 4).unwrap()));
    let projs = points_q.map(|q| proj_from_normal(q));//spaces_q
        //.map(|mat| 
        //    na::linalg::QR::new(mat).q())
        //    .map(|q| q*q.transpose());
    (points_clone.collect(), projs.collect())
}

fn proj_from_normal(point: Vector6<FLO>) -> Matrix4<FLO>{
    let normal: Vector4<FLO> = Vector4::from_vec(vec![FLO::cos(PI*point[2]/180.0),FLO::sin(PI*point[2]/180.0),-FLO::cos(PI*point[5]/180.0),-FLO::sin(PI*point[5]/180.0)]);
    //both of the following methods work the one used now is faster
    // println!("normal: {}", point[2]);
    // let mat: Matrix4x3<FLO> = Matrix4x3::new(
    //     -normal[1],-normal[2],-normal[3],
    //     normal[0],normal[3],-normal[2],
    //     -normal[3],normal[0],normal[1],
    //     normal[2],-normal[1], normal[0]
    // );
    // let q = na::linalg::QR::new(mat).q();//.map(|q| q*q.transpose())
    // q*q.transpose()
    -Matrix4::new(
        normal[0]*normal[0]-2.0,normal[0]*normal[1],normal[0]*normal[2],normal[0]*normal[3],
        normal[1]*normal[0],normal[1]*normal[1]-2.0,normal[1]*normal[2],normal[1]*normal[3],
        normal[2]*normal[0],normal[2]*normal[1],normal[2]*normal[2]-2.0,normal[2]*normal[3],
        normal[3]*normal[0],normal[3]*normal[1],normal[3]*normal[2],normal[3]*normal[3]-2.0,
    )/2.0
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
                if 100.0*(coords(&(&arr[chunk[j] as usize] - &arr[chunk[k] as usize])))
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

fn define_f<'a>(arr_at_clique: &'a Vec<Vector4<FLO>>, mats: &'a Vec<Matrix4<FLO>>, divisor_r: &'a FLO) 
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
            .map(|a| a/ *divisor_r)
            .zip(mats)
            .map(|(arr,mat)| 
                {move |y: Vector4<FLO>| (mat*y + arr,mu(&y, &arr))});
    let phis = projs.map(|func| {move |y: Vector4<FLO>| func(y).1*func(y).0 + (1.0-func(y).1)*y});
    //phis.clone().fold(Vector4::from_vec(vec![58.0, 112.0, 166.0, 54.0]), move|acc, phi| {println!("{:?}",phi(acc)); phi(acc)});
    //phis.clone().fold(Vector4::from_vec(vec![138.0, 104.0, 71.0, 142.0]), move|acc, phi| {println!("{:?}",phi(acc)); phi(acc)});
    Box::new(move |y: Vector4<FLO>| phis.clone().fold(y, move|acc, phi| {//print!("{}",phi(acc)); 
    phi(acc)}).map(|v| v* *divisor_r))
}

fn check_r (max_clique: Vec<NAB>, points: &Vec<Vector6<FLO>>, r: FLO) {
    let mut counter = 0;
    let my_points: Vec<Vector6<FLO>> = points.into_iter().map(|p| p/r).collect();
    let points_q = max_clique.iter().map(|p| my_points[*p as usize]);
    let mut x_ones = points_q.clone().map(|q| ball_rad_r(&coords(&q), &my_points.iter().map(|q| coords(&q)).collect(),  1.0));
    let mut projs = points_q.clone().map(|q| proj_from_normal(q*r));
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
            max = max.max((&(coords(&q)+w)-proj*&w).norm());
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

#[derive(Clone)]
struct FuncToMin<'a> {
    point_to_find: &'a Vector4<FLO>,
    fun: &'a Box<dyn Fn(Vector4<FLO>) -> Vector4<FLO> + 'a>,
}

impl FuncToMin<'_> {
    fn norm_func(&self, p: &Vector4<FLO>) -> FLO {
        FLO::sqrt(
            ((self.fun)(*p)[0]-self.point_to_find[0]).powi(2)+((self.fun)(*p)[1]-self.point_to_find[1]).powi(2)
        )
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
        //print!("{} ", self.norm_func(p));
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
        //println!("{}", grad);
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
        //println!("{}", grad);

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

fn solve<'a>(points_at_clique: &Vec<Vector4<FLO>>, func: &'a Box<dyn Fn(Vector4<FLO>) -> Vector4<FLO> + 'a>, point_to_find: &Vector4<FLO>, divisor_r: FLO) {
    let cost = FuncToMin {point_to_find, fun: func};

    let mut best_point: Vector4<FLO> = points_at_clique[0];
    let mut best_val: FLO = 10e10;

    for start in points_at_clique {
    
        let cp = argmin::solver::trustregion::CauchyPoint::new();
        let tr = TrustRegion::new(cp).with_max_radius(DELTA*divisor_r).unwrap().with_radius(divisor_r*DELTA/100.0).unwrap();
    
        //let linesearch: MoreThuenteLineSearch<Vector4<FLO>, Vector4<FLO>, FLO> = MoreThuenteLineSearch::new()
        //    .with_bounds(DELTA/100000.0,DELTA/10000.0).expect("msg");
        //let solver = SteepestDescent::new(linesearch);
    
        let res = Executor::new(cost.clone(), tr)
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
                .param(*start)
                // Set maximum iterations to 10
                // (optional, set to `std::u64::MAX` if not provided)
                .max_iters(10)
                // Set target cost. The solver stops when this cost
                // function value is reached (optional)
                //.target_cost(0.0)
        )
        // run the solver on the defined problem
        .run().unwrap();
        let best_res = res.state.get_best_cost();
        if best_res < best_val {
            best_point = *res.state.get_best_param().unwrap();
            best_val = best_res;
        }
        // println!("{}", res);
        // println!("starting point was {}", start);
    }
    println!("best val: {} and best point: f({}) = {}, when trying to find {}", best_val, best_point, func(best_point), point_to_find);
}

fn lin_solve<'a>(points_at_clique: &Vec<Vector4<FLO>>, func: &'a Box<dyn Fn(Vector4<FLO>) -> Vector4<FLO> + 'a>, point_to_find: &Vector4<FLO>) {
    let cost = FuncToMin {point_to_find, fun: func};
    let mut best_point: Vector4<FLO> = points_at_clique[0];
    // println!("f(curr_point)={}", func(best_point));
    // println!("f(curr_point)={}", func(best_point/2000000.0));

    let mut best_val: FLO = 10e10;
    for start in points_at_clique {
        let mut cur_point = *start;
        println!("f(curr_point)={}", func(cur_point));
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
    let max_clique = //vec![[756, 530, 1439, 1342, 551, 1410, 603, 492, 1389, 580, 1106, 1045, 2, 798, 686, 1058, 27, 410, 351, 1143, 1175, 313, 840, 981, 77, 657, 1207, 1210, 252, 845, 1312, 280, 629, 216, 129, 953, 464, 817, 1248, 438, 899, 868, 873, 159, 381, 352, 103, 1182, 927, 1278, 185, 186, 1263, 894, 32, 223, 1289, 1158, 162, 1313, 932, 956, 1365, 247, 604, 783, 134, 986, 1032, 513, 1134, 1008, 491, 269, 112, 626, 403, 54, 291, 74, 423, 332, 646, 310, 763, 687, 1114, 669, 742, 1095, 1075, 713, 1337, 1339, 451, 449, 711, 1079, 1235]67, 1102, 370, 1098, 374, 1094, 33, 37, 41, 1072, 45, 1062, 394, 1059, 1058, 398, 402, 1055, 406, 1054, 1051, 1047, 416, 1043, 420, 1039, 49, 1027, 430, 1023, 1019, 1015, 1011, 1007, 997, 993, 883, 882, 871, 870, 867, 866, 535, 546, 547, 856, 855, 558, 559, 562, 563, 566, 567, 852, 851, 848, 578, 579, 847, 844, 843, 589, 840, 839, 599, 836, 835, 832, 831, 828, 827, 824, 823, 820, 819, 816, 633, 815, 812, 811, 808, 807, 804, 800, 796, 661, 792, 664, 788, 778, 774, 770, 678, 766, 682, 762, 686, 758, 690, 1178, 694, 753, 53, 698, 749, 702, 743, 706, 731, 727, 710, 63, 714, 1259, 717, 657, 720, 62, 77, 5, 1434, 78, 79, 1430, 59, 7, 58, 57, 738, 1428, 82, 85, 705, 1424, 1420, 54, 88, 1418, 1417, 697, 89, 242, 1415, 92, 759, 687, 763, 93, 94, 1411, 681, 95, 1409, 98, 773, 99, 1405, 674, 101, 781, 782, 784, 785, 1401, 670, 668, 667, 791, 793, 663, 660, 588, 797, 656, 104, 801, 803, 653, 105, 650, 647, 646, 645, 1397, 639, 638, 637, 636, 630, 629, 627, 107, 624, 621, 620, 619, 616, 613, 612, 611, 610, 605, 604, 603, 600, 598, 595, 594, 593, 590, 493, 585, 584, 583, 582, 574, 573, 1393, 571, 570, 553, 552, 551, 550, 541, 540, 860, 861, 862, 863, 539, 536, 534, 531, 530, 529, 528, 523, 522, 1, 520, 876, 877, 878, 879, 517, 516, 515, 514, 511, 886, 887, 889, 890, 510, 509, 894, 895, 896, 897, 508, 505, 504, 503, 902, 903, 904, 905, 502, 499, 908, 909, 910, 911, 498, 497, 914, 915, 917, 918, 496, 1391, 921, 922, 923, 492, 926, 927, 928, 929, 491, 490, 932, 933, 934, 935, 487, 486, 938, 939, 940, 941, 485, 484, 944, 945, 947, 948, 481, 108, 951, 952, 953, 480, 479, 956, 957, 958, 959, 478, 475, 962, 963, 964, 965, 474, 473, 472, 469, 970, 971, 972, 973, 468, 466, 465, 978, 979, 111, 981, 982, 462, 461, 986, 987, 988, 989, 460, 113, 1385, 456, 994, 455, 454, 998, 451, 114, 1001, 1003, 1004, 450, 1383, 449, 448, 1010, 443, 1012, 442, 440, 1016, 439, 117, 436, 1022, 1024, 434, 433, 429, 1028, 120, 1030, 46, 426, 1377, 1034, 50, 424, 1038, 121, 1042, 419, 1375, 417, 1048, 122, 1050, 412, 410, 123, 407, 403, 126, 397, 393, 391, 1063, 69, 127, 128, 1369, 1069, 71, 129, 385, 1367, 132, 133, 134, 135, 138, 1361, 139, 1358, 381, 1356, 1355, 1353, 380, 141, 144, 378, 145, 377, 1347, 373, 1101, 1346, 32, 1345, 369, 1344, 147, 1110, 150, 151, 1113, 1339, 360, 152, 358, 1336, 352, 1122, 349, 1334, 1125, 1127, 1128, 348, 347, 153, 1134, 1331, 1330, 1328, 1327, 1140, 156, 157, 1143, 1144, 1145, 334, 1148, 1150, 332, 28, 158, 26, 1158, 1323, 1322, 1163, 1164, 1165, 324, 1321, 318, 1320, 314, 310, 309, 159, 307, 162, 304, 301, 1317, 1316, 296, 1182, 208, 1184, 294, 293, 1188, 289, 1190, 1314, 286, 284, 283, 277, 276, 275, 274, 269, 268, 1202, 74, 1204, 266, 265, 261, 260, 1311, 258, 1212, 1213, 1214, 1215, 255, 1310, 252, 1309, 250, 247, 1308, 244, 755, 241, 239, 235, 234, 233, 1232, 1233, 1234, 1235, 232, 229, 225, 224, 1242, 1243, 1244, 1245, 221, 220, 219, 218, 211, 1252, 1253, 1254, 1255, 210, 209, 1258, 66, 1260, 1261, 163, 201, 200, 199, 1266, 1267, 1268, 1269, 198, 185, 1272, 1273, 1274, 1275, 184, 183, 1278, 1279, 1280, 1281, 182, 175, 1284, 1285, 1286, 1287, 174, 173, 1290, 1291, 1292, 1293, 172, 167, 1296, 1297, 1298, 166, 1301, 1302, 1305, 1304, 1065, 328, 164, 245, 251, 253, 259, 267, 285, 287, 1315, 297, 299, 306, 308, 316, 322, 1162, 1160, 401, 1142, 338, 340, 342, 1329, 1303, 344, 346, 1130, 1124, 1335, 1118, 1337, 1116, 1114, 1112, 1111, 1109, 68, 1107, 1105, 1103, 1097, 146, 1095, 1093, 1091, 1089, 1087, 140, 36, 1085, 1083, 38, 1359, 1081, 1079, 40, 1077, 72, 1075, 42, 1073, 1071, 44, 70, 1136, 165, 671, 413, 1044, 423, 425, 427, 435, 437, 1018, 116, 1006, 1000, 457, 411, 463, 980, 110, 950, 920, 521, 572, 626, 628, 642, 644, 652, 654, 655, 409, 783, 673, 777, 775, 100, 675, 677, 679, 769, 767, 765, 685, 691, 693, 695, 750, 701, 748, 746, 87, 703, 86, 744, 742, 84, 740, 56, 734, 732, 730, 728, 709, 726, 711, 724, 722, 713, 715, 1159];
    chunk_clique(chunk_size, &(0..(len as NAB)).collect::<Vec<NAB>>(), &arr, divisor_r, 2);

    //println!("Found max clique of len {}: {:?}", max_clique.len(), max_clique);

    let now2 = std::time::Instant::now();
    let vals = get_qs(max_clique.clone(), &arr);

    // for ma in &vals.1 {
    //     println!("{}", ma[(0,0)]);
    // }
    //println!("{:?} {:?}", &vals.0[0], &vals.0[40]);

    let elapsed_time2 = now2.elapsed();
    println!("It took {} micro seconds to calc {} qr decomps", elapsed_time2.as_micros(), vals.0.len());
    //let start = &vals.0[9].clone(); //4
    let point_to_find = &vals.0[21].clone();//8

    let func = define_f(&vals.0, &vals.1, &divisor_r);
   // println!("{}", func(*point_to_find));
   // let cost = FuncToMin {point_to_find: &point_to_find, fun: &func};
    
    //println!("{}", &cost.norm_func(&start));

    // let root_drawing_area = BitMapBackend::new("/home/leo/Documents/work/WFtranslator/feffer/images/0.1.png", (1024, 768))
    // .into_drawing_area();

    // root_drawing_area.fill(&WHITE).unwrap();

    // let mut chart = ChartBuilder::on(&root_drawing_area)
    //     .build_cartesian_2d(-DELTA..DELTA, (cost.norm_func(&start)-0.1)..(cost.norm_func(&start)+0.1))
    //     .unwrap();

    // chart.draw_series(LineSeries::new(
    //     (-(f64::ceil(DELTA*1000.0) as i32)..(f64::ceil(DELTA*1000.0) as i32)).map(|x| x as f64/ 1000.0).map(|x| (x, cost.norm_func(&(start + Vector4::from_vec(vec![0.0,x,0.0,0.0]))))),
    //     &RED
    // )).unwrap();
    //check_r(max_clique, &arr, divisor_r);

    // lin_solve(&vals.0, &func, &point_to_find);
    // println!("point to find: {}",point_to_find);

    solve(&vals.0, &func, &point_to_find, divisor_r);
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
