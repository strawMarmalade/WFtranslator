use argmin::core::{CostFunction, Error, Gradient, Executor, State};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use na::{Vector6, Vector4, Matrix4x3, Matrix4};
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
use std::f32::consts::PI;
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
type FLO = f32;
const DELTA: FLO = 0.016; //want C_{18}n\delta < C_{18}n 1/(n*20) = 6/20 
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
    let normal: Vector4<FLO> = Vector4::from_vec(vec![FLO::cos(point[5]/180.0*PI),FLO::sin(point[5]/180.0*PI),FLO::cos(point[2]/180.0*PI),FLO::sin(point[2]/180.0*PI)]);
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
    println!("Avg ball: {}", (avg_ball as f32)/(max_clique.len() as f32));
    println!("max ball {}", max_ball);
    println!("counter: {}", counter);
    println!("clique length: {}", max_clique.len());
    //let max_dists = x_ones.zip(projs).zip(points_q).map(|(wp,q)| wp.0.iter().map(|w| coords(&w)-wp.1*coords(&w) + coords(&q)));
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
            divisor_r = args[3].parse::<f32>().unwrap();
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
    let max_clique = chunk_clique(chunk_size, &(0..(len as NAB)).collect::<Vec<NAB>>(), &arr, divisor_r, 2);

    println!("Found max clique of len {}: {:?}", max_clique.len(), max_clique);

    let now2 = std::time::Instant::now();
    let vals = get_qs(max_clique.clone(), &arr);
    let elapsed_time2 = now2.elapsed();
    println!("It took {} micro seconds to calc {} qr decomps", elapsed_time2.as_micros(), vals.0.len());
    let start = coords(&vals.0[0].clone());

    let func = define_f(&vals.0, &vals.1);
    let point_to_find: Vector4<FLO> = Vector4::from_vec(vec![0.04,1.0,32.0,1.0]);
    println!("{}", func(start));

    

    check_r(max_clique, &arr, divisor_r);

    struct FuncToMin<'a> {
        point_to_find: Vector4<FLO>,
        base_point: Vector4<FLO>,
        fun: Box<dyn Fn(Vector4<FLO>) -> Vector4<FLO> + 'a>,
    }

    impl CostFunction for FuncToMin<'_> {
        /// Type of the parameter vector
        type Param = Vector4<FLO>;
        /// Type of the return value computed by the cost function
        type Output = FLO;
        /// Apply the cost function to a parameter `p`
        fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            //Outside of the ball of radius one around this point I wanna make the cost super high
            if (p-self.base_point).norm() > DELTA {
                return Ok(((self.fun)(*p)[2]-self.point_to_find[2]).powi(2)+((self.fun)(*p)[3]-self.point_to_find[3]).powi(2)*1000.0+1000.0);
            }
            // Evaluate 2D Rosenbrock function
            Ok(((self.fun)(*p)[2]-self.point_to_find[2]).powi(2)+((self.fun)(*p)[3]-self.point_to_find[3]).powi(2))
        }
    }

    impl Gradient for FuncToMin<'_> {
        /// Type of the parameter vector
        type Param = Vector4<FLO>;
        /// Type of the gradient
        type Gradient = Vector4<FLO>;

        /// Compute the gradient at parameter `p`.
        fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
            // Compute gradient of 2D Rosenbrock function
            let val1 = (self.fun)(*p + Vector4::from_vec(vec![10e-7,0.0,0.0,0.0]));
            let val2 = (self.fun)(*p - Vector4::from_vec(vec![10e-7,0.0,0.0,0.0]));
            let val3 = (self.fun)(*p + Vector4::from_vec(vec![0.0,10e-7,0.0,0.0]));
            let val4 = (self.fun)(*p - Vector4::from_vec(vec![0.0,10e-7,0.0,0.0]));
            let val5 = (self.fun)(*p + Vector4::from_vec(vec![0.0,0.0,10e-7,0.0]));
            let val6 = (self.fun)(*p - Vector4::from_vec(vec![0.0,0.0,10e-7,0.0]));
            let val7 = (self.fun)(*p + Vector4::from_vec(vec![0.0,0.0,0.0,10e-7]));
            let val8 = (self.fun)(*p - Vector4::from_vec(vec![0.0,0.0,0.0,10e-7]));
            Ok(Vector4::from_vec(vec![2.0/10e-7*(
                (val1[2]-self.point_to_find[2]).powi(2)
                +(val1[3]-self.point_to_find[3]).powi(2)
                -(val2[2]-self.point_to_find[2]).powi(2)
                -(val2[3]-self.point_to_find[3]).powi(2)),
                2.0/10e-7*(
                (val3[2]-self.point_to_find[2]).powi(2)
                +(val3[3]-self.point_to_find[3]).powi(2)
                -(val4[2]-self.point_to_find[2]).powi(2)
                -(val4[3]-self.point_to_find[3]).powi(2)),
                2.0/10e-7*(
                (val5[2]-self.point_to_find[2]).powi(2)
                +(val5[3]-self.point_to_find[3]).powi(2)
                -(val6[2]-self.point_to_find[2]).powi(2)
                -(val6[3]-self.point_to_find[3]).powi(2)),
                2.0/10e-7*(
                (val7[2]-self.point_to_find[2]).powi(2)
                +(val7[3]-self.point_to_find[3]).powi(2)
                -(val8[2]-self.point_to_find[2]).powi(2)
                -(val8[3]-self.point_to_find[3]).powi(2))]))
        }
    }
    //let func2 = func.clone();
    //let fun2 = Box::new(func);
    println!("{}", ((func)(start)[2]-point_to_find[2]).powi(2)+((func)(start)[3]-point_to_find[3]).powi(2));
    let cost = FuncToMin {point_to_find, base_point:start, fun: func};
    let linesearch: MoreThuenteLineSearch<Vector4<FLO>, Vector4<FLO>, FLO> = MoreThuenteLineSearch::new();
    let solver = SteepestDescent::new(linesearch);

    let res = Executor::new(cost, solver)
    // Via `configure`, one has access to the internally used state.
    // This state can be initialized, for instance by providing an
    // initial parameter vector.
    // The maximum number of iterations is also set via this method.
    // In this particular case, the state exposed is of type `IterState`.
    // The documentation of `IterState` shows how this struct can be
    // manipulated.
    // Population based solvers use `PopulationState` instead of 
    // `IterState`.
    .configure(|state|
        state
            // Set initial parameters (depending on the solver,
            // this may be required)
            .param(point_to_find)
            // Set maximum iterations to 10
            // (optional, set to `std::u64::MAX` if not provided)
            .max_iters(10)
            // Set target cost. The solver stops when this cost
            // function value is reached (optional)
            .target_cost(0.0)
    )
    // run the solver on the defined problem
    .run().expect("Thing");
    println!("{}", res);
    // Best parameter vector
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
