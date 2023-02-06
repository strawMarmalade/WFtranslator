use argmin::core::Executor;
use argmin::solver::quasinewton::SR1TrustRegion;
//use plotters::prelude::*;
use argmin::solver::trustregion::CauchyPoint;
use minimizers::FuncMin;
use nalgebra as na;
use na::{Matrix4, Vector4, Vector6};
use ndarray::Array1;
use rayon::prelude::*;
use std::env;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::mpsc::channel;
use std::sync::mpsc::{Receiver, Sender};

mod pmcgraph;
mod minimizers;
use crate::minimizers::FindingR;
use crate::pmcgraph::PmcGraph;

type Nab = u32;
type Flo = f64;
const DELTAMOD: Flo = 0.001;//0.0016; //want C_{18}n\delta < C_{18}n 1/(n*20) = 6/20
                                //const DIV: FLO = 1.0;//1.0;

fn mu(x_point: &Vector4<Flo>, q_point: &Vector4<Flo>) -> Flo {
    let dist: Flo = (x_point - q_point).norm();
    //println!("{}", dist);
    if 3.0*dist <= 1.01 {
        return 1.0;
    }
    else if 2.0*dist >= 1.01 {
        return 0.0;
    }
    (1.0 / (dist - 1.0 / 3.0)).exp()
        / ((1.0 / (dist - 1.0 / 3.0)).exp() + (1.0 / (1.0 / 2.0 - dist)).exp())
}

/*
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
*/

fn get_qs(max_clique: Vec<Nab>, points: &[Vector6<Flo>]) -> (Vec<Vector4<Flo>>, Vec<Matrix4<Flo>>) {
    let points_q = max_clique.iter().map(|p| points[*p as usize]);
    let points_clone = points_q.clone().map(|q| coords(&q));
    //let x_ones = points_q.clone().map(|q| ball_rad_r(&q, points,  40.0));
    //let spaces_q = points_q.zip(x_ones).map(|(q,x)| na::Matrix6x4::from_columns(&find_disc(&q, &x, 4).unwrap()));
    let projs = points_q.map(proj_from_normal); //
    (points_clone.collect(), projs.collect())
}

fn proj_from_normal(point: Vector6<Flo>) -> Matrix4<Flo> {
    let normal: Vector4<Flo> = Vector4::from_vec(vec![
        Flo::cos(PI * point[4] / 180.0),
        Flo::sin(PI * point[4] / 180.0),
        -Flo::cos(PI * point[5] / 180.0),
        -Flo::sin(PI * point[5] / 180.0),
    ]);
    /*
    both of the following methods work the one used now is faster
    println!("normal: {}", point[2]);
    let mat: Matrix4x3<FLO> = Matrix4x3::new(
        -normal[1],-normal[2],-normal[3],
        normal[0],normal[3],-normal[2],
        -normal[3],normal[0],normal[1],
        normal[2],-normal[1], normal[0]
    );
    let q = na::linalg::QR::new(mat).q();//.map(|q| q*q.transpose())
    q*q.transpose()
    */
    -Matrix4::new(
        normal[0] * normal[0] - 2.0,
        normal[0] * normal[1],
        normal[0] * normal[2],
        normal[0] * normal[3],
        normal[1] * normal[0],
        normal[1] * normal[1] - 2.0,
        normal[1] * normal[2],
        normal[1] * normal[3],
        normal[2] * normal[0],
        normal[2] * normal[1],
        normal[2] * normal[2] - 2.0,
        normal[2] * normal[3],
        normal[3] * normal[0],
        normal[3] * normal[1],
        normal[3] * normal[2],
        normal[3] * normal[3] - 2.0,
    ) / 2.0
}

fn chunk_clique(
    chunk_size: u32,
    verts: Vec<Nab>,
    arr: &[Vector4<Flo>],
    //divisor_r: Flo,
    increase_factor: u32,
) -> Vec<Nab> {
    let mut collected_verts: Vec<Nab> = vec![];
    let chunks = verts.chunks(chunk_size as usize);

    let chunk_amount = chunks.len();

    for chunk in chunks {
        let chunk_len = chunk.len();
        let mut edgs: Vec<(Nab, Nab)> = vec![];
        for j in 0..chunk_len {
            for k in 0..j {
                if (&(arr[chunk[j] as usize] - arr[chunk[k] as usize])).norm()
                    >= 1.0/100.0
                {
                    edgs.push((chunk[j] as Nab, chunk[k] as Nab));
                }
            }
        }

        // println!(
        //     "\tWe have #verts= {}, #edges={}, density={}.",
        //     chunk_len,
        //     edgs.len(),
        //     ((2 * edgs.len()) as Flo) / ((chunk_len as Flo) * (chunk_len - 1) as Flo)
        // );
        if chunk_len*(chunk_len-1)/2 as usize == edgs.len()
        {
            println!("All are 1/100 distant");
            collected_verts.extend(chunk);
        }
        else {
            println!("Not all are 1/100 distant");
            let graph = PmcGraph::new(chunk.to_vec(), edgs);
            //let now2 = std::time::Instant::now();
            collected_verts.extend(
                graph
                    .search_bounds()
                    .into_iter()
                    .map(|val| chunk[val as usize]),
            );
        }

        //let elapsed_time = now2.elapsed();
        // println!(
        //     "\tIt took {} milliseconds to compute the clique\n",
        //     elapsed_time.as_millis(),
        // );
    }
    if chunk_amount == 1 {
        return collected_verts;
    }
    chunk_clique(
        chunk_size * increase_factor,
        collected_verts,
        arr,
        //divisor_r,
        increase_factor,
    )
}

fn read_file_to_mat(file_path: &String) -> Vec<Vector6<Flo>> {
    let f = BufReader::new(File::open(file_path).unwrap());

    f.lines()
        .map(|l| {
            Vector6::from_iterator(
                l.unwrap()
                    .split_whitespace()
                    .map(|number| number.parse::<Flo>().unwrap()),
            )
        })
        .collect()
}

fn coords(point: &Vector6<Flo>) -> Vector4<Flo> {
    Vector4::from_vec(vec![point[0], point[1], point[2], point[3]])
}

fn define_f<'a>(
    arr_at_clique: &'a [Vector4<Flo>],
    mats: &'a [Matrix4<Flo>],
    //divisor_r: Flo,
) -> Box<dyn Fn(Vector4<Flo>) -> Vector4<Flo> + 'a + Sync> {
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
    Box::new(move |y: Vector4<Flo>| {
        arr_at_clique
        .iter()
        .zip(mats)
        .map(|(arr, mat)| move |y: Vector4<Flo>|
            {
                let m = mu(&y, &arr);
                //println!("{}", (y-arr).norm());
                //println!("{}, {}",y, m* (mat * (y-arr) + arr) + (1.0- m)*y);
                m* (mat * (y-arr) + arr) + (1.0- m)*y
            })
        .fold(y, move |acc, phi| 
            {
                //print!("{}",phi(acc));
                phi(acc)
            })
    })
}

/*
fn solve<'a>(
    points_at_clique: &[Vector4<Flo>],
    func: &'a (dyn Fn(Vector4<Flo>) -> Vector4<Flo> + 'a + Sync),
    point_to_find: &Vector4<Flo>,
    divisor_r: Flo,
) -> (Flo, Vector4<Flo>) {
    let cost = FuncToMin {
        point_to_find,
        fun: func,
    };

    let mut best_point: Vector4<Flo> = points_at_clique[0];
    let mut best_val: Flo = 10e10;

    for start in points_at_clique {
        let cp = argmin::solver::trustregion::CauchyPoint::new();
        let tr = TrustRegion::new(cp)
            .with_max_radius(DELTA * divisor_r)
            .unwrap()
            .with_radius(divisor_r * DELTA / 100.0)
            .unwrap();

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
            .configure(
                |state| {
                    state
                        // Set initial parameters (depending on the solver,
                        // this may be required)
                        .param(*start)
                        // Set maximum iterations to 10
                        // (optional, set to `std::u64::MAX` if not provided)
                        .max_iters(10)
                }, // Set target cost. The solver stops when this cost
                   // function value is reached (optional)
                   //.target_cost(0.0)
            )
            // run the solver on the defined problem
            .run()
            .unwrap();
        let best_res = res.state.get_best_cost();
        if best_res < best_val {
            best_point = *res.state.get_best_param().unwrap();
            best_val = best_res;
        }
        // println!("{}", res);
        // println!("starting point was {}", start);
    }
    //println!("best val: {} and best point: f({}) = {}, when trying to find {}", best_val, best_point, func(best_point), point_to_find);
    (best_val, best_point)
}
*/

fn match_to_input(
    points_at_clique: &[Vector4<Flo>],
    cost: FindingR, 
    //divisor_r: Flo,
    delta_mult: Flo,
    max_iters: usize,
) -> (Flo, Vector4<Flo>) {
    let mut best_val: Flo = 10e10;
    let mut best_param: Vector4<Flo> = Vector4::from_vec(vec![0.0, 0.0, 0.0, 0.0]);

    //MAKE THIS PARALLEL
    for start in points_at_clique {
        let star = Array1::from_vec(vec![start[0], start[1], start[2], start[3]]);

        // let linesearch: BacktrackingLineSearch<Array1<FLO>, Array1<FLO>,_, FLO> = BacktrackingLineSearch::new(ArmijoCondition::new(0.0001f64).unwrap());
        //     //.with_bounds(divisor_r*DELTA/500.0,divisor_r*DELTA/10.0).unwrap();
        // let cp: DFP<_, FLO> = argmin::solver::quasinewton::DFP::new(linesearch);
        // let inv_hessian: Array2<FLO> = Array2::eye(4);
        //let state = IterState::new()
        // .param(star)
        // .inv_hessian(inv_hessian);
        // //let tr = TrustRegion::new(cp).with_max_radius(divisor_r*DELTA).unwrap().with_radius(divisor_r*DELTA/100.0).unwrap();
        // let (mut state_out, kv) = cp.init(&mut Problem::new(cost), state).unwrap();
        let cp: CauchyPoint<Flo> = argmin::solver::trustregion::CauchyPoint::new();

        let sr1: SR1TrustRegion<_, f64> = SR1TrustRegion::new(cp).with_radius(delta_mult*DELTAMOD);
        let res = Executor::new(cost.clone(), sr1)
            .configure(|state| state.param(star).max_iters(max_iters as u64))
            // run the solver on the defined problem
            .run()
            .unwrap();
        let best_res = res.state().best_cost;
        if best_res < best_val {
            best_val = best_res;
            best_param = Vector4::from_column_slice(
                res.state().best_param.as_ref().unwrap().as_slice().unwrap(),
            );
        }
    }
    //     let cost = FindingR {point_to_find, fun: func};

    // for start in points_at_clique {

    //     let cp = argmin::solver::trustregion::CauchyPoint::new();
    //     let tr = TrustRegion::new(cp).with_max_radius(divisor_r*DELTA).unwrap().with_radius(divisor_r*DELTA/100.0).unwrap();
    //     let res = Executor::new(cost, tr)
    //     .configure(|state|
    //         state
    //             .param(*start)
    //             .max_iters(30)
    //     )
    //     // run the solver on the defined problem
    //     .run().unwrap();
    //     let best_res = res.state().best_cost;
    //     if best_res < best_val {
    //         best_val = best_res;
    //         best_param = res.state().best_param.unwrap();
    //     }
    // }
    (best_val, best_param)
}

fn match_to_input2(
    points_at_clique: &[Vector4<Flo>],
    cost: FuncMin, 
    //divisor_r: Flo,
    delta_mult: Flo,
    max_iters: usize,
) -> (Flo, Vector4<Flo>) {
    let mut best_val: Flo = 10e10;
    let mut best_param: Vector4<Flo> = Vector4::from_vec(vec![0.0, 0.0, 0.0, 0.0]);

    //MAKE THIS PARALLEL
    for start in points_at_clique {
        let star = Array1::from_vec(vec![start[0], start[1], start[2], start[3]]);

        // let linesearch: BacktrackingLineSearch<Array1<FLO>, Array1<FLO>,_, FLO> = BacktrackingLineSearch::new(ArmijoCondition::new(0.0001f64).unwrap());
        //     //.with_bounds(divisor_r*DELTA/500.0,divisor_r*DELTA/10.0).unwrap();
        // let cp: DFP<_, FLO> = argmin::solver::quasinewton::DFP::new(linesearch);
        // let inv_hessian: Array2<FLO> = Array2::eye(4);
        //let state = IterState::new()
        // .param(star)
        // .inv_hessian(inv_hessian);
        // //let tr = TrustRegion::new(cp).with_max_radius(divisor_r*DELTA).unwrap().with_radius(divisor_r*DELTA/100.0).unwrap();
        // let (mut state_out, kv) = cp.init(&mut Problem::new(cost), state).unwrap();
        let cp: CauchyPoint<Flo> = argmin::solver::trustregion::CauchyPoint::new();

        let sr1: SR1TrustRegion<_, f64> = SR1TrustRegion::new(cp).with_radius(delta_mult*DELTAMOD);
        let res = Executor::new(cost.clone(), sr1)
            .configure(|state| state.param(star).max_iters(max_iters as u64))
            // run the solver on the defined problem
            .run()
            .unwrap();
        let best_res = res.state().best_cost;
        if best_res < best_val {
            best_val = best_res;
            best_param = Vector4::from_column_slice(
                res.state().best_param.as_ref().unwrap().as_slice().unwrap(),
            );
        }
    }
    //     let cost = FindingR {point_to_find, fun: func};

    // for start in points_at_clique {

    //     let cp = argmin::solver::trustregion::CauchyPoint::new();
    //     let tr = TrustRegion::new(cp).with_max_radius(divisor_r*DELTA).unwrap().with_radius(divisor_r*DELTA/100.0).unwrap();
    //     let res = Executor::new(cost, tr)
    //     .configure(|state|
    //         state
    //             .param(*start)
    //             .max_iters(30)
    //     )
    //     // run the solver on the defined problem
    //     .run().unwrap();
    //     let best_res = res.state().best_cost;
    //     if best_res < best_val {
    //         best_val = best_res;
    //         best_param = res.state().best_param.unwrap();
    //     }
    // }
    (best_val, best_param)
}

fn error_in_f<'a>(arr4: &[Vector4<Flo>], delta_mult: Flo, max_iter: usize, func: &'a (dyn Fn(Vector4<Flo>) -> Vector4<Flo> + 'a + Sync)) -> Flo {
    let mut avg_small_dist: Flo = 0.0;
    let amount: usize = 127;
    let (sender_glo, receiver): (Sender<Flo>, Receiver<Flo>) = channel();

    arr4.par_iter()
        .skip(arr4.len() - amount)
        .for_each_with(sender_glo, |sender, p| {
            let cost = FindingR {
                point_to_find: &p,
                fun: func,
            };
            let ret = match_to_input(&arr4, cost, delta_mult, max_iter);
            // println!(
            //     "When trying to find {}, we instead found {}",
            //     p,
            //     func(ret.1)
            // );
            sender.send(ret.0).unwrap();
        });
    for _ in 0..amount {
        let results = receiver.recv().unwrap();
        avg_small_dist += results;
    }
    avg_small_dist /= amount as Flo;
    //println!("Final avg dist is {avg_small_dist}");
    avg_small_dist
}

fn solve_for<'a>(p: Vector4<Flo>, arr4: &[Vector4<Flo>], delta_mult: Flo, max_iter: usize, func: &'a (dyn Fn(Vector4<Flo>) -> Vector4<Flo> + 'a + Sync)) -> Flo {
    let cost = FuncMin {
        point_to_find: &p,
        fun: func,
    };
    let ret = match_to_input2(&arr4, cost, delta_mult, max_iter);
    ret.0
}

fn divide(arr: Vector6<Flo>, divisor: Flo) -> Vector6<Flo> {
    Vector6::from_vec(vec![arr[0]/divisor, arr[1]/divisor, arr[2]/divisor, arr[3]/divisor, arr[4], arr[5]])
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut chunk_size: Nab = 0;
    let mut divisor_r: Flo = 1.0;

    match args.len() {
        1 => println!("Give me a file!"),
        2 => (), //corresponds to just a filename so then we don't chunk
        3 => chunk_size = args[2].parse::<usize>().unwrap() as Nab,
        _ => {
            chunk_size = args[2].parse::<usize>().unwrap() as Nab;
            divisor_r = args[3].parse::<Flo>().unwrap();
        }
    }
    //println!("We are choosing r to be {divisor_r}");
    let mut now: std::time::Instant = std::time::Instant::now();
    let file_path = &args[1];

    let mut arr = read_file_to_mat(file_path);
    let mut len = arr.len();
    //array is sorted beforehand and full duplicates are already removed. If, however, the coordinates
    //are the same, but angles are different by 1 or so due to machine erros, we deduplicate them here
    //dedup
    let mut k: usize = 0;
    while k < len-1 {
        if arr[k][0]==arr[k+1][0] && arr[k][1]==arr[k+1][1] && arr[k][2]==arr[k+1][2] && arr[k][3]==arr[k+1][3] {
            arr.remove(k+1);
            len -= 1;
        }
        else {
            k +=1;
        }
    }
    //find_r(&arr);

    // println!(
    //     "Reading the file of points took {} milliseconds and we have {} many points",
    //     elapsed_time.as_millis(),
    //     len
    // );
    if chunk_size > 0 {
        //println!("We are in chunking mode with chunk size {chunk_size}");
    } else {
        //println!("We are in non-chunking mode");
        chunk_size = len as Nab + 1;
    }

    // let file_len = file_path.len();
    // let mut skip = 0;
    // if file_len > 10 {
    //     skip = file_len-10;
    // }
    // let name = file_path.chars().skip(skip).collect::<String>();
    // println!("DataFile  , #Points  , ChunkSize, #TestedPo, MaxIter  , divisor_r, DeltaMut , AvgDist   , Time     ");

    // //start r at 1000.0
    // for r in 1..=20{
    //     let now3 = std::time::Instant::now();
    //     divisor_r = 1000.0+(r as Flo)*50.0;
    //     let max_clique = chunk_clique(
    //         chunk_size,
    //         (0..(len as Nab)).collect::<Vec<Nab>>(),
    //         &arr,
    //         divisor_r,
    //         2,
    //     );
    //     let vals = get_qs(max_clique, &arr);
    //     // vals.0 = vals.0.iter().map(|v| v/divisor_r).collect();
    //     let func = define_f(&vals.0, &vals.1, &divisor_r);
    //     println!("At next r, and it took {}s to get the func", now3.elapsed().as_secs());
    //     let arr4 = arr.iter().map(coords).collect::<Vec<Vector4<Flo>>>();

    //     for l in 1..=7 {
    //         for k in 1..17 {
    //             let now2 = std::time::Instant::now();
    //             let err = error_in_f(&arr4, divisor_r, k as Flo, 10*l, &func);
    //             println!("{name:>10},{len:>10},{chunk_size:>10},       127,{:>10},{divisor_r:>10},{k:>10}, {:>10},{:>9}s", 10*l,format!("{:.2}", err), now2.elapsed().as_secs());
    //         }
    //         println!(" ");
    //     }
    // }

    //divisor_r = 200.0;
    let arr_div = arr.iter().map(|a| divide(*a, divisor_r)).collect::<Vec<Vector6<Flo>>>(); 
    
    let arr4 = arr_div.iter().map(coords).collect::<Vec<Vector4<Flo>>>();

    let max_clique = chunk_clique(
        chunk_size,
        (0..(len as Nab)).collect::<Vec<Nab>>(),
        &arr4,
        //divisor_r,
        2,
    );
    let vals = get_qs(max_clique, &arr_div);
    // vals.0 = vals.0.iter().map(|v| v/divisor_r).collect();
    println!("{}", vals.0[0]*divisor_r);
    let func = define_f(&vals.0, &vals.1);
    println!("{}", func(vals.0[0])*divisor_r);

    //let point_to_find = coords(&arr[21]);//8

    // let mut best_r: FLO = 1000.0;
    // let mut smallest_dist: FLO = 10e10;
    // let mut best_input: Vector4<FLO> = Vector4::from_vec(vec![0.0,0.0,0.0,0.0]);
    // let mut best_out: Vector4<FLO> = Vector4::from_vec(vec![0.0,0.0,0.0,0.0]);
    //for k in 0..20 {
    //    divisor_r = 1000.0+ (k as FLO)*50.0;
    //    println!("{}", divisor_r);

    // if val < smallest_dist {
    //     best_input = input;
    //     best_out = output;
    //     best_r = r;
    //     smallest_dist = val;
    //     print!("With divisor {}, we get a distance of {}", best_r, smallest_dist);
    //     print!("where f({}) = {}, which is supposed to be {}", best_input, best_out, point_to_find);
    // }
    //}
    // println!("This is the very final best one!");
    // print!("With divisor {}, we get a distance of {}", best_r, smallest_dist);
    // print!("where f({}) = {}, which is supposed to be {}", best_input, best_out, point_to_find);

    //println!("Found max clique of len {}: {:?}", max_clique.len(), max_clique);

    //let now2 = std::time::Instant::now();

    // for ma in &vals.1 {
    //     println!("{}", ma[(0,0)]);
    // }
    //println!("{:?} {:?}", &vals.0[0], &vals.0[40]);

    //let elapsed_time2 = now2.elapsed();
    //println!("It took {} micro seconds to calc {} qr decomps", elapsed_time2.as_micros(), vals.0.len());
    //let start = &vals.0[9].clone(); //4

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

    let elapsed_time = now.elapsed();
    println!("The total process took {} seconds.", elapsed_time.as_secs());
}
