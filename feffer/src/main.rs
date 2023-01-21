use std::f64;
use rand::{self, distributions, prelude::Distribution};
use ndarray::{self, Array1};
use std::time;

static DIM: usize = 8;

// fn dot_product(a: &[f64], b: &[f64]) -> f64 {
//     // Calculate the dot product of two vectors. 
//     assert_eq!(a.len(), b.len()); 
//     let mut product: f64 = 0.0;
//     for i in 0..a.len() {
//         product += a[i] * b[i];
//     }
//     return product;
// }

// fn dist(x_point: &[f64], q_point: &[f64]) -> f64{
//     let diff = x_point - q_point; 
//     let sum: f64 = dot_product(diff,diff);
//     return sum.sqrt();
// }

// fn maxDotFunc(x_point: &Array1<f64>, q_point: &Array1<f64>) -> f64 {
//     f64::abs(x_point.dot(q_point)/x_point.dot(x_point))
// }

fn mu(x_point: &Array1<f64>, q_point: &Array1<f64>) -> f64{
    let diff: Array1<f64> = x_point-q_point;
    let dist: f64 = diff.dot(&diff);
    (1.0/(dist-1.0/3.0)).exp()/((1.0/(dist-1.0/3.0)).exp() + (1.0/(1.0/2.0-dist)).exp())
}

fn calc_dist_from1(x0_point: &Array1<f64>, x_point: &Array1<f64>, Y: &[Array1<f64>]) -> f64{
    if Y.len() != 0 {
        let max_y: f64 = Y
        .into_iter()
        .map(|y| 
            f64::abs(x_point.dot(y)/x_point.dot(x_point))).reduce(f64::max).unwrap();
        max_y
        .max(f64::abs(1.0-(x0_point-x_point).dot(&(x0_point-x_point))))
    }
    else {
        f64::abs(1.0-(x0_point-x_point).dot(&(x0_point-x_point)))
    }
}

fn argmin(nets: &Vec<f64>) -> usize {
    *nets
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index).get_or_insert(0_usize)
}

fn find_disc2(x0_point: &Array1<f64>, X: &[Array1<f64>], n: u32) -> Vec<Array1<f64>> 
{
    let mut output: Vec<Array1<f64>> = vec![];
    for i in 0..n{
        output.append(&mut 
            vec![X[
                argmin(&mut X
                    .into_iter()
                    .map(|x| 
                        calc_dist_from1(x0_point, x, &output)).collect::<Vec<f64>>())]
                        .clone()]);
    }
    output
}

fn find_disc(x0_point: &Array1<f64>, X: &[Array1<f64>], n: u32) -> Vec<Array1<f64>>
{
    let mut output: Vec<Array1<f64>> = vec![];
    for i in 0..n{
        let distance_vec: Vec<f64> = X
        .into_iter()
        .map(|x| 
            calc_dist_from1(x0_point, x, &output)).collect::<Vec<f64>>();
        let mut current_min: f64 = distance_vec[0];
        let mut current_index: usize = 0;
        for j in 0..distance_vec.len(){
            if distance_vec[j] < current_min {
                current_index = j;
                current_min = distance_vec[j];
            }
        }
        output.append(&mut vec![X[current_index].clone()]);
    }
    output
}

fn main() {
    let range: distributions::Uniform<f64> = distributions::Uniform::from(0.0..1.0);
    let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
    let x_vals: [Array1<f64>; 1000] = [(); 1000].map(|_| range.sample_iter(&mut rng).take(DIM).collect());
    
    let xs: Array1<f64> = range.sample_iter(&mut rng).take(DIM).collect();
    let mut now: std::time::Instant = std::time::Instant::now();
    let res1: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>> = find_disc(&xs, &x_vals, 60);
    let elapsed_time = now.elapsed();
    println!("Running slow_function() took {} seconds.", elapsed_time.as_millis());
    now = std::time::Instant::now();
    let res2: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>> = find_disc2(&xs, &x_vals, 60);
    let elapsed_time = now.elapsed();

    println!("Running slow_function() took {} seconds.", elapsed_time.as_millis());
    println!("First objects are {:?} and {:?}", res1[0], res2[0])
}