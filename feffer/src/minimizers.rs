use argmin::core::{CostFunction, Error, Gradient, Hessian};
//use plotters::prelude::*;
use finitediff::FiniteDiff;
use na::Vector4;
use nalgebra as na;
use ndarray::{Array1, Array2};

type Flo = f64;


#[derive(Clone)]
pub struct FindingR<'a> {
    pub point_to_find: &'a Vector4<Flo>,
    pub fun: &'a (dyn Fn(Vector4<Flo>) -> Vector4<Flo> + 'a + Sync),
}

impl FindingR<'_> {
    fn norm_func(&self, p: &Vector4<Flo>) -> Flo {
        ((self.fun)(*p) - self.point_to_find).norm()
        // FLO::sqrt(
        //     ((self.fun)(*p)[0]-self.point_to_find[0]).powi(2)+((self.fun)(*p)[1]-self.point_to_find[1]).powi(2)
        // )
        //((self.fun)(*p) - self.point_to_find).norm()
    }
    fn array_norm_func(&self, y: &Array1<Flo>) -> Flo {
        self.norm_func(&Vector4::from_column_slice(y.as_slice().unwrap()))
    }
    // fn array_gradient(&self, y: &Array1<FLO>) -> Array1<FLO> {
    //     self.gradient(&Vector4::from_column_slice(y.as_slice().unwrap())).unwrap()
    // }
}

impl CostFunction for FindingR<'_> {
    /// Type of the parameter vector
    type Param = Array1<Flo>;
    /// Type of the return value computed by the cost function
    type Output = Flo;
    /// Apply the cost function to a parameter `p`
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        //print!("{} ", self.norm_func(p));
        Ok(self.array_norm_func(p))
    }
}

impl Gradient for FindingR<'_> {
    /// Type of the parameter vector
    type Param = Array1<Flo>;
    /// Type of the gradient
    type Gradient = Array1<Flo>;
    /// Compute the gradient at parameter `p`.
    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        let grad = FiniteDiff::forward_diff(p, &{ |x| self.array_norm_func(x) });
        //println!("{}", grad);
        Ok(grad)
        //Ok(Vector4::from_column_slice(grad.as_slice().unwrap()))
    }
}

impl Hessian for FindingR<'_> {
    type Param = Array1<Flo>;
    /// Type of the gradient
    type Hessian = Array2<Flo>;

    /// Compute gradient of rosenbrock function
    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        //let func2 = {|y: &Array1<FLO>| self.norm_func(Vector4::from_column_slice(y.as_slice().unwrap()))};
        let grad = FiniteDiff::forward_hessian(p, &{ |x| self.gradient(x).unwrap() }); //self.array_norm_func(x.clone()) });
        //let grad = FiniteDiff::forward_hessian_nograd(&y, &func2);
        //println!("{}", grad);
        Ok(grad)
        // Ok(Matrix4::new(
        //     grad[(0,0)], grad[(0,1)], grad[(0,2)],grad[(0,3)],
        //     grad[(1,0)], grad[(1,1)], grad[(1,2)],grad[(1,3)],
        //     grad[(2,0)], grad[(2,1)], grad[(2,2)],grad[(2,3)],
        //     grad[(3,0)], grad[(3,1)], grad[(3,2)],grad[(3,3)],
        // ))
        //Ok(Matrix4::from_column_slice(grad.as_slice().unwrap()))
        //Ok(rosenbrock_2d_hessian(param, 1.0, 100.0))
    }
}


#[derive(Clone)]
pub struct FuncMin<'a> {
    pub point_to_find: &'a Vector4<Flo>,
    pub fun: &'a (dyn Fn(Vector4<Flo>) -> Vector4<Flo> + 'a + Sync),
}

impl FuncMin<'_> {
    fn norm_func(&self, p: &Vector4<Flo>) -> Flo {
        Flo::sqrt(
            ((self.fun)(*p)[0] - self.point_to_find[0]).powi(2)
                + ((self.fun)(*p)[1] - self.point_to_find[1]).powi(2),
        )
        // FLO::sqrt(
        //     ((self.fun)(*p)[0]-self.point_to_find[0]).powi(2)+((self.fun)(*p)[1]-self.point_to_find[1]).powi(2)
        // )
        //((self.fun)(*p) - self.point_to_find).norm()
    }
    fn array_norm_func(&self, y: &Array1<Flo>) -> Flo {
        self.norm_func(&Vector4::from_column_slice(y.as_slice().unwrap()))
    }
    // fn array_gradient(&self, y: &Array1<FLO>) -> Array1<FLO> {
    //     self.gradient(&Vector4::from_column_slice(y.as_slice().unwrap())).unwrap()
    // }
}

impl CostFunction for FuncMin<'_> {
    /// Type of the parameter vector
    type Param = Array1<Flo>;
    /// Type of the return value computed by the cost function
    type Output = Flo;
    /// Apply the cost function to a parameter `p`
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        //print!("{} ", self.norm_func(p));
        Ok(self.array_norm_func(p))
    }
}

impl Gradient for FuncMin<'_> {
    /// Type of the parameter vector
    type Param = Array1<Flo>;
    /// Type of the gradient
    type Gradient = Array1<Flo>;
    /// Compute the gradient at parameter `p`.
    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        let grad = FiniteDiff::forward_diff(p, &{ |x| self.array_norm_func(x) });
        //println!("{}", grad);
        Ok(grad)
        //Ok(Vector4::from_column_slice(grad.as_slice().unwrap()))
    }
}

impl Hessian for FuncMin<'_> {
    type Param = Array1<Flo>;
    /// Type of the gradient
    type Hessian = Array2<Flo>;

    /// Compute gradient of rosenbrock function
    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        //let func2 = {|y: &Array1<FLO>| self.norm_func(Vector4::from_column_slice(y.as_slice().unwrap()))};
        let grad = FiniteDiff::forward_hessian(p, &{ |x| self.gradient(x).unwrap() }); //self.array_norm_func(x.clone()) });
        //let grad = FiniteDiff::forward_hessian_nograd(&y, &func2);
        //println!("{}", grad);
        Ok(grad)
        // Ok(Matrix4::new(
        //     grad[(0,0)], grad[(0,1)], grad[(0,2)],grad[(0,3)],
        //     grad[(1,0)], grad[(1,1)], grad[(1,2)],grad[(1,3)],
        //     grad[(2,0)], grad[(2,1)], grad[(2,2)],grad[(2,3)],
        //     grad[(3,0)], grad[(3,1)], grad[(3,2)],grad[(3,3)],
        // ))
        //Ok(Matrix4::from_column_slice(grad.as_slice().unwrap()))
        //Ok(rosenbrock_2d_hessian(param, 1.0, 100.0))
    }
}