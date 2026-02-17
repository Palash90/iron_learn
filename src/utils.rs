use std::f64::consts::PI;

use crate::numeric::FloatingPoint;

pub fn get_current_lr<D>(base_lr: D, lr_adjustment: bool, lr_min: D, total_timeline: usize, global_i: usize) -> D
where
    D: FloatingPoint,
{
    let cos_term = ((D::from_f64(PI) * D::from_u32(global_i as u32)) / D::from_u32(total_timeline as u32)).cos();

    let decay_factor = D::from_f64(0.5) * (D::one() + cos_term);
    match lr_adjustment {
        true => lr_min + (base_lr - lr_min) * decay_factor,
        false => base_lr,
    }
}