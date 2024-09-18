// Copyright (C) 2024 GLStudios
// SPDX-License-Identifier: LGPL-2.1-only

use std::marker::ConstParamTy;

pub mod vec3;
pub type DVec3A = vec3::Vec3A<f64>;

#[repr(usize)]
#[derive(ConstParamTy, PartialEq, Eq, Copy, Clone)]
pub enum SwizzleLoc {
    X = 0,
    Y = 1,
    Z = 2,
}

const fn to_swizzle<const N: usize, const M: usize>(index: [SwizzleLoc; N]) -> [usize; M] {
    let mut output = [3; M];
    let mut i = 0;
    while i < N {
        output[i] = index[i] as usize;
        i += 1;
    }
    output
}
