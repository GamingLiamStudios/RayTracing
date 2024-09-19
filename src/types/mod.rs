// Copyright (C) 2024 GLStudios
// SPDX-License-Identifier: LGPL-2.1-only

use std::marker::ConstParamTy;

use glam::DAffine3;

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

#[derive(Debug, Clone, Copy)]
pub struct DAffine3A {
    x:           DVec3A,
    y:           DVec3A,
    z:           DVec3A,
    translation: DVec3A,
}

impl DAffine3A {
    pub fn from_daffine3(input: DAffine3) -> Self {
        Self {
            x:           DVec3A::from_dvec3(input.x_axis),
            y:           DVec3A::from_dvec3(input.y_axis),
            z:           DVec3A::from_dvec3(input.z_axis),
            translation: DVec3A::from_dvec3(input.translation),
        }
    }

    #[inline]
    pub fn transform_vector3(
        self,
        vector: DVec3A,
    ) -> DVec3A {
        self.x * vector[0] + self.y * vector[1] + self.z * vector[2]
    }

    #[inline]
    pub fn transform_point3(
        self,
        point: DVec3A,
    ) -> DVec3A {
        self.x * point[0] + self.y * point[1] + self.z * point[2] + self.translation
    }
}
