// Copyright (C) 2024 GLStudios
// SPDX-License-Identifier: LGPL-2.1-only

mod fixed;

use core::f32;
use std::arch::x86_64::{
    __m128,
    _mm_blendv_ps,
};

pub use fixed::{
    Static,
    StaticBuilder,
};
use glam::Vec3A;

use crate::render::{
    Object,
    Ray,
};

#[repr(C)]
union Vec3AInternal {
    i: __m128,
    v: Vec3A,
}

#[derive(Clone, Copy, Debug)]
pub struct BoundingBox {
    max: Vec3A,
    min: Vec3A,
}

impl BoundingBox {
    pub const fn new() -> Self {
        Self {
            max: Vec3A::splat(f32::MIN),
            min: Vec3A::splat(f32::MAX),
        }
    }

    #[inline]
    pub fn intersects(
        &self,
        ray: &Ray,
    ) -> Option<f32> {
        // TODO: Check if blendv is actually any faster than min-max swaps
        // Theoretically should be faster according to intel docs
        // (4 vs 2 latency, .5 vs .66 throughput)
        let bmin = unsafe {
            Vec3AInternal {
                i: _mm_blendv_ps(
                    Vec3AInternal { v: self.min }.i,
                    Vec3AInternal { v: self.max }.i,
                    Vec3AInternal { v: ray.inv_dir }.i,
                ),
            }
            .v
        };
        let bmax = unsafe {
            Vec3AInternal {
                i: _mm_blendv_ps(
                    Vec3AInternal { v: self.max }.i,
                    Vec3AInternal { v: self.min }.i,
                    Vec3AInternal { v: ray.inv_dir }.i,
                ),
            }
            .v
        };

        let tmin = ((bmin - ray.origin) * ray.inv_dir).max_element().max(0.0);
        let tmax = ((bmax - ray.origin) * ray.inv_dir).min_element();

        if tmin > tmax {
            None
        } else {
            Some(tmin)
        }
    }

    #[inline]
    fn grow_to_include_point(
        &mut self,
        point: Vec3A,
    ) {
        self.max = self.max.max(point);
        self.min = self.min.min(point);
    }

    pub fn half_surface_area(&self) -> f32 {
        let size = self.max - self.min;
        (size.x * size.y).abs() + (size.y * size.z).abs() + (size.z * size.x).abs()
    }

    pub fn grow_to_include(
        &mut self,
        object: &Object,
    ) {
        match object {
            Object::Sphere { center, radius } => {
                self.grow_to_include_point(center - Vec3A::splat(*radius));
                self.grow_to_include_point(center + Vec3A::splat(*radius));
            },
            Object::Triangle { a, b, c } => {
                self.grow_to_include_point(a.pos);
                self.grow_to_include_point(b.pos);
                self.grow_to_include_point(c.pos);
            },
        }
    }
}
