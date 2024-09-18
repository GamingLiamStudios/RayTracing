// Copyright (C) 2024 GLStudios
// SPDX-License-Identifier: LGPL-2.1-only

mod fixed;

use core::f64;

pub use fixed::{
    Static,
    StaticBuilder,
};

use crate::{
    render::{
        Object,
        Ray,
    },
    types::{
        DVec3A,
        SwizzleLoc,
    },
    ACNE_MIN,
};

#[derive(Clone, Copy, Debug)]
pub struct BoundingBox {
    max: DVec3A,
    min: DVec3A,
}

impl BoundingBox {
    pub const fn new() -> Self {
        Self {
            max: DVec3A::splat(f64::MIN),
            min: DVec3A::splat(f64::MAX),
        }
    }

    #[inline]
    pub fn intersects(
        &self,
        ray: &Ray,
    ) -> Option<f64> {
        // TODO: Check if blendv is actually any faster than min-max swaps
        // Theoretically should be faster according to intel docs
        // (4 vs 2 latency, .5 vs .66 throughput)

        let bmin = DVec3A::blend_sign(self.min, self.max, ray.inv_dir);
        let bmax = DVec3A::blend_sign(self.max, self.min, ray.inv_dir);

        let tmin = ((bmin - ray.origin) * ray.inv_dir).max_element().max(0.0);
        let tmax = ((bmax - ray.origin) * ray.inv_dir)
            .min_element()
            .min(f64::INFINITY);

        if tmin > tmax {
            None
        } else {
            Some(tmin.max(ACNE_MIN))
        }
    }

    #[inline]
    fn grow_to_include_point(
        &mut self,
        point: DVec3A,
    ) {
        self.max = self.max.max(point);
        self.min = self.min.min(point);
    }

    pub fn half_surface_area(&self) -> f64 {
        use SwizzleLoc::{
            X,
            Y,
            Z,
        };
        let size = self.max - self.min;
        size.dot(size.permute::<{ [Y, Z, X] }>())
    }

    pub fn grow_to_include(
        &mut self,
        object: &Object,
    ) {
        match object {
            Object::Sphere { center, radius } => {
                self.grow_to_include_point(center - *radius);
                self.grow_to_include_point(center + *radius);
            },
            Object::Triangle { a, b, c } => {
                self.grow_to_include_point(a.pos);
                self.grow_to_include_point(b.pos);
                self.grow_to_include_point(c.pos);
            },
        }
    }

    pub fn merge(
        &mut self,
        right: &Self,
    ) {
        self.min = self.min.min(right.min);
        self.max = self.max.max(right.max);
    }
}
