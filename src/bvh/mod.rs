// Copyright (C) 2024 GLStudios
// SPDX-License-Identifier: LGPL-2.1-only

mod fixed;

use core::f64;

pub use fixed::{
    Static,
    StaticBuilder,
};
use glam::{
    DVec3,
    Vec3Swizzles,
};

use crate::{
    render::{
        Object,
        Ray,
    },
    ACNE_MIN,
};

#[inline]
fn min<T: PartialOrd>(
    left: T,
    right: T,
) -> T {
    if left < right {
        left
    } else {
        right
    }
}

#[inline]
fn max<T: PartialOrd>(
    left: T,
    right: T,
) -> T {
    if left > right {
        left
    } else {
        right
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BoundingBox {
    max: DVec3,
    min: DVec3,
}

impl BoundingBox {
    pub const fn new() -> Self {
        Self {
            max: DVec3::splat(f64::MIN),
            min: DVec3::splat(f64::MAX),
        }
    }

    #[inline]
    pub fn intersects(
        &self,
        ray: &Ray,
    ) -> Option<f64> {
        let t0 = ((self.min - ray.origin) * ray.inv_dir).to_array();
        let t1 = ((self.max - ray.origin) * ray.inv_dir).to_array();

        let mut tmin = 0.0;
        let mut tmax = f64::INFINITY;
        for (&t0, &t1) in t0.iter().zip(t1.iter()) {
            tmin = min(max(t0, tmin), max(t1, tmin));
            tmax = max(min(t0, tmax), min(t1, tmax));
        }

        if tmin > tmax {
            None
        } else {
            Some(tmin.max(ACNE_MIN))
        }
    }

    #[inline]
    fn grow_to_include_point(
        &mut self,
        point: DVec3,
    ) {
        self.max = self.max.max(point);
        self.min = self.min.min(point);
    }

    pub fn half_surface_area(&self) -> f64 {
        let size = self.max - self.min;
        size.dot(size.yzx())
    }

    pub fn grow_to_include(
        &mut self,
        object: &Object,
    ) {
        match object {
            Object::Sphere { center, radius } => {
                self.grow_to_include_point(center - DVec3::splat(*radius));
                self.grow_to_include_point(center + DVec3::splat(*radius));
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
