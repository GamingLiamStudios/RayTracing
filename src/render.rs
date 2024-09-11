// Copyright (C) 2024 GLStudios
// SPDX-License-Identifier: LGPL-2.1-only

use core::f32;
use std::{
    collections::BTreeMap,
    ops::Bound,
};

use glam::{
    Affine3A,
    Mat4,
    Vec3A,
};
use rand::{
    rngs::ThreadRng,
    Rng,
};
use slotmap::SlotMap;

use crate::bvh::Static;

#[derive(Clone, Copy)]
pub struct Ray {
    pub origin:    Vec3A,
    pub direction: Vec3A,
    pub inv_dir:   Vec3A,
}

impl Ray {
    pub fn new(
        origin: Vec3A,
        direction: Vec3A,
    ) -> Self {
        Self {
            origin,
            direction,
            inv_dir: 1.0 / direction,
        }
    }

    #[inline]
    pub fn set_direction(
        &mut self,
        new: Vec3A,
    ) {
        self.direction = new;
        self.inv_dir = 1.0 / self.direction;
    }
}

pub struct HitRecord {
    pub along:  f32,
    pub point:  Vec3A,
    pub normal: Vec3A,
}

slotmap::new_key_type! {
    pub struct MaterialKey;
    pub struct ObjectKey;
}

pub struct Scene {
    // TODO: Make this a Top Level BVH
    pub materials: SlotMap<MaterialKey, Material>,
    pub objects:   SlotMap<ObjectKey, (Static, Vec<Affine3A>)>,
}

pub struct Material {
    pub diffuse:   Vec3A,
    pub emmitance: Vec3A,

    pub smoothness: f32,
    pub radiance:   f32,
}

#[derive(Clone, Copy)]
pub struct Vertex {
    pub(crate) pos:    Vec3A,
    pub(crate) normal: Vec3A,
    // TODO: uv: Vec2,
}

// TODO: Benchmark differences between Enum and Trait for storing render objects
#[derive(Clone, Copy)]
pub enum Object {
    // TODO: Support Ellipsoid
    Sphere { center: Vec3A, radius: f32 },
    Triangle { a: Vertex, b: Vertex, c: Vertex },
}

impl Object {
    pub fn center(&self) -> Vec3A {
        match self {
            Self::Sphere { center, radius: _ } => *center,
            Self::Triangle { a, b, c } => (a.pos + b.pos + c.pos) / 3.0,
        }
    }
}

#[inline]
pub fn random_unit_sphere(rng: &mut ThreadRng) -> Vec3A {
    loop {
        // Rejection sampling
        let x = rng.gen_range(-1f32..1f32);
        let y = rng.gen_range(-1f32..1f32);
        let z = rng.gen_range(-1f32..1f32);

        let vector = Vec3A::new(x, y, z);
        if vector.length_squared() <= 1f32 {
            return vector.normalize_or_zero();
        }
    }
}

impl Scene {
    pub fn new() -> Self {
        Self {
            objects:   SlotMap::with_key(),
            materials: SlotMap::with_key(),
        }
    }

    pub fn create_material(
        &mut self,
        mat: Material,
    ) -> MaterialKey {
        self.materials.insert(mat)
    }

    pub fn insert_object(
        &mut self,
        object: Static,
        transform: Affine3A,
    ) -> ObjectKey {
        self.objects.insert((object, vec![transform]))
    }

    pub fn add_instance(
        &mut self,
        object: ObjectKey,
        transform: Affine3A,
    ) {
        let Some((_, instances)) = self.objects.get_mut(object) else {
            return;
        };
        instances.push(transform);
    }

    pub fn material(
        &self,
        mat_idx: MaterialKey,
    ) -> Option<&Material> {
        self.materials.get(mat_idx)
    }

    pub fn hit_scene(
        &self,
        ray: &Ray,
        mut search_range: (Bound<f32>, Bound<f32>),
    ) -> Option<(HitRecord, MaterialKey)> {
        let mut closest = None;

        for (bvh, transforms) in self.objects.values() {
            for transform in transforms {
                let transformed_ray = Ray::new(
                    transform.inverse().transform_point3a(ray.origin),
                    transform.inverse().transform_vector3a(ray.direction),
                );
                let Some(mut hit) = bvh.hit_scene(&transformed_ray, search_range) else {
                    continue;
                };

                hit.0.point = transform.transform_point3a(hit.0.point);

                search_range.1 = Bound::Included(hit.0.along);
                closest = Some(hit);
            }
        }

        closest
    }
}

impl Material {
    pub fn bounce_ray(
        &self,
        rng: &mut ThreadRng,
        ray: &Ray,
        normal: Vec3A,
    ) -> Vec3A {
        let diffuse = ray.origin + normal + random_unit_sphere(rng);
        let specular = ray.direction.normalize().reflect(normal);
        diffuse.lerp(specular, self.smoothness)
    }
}
