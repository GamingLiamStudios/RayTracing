// Copyright (C) 2024 GLStudios
// SPDX-License-Identifier: LGPL-2.1-only

use core::f64;
use std::ops::Bound;

use glam::{
    DAffine3,
    DVec3,
};
use rand::{
    rngs::ThreadRng,
    Rng,
};
use slotmap::SlotMap;

use crate::bvh::Static;

#[derive(Clone, Copy)]
pub struct Ray {
    pub origin:    DVec3,
    pub direction: DVec3,
    pub inv_dir:   DVec3,
}

impl Ray {
    pub fn new(
        origin: DVec3,
        direction: DVec3,
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
        new: DVec3,
    ) {
        self.direction = new;
        self.inv_dir = 1.0 / self.direction;
    }
}

pub struct HitRecord {
    pub along:  f64,
    pub normal: DVec3,
}

slotmap::new_key_type! {
    pub struct MaterialKey;
    pub struct ObjectKey;
}

#[derive(Debug)]
pub struct Instance {
    transform: DAffine3,
    materials: Vec<MaterialKey>,
}

#[derive(Debug)]
pub struct Scene {
    // TODO: Make this a Top Level BVH
    pub materials: SlotMap<MaterialKey, Material>,
    pub objects:   SlotMap<ObjectKey, (Static, Vec<Instance>)>,
}

#[derive(Debug)]
pub struct Material {
    pub diffuse:   DVec3,
    pub emmitance: DVec3,

    pub smoothness: f64,
    pub radiance:   f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub(crate) pos:    DVec3,
    pub(crate) normal: DVec3,
    // TODO: uv: DVec2,
}

// TODO: Benchmark differences between Enum and Trait for storing render objects
#[derive(Clone, Copy, Debug)]
pub enum Object {
    // TODO: Support Ellipsoid
    Sphere { center: DVec3, radius: f64 },
    Triangle { a: Vertex, b: Vertex, c: Vertex },
}

impl Object {
    pub fn center(&self) -> DVec3 {
        match self {
            Self::Sphere { center, radius: _ } => *center,
            Self::Triangle { a, b, c } => (a.pos + b.pos + c.pos) / 3.0,
        }
    }
}

#[inline]
pub fn random_unit_sphere(rng: &mut ThreadRng) -> DVec3 {
    loop {
        // Rejection sampling
        let x = rng.gen_range(-1.0..1.0);
        let y = rng.gen_range(-1.0..1.0);
        let z = rng.gen_range(-1.0..1.0);

        let vector = DVec3::new(x, y, z);
        if vector.length_squared() <= 1.0 {
            return vector;
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
        transform: DAffine3,
        materials: Vec<MaterialKey>,
    ) -> ObjectKey {
        let objects = object.objects.len();
        let id = self.objects.insert((object, vec![Instance {
            transform,
            materials,
        }]));
        let (scale, rotation, translation) = transform.to_scale_rotation_translation();
        tracing::debug!(?id, objects, instances = 1, transform.scale = ?scale, transform.rotation = ?rotation, transform.translation = ?translation);
        id
    }

    pub fn add_instance(
        &mut self,
        object: ObjectKey,
        transform: DAffine3,
        materials: Vec<MaterialKey>,
    ) {
        let Some((_, instances)) = self.objects.get_mut(object) else {
            return;
        };
        instances.push(Instance {
            transform,
            materials,
        });

        let (scale, rotation, translation) = transform.to_scale_rotation_translation();
        tracing::debug!(id = ?object, instances = instances.len(), transform.scale = ?scale, transform.rotation = ?rotation, transform.translation = ?translation);
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
        mut search_range: (Bound<f64>, Bound<f64>),
    ) -> Option<(HitRecord, MaterialKey)> {
        let mut closest = None;

        for (bvh, transforms) in self.objects.values() {
            for Instance {
                transform,
                materials,
            } in transforms
            {
                let transformed_ray = Ray::new(
                    transform.inverse().transform_point3(ray.origin),
                    transform.inverse().transform_vector3(ray.direction),
                );
                let Some((hit_record, material)) = bvh.hit_scene(&transformed_ray, search_range)
                else {
                    continue;
                };

                search_range.1 = Bound::Included(hit_record.along);
                closest = Some((hit_record, materials[material]));
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
        normal: DVec3,
    ) -> DVec3 {
        let diffuse = ray.origin + normal + random_unit_sphere(rng);
        let specular = ray.direction.normalize().reflect(normal);
        diffuse.lerp(specular, self.smoothness)
    }
}
