// TODO: Benchmark differences between Enum and Trait for storing render objects

use core::f32;
use std::ops::Bound;

use glam::Vec3A;
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
            direction: direction.normalize(),
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
    pub normal: Vec3A,
}

pub struct Scene {
    pub materials: SlotMap<slotmap::DefaultKey, Material>,
    pub objects:   Vec<Static>,
}

pub struct Material {
    pub diffuse:   Vec3A,
    pub emmitance: Vec3A,

    pub smoothness: f32,
    pub radiance:   f32,
}

pub struct Vertex {
    pub(crate) pos:    Vec3A,
    pub(crate) normal: Vec3A,
    // TODO: uv: Vec2,
}

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
            objects:   Vec::new(),
            materials: SlotMap::new(),
        }
    }

    pub fn add_material(
        &mut self,
        mat: Material,
    ) -> slotmap::DefaultKey {
        self.materials.insert(mat)
    }

    pub fn add_object(
        &mut self,
        object: Static,
    ) {
        self.objects.push(object);
    }

    pub fn material(
        &self,
        mat_idx: slotmap::DefaultKey,
    ) -> Option<&Material> {
        self.materials.get(mat_idx)
    }

    pub fn hit_scene(
        &self,
        ray: &Ray,
        mut search_range: (Bound<f32>, Bound<f32>),
    ) -> Option<(HitRecord, slotmap::DefaultKey)> {
        let mut closest = None;

        for bvh in &self.objects {
            let Some(hit) = bvh.hit_scene(ray, search_range) else {
                continue;
            };

            search_range.1 = Bound::Included(hit.0.along);
            closest = Some(hit);
        }

        closest
    }
}

impl Material {
    pub fn bounce_ray(
        &self,
        rng: &mut ThreadRng,
        incoming: &Vec3A,
        normal: Vec3A,
    ) -> Vec3A {
        let mut diffuse = (normal + random_unit_sphere(rng)).normalize();
        if diffuse.length_squared() <= f32::EPSILON {
            println!("Near zero diffuse");
            diffuse = normal;
        }
        let specular = incoming.reflect(normal);
        diffuse.lerp(specular, self.smoothness)
    }
}
