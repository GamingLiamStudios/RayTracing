// Copyright (C) 2024 GLStudios
// SPDX-License-Identifier: LGPL-2.1-only

#![feature(array_chunks)]
#![feature(new_range_api)]
#![feature(generic_arg_infer)]
#![feature(iter_map_windows)]
#![feature(iter_array_chunks)]
#![feature(const_mut_refs)]
#![feature(allocator_api)]

use core::f32;
use std::{
    io::Write,
    ops::Bound,
};

use glam::{
    Affine3A,
    Quat,
    Vec2,
    Vec3,
    Vec3A,
};
use gltf::mesh::Mode;
use indicatif::{
    ParallelProgressIterator,
    ProgressBar,
};
use rand::{
    rngs::ThreadRng,
    Rng,
};
use rayon::iter::{
    IndexedParallelIterator,
    IntoParallelRefMutIterator,
    ParallelBridge,
    ParallelIterator,
};
use render::{
    Material,
    Object,
    Vertex,
};
use rgb::Rgb;
use tracing::{
    event,
    info,
    instrument,
};

mod bvh;
mod render;
mod types;

pub const ACNE_MIN: f32 = 0.01;

pub const WIDTH: u32 = 800;
pub const HEIGHT: u32 = 600;

pub const SAMPLES: u32 = 50;
pub const BOUNCES: usize = 500;

#[inline]
pub fn random_unit_circle(rng: &mut ThreadRng) -> Vec2 {
    loop {
        // Rejection sampling
        let x = rng.gen_range(-1f32..1f32);
        let y = rng.gen_range(-1f32..1f32);

        let vector = Vec2::new(x, y);
        if vector.length_squared() <= 1f32 {
            return vector.normalize_or_zero();
        }
    }
}

#[inline]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn process_ray(mut input: Vec3A) -> Rgb<u16> {
    // Post-processing
    // "Tone Mapping"
    input = input.clamp(Vec3A::splat(0f32), Vec3A::splat(1f32));
    // Gamma correction
    input = input.powf(1.0 / 1.8);

    // TODO: How to actually HDR/tonemap

    // Saturating cast - Will auto-clamp within bounds of u16
    input *= f32::from(u16::MAX);
    Rgb::new(input.x as u16, input.y as u16, input.z as u16)
}

#[allow(clippy::cast_precision_loss)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::many_single_char_names)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fmt_subscriber = tracing_subscriber::fmt::Subscriber::builder()
        .with_max_level(tracing::Level::TRACE)
        .finish();
    tracing::subscriber::set_global_default(fmt_subscriber)?;

    let mut scene = render::Scene::new();
    let mut builder = bvh::StaticBuilder::new();

    println!("Building scene...");
    let begin_time = std::time::Instant::now();

    let bunny_id = {
        let material = Material {
            diffuse:    Vec3A::new(0.1, 0.3, 0.3),
            emmitance:  Vec3A::splat(1.0),
            smoothness: 0.3,
            radiance:   0.0,
        };
        let material = scene.create_material(material);
        let (document, buffers, _) = gltf::import("models/bunny.glb")?;

        for mesh in document.meshes() {
            for prim in mesh.primitives() {
                if prim.mode() != Mode::Triangles {
                    continue;
                }

                let reader = prim.reader(|buf| buffers.get(buf.index()).map(|d| &*d.0));
                let Some(positions) = reader.read_positions() else {
                    println!("No positions attached to triangle mesh");
                    continue;
                };
                let positions = positions.collect::<Vec<_>>();

                let Some(normals) = reader.read_normals() else {
                    println!("No normals attached to triangle mesh");
                    continue;
                };
                let normals = normals.collect::<Vec<_>>();

                let Some(indices) = reader.read_indices() else {
                    println!("No indices attached to triangle mesh");
                    continue;
                };

                let mut tris = 0;
                for [a, b, c] in indices.into_u32().map(|v| v as usize).array_chunks() {
                    let tri = Object::Triangle {
                        a: Vertex {
                            pos:    Vec3A::from_array(positions[a]),
                            normal: Vec3A::from_array(normals[a]),
                        },
                        b: Vertex {
                            pos:    Vec3A::from_array(positions[b]),
                            normal: Vec3A::from_array(normals[b]),
                        },
                        c: Vertex {
                            pos:    Vec3A::from_array(positions[c]),
                            normal: Vec3A::from_array(normals[c]),
                        },
                    };
                    builder.append(tri, material);
                    tris += 1;
                }
                println!("Added mesh with {tris} tris");
            }
        }

        println!("Object contains {} primatives", builder.len());
        scene.insert_object(
            builder.build(),
            Affine3A::from_scale_rotation_translation(
                Vec3::splat(1.0),
                Quat::from_euler(
                    glam::EulerRot::XYZ,
                    90.0f32.to_radians(),
                    0.0f32.to_radians(),
                    90.0f32.to_radians(),
                ),
                Vec3::new(-0.3, 0.0, 0.05),
            ),
        )
    };

    scene.add_instance(
        bunny_id,
        Affine3A::from_scale_rotation_translation(
            Vec3::splat(1.0),
            Quat::from_euler(
                glam::EulerRot::XYZ,
                90.0f32.to_radians(),
                0.0f32.to_radians(),
                -90.0f32.to_radians(),
            ),
            Vec3::new(0.0, 0.0, 0.05),
        ),
    );

    let _sphere_id = {
        // Earth
        let mut builder = bvh::StaticBuilder::new();

        let object = render::Object::Sphere {
            center: Vec3A::new(0.0, -1001.0, 0.0),
            radius: 1000f32,
        };
        let material = render::Material {
            diffuse:    Vec3A::new(0.5, 0.5, 0.5),
            smoothness: 0f32,

            emmitance: Vec3A::splat(1f32),
            radiance:  0f32,
        };

        let material = scene.create_material(material);
        builder.append(object, material);
        scene.insert_object(builder.build(), Affine3A::IDENTITY)
    };

    println!("Scene built in {}ms", begin_time.elapsed().as_millis());

    let sample_ratio = 1f32 / SAMPLES as f32;

    let mut render_buffer = vec![rgb::Rgb::<u16>::default(); WIDTH as usize * HEIGHT as usize];

    let fov = 20f32;
    let focal_length = 10.0;
    let defocus = 0.00;

    let camera_position = Vec3A::new(0.5, 0.12, -0.2);
    let look_at = Vec3A::new(0.0, 0.02, 0.0);
    let camera_up = Vec3A::new(0.0, 1.0, 0.0);

    let viewport_height = 2f32 * (fov.to_radians() / 2.0).tan() * focal_length;
    let viewport_width = viewport_height * (WIDTH as f32 / HEIGHT as f32);

    let focal_w = (camera_position - look_at).normalize_or_zero();
    let focal_u = camera_up.cross(focal_w).normalize_or_zero();
    let focal_v = focal_w.cross(focal_u);

    let viewport_u = focal_u * viewport_width;
    let viewport_v = -focal_v * viewport_height;

    let delta_u = viewport_u / WIDTH as f32;
    let delta_v = viewport_v / HEIGHT as f32;

    let viewport_origin =
        (camera_position - (focal_length * focal_w) - viewport_u / 2.0 - viewport_v / 2.0)
            + 0.5f32 * (delta_u + delta_v);

    let defocus_radi = focal_length * (defocus / 2f32).to_radians().tan();
    let defocus_u = focal_u * defocus_radi;
    let defocus_v = focal_v * defocus_radi;

    println!("Beginning render");
    let begin_time = std::time::Instant::now();

    let bar = ProgressBar::new(u64::from(HEIGHT) * u64::from(WIDTH));

    //let mut rng = rand::thread_rng();
    render_buffer
        .par_iter_mut()
        .enumerate()
        .progress_with(bar)
        .for_each(|(idx, px)| {
            let y = idx / WIDTH as usize;
            let x = idx % WIDTH as usize;

            let pixel_center = viewport_origin + x as f32 * delta_u + y as f32 * delta_v;
            //println!("{:?} {:?}", ray.origin, ray.direction);

            // TODO: Importance sampling
            let colored = (0..SAMPLES)
                .par_bridge()
                .fold(
                    || Vec3A::splat(0f32),
                    |colored, _| {
                        let mut rng = rand::thread_rng();

                        let aa_shift =
                            rng.gen_range(-0.5..0.5) * delta_u + rng.gen_range(-0.5..0.5) * delta_v;
                        let origin = camera_position
                            + if defocus_radi <= 0.0 {
                                Vec3A::splat(0.0)
                            } else {
                                let rand_circ = random_unit_circle(&mut rng);
                                defocus_u * rand_circ.x + defocus_v * rand_circ.y
                            };

                        let mut ray = render::Ray::new(origin, pixel_center + aa_shift - origin);

                        let mut ray_color = Vec3A::splat(1f32);
                        let mut sample_color = Vec3A::splat(0f32);
                        for _ in 0..BOUNCES {
                            // TODO: Better shadow acne handling
                            let Some((
                                render::HitRecord {
                                    along,
                                    point,
                                    normal,
                                },
                                mat_idx,
                            )) = scene
                                .hit_scene(&ray, (Bound::Included(ACNE_MIN), Bound::Unbounded))
                            else {
                                let a = 0.5f32 * (ray.direction.normalize().y + 1f32);
                                let sky = Vec3A::splat(1f32) * (1f32 - a)
                                    + Vec3A::new(0.5f32, 0.7f32, 1f32) * a;
                                sample_color += sky * ray_color;
                                break;
                            };

                            let material =
                                scene.material(mat_idx).expect("Material does not exist");

                            // Bounce
                            ray.origin += ray.direction * along;
                            ray.set_direction(material.bounce_ray(&mut rng, &ray, normal));

                            sample_color += material.emmitance * material.radiance * ray_color;
                            ray_color *= material.diffuse;
                        }

                        colored + sample_color * sample_ratio
                    },
                )
                .sum();

            *px = process_ray(colored);
            //bar.inc(1);
        });

    //bar.finish_and_clear();

    println!("Rendered in {:.2}s", begin_time.elapsed().as_secs_f32());

    // Write results to a PNG
    let mut encoder = png::Encoder::new(std::fs::File::create("render.png")?, WIDTH, HEIGHT);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Sixteen);
    encoder.set_source_gamma(png::ScaledFloat::new(1.0 / 1.8));
    encoder.set_srgb(png::SrgbRenderingIntent::Perceptual);
    let mut writer = encoder.write_header()?;
    let mut stream = writer.stream_writer()?;

    // Hopefully this gets optimized properly by the compiler to some AVX
    for px in bytemuck::must_cast_slice::<_, u16>(&render_buffer) {
        let bytes_written = stream.write(&px.to_be_bytes())?;
        debug_assert!(bytes_written == size_of::<u16>());
    }

    stream.finish()?;
    writer.finish()?;

    Ok(())
}
