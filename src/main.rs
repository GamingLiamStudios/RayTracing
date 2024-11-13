// Copyright (C) 2024 GLStudios
// SPDX-License-Identifier: LGPL-2.1-only

//#![feature(array_chunks)]
//#![feature(new_range_api)]
//#![feature(generic_arg_infer)]
//#![feature(iter_map_windows)]
#![feature(iter_array_chunks)]
//#![feature(const_mut_refs)]
//#![feature(allocator_api)]

use core::f64;
use std::{
    io::Write,
    ops::Bound, sync::atomic::AtomicUsize,
};

use glam::{
    DAffine3,
    DQuat,
    DVec2,
    DVec3,
    Vec3,
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

mod bvh;
mod render;
mod types;

pub const ACNE_MIN: f64 = f32::EPSILON as f64;

pub const WIDTH: u32 = 800;
pub const HEIGHT: u32 = 600;

pub const SAMPLES: u32 = 50;
pub const BOUNCES: usize = 50;

#[inline]
pub fn random_unit_circle(rng: &mut ThreadRng) -> DVec2 {
    loop {
        // Rejection sampling
        let x = rng.gen_range(-1.0..1.0);
        let y = rng.gen_range(-1.0..1.0);

        let vector = DVec2::new(x, y);
        if vector.length_squared() <= 1.0 {
            return vector;
        }
    }
}

#[inline]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn process_ray(mut input: DVec3) -> Rgb<u16> {
    // Post-processing
    // "Tone Mapping"
    input = input.clamp(DVec3::splat(0.0), DVec3::splat(1.0));
    // Gamma correction
    input = input.powf(1.0 / 1.8);

    // TODO: How to actually HDR/tonemap

    // Saturating cast - Will auto-clamp within bounds of u16
    input *= f64::from(u16::MAX);
    Rgb::new(input.x as u16, input.y as u16, input.z as u16)
}

#[allow(clippy::cast_precision_loss)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::many_single_char_names)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fmt_subscriber = tracing_subscriber::fmt::Subscriber::builder()
        .with_max_level(tracing::Level::DEBUG)
        .finish();
    tracing::subscriber::set_global_default(fmt_subscriber)?;

    let mut scene = render::Scene::new();
    let mut builder = bvh::StaticBuilder::new();

    tracing::debug_span!("PrimarySceneBuild").in_scope(|| {
        tracing::info!("Building Scene...");
        let begin_time = std::time::Instant::now();

        let bunny1_mat = {
            let material = Material {
                diffuse:    DVec3::splat(0.3),
                emmitance:  DVec3::splat(1.0),
                smoothness: 0.7,
                radiance:   0.0,
            };
            scene.create_material(material)
        };

        let bunny_id = {
            let bunny_span = tracing::debug_span!("BunnyModel");
            let _guard = bunny_span.enter();

            let (document, buffers, _) =
                gltf::import("models/dragon.glb").expect("Failed to load model");

            for mesh in document.meshes() {
                for prim in mesh.primitives() {
                    if prim.mode() != Mode::Triangles {
                        continue;
                    }

                    let reader = prim.reader(|buf| buffers.get(buf.index()).map(|d| &*d.0));
                    let Some(positions) = reader.read_positions() else {
                        tracing::warn!("No positions attached to triangle mesh");
                        continue;
                    };
                    let positions = positions.collect::<Vec<_>>();

                    let Some(normals) = reader.read_normals() else {
                        tracing::warn!("No normals attached to triangle mesh");
                        continue;
                    };
                    let normals = normals.collect::<Vec<_>>();

                    let Some(indices) = reader.read_indices() else {
                        tracing::warn!("No indices attached to triangle mesh");
                        continue;
                    };

                    tracing::trace!(num_positions = positions.len(), num_normals = normals.len());

                    let mut tris = 0;
                    for [a, b, c] in indices.into_u32().map(|v| v as usize).array_chunks() {
                        let tri = Object::Triangle {
                            a: Vertex {
                                pos:    Vec3::from_array(positions[a]).as_dvec3(),
                                normal: Vec3::from_array(normals[a]).as_dvec3(),
                            },
                            b: Vertex {
                                pos:    Vec3::from_array(positions[b]).as_dvec3(),
                                normal: Vec3::from_array(normals[b]).as_dvec3(),
                            },
                            c: Vertex {
                                pos:    Vec3::from_array(positions[c]).as_dvec3(),
                                normal: Vec3::from_array(normals[c]).as_dvec3(),
                            },
                        };
                        builder.append(tri, 0);
                        tris += 1;
                    }

                    tracing::trace!(tris, "Mesh contains {} triangles", tris);
                }
            }

            tracing::debug!(num_primatives = builder.len());
            scene.insert_object(
                builder.build().expect("Zero sized scene"),
                DAffine3::from_scale_rotation_translation(
                    DVec3::splat(1.0),
                    DQuat::from_euler(
                        glam::EulerRot::XYZ,
                        90.0_f64.to_radians(),
                        0.0_f64.to_radians(),
                        90.0_f64.to_radians(),
                    ),
                    DVec3::new(-0.3, -0.01, 0.05),
                ),
                vec![bunny1_mat],
            )
        };

        let bunny2_mat = {
            let material = Material {
                diffuse:    DVec3::new(0.7, 0.3, 0.3),
                emmitance:  DVec3::splat(1.0),
                smoothness: 0.0,
                radiance:   0.0,
            };
            scene.create_material(material)
        };

        scene.add_instance(
            bunny_id,
            DAffine3::from_scale_rotation_translation(
                DVec3::splat(1.0),
                DQuat::from_euler(
                    glam::EulerRot::XYZ,
                    90.0f64.to_radians(),
                    0.0f64.to_radians(),
                    -90.0f64.to_radians(),
                ),
                DVec3::new(0.0, 0.0, 0.05),
            ),
            vec![bunny2_mat],
        );

        let _sphere_id = {
            // Earth
            let mut builder = bvh::StaticBuilder::new();

            let object = render::Object::Sphere {
                center: DVec3::new(0.0, -1000.065, 0.0),
                radius: 1000.0,
            };
            let material = render::Material {
                diffuse:    DVec3::new(0.5, 0.5, 0.5),
                smoothness: 0.0,

                emmitance: DVec3::splat(1.0),
                radiance:  0.0,
            };

            let material = scene.create_material(material);
            builder.append(object, 0);
            scene.insert_object(builder.build().expect("Zero sized scene"), DAffine3::IDENTITY, vec![material])
        };

        let _unit_sphere = {
            // light
            let mut builder = bvh::StaticBuilder::new();

            let object = render::Object::Sphere {
                center: DVec3::new(0.0, 0.0, 0.0),
                radius: 25.0,
            };
            let material = render::Material {
                diffuse:    DVec3::new(0.5, 0.5, 0.5),
                smoothness: 0.0,

                emmitance: DVec3::splat(1.0),
                radiance:  2.0,
            };

            let material = scene.create_material(material);
            builder.append(object, 0);
            scene.insert_object(builder.build().expect("Zero sized scene"), DAffine3::from_translation(DVec3::new(-2.0, 50.0, -2.0)), vec![material])
        };

        {
            let mut builder = bvh::StaticBuilder::new();

            let object = render::Object::Sphere {
                center: DVec3::new(-0.4, 0.09, 0.5),
                radius: 0.15,
            };
            let material = render::Material {
                diffuse:    DVec3::new(0.5, 0.5, 0.5),
                smoothness: 1.0,

                emmitance: DVec3::splat(1.0),
                radiance:  0.0,
            };

            let material = scene.create_material(material);
            builder.append(object, 0);
            scene.insert_object(builder.build().expect("Zero sized scene"), DAffine3::IDENTITY, vec![material]);
        }

        tracing::info!(elapsed = ?begin_time.elapsed(), "Scene Built in {:.2}s", begin_time.elapsed().as_secs_f64());
    });

    let sample_ratio = 1.0 / f64::from(SAMPLES);

    let mut render_buffer = vec![rgb::Rgb::<u16>::default(); WIDTH as usize * HEIGHT as usize];

    let fov = 20f64;
    let focal_length = 1.0;
    let defocus = 0.05;

    let camera_position = DVec3::new(0.3, 0.07, -1.0);
    let look_at = DVec3::new(-0.15, 0.07, 0.0);
    let camera_up = DVec3::new(0.0, 1.0, 0.0);

    let viewport_height = 2.0 * (fov.to_radians() / 2.0).tan() * focal_length;
    let viewport_width = viewport_height * (f64::from(WIDTH) / f64::from(HEIGHT));

    let focal_w = (camera_position - look_at).normalize_or_zero();
    let focal_u = camera_up.cross(focal_w).normalize_or_zero();
    let focal_v = focal_w.cross(focal_u);

    let viewport_u = focal_u * viewport_width;
    let viewport_v = -focal_v * viewport_height;

    let delta_u = viewport_u / f64::from(WIDTH);
    let delta_v = viewport_v / f64::from(HEIGHT);

    let viewport_origin = camera_position - (focal_length * focal_w)
        + 0.5 * (delta_u + delta_v - viewport_u - viewport_v);

    let defocus_radi = focal_length * (defocus / 2.0f64).to_radians().tan();
    let defocus_u = focal_u * defocus_radi;
    let defocus_v = focal_v * defocus_radi;

    tracing::info_span!("SceneRender").in_scope(|| {
        tracing::info!(width = WIDTH, height = HEIGHT, "Beginning render of size {WIDTH}x{HEIGHT}");
        tracing::debug!(?camera_position, ?look_at, ?camera_up, fov, focal_length, defocus);

        let begin_time = std::time::Instant::now();

        let ray_count = AtomicUsize::new(0);

        //let mut rng = rand::thread_rng();
        // TODO: Better load balancing - See Ray Tracing Gems Ch10
        render_buffer
            .par_iter_mut()
            .enumerate()
            .progress_with(ProgressBar::new(u64::from(HEIGHT) * u64::from(WIDTH)))
            .for_each(|(idx, px)| {
                let y = idx / WIDTH as usize;
                let x = idx % WIDTH as usize;

                let pixel_center = viewport_origin + x as f64 * delta_u + y as f64 * delta_v;
                //println!("{:?} {:?}", ray.origin, ray.direction);

                let colored = (0..SAMPLES)
                    .par_bridge()
                    .fold(
                        || DVec3::splat(0.0),
                        |colored, _| {
                            let mut rng = rand::thread_rng();

                            let aa_shift = rng.gen_range(-0.5..0.5) * delta_u
                                + rng.gen_range(-0.5..0.5) * delta_v;
                            let origin = camera_position
                                + if defocus_radi <= 0.0 {
                                    DVec3::splat(0.0)
                                } else {
                                    let rand_circ = random_unit_circle(&mut rng);
                                    defocus_u * rand_circ.x + defocus_v * rand_circ.y
                                };

                            let mut ray =
                                render::Ray::new(origin, pixel_center + aa_shift - origin);

                            let mut ray_color = DVec3::splat(1.0);
                            let mut sample_color = DVec3::splat(0.0);
                            let mut bounces = 0;
                            for _ in 0..BOUNCES {
                                bounces += 1;

                                let _ = ray_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                                // TODO: Better self-intersection handling
                                // See: Ray Tracing Gems Ch6
                                let Some((render::HitRecord { along: _, surface, normal }, mat_idx)) = scene
                                    .hit_scene(&ray, (Bound::Included(ACNE_MIN), Bound::Unbounded))
                                else {
                                    let a = 0.5 * (ray.direction.normalize().y + 1.0);
                                    let sky = DVec3::splat(1.0) * (1.0 - a)
                                        + DVec3::new(0.5, 0.7, 1.0) * a;
                                    sample_color += sky * ray_color;
                                    break;
                                };

                                let material =
                                    scene.material(mat_idx).expect("Material does not exist");

                                // Bounce
                                ray.origin = surface;
                                ray.set_direction(material.bounce_ray(&mut rng, &ray, normal));

                                sample_color += material.emmitance * material.radiance * ray_color;
                                ray_color *= material.diffuse;
                            }

                            tracing::trace!(bounces);
                            colored + sample_color * sample_ratio
                        },
                    )
                    .sum();

                *px = process_ray(colored);
                //bar.inc(1);
            });

        //bar.finish_and_clear();
        let rays = ray_count.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1_000_000.0;

        tracing::info!(elapsed = ?begin_time.elapsed(), "Scene Rendered in {:.2}s ({rays:.2} mray/s avg)", begin_time.elapsed().as_secs_f64());
    });

    // Write results to a PNG
    tracing::debug_span!("FileWriter::PNG").in_scope(|| {
        let path = std::fs::File::create("render.png")?;
        tracing::info!(?path, "Writing render to file {path:?}");

        let mut encoder = png::Encoder::new(path, WIDTH, HEIGHT);
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
    })
}
