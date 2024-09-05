#![feature(array_chunks)]
#![feature(new_range_api)]
#![feature(generic_arg_infer)]
#![feature(iter_map_windows)]
#![feature(const_mut_refs)]

use core::f32;
use std::{
    io::Write,
    ops::Bound,
};

use glam::{
    Vec2,
    Vec3A,
};
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
use rgb::Rgb;
use tracing::{
    event,
    info,
    instrument,
};

mod bvh;
mod render;

pub const ACNE_MIN: f32 = 0.01;

pub const WIDTH: u32 = 800;
pub const HEIGHT: u32 = 600;

pub const SAMPLES: u32 = 50;
pub const BOUNCES: usize = 50;

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
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fmt_subscriber = tracing_subscriber::fmt::Subscriber::builder()
        .with_max_level(tracing::Level::TRACE)
        .finish();
    tracing::subscriber::set_global_default(fmt_subscriber)?;

    let mut scene = render::Scene::new();
    let mut builder = bvh::StaticBuilder::new();

    println!("Building scene...");
    let begin_time = std::time::Instant::now();

    {
        // Earth
        let object = render::Object::Sphere {
            center: Vec3A::new(0.0, -1000.0, 0.0),
            radius: 1000f32,
        };
        let material = render::Material {
            diffuse:    Vec3A::new(0.5, 0.5, 0.5),
            smoothness: 0f32,

            emmitance: Vec3A::splat(1f32),
            radiance:  0f32,
        };

        let material = scene.add_material(material);
        builder.append(object, material);
    }

    {
        // A
        let object = render::Object::Sphere {
            center: Vec3A::new(-4.0, 1.0, 0.0),
            radius: 1.0,
        };
        let material = render::Material {
            diffuse:    Vec3A::new(0.4, 0.2, 0.1),
            smoothness: 0f32,

            emmitance: Vec3A::splat(1f32),
            radiance:  0f32,
        };

        let material = scene.add_material(material);
        builder.append(object, material);
    }

    {
        // B
        let object = render::Object::Sphere {
            center: Vec3A::new(0.0, 1.0, 0.0),
            radius: 1.0,
        };
        let material = render::Material {
            diffuse:    Vec3A::new(0.7, 0.6, 0.5),
            smoothness: 1.0f32,

            emmitance: Vec3A::splat(1f32),
            radiance:  0f32,
        };

        let material = scene.add_material(material);
        builder.append(object, material);
    }

    let mut rng = rand::thread_rng();
    for a in -11..11 {
        for b in -11..11 {
            let material = rng.gen_range(0.0..1.0);
            let center = Vec3A::new(
                0.9f32.mul_add(rng.gen_range(0.0..1.0), a as f32),
                0.2,
                0.9f32.mul_add(rng.gen_range(0.0..1.0), b as f32),
            );

            if (center - Vec3A::new(4.0, 0.2, 0.0)).length() > 0.9 {
                match material {
                    ..0.8 => {
                        // Diffuse
                        let albedo = Vec3A::new(
                            rng.gen_range(0.0..1.0),
                            rng.gen_range(0.0..1.0),
                            rng.gen_range(0.0..1.0),
                        ) * Vec3A::new(
                            rng.gen_range(0.0..1.0),
                            rng.gen_range(0.0..1.0),
                            rng.gen_range(0.0..1.0),
                        );

                        let mat_idx = scene.add_material(render::Material {
                            diffuse:    albedo,
                            emmitance:  Vec3A::splat(0.0),
                            smoothness: 0.0,
                            radiance:   0.0,
                        });
                        builder.append(
                            render::Object::Sphere {
                                center,
                                radius: 0.2,
                            },
                            mat_idx,
                        );
                    },
                    0.8.. => {
                        // Metal
                        let albedo = Vec3A::new(
                            rng.gen_range(0.5..1.0),
                            rng.gen_range(0.5..1.0),
                            rng.gen_range(0.5..1.0),
                        );
                        let smoothness = rng.gen_range(0.0..0.5);

                        let mat_idx = scene.add_material(render::Material {
                            diffuse: albedo,
                            emmitance: Vec3A::splat(0.0),
                            smoothness,
                            radiance: 0.0,
                        });
                        builder.append(
                            render::Object::Sphere {
                                center,
                                radius: 0.2,
                            },
                            mat_idx,
                        );
                    },
                    _ => unreachable!(),
                }
            }
        }
    }

    println!("Scene contains {} objects", builder.objects.len());
    scene.add_object(builder.build());

    println!("Scene built in {}ms", begin_time.elapsed().as_millis());

    let sample_ratio = 1f32 / SAMPLES as f32;

    let mut render_buffer = vec![rgb::Rgb::<u16>::default(); WIDTH as usize * HEIGHT as usize];

    let fov = 20f32;
    let focal_length = 10.0;
    let defocus = 0.6;

    let camera_position = Vec3A::new(13.0, 2.0, 3.0);
    let look_at = Vec3A::new(0.0, 0.0, 0.0);
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

            let unit_direction = (pixel_center - camera_position).normalize_or_zero();
            let a = 0.5f32 * (unit_direction.y + 1f32);
            let sky = Vec3A::splat(1f32) * (1f32 - a) + Vec3A::new(0.5f32, 0.7f32, 1f32) * a;

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
                            let Some((render::HitRecord { along, normal }, mat_idx)) = scene
                                .hit_scene(&ray, (Bound::Included(ACNE_MIN), Bound::Unbounded))
                            else {
                                sample_color += sky * ray_color;
                                break;
                            };

                            let material =
                                scene.material(mat_idx).expect("Material does not exist");

                            // Bounce
                            ray.origin += ray.direction * along;
                            ray.set_direction(material.bounce_ray(
                                &mut rng,
                                &ray.direction,
                                normal,
                            ));

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
