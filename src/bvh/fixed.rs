// Copyright (C) 2024 GLStudios
// SPDX-License-Identifier: LGPL-2.1-only

use std::{
    num::NonZeroU32,
    ops::{
        Bound,
        RangeBounds,
    },
    sync::{
        atomic::AtomicUsize,
        Arc,
    },
};

use super::BoundingBox;
use crate::render::{
    HitRecord,
    Object,
    Ray,
};

pub struct StaticBuilder {
    objects:     Vec<(Object, usize)>,
    root_bounds: BoundingBox,
}

impl StaticBuilder {
    const BINS: usize = 8;
    const BVH_MAX_LEAF: usize = 5;

    pub const fn new() -> Self {
        Self {
            objects:     Vec::new(),
            root_bounds: BoundingBox::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn append(
        &mut self,
        object: Object,
        material: usize,
    ) -> &mut Self {
        self.root_bounds.grow_to_include(&object);
        self.objects.push((object, material));
        self
    }

    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::too_many_lines)]
    fn subdivide(
        root_node: &mut BvhNode,
        objects: &mut [(Object, usize)],
        depth: usize,
        stats: &Arc<(AtomicUsize, AtomicUsize, AtomicUsize, AtomicUsize)>,
    ) {
        let BvhNode::Leaf {
            bounds: root_bounds,
            objects: _,
            index,
        } = root_node
        else {
            tracing::warn!(depth, "Internal node in BVH build stack");
            return;
        };

        if objects.len() <= Self::BVH_MAX_LEAF {
            let (_, leaf_count, depth_sum, max_depth) = stats.as_ref();
            leaf_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            depth_sum.fetch_add(depth, std::sync::atomic::Ordering::Relaxed);
            max_depth.fetch_max(depth, std::sync::atomic::Ordering::Relaxed);
            return;
        }

        let bin_step = Self::BINS as f64 / (root_bounds.max - root_bounds.min);
        let lowest_cost = (0..3)
            .filter_map(|axis| {
                let bound_bins = objects.iter().fold(
                    vec![(BoundingBox::new(), 0u32); Self::BINS],
                    |mut bins, (obj, _)| {
                        let center = obj.center()[axis];
                        let index = (((center - root_bounds.min[axis]) * bin_step[axis]) as usize)
                            .min(bins.len() - 1);
                        let (bin, bin_len) = &mut bins[index];
                        bin.grow_to_include(obj);
                        *bin_len += 1;

                        bins
                    },
                );

                // Left + Right sweep arrays
                let left_costs = bound_bins
                    .iter()
                    .scan(
                        (BoundingBox::new(), 0.0),
                        |(state_bounds, state_num), (bounds, num)| {
                            state_bounds.merge(bounds);
                            *state_num += f64::from(*num);

                            Some((
                                *state_bounds,
                                if *state_num == 0.0 {
                                    0.0
                                } else {
                                    state_bounds.half_surface_area() * *state_num
                                },
                            ))
                        },
                    )
                    .take(Self::BINS - 1)
                    .collect::<Vec<_>>();
                let mut right_costs = bound_bins
                    .iter()
                    .rev()
                    .scan(
                        (BoundingBox::new(), 0.0),
                        |(state_bounds, state_num), (bounds, num)| {
                            state_bounds.merge(bounds);
                            *state_num += f64::from(*num);

                            Some((
                                *state_bounds,
                                if *state_num == 0.0 {
                                    0.0
                                } else {
                                    state_bounds.half_surface_area() * *state_num
                                },
                            ))
                        },
                    )
                    .take(Self::BINS - 1)
                    .collect::<Vec<_>>();
                right_costs.reverse();

                left_costs
                    .iter()
                    .zip(right_costs.iter())
                    .enumerate()
                    .filter_map(|(idx, ((_, left), (_, right)))| {
                        if *left > 0.0 && *right > 0.0 {
                            Some((idx, left + right))
                        } else {
                            None
                        }
                    })
                    .min_by(|(_, left), (_, right)| left.total_cmp(right))
                    .map(|(split_idx, cost)| {
                        (
                            cost,
                            (1.0 / bin_step[axis])
                                .mul_add((split_idx + 1) as f64, root_bounds.min[axis]),
                            axis,
                            left_costs[split_idx].0,
                            right_costs[split_idx].0,
                        )
                    })
            })
            .min_by(|(left, ..), (right, ..)| left.total_cmp(right));

        let Some((_, split_point, axis, left, right)) = lowest_cost else {
            // Just leave it
            tracing::warn!("Couldn't split further. Size {} leaf exists", objects.len());

            let (_, leaf_count, depth_sum, max_depth) = stats.as_ref();
            leaf_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            depth_sum.fetch_add(depth, std::sync::atomic::Ordering::Relaxed);
            max_depth.fetch_max(depth, std::sync::atomic::Ordering::Relaxed);
            return;
        };

        let mut right_start = 0;
        for i in 0..objects.len() {
            let (obj, _) = &objects[i];
            if obj.center()[axis] >= split_point {
                continue;
            }

            objects.swap(right_start, i);
            right_start += 1;
        }

        if right_start == 0 || right_start == objects.len() {
            tracing::warn!("Couldn't split further. Size {} leaf exists", objects.len());

            let (_, leaf_count, depth_sum, max_depth) = stats.as_ref();
            leaf_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            depth_sum.fetch_add(depth, std::sync::atomic::Ordering::Relaxed);
            max_depth.fetch_max(depth, std::sync::atomic::Ordering::Relaxed);
            return;
        }

        tracing::trace!(
            depth,
            split_point,
            axis = match axis {
                0 => "Axis::X",
                1 => "Axis::Y",
                2 => "Axis::Z",
                _ => unreachable!(),
            },
            left = right_start,
            right = objects.len() - right_start,
        );

        // Split objects into mut arrays
        let index = *index;

        let (left_objects, right_objects) = objects.split_at_mut(right_start);
        let mut left = BvhNode::Leaf {
            bounds: left,
            objects: unsafe {
                NonZeroU32::new_unchecked(left_objects.len().try_into().expect("Leaf too large"))
            },
            index,
        };
        let mut right = BvhNode::Leaf {
            bounds:  right,
            objects: unsafe {
                NonZeroU32::new_unchecked(right_objects.len().try_into().expect("Leaf too large"))
            },
            index:   index + u32::try_from(left_objects.len()).expect("Scene too large"),
        };

        rayon::join(
            || Self::subdivide(&mut left, left_objects, depth + 1, stats),
            || Self::subdivide(&mut right, right_objects, depth + 1, stats),
        );

        *root_node = BvhNode::Node {
            bounds:   *root_bounds,
            children: Box::new([left, right]),
        };

        let (node_count, ..) = stats.as_ref();
        node_count.fetch_add(2, std::sync::atomic::Ordering::Relaxed);
    }

    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::too_many_lines)]
    pub fn build(mut self) -> Option<Static> {
        tracing::trace!(num_objects = self.objects.len(), bounds = ?self.root_bounds);

        if self.objects.is_empty() {
            return None;
        }

        let mut root = BvhNode::Leaf {
            bounds:  self.root_bounds,
            objects: NonZeroU32::new(self.objects.len().try_into().expect("Scene too large"))
                .expect("No objects in scene"),
            index:   0,
        };

        let stats = Arc::new((
            AtomicUsize::new(1),
            AtomicUsize::new(0),
            AtomicUsize::new(0),
            AtomicUsize::new(usize::MIN),
        ));

        Self::subdivide(&mut root, &mut self.objects, 1, &stats);

        let (node_count, leaf_count, depth_sum, max_depth) = stats.as_ref();
        let node_count = node_count.load(std::sync::atomic::Ordering::Relaxed);
        let leaf_count = leaf_count.load(std::sync::atomic::Ordering::Relaxed);
        let depth_sum = depth_sum.load(std::sync::atomic::Ordering::Relaxed);
        let max_depth = max_depth.load(std::sync::atomic::Ordering::Relaxed);

        tracing::debug!(
            nodes = node_count,
            leaves = leaf_count,
            avg_depth = depth_sum as f64 / leaf_count as f64,
            max_depth
        );

        Some(Static {
            objects: self.objects,
            root,
        })
    }
}

// TODO: Investigate cache alignment (move BB to Vec3?)
#[derive(Debug)]
enum BvhNode {
    Node {
        bounds:   BoundingBox,
        children: Box<[Self; 2]>,
    },
    Leaf {
        bounds:  BoundingBox,
        objects: NonZeroU32,
        index:   u32,
    },
}

impl BvhNode {
    pub fn intersects(
        &self,
        ray: &Ray,
    ) -> Option<f64> {
        match self {
            Self::Leaf { bounds, .. } | Self::Node { bounds, .. } => bounds.intersects(ray),
        }
    }
}

#[derive(Debug)]
pub struct Static {
    pub(crate) objects: Vec<(Object, usize)>,
    root:               BvhNode,
}

impl Static {
    pub fn hit_scene(
        &self,
        ray: &Ray,
        mut search_range: (Bound<f64>, Bound<f64>),
    ) -> Option<(HitRecord, usize)> {
        let mut closest = None;

        // Probably good to do a quick check here with root node
        // Since loop below doesn't actually do any checks for the root node
        self.root.intersects(ray)?;

        let mut max_depth_searched = 0;
        let mut hit_depth = 0;
        let mut boxes_tested = 1;
        let mut prims_tested = 0;

        let mut search_space = vec![(&self.root, 1)];
        while let Some((node, depth)) = search_space.pop() {
            max_depth_searched = max_depth_searched.max(depth);

            match node {
                BvhNode::Leaf {
                    bounds: _,
                    objects,
                    index,
                } => {
                    let objects: u32 = (*objects).into();
                    let Some(new_closest) =
                        self.hit_closest(ray, search_range, *index as usize, objects as usize)
                    else {
                        continue;
                    };

                    prims_tested += objects as usize;
                    hit_depth = depth;

                    search_range.1 = Bound::Included(new_closest.0.along);
                    closest = Some(new_closest);
                    continue;
                },
                BvhNode::Node {
                    bounds: _,
                    children,
                } => {
                    let mut intersections = children
                        .iter()
                        .filter_map(|node| {
                            boxes_tested += 1;

                            let next_intersect = node.intersects(ray)?;

                            if !search_range.contains(&next_intersect) {
                                return None;
                            }

                            Some((next_intersect, node))
                        })
                        .collect::<Vec<_>>();
                    intersections.sort_by(|(left, _), (right, _)| left.total_cmp(right));
                    search_space.extend(intersections.iter().map(|(_, node)| (*node, depth + 1)));
                },
            }
        }

        tracing::trace!(max_depth_searched, hit_depth, boxes_tested, prims_tested);

        closest
    }

    fn hit_closest(
        &self,
        ray: &Ray,
        mut search_range: (Bound<f64>, Bound<f64>),
        start: usize,
        len: usize,
    ) -> Option<(HitRecord, usize)> {
        let mut closest = None;

        for (obj, mat_idx) in &self.objects[start..start + len] {
            let matched = match obj {
                Object::Sphere { center, radius } => {
                    let center_offset = center - ray.origin;

                    // Uses concepts from Ray Tracing Gems ch7, but with custom spin
                    // https://www.desmos.com/calculator/a7rvtxusuh

                    let midpoint = center_offset.dot(ray.direction) / ray.direction.length();
                    let discrim = midpoint.mul_add(midpoint, radius * radius)
                        - center_offset.length_squared();

                    // if < 0.0, then it does not hit the sphere at all
                    if discrim < 0.0 {
                        continue;
                    }

                    let sqrt_d = discrim.sqrt();
                    let mut along = (midpoint - sqrt_d) / ray.direction.length();
                    if !search_range.contains(&along) {
                        along = (midpoint + sqrt_d) / ray.direction.length();
                        if !search_range.contains(&along) {
                            continue;
                        }
                    }

                    let surface = ray.origin + along * ray.direction;
                    let normal = (surface - center) / radius;

                    (
                        HitRecord {
                            along,
                            surface,
                            normal,
                        },
                        *mat_idx,
                    )
                },
                Object::Triangle { a, b, c } => {
                    let e1 = b.pos - a.pos;
                    let e2 = c.pos - a.pos;

                    let ray_cross_e2 = ray.direction.cross(e2);
                    let det = e1.dot(ray_cross_e2);

                    if (-f64::EPSILON..f64::EPSILON).contains(&det) {
                        continue; // This ray is parallel to this triangle.
                    }

                    let inv_det = 1.0 / det;
                    let s = ray.origin - a.pos;
                    let u = inv_det * s.dot(ray_cross_e2);
                    if !(0.0..=1.0).contains(&u) {
                        continue;
                    }

                    let s_cross_e1 = s.cross(e1);
                    let v = inv_det * ray.direction.dot(s_cross_e1);
                    if v < 0.0 || u + v > 1.0 {
                        continue;
                    }

                    // At this stage we can compute t to find out where the intersection point is on
                    // the line.
                    let t = inv_det * e2.dot(s_cross_e1);
                    if !search_range.contains(&t) {
                        continue;
                    }

                    let surface = a.pos * u + b.pos * v + c.pos * (1.0 - u - v);
                    let normal =
                        (a.normal * u + b.normal * v + c.normal * (1.0 - u - v)).normalize();

                    (
                        HitRecord {
                            along: t,
                            surface,
                            normal,
                        },
                        *mat_idx,
                    )
                },
            };

            search_range.1 = Bound::Included(matched.0.along);
            closest = Some(matched);
        }

        closest
    }
}
