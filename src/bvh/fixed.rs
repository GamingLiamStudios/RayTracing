// Copyright (C) 2024 GLStudios
// SPDX-License-Identifier: LGPL-2.1-only

use std::ops::{
    Bound,
    RangeBounds,
};

use rayon::{
    current_num_threads,
    iter::{
        IndexedParallelIterator,
        IntoParallelIterator,
        ParallelBridge,
        ParallelIterator,
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
    const BVH_MAX_LEAF: u32 = 5;

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
    #[allow(clippy::too_many_lines)]
    pub fn build(mut self) -> Static {
        tracing::trace!(num_objects = self.objects.len(), bounds = ?self.root_bounds);

        // Top-down building - divide and conquer
        let mut tree = vec![BvhNode {
            bounds:  self.root_bounds,
            objects: u32::try_from(self.objects.len()).expect("Scene too large"),
            index:   0,
        }];

        let mut leaf = 0usize;
        let mut depth_sum = 0;
        let mut max_depth = usize::MIN;

        // TODO: Further subdivide this w/ rayon::join or something
        // would have to like store jump offsets instead of indices into the tree array
        // + since we use rayon internally, idefk if it'll make it faster
        let mut search_nodes = vec![(0, 1)];
        while let Some((idx, depth)) = search_nodes.pop() {
            let tree_node = &tree[idx];
            if tree_node.objects <= Self::BVH_MAX_LEAF {
                //|| depth >= 3 {
                leaf += 1;
                depth_sum += depth;
                max_depth = max_depth.max(depth);
                continue;
            }

            let objects = &self.objects
                [tree_node.index as usize..tree_node.index as usize + tree_node.objects as usize];

            let bins = current_num_threads();
            let bin_step = tree_node.bounds.max - tree_node.bounds.min;

            let lowest_cost = (0..bins)
                .map(|i| tree_node.bounds.min + bin_step * ((i + 1) as f64 / bins as f64))
                .par_bridge()
                .flat_map(|v| v.to_array().into_par_iter().enumerate())
                .filter_map(|(axis, split_point)| {
                    // calculate cost of either side
                    let (left, left_count, right) = objects.iter().fold(
                        (BoundingBox::new(), 0, BoundingBox::new()),
                        |(mut left, mut left_count, mut right), (obj, _)| {
                            if obj.center()[axis] > split_point {
                                right.grow_to_include(obj);
                            } else {
                                left.grow_to_include(obj);
                                left_count += 1;
                            }
                            (left, left_count, right)
                        },
                    );

                    let objects_u32 = u32::try_from(objects.len())
                        .expect("More than u32::MAX objects in StaticBuilder");
                    if left_count == 0 || left_count == objects_u32 {
                        return None;
                    }

                    let left_sa = left.half_surface_area() * f64::from(left_count);
                    let right_sa = right.half_surface_area() * f64::from(objects_u32 - left_count);

                    Some((left_sa + right_sa, split_point, axis, left, right))
                })
                .min_by(|(left, ..), (right, ..)| left.total_cmp(right));

            let Some((_, split_point, axis, left, right)) = lowest_cost else {
                // Just leave it
                tracing::warn!(
                    "Couldn't split further. Size {} leaf exists",
                    tree_node.objects
                );
                leaf += 1;
                depth_sum += depth;
                max_depth = max_depth.max(depth);
                continue;
            };

            let objects = &mut self.objects
                [tree_node.index as usize..tree_node.index as usize + tree_node.objects as usize];

            let mut right_start = 0;
            for i in 0..tree_node.objects {
                let (obj, _) = &objects[i as usize];
                if obj.center()[axis] > split_point {
                    continue;
                }

                objects.swap(right_start as usize, i as usize);
                right_start += 1;
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
                right = tree_node.objects - right_start,
            );

            search_nodes.push((tree.len(), depth + 1));
            search_nodes.push((tree.len() + 1, depth + 1));

            let objects = tree_node.objects;
            let index = tree_node.index;

            tree.push(BvhNode {
                bounds: left,
                objects: right_start,
                index,
            });

            tree.push(BvhNode {
                bounds:  right,
                objects: objects - right_start,
                index:   index + right_start,
            });

            tree[idx].index = u32::try_from(tree.len() - 2).expect("BVH too large");
            tree[idx].objects = 0;
        }

        tracing::debug!(
            nodes = tree.len(),
            leaves = leaf,
            avg_depth = depth_sum as f64 / leaf as f64,
            max_depth
        );

        Static {
            objects: self.objects,
            tree,
        }
    }
}

// TODO: Investigate cache alignment (move BB to Vec3?)
#[derive(Debug)]
struct BvhNode {
    bounds:  BoundingBox,
    objects: u32,
    index:   u32,
}

#[derive(Debug)]
pub struct Static {
    pub(crate) objects: Vec<(Object, usize)>,
    tree:               Vec<BvhNode>,
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
        self.tree[0].bounds.intersects(ray)?;

        let mut max_depth_searched = 0;
        let mut hit_depth = 0;
        let mut boxes_tested = 1;
        let mut prims_tested = 0;

        let mut search_space = vec![(0, 1)];
        while let Some((next_search, depth)) = search_space.pop() {
            let node = &self.tree[next_search];
            let index = node.index as usize;

            max_depth_searched = max_depth_searched.max(depth);

            // Is this a leaf node?
            if node.objects > 0 {
                let Some(new_closest) =
                    self.hit_closest(ray, search_range, index, node.objects as usize)
                else {
                    continue;
                };

                prims_tested += node.objects as usize;
                hit_depth = depth;

                search_range.1 = Bound::Included(new_closest.0.along);
                closest = Some(new_closest);
                continue;
            }

            // This a tree node
            //println!();
            let mut closest = None;
            for node_idx in index..index + 2 {
                boxes_tested += 1;

                let Some(next_intersect) = self.tree[node_idx].bounds.intersects(ray) else {
                    continue;
                };

                if !search_range.contains(&next_intersect) {
                    continue;
                }

                //println!("{node_idx} {next_intersect} {closest:?}");

                search_space.push((node_idx, depth + 1));
                let Some(prev_intersect) = closest else {
                    closest = Some(next_intersect);
                    continue;
                };

                if prev_intersect < next_intersect {
                    let idx = search_space.len();
                    search_space.swap(idx - 1, idx - 2);
                    //closest = Some(next_intersect);
                } else {
                    closest = Some(next_intersect);
                }
            }
            //println!("{:?}", &search_space);
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
                    let offset_origin = center - ray.origin;
                    let a = ray.direction.length_squared();
                    let h = ray.direction.dot(offset_origin);
                    let c = radius.mul_add(-radius, offset_origin.length_squared());

                    let discrim = h.mul_add(h, -(a * c));
                    if discrim < 0.0 {
                        continue;
                    }

                    let sqrt_d = discrim.sqrt();
                    let mut root = (h - sqrt_d) / a;
                    if !search_range.contains(&root) {
                        root = (h + sqrt_d) / a;
                        if !search_range.contains(&root) {
                            continue;
                        }
                    }

                    // TODO: Better self-intersection testing (shadow acne)

                    let intersection_point = ray.origin + ray.direction * root;
                    let normal = (intersection_point - center) / radius;

                    (
                        HitRecord {
                            along: root,
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

                    (
                        HitRecord {
                            along:  t,
                            normal: (a.normal * u + b.normal * v + c.normal * (1.0 - u - v))
                                .normalize_or_zero(),
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
