// Copyright (C) 2024 GLStudios
// SPDX-License-Identifier: LGPL-2.1-only

use std::ops::{
    Bound,
    RangeBounds,
};

use slotmap::DefaultKey;

use super::BoundingBox;
use crate::render::{
    HitRecord,
    Object,
    Ray,
};

pub struct StaticBuilder {
    pub objects: Vec<(Object, DefaultKey)>,
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

    pub fn append(
        &mut self,
        object: Object,
        material: DefaultKey,
    ) -> &mut Self {
        self.root_bounds.grow_to_include(&object);
        self.objects.push((object, material));
        self
    }

    #[allow(clippy::cast_precision_loss)]
    pub fn build(mut self) -> Static {
        // Top-down building - divide and conquer
        let mut tree = vec![BvhNode {
            bounds:  self.root_bounds,
            objects: u32::try_from(self.objects.len()).expect("Scene too large"),
            index:   0,
        }];

        let mut leaf = 0usize;
        let mut depth_sum = 0;
        let mut max_depth = usize::MIN;

        let mut search_nodes = vec![(0, 0)];
        while let Some((idx, depth)) = search_nodes.pop() {
            let tree_node = &tree[idx];
            if tree_node.objects <= Self::BVH_MAX_LEAF {
                //|| depth >= 3 {
                leaf += 1;
                depth_sum += depth;
                max_depth = max_depth.max(depth);
                continue;
            }

            let mut part_point = tree_node.index;
            let mut left = BoundingBox::new();
            let mut right = BoundingBox::new();
            for i in tree_node.index..tree_node.index + tree_node.objects {
                let (object, _) = &self.objects[i as usize];

                let mut left_grown = left;
                left_grown.grow_to_include(object);
                let mut right_grown = right;
                right_grown.grow_to_include(object);

                let left_sa = f64::from(left_grown.half_surface_area())
                    / f64::from(part_point - tree_node.index + 1);
                let right_sa =
                    f64::from(right_grown.half_surface_area()) / f64::from(i - part_point + 1);

                if left_sa > right_sa {
                    right = right_grown;
                } else {
                    left = left_grown;
                    self.objects.swap(part_point as usize, i as usize);
                    part_point += 1;
                }
            }

            if part_point - tree_node.index == 0
                || tree_node.index + tree_node.objects - part_point == 0
            {
                // Just leave it
                println!(
                    "Couldn't split further. size {} leaf exists",
                    tree_node.objects
                );
                leaf += 1;
                depth_sum += depth;
                max_depth = max_depth.max(depth);
                continue;
            }

            search_nodes.push((tree.len(), depth + 1));
            search_nodes.push((tree.len() + 1, depth + 1));

            tree.push(BvhNode {
                bounds:  left,
                objects: part_point - tree_node.index,
                index:   tree_node.index,
            });

            let tree_node = &tree.last().expect("No items in tree");
            tree.push(BvhNode {
                bounds:  right,
                objects: tree[idx].objects - tree_node.objects,
                index:   part_point,
            });

            tree[idx].index = u32::try_from(tree.len() - 2).expect("BVH too large");
            tree[idx].objects = 0;
        }

        println!("{} BVH nodes generated", tree.len());
        println!(
            "{leaf} Leafs generated\n\t- Avg Depth: {:.2}\n\t- Max Depth: {max_depth}",
            depth_sum as f64 / leaf as f64
        );

        Static {
            objects: self.objects,
            tree,
        }
    }
}

// TODO: Investigate perf difference of Binary vs B-trees
// since RAM is quite fast and we aren't doing static building (hopefully)
// it Binary trees should roughly be as fast?
#[derive(Debug)]
struct BvhNode {
    bounds:  BoundingBox,
    objects: u32,
    index:   u32,
}

pub struct Static {
    objects: Vec<(Object, DefaultKey)>,
    tree:    Vec<BvhNode>,
}

impl Static {
    pub fn hit_scene(
        &self,
        ray: &Ray,
        mut search_range: (Bound<f32>, Bound<f32>),
    ) -> Option<(HitRecord, slotmap::DefaultKey)> {
        let mut closest = None;

        // Probably good to do a quick check here with root node
        // Since loop below doesn't actually do any checks for the root node
        self.tree[0].bounds.intersects(ray)?;

        let mut search_space = vec![0];
        while let Some(next_search) = search_space.pop() {
            let node = &self.tree[next_search];
            let index = node.index as usize;

            // Is this a leaf node?
            if node.objects > 0 {
                let Some(new_closest) =
                    self.hit_closest(ray, search_range, index, node.objects as usize)
                else {
                    continue;
                };

                search_range.1 = Bound::Included(new_closest.0.along);
                closest = Some(new_closest);
                continue;
            }

            // This a tree node
            //println!();
            let mut closest = None;
            for node_idx in index..index + 2 {
                let Some(next_intersect) = self.tree[node_idx].bounds.intersects(ray) else {
                    continue;
                };

                if !search_range.contains(&next_intersect) {
                    continue;
                }

                //println!("{node_idx} {next_intersect} {closest:?}");

                search_space.push(node_idx);
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

        closest
    }

    fn hit_closest(
        &self,
        ray: &Ray,
        mut search_range: (Bound<f32>, Bound<f32>),
        start: usize,
        len: usize,
    ) -> Option<(HitRecord, slotmap::DefaultKey)> {
        let mut closest = None;

        for (obj, mat_idx) in &self.objects[start..start + len] {
            let matched = match obj {
                Object::Sphere { center, radius } => {
                    let offset_origin = center - ray.origin;
                    let a = ray.direction.length_squared();
                    let h = ray.direction.dot(offset_origin);
                    let c = radius.mul_add(-radius, offset_origin.length_squared());

                    let discrim = h.mul_add(h, -(a * c));
                    if discrim < 0f32 {
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
                    unimplemented!()
                },
            };

            search_range.1 = Bound::Included(matched.0.along);
            closest = Some(matched);
        }

        closest
    }
}
