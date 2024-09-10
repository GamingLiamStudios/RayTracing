// Copyright (C) 2024 GLStudios
// SPDX-License-Identifier: LGPL-2.1-only

use std::{num::NonZeroUsize, ptr::NonNull};

use slotmap::SlotMap;

use crate::render::{
    MaterialKey,
    Object,
};

use super::BoundingBox;

slotmap::new_key_type! {
    pub struct TreeSceneObject;
}

// Weight-balanced Binary Tree to dynamically recompute
// Bounding Boxes based on what is most optimal placement
// through a iterative-bottom-up construction technique
enum TreeNode {
    Leaf {
        object: TreeSceneObject,
    },
    Internal {
        bounds: BoundingBox,
        children: NonZeroUsize,

        left: Option<NonNull<Self>>,
        right: Option<NonNull<Self>>,
    }   
}

// wait fuck this is just TSP (optimization problem)
struct TreeScene {
    objects: SlotMap<TreeSceneObject, (Object, MaterialKey)>,
}
