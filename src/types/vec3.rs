// Copyright (C) 2024 GLStudios
// SPDX-License-Identifier: LGPL-2.1-only

use std::{
    iter::Sum,
    ops::{
        Add,
        AddAssign,
        Div,
        DivAssign,
        Index,
        IndexMut,
        Mul,
        MulAssign,
        Neg,
        Sub,
        SubAssign,
    },
    simd::{
        cmp::SimdPartialOrd,
        num::SimdFloat,
        Mask,
        MaskElement,
        Simd,
        SimdElement,
        StdFloat,
        Swizzle,
    },
};

use glam::DVec3;

use super::SwizzleLoc;

#[inline]
fn min<T: PartialOrd>(
    left: T,
    right: T,
) -> T {
    if left < right {
        left
    } else {
        right
    }
}

#[inline]
fn max<T: PartialOrd>(
    left: T,
    right: T,
) -> T {
    if left > right {
        left
    } else {
        right
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct Vec3A<T: SimdElement>(Simd<T, 4>);

impl<T: SimdElement> Vec3A<T> {
    #[inline]
    pub const fn splat(v: T) -> Self {
        // Could actually be slower than ::splat due to LLVM weirdness
        Self(Simd::from_array([v; 4]))
    }

    #[inline]
    pub const fn to_array(self) -> [T; 3] {
        let [x, y, z, _] = self.0.to_array();
        [x, y, z]
    }
}

impl Vec3A<f64> {
    #[inline]
    pub const fn from_dvec3(v: DVec3) -> Self {
        Self::from_array(v.to_array())
    }

    #[inline]
    pub const fn to_dvec3(self) -> DVec3 {
        glam::DVec3::from_array(self.to_array())
    }

    #[inline]
    pub fn map_dvec3(
        self,
        map: impl FnOnce(DVec3) -> DVec3,
    ) -> Self {
        Self::from_dvec3(map(self.to_dvec3()))
    }
}

impl<T: SimdElement> Vec3A<T> {
    #[inline]
    pub const fn new(
        x: T,
        y: T,
        z: T,
    ) -> Self
    where
        T: num_traits::ConstZero,
    {
        Self(Simd::from_array([x, y, z, T::ZERO]))
    }

    #[inline]
    pub const fn from_array(v: [T; 3]) -> Self
    where
        T: num_traits::ConstZero,
    {
        Self::new(v[0], v[1], v[2])
    }

    #[inline]
    pub fn permute<const INDEX: [SwizzleLoc; 3]>(self) -> Self {
        struct Impl<const INDEX: [SwizzleLoc; 3]>;
        impl<const INDEX: [SwizzleLoc; 3]> Swizzle<4> for Impl<INDEX> {
            const INDEX: [usize; 4] = super::to_swizzle(INDEX);
        }

        Self(Impl::<INDEX>::swizzle(self.0))
    }

    #[inline]
    pub fn x(self) -> T
    where
        Self: Index<usize, Output = T>,
    {
        self[0]
    }

    #[inline]
    pub fn y(self) -> T
    where
        Self: Index<usize, Output = T>,
    {
        self[1]
    }

    #[inline]
    pub fn z(self) -> T
    where
        Self: Index<usize, Output = T>,
    {
        self[2]
    }
}

impl<U: MaskElement, T: SimdElement<Mask = U>> Vec3A<T>
where
    Simd<T, 4>: SimdPartialOrd<Mask = Mask<U, 4>>,
{
    /// If equal, will prefer items on the right over items on the left
    #[inline]
    pub fn max(
        self,
        other: Self,
    ) -> Self {
        Self(self.0.simd_gt(other.0).select(self.0, other.0))
    }

    /// If equal, will prefer items on the right over items on the left
    #[inline]
    pub fn min(
        self,
        other: Self,
    ) -> Self {
        Self(self.0.simd_lt(other.0).select(self.0, other.0))
    }

    #[inline]
    pub fn clamp(
        self,
        min: Self,
        max: Self,
    ) -> Self {
        self.min(max).max(min)
    }

    #[inline]
    pub fn max_element(self) -> T
    where
        T: PartialOrd,
        Self: Index<usize, Output = T>,
    {
        use SwizzleLoc::{
            Y,
            Z,
        };

        // XZ, YZ, ZY, W
        // max(XZ, YZ) = XYZ
        let upper = self.max(self.permute::<{ [Z, Z, Y] }>());
        max(upper[0], upper[1])
    }

    #[inline]
    pub fn min_element(self) -> T
    where
        T: PartialOrd,
        Self: Index<usize, Output = T>,
    {
        use SwizzleLoc::{
            Y,
            Z,
        };

        // XZ, YZ, ZY, W
        // min(XZ, YZ) = XYZ
        let upper = self.min(self.permute::<{ [Z, Z, Y] }>());
        min(upper[0], upper[1])
    }
}

impl<T: SimdElement + Sub<Output = T>> Vec3A<T>
where
    Simd<T, 4>: Mul<Output = Simd<T, 4>>,
    <Simd<T, 4> as Mul>::Output: SimdFloat<Scalar = T>,
{
    #[inline]
    pub fn powf(
        self,
        pow: T,
    ) -> Self
    where
        Simd<T, 4>: StdFloat,
    {
        // TODO: Investigate accuracy
        Self((Self::splat(pow).0 * self.0.ln()).exp())
    }

    #[inline]
    pub fn lerp(
        self,
        rhs: Self,
        t: T,
    ) -> Self
    where
        T: num_traits::ConstOne,
        Simd<T, 4>: StdFloat,
        Self: Add<Output = Self>,
    {
        self * (T::ONE - t) + rhs * t
    }

    #[inline]
    pub fn dot(
        self,
        other: Self,
    ) -> T
    where
        <Simd<T, 4> as Mul>::Output: Index<usize, Output = T>,
    {
        let prod = self.0 * other.0;
        prod.reduce_sum() - prod[3]
        // Ensure no garbage in `w` term affects output
    }

    #[inline]
    pub fn length_squared(self) -> T {
        self.dot(self)
    }

    #[inline]
    pub fn length(&self) -> T
    where
        T: num_traits::Float,
    {
        self.length_squared().sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self
    where
        T: num_traits::Float,
        Self: Div<T, Output = Self>,
    {
        let length = self.length();
        self / length
    }

    #[inline]
    pub fn reflect(
        self,
        normal: Self,
    ) -> Self
    where
        T: num_traits::Float,
        Self: Mul<T, Output = Self> + Sub<Output = Self>,
    {
        self - normal
            * self.dot(normal)
            * T::from(2.0).expect("Failed to cast to expected float type")
    }

    #[inline]
    pub fn cross(
        self,
        rhs: Self,
    ) -> Self
    where
        Self: Mul<Output = Self> + Sub<Output = Self>,
    {
        // x  <-  a.y*b.z - a.z*b.y
        // y  <-  a.z*b.x - a.x*b.z
        // z  <-  a.x*b.y - a.y*b.x
        // (self.zxy() * rhs - self * rhs.zxy()).zxy()
        use SwizzleLoc::{
            X,
            Y,
            Z,
        };
        let lhs_zxy = self.permute::<{ [Z, X, Y] }>();
        let rhs_zxy = rhs.permute::<{ [Z, X, Y] }>();

        (lhs_zxy * rhs - self * rhs_zxy).permute::<{ [Z, X, Y] }>()
    }
}

impl<U: MaskElement, T: SimdElement<Mask = U>> Vec3A<T>
where
    Simd<T, 4>: SimdFloat<Mask = Mask<U, 4>>,
{
    pub fn blend_sign(
        if_positive: Self,
        if_negative: Self,
        select: Self,
    ) -> Self {
        // should optimize to blendv on AVX systems
        Self(
            select
                .0
                .is_sign_negative()
                .select(if_negative.0, if_positive.0),
        )
    }
}

macro_rules! impl_operations {
    ($type:ident $oper:ident) => {
        paste::paste! {
            impl<T: SimdElement> $oper for $type<T>
            where
                Simd<T, 4>: $oper<Output = Simd<T, 4>>,
            {
                type Output = $type<T>;

                #[inline]
                fn [<$oper:lower>](
                    self,
                    rhs: Self,
                ) -> Self::Output {
                    $type($oper::[<$oper:lower>](self.0, rhs.0))
                }
            }

            impl<T: SimdElement> $oper<&Self> for $type<T>
            where
                Simd<T, 4>: $oper<Output = Simd<T, 4>>,
            {
                type Output = $type<T>;

                #[inline]
                fn [<$oper:lower>](
                    self,
                    rhs: &Self,
                ) -> Self::Output {
                    $type($oper::[<$oper:lower>](self.0, rhs.0))
                }
            }


            impl<T: SimdElement> $oper<$type<T>> for &$type<T>
            where
                Simd<T, 4>: $oper<Output = Simd<T, 4>>,
            {
                type Output = $type<T>;

                #[inline]
                fn [<$oper:lower>](
                    self,
                    rhs: $type<T>,
                ) -> Self::Output {
                    $type($oper::[<$oper:lower>](self.0, rhs.0))
                }
            }

            impl<T: SimdElement> $oper for &$type<T>
            where
                Simd<T, 4>: $oper<Output = Simd<T, 4>>,
            {
                type Output = $type<T>;

                #[inline]
                fn [<$oper:lower>](
                    self,
                    rhs: Self,
                ) -> Self::Output {
                    $type($oper::[<$oper:lower>](self.0, rhs.0))
                }
            }

            impl<T: SimdElement> $oper<T> for $type<T>
            where
                Simd<T, 4>: $oper<Output = Simd<T, 4>>,
            {
                type Output = $type<T>;

                #[inline]
                fn [<$oper:lower>](
                    self,
                    rhs: T,
                ) -> Self::Output {
                    $oper::[<$oper:lower>](self, $type::splat(rhs))
                }
            }

            impl<T: SimdElement> $oper<T> for &$type<T>
            where
                Simd<T, 4>: $oper<Output = Simd<T, 4>>,
            {
                type Output = $type<T>;

                #[inline]
                fn [<$oper:lower>](
                    self,
                    rhs: T,
                ) -> Self::Output {
                    $oper::[<$oper:lower>](self, &$type::splat(rhs))
                }
            }

            impl<T: SimdElement> $oper<&T> for $type<T>
            where
                Simd<T, 4>: $oper<Output = Simd<T, 4>>,
            {
                type Output = $type<T>;

                #[inline]
                fn [<$oper:lower>](
                    self,
                    rhs: &T,
                ) -> Self::Output {
                    $oper::[<$oper:lower>](self, $type::splat(*rhs))
                }
            }

            impl<T: SimdElement> $oper<&T> for &$type<T>
            where
                Simd<T, 4>: $oper<Output = Simd<T, 4>>,
            {
                type Output = $type<T>;

                #[inline]
                fn [<$oper:lower>](
                    self,
                    rhs: &T,
                ) -> Self::Output {
                    $oper::[<$oper:lower>](self, &$type::splat(*rhs))
                }
            }

            impl<T: SimdElement> [<$oper Assign>] for $type<T>
            where
                Simd<T, 4>: [<$oper Assign>]<Simd<T, 4>>,
            {
                #[inline]
                fn [<$oper:lower _assign>](
                    &mut self,
                    rhs: Self,
                ) {
                    [<$oper Assign>]::[<$oper:lower _assign>](&mut self.0, rhs.0);
                }
            }

            impl<T: SimdElement> [<$oper Assign>]<&$type<T>> for $type<T>
            where
                Simd<T, 4>: [<$oper Assign>]<Simd<T, 4>>,
            {
                #[inline]
                fn [<$oper:lower _assign>](
                    &mut self,
                    rhs: &Self,
                ) {
                    [<$oper Assign>]::[<$oper:lower _assign>](&mut self.0, rhs.0)
                }
            }

            impl<T: SimdElement> [<$oper Assign>]<T> for $type<T>
            where
                Simd<T, 4>: [<$oper Assign>]<Simd<T, 4>>,
            {
                #[inline]
                fn [<$oper:lower _assign>](
                    &mut self,
                    rhs: T,
                ) {
                    [<$oper Assign>]::[<$oper:lower _assign>](self, $type::splat(rhs))
                }
            }
        }
    };
}

impl_operations!(Vec3A Add);
impl_operations!(Vec3A Sub);
impl_operations!(Vec3A Mul);
impl_operations!(Vec3A Div);

// Rust orphan rules are fucking annoying
// so we can't do T + Vec3A

impl<T: SimdElement> Neg for Vec3A<T>
where
    Simd<T, 4>: Neg<Output = Simd<T, 4>>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl<T: SimdElement> Neg for &Vec3A<T>
where
    Simd<T, 4>: Neg<Output = Simd<T, 4>>,
{
    type Output = Vec3A<T>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vec3A(-self.0)
    }
}

impl<T: SimdElement, Idx> Index<Idx> for Vec3A<T>
where
    Simd<T, 4>: Index<Idx>,
{
    type Output = <Simd<T, 4> as Index<Idx>>::Output;

    #[inline]
    fn index(
        &self,
        index: Idx,
    ) -> &Self::Output {
        &self.0[index]
    }
}

impl<T: SimdElement, Idx> IndexMut<Idx> for Vec3A<T>
where
    Simd<T, 4>: IndexMut<Idx>,
{
    #[inline]
    fn index_mut(
        &mut self,
        index: Idx,
    ) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T: SimdElement + num_traits::ConstZero> Sum for Vec3A<T>
where
    Self: Add<Output = Self>,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::splat(T::ZERO), |acc, v| acc + v)
    }
}
