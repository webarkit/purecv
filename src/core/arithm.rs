/*
 *  arithm.rs
 *  purecv
 *
 *  This file is part of purecv - OpenCV.
 *
 *  purecv is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  purecv is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with purecv.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  As a special exception, the copyright holders of this library give you
 *  permission to link this library with independent modules to produce an
 *  executable, regardless of the license terms of these independent modules, and to
 *  copy and distribute the resulting executable under terms of your choice,
 *  provided that you also meet, for each linked independent module, the terms and
 *  conditions of the license of that module. An independent module is a module
 *  which is neither derived from nor based on this library. If you modify this
 *  library, you may extend this exception to your version of the library, but you
 *  are not obligated to do so. If you do not wish to do so, delete this exception
 *  statement from your version.
 *
 *  Copyright 2026 WebARKit.
 *
 *  Author(s): Walter Perdan @kalwalt https://github.com/kalwalt
 *
 */

use crate::core::error::{PureCvError, Result};
use crate::core::types::{CmpTypes, NormTypes, ReduceTypes, Scalar};
use crate::core::{DataType, Matrix};
use num_traits::{Bounded, FromPrimitive, Num, SaturatingAdd, SaturatingSub, ToPrimitive};
use std::ops::{BitAnd, BitOr, BitXor, Not, Sub};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::core::simd::SimdElement;

/// Internal macro to handle feature-gated loop execution for binary operations.
/// Handles 4 combinations: simd+parallel, simd-only, parallel-only, scalar-only.
///
/// # Usage
///
/// ```ignore
/// // Without SIMD fast-path (scalar body only):
/// binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 + s2);
///
/// // With SIMD fast-path:
/// binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 + s2, simd: simd_add);
/// ```
macro_rules! binary_op {
    // Variant WITH simd fast-path
    ($dst:expr, $src1:expr, $src2:expr, $t_dst:ty, $t_src:ty, |$d:ident, $s1:ident, $s2:ident| $body:expr, simd: $simd_fn:ident) => {
        #[cfg(feature = "simd")]
        {
            // SIMD fast-path: only when dst type == src type and type has SIMD support
            if std::any::TypeId::of::<$t_dst>() == std::any::TypeId::of::<$t_src>()
                && <$t_src as SimdElement>::has_simd()
            {
                // Try the SIMD kernel. If it returns false, fall back to scalar.
                let simd_done;

                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    use std::sync::atomic::{AtomicBool, Ordering};

                    let chunk_size = ($dst.data.len() / rayon::current_num_threads()).max(1024);
                    let all_ok = AtomicBool::new(true);

                    $dst.data
                        .par_chunks_mut(chunk_size)
                        .enumerate()
                        .for_each(|(idx, dst_chunk)| {
                            let offset = idx * chunk_size;
                            let len = dst_chunk.len();
                            // Transmute the dst chunk to $t_src for the SIMD call
                            // This is safe because we verified $t_dst == $t_src above
                            let dst_as_src: &mut [$t_src] = unsafe {
                                std::slice::from_raw_parts_mut(
                                    dst_chunk.as_mut_ptr() as *mut $t_src,
                                    len,
                                )
                            };
                            let ok = <$t_src>::$simd_fn(
                                dst_as_src,
                                &$src1.data[offset..offset + len],
                                &$src2.data[offset..offset + len],
                            );
                            if !ok {
                                all_ok.store(false, Ordering::Relaxed);
                            }
                        });

                    simd_done = all_ok.load(Ordering::Relaxed);
                }
                #[cfg(not(feature = "parallel"))]
                {
                    let dst_as_src: &mut [$t_src] = unsafe {
                        std::slice::from_raw_parts_mut(
                            $dst.data.as_mut_ptr() as *mut $t_src,
                            $dst.data.len(),
                        )
                    };
                    simd_done = <$t_src>::$simd_fn(
                        dst_as_src,
                        &$src1.data,
                        &$src2.data,
                    );
                }

                if !simd_done {
                    // SIMD kernel declined — fall back to scalar loop
                    binary_op!(@scalar $dst, $src1, $src2, $t_dst, $t_src, |$d, $s1, $s2| $body);
                }
            } else {
                // Fallback to scalar loop when types differ or no SIMD support
                binary_op!(@scalar $dst, $src1, $src2, $t_dst, $t_src, |$d, $s1, $s2| $body);
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            binary_op!(@scalar $dst, $src1, $src2, $t_dst, $t_src, |$d, $s1, $s2| $body);
        }
    };

    // Variant WITHOUT simd fast-path (original behavior)
    ($dst:expr, $src1:expr, $src2:expr, $t_dst:ty, $t_src:ty, |$d:ident, $s1:ident, $s2:ident| $body:expr) => {
        binary_op!(@scalar $dst, $src1, $src2, $t_dst, $t_src, |$d, $s1, $s2| $body);
    };

    // Internal: scalar-only dispatch (parallel or sequential)
    (@scalar $dst:expr, $src1:expr, $src2:expr, $t_dst:ty, $t_src:ty, |$d:ident, $s1:ident, $s2:ident| $body:expr) => {
        #[cfg(feature = "parallel")]
        {
            $dst.data
                .par_iter_mut()
                .zip($src1.data.par_iter())
                .zip($src2.data.par_iter())
                .for_each(|((d_raw, &s1_raw), &s2_raw)| {
                    let $d: &mut $t_dst = d_raw;
                    let $s1: $t_src = s1_raw;
                    let $s2: $t_src = s2_raw;
                    $body
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            $dst.data
                .iter_mut()
                .zip($src1.data.iter())
                .zip($src2.data.iter())
                .for_each(|((d_raw, &s1_raw), &s2_raw)| {
                    let $d: &mut $t_dst = d_raw;
                    let $s1: $t_src = s1_raw;
                    let $s2: $t_src = s2_raw;
                    $body
                });
        }
    };
}

/// Internal macro to handle feature-gated loop execution for unary operations.
macro_rules! binary_op_scalar {
    ($dst:expr, $src:expr, $scalar:expr, $t_dst:ty, $t_src:ty, |$d:ident, $s:ident, $sc:ident| $body:expr) => {
        let channels = $src.channels as usize;
        let scalar_data = $scalar.v;

        #[cfg(feature = "parallel")]
        {
            $dst.data
                .par_chunks_exact_mut(channels)
                .zip($src.data.par_chunks_exact(channels))
                .for_each(|(d_chunk, s_chunk)| {
                    for i in 0..channels {
                        let $d: &mut $t_dst = &mut d_chunk[i];
                        let $s: $t_src = s_chunk[i];
                        let $sc = scalar_data[i];
                        $body
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            $dst.data
                .chunks_exact_mut(channels)
                .zip($src.data.chunks_exact(channels))
                .for_each(|(d_chunk, s_chunk)| {
                    for i in 0..channels {
                        let $d: &mut $t_dst = &mut d_chunk[i];
                        let $s: $t_src = s_chunk[i];
                        let $sc = scalar_data[i];
                        $body
                    }
                });
        }
    };
}

macro_rules! unary_op {
    // Variant WITH simd fast-path
    ($dst:expr, $src:expr, $t_dst:ty, $t_src:ty, |$d:ident, $s:ident| $body:expr, simd: $simd_fn:ident) => {
        #[cfg(feature = "simd")]
        {
            if std::any::TypeId::of::<$t_dst>() == std::any::TypeId::of::<$t_src>()
                && <$t_src as SimdElement>::has_simd()
            {
                // Try the SIMD kernel. If it returns false, fall back to scalar.
                let simd_done;

                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    use std::sync::atomic::{AtomicBool, Ordering};

                    let chunk_size = ($dst.data.len() / rayon::current_num_threads()).max(1024);
                    let all_ok = AtomicBool::new(true);

                    $dst.data
                        .par_chunks_mut(chunk_size)
                        .enumerate()
                        .for_each(|(idx, dst_chunk)| {
                            let offset = idx * chunk_size;
                            let len = dst_chunk.len();
                            let dst_as_src: &mut [$t_src] = unsafe {
                                std::slice::from_raw_parts_mut(
                                    dst_chunk.as_mut_ptr() as *mut $t_src,
                                    len,
                                )
                            };
                            let ok = <$t_src>::$simd_fn(
                                dst_as_src,
                                &$src.data[offset..offset + len],
                            );
                            if !ok {
                                all_ok.store(false, Ordering::Relaxed);
                            }
                        });

                    simd_done = all_ok.load(Ordering::Relaxed);
                }
                #[cfg(not(feature = "parallel"))]
                {
                    let dst_as_src: &mut [$t_src] = unsafe {
                        std::slice::from_raw_parts_mut(
                            $dst.data.as_mut_ptr() as *mut $t_src,
                            $dst.data.len(),
                        )
                    };
                    simd_done = <$t_src>::$simd_fn(
                        dst_as_src,
                        &$src.data,
                    );
                }

                if !simd_done {
                    // SIMD kernel declined — fall back to scalar loop
                    unary_op!(@scalar $dst, $src, $t_dst, $t_src, |$d, $s| $body);
                }
            } else {
                unary_op!(@scalar $dst, $src, $t_dst, $t_src, |$d, $s| $body);
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            unary_op!(@scalar $dst, $src, $t_dst, $t_src, |$d, $s| $body);
        }
    };

    // Variant WITHOUT simd fast-path (original behavior)
    ($dst:expr, $src:expr, $t_dst:ty, $t_src:ty, |$d:ident, $s:ident| $body:expr) => {
        unary_op!(@scalar $dst, $src, $t_dst, $t_src, |$d, $s| $body);
    };

    // Internal: scalar-only dispatch
    (@scalar $dst:expr, $src:expr, $t_dst:ty, $t_src:ty, |$d:ident, $s:ident| $body:expr) => {
        #[cfg(feature = "parallel")]
        {
            $dst.data
                .par_iter_mut()
                .zip($src.data.par_iter())
                .for_each(|(d_raw, &s_raw)| {
                    let $d: &mut $t_dst = d_raw;
                    let $s: $t_src = s_raw;
                    $body
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            $dst.data
                .iter_mut()
                .zip($src.data.iter())
                .for_each(|(d_raw, &s_raw)| {
                    let $d: &mut $t_dst = d_raw;
                    let $s: $t_src = s_raw;
                    $body
                });
        }
    };
}

/// Adds a scalar value to a matrix: dst = src1 + scalar
pub fn add_scalar<T>(src1: &Matrix<T>, scalar: Scalar<f64>) -> Result<Matrix<T>>
where
    T: DataType
        + SaturatingAdd
        + FromPrimitive
        + ToPrimitive
        + Send
        + Sync
        + 'static
        + Default
        + Copy,
{
    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);
    binary_op_scalar!(&mut dst, src1, scalar, T, T, |d, s, sc| {
        *d = s.saturating_add(&T::from_f64(sc).unwrap_or_default());
    });
    Ok(dst)
}

/// Subtracts a scalar value from a matrix: dst = src1 - scalar
pub fn subtract_scalar<T>(src1: &Matrix<T>, scalar: Scalar<f64>) -> Result<Matrix<T>>
where
    T: DataType
        + SaturatingSub
        + FromPrimitive
        + ToPrimitive
        + Send
        + Sync
        + 'static
        + Default
        + Copy,
{
    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);
    binary_op_scalar!(&mut dst, src1, scalar, T, T, |d, s, sc| {
        *d = s.saturating_sub(&T::from_f64(sc).unwrap_or_default());
    });
    Ok(dst)
}

/// Multiplies a matrix by a scalar value: dst = src1 * scalar
pub fn multiply_scalar<T>(src1: &Matrix<T>, scalar: Scalar<f64>) -> Result<Matrix<T>>
where
    T: DataType + FromPrimitive + ToPrimitive + Send + Sync + 'static + Default + Copy,
{
    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);
    binary_op_scalar!(&mut dst, src1, scalar, T, T, |d, s, sc| {
        let val = s.to_f64().unwrap_or_default() * sc;
        *d = T::from_f64(val).unwrap_or_default();
    });
    Ok(dst)
}

/// Divides a matrix by a scalar value: dst = src1 / scalar
pub fn divide_scalar<T>(src1: &Matrix<T>, scalar: Scalar<f64>) -> Result<Matrix<T>>
where
    T: DataType + FromPrimitive + ToPrimitive + Send + Sync + 'static + Default + Copy,
{
    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);
    binary_op_scalar!(&mut dst, src1, scalar, T, T, |d, s, sc| {
        if sc != 0.0 {
            let val = s.to_f64().unwrap_or_default() / sc;
            *d = T::from_f64(val).unwrap_or_default();
        } else {
            *d = T::default();
        }
    });
    Ok(dst)
}

/// Calculates the per-element sum of two matrices.
///
/// dst = src1 + src2
pub fn add<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + Default + SimdElement + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 + s2, simd: simd_add);

    Ok(dst)
}

/// Calculates the per-element difference between two matrices.
///
/// dst = src1 - src2
pub fn subtract<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + Default + SimdElement + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 - s2, simd: simd_sub);

    Ok(dst)
}

/// Calculates the per-element product of two matrices.
///
/// dst = src1 * src2
pub fn multiply<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + Default + SimdElement + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 * s2, simd: simd_mul);

    Ok(dst)
}

/// Calculates the per-element quotient of two matrices.
///
/// dst = src1 / src2
pub fn divide<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + Default + SimdElement + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| {
        if !s2.is_zero() {
            *d = s1 / s2;
        } else {
            *d = T::zero();
        }
    }, simd: simd_div);

    Ok(dst)
}

/// Calculates the per-element bit-wise conjunction of two matrices.
///
/// dst = src1 & src2
pub fn bitwise_and<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + BitAnd<Output = T> + Default + SimdElement + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 & s2, simd: simd_bitwise_and);

    Ok(dst)
}

/// Calculates the per-element bit-wise disjunction of two matrices.
///
/// dst = src1 | src2
pub fn bitwise_or<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + BitOr<Output = T> + Default + SimdElement + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 | s2, simd: simd_bitwise_or);

    Ok(dst)
}

/// Calculates the per-element bit-wise "exclusive or" operation on two matrices.
///
/// dst = src1 ^ src2
pub fn bitwise_xor<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + BitXor<Output = T> + Default + SimdElement + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 ^ s2, simd: simd_bitwise_xor);

    Ok(dst)
}

/// Calculates the per-element bit-wise conjunction of a matrix and a scalar.
pub fn bitwise_and_scalar<T>(src1: &Matrix<T>, scalar: Scalar<T>) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + BitAnd<Output = T> + Default + 'static,
{
    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);
    binary_op_scalar!(&mut dst, src1, scalar, T, T, |d, s, sc| *d = s & sc);
    Ok(dst)
}

/// Calculates the per-element bit-wise disjunction of a matrix and a scalar.
pub fn bitwise_or_scalar<T>(src1: &Matrix<T>, scalar: Scalar<T>) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + BitOr<Output = T> + Default + 'static,
{
    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);
    binary_op_scalar!(&mut dst, src1, scalar, T, T, |d, s, sc| *d = s | sc);
    Ok(dst)
}

/// Calculates the per-element bit-wise "exclusive or" operation on a matrix and a scalar.
pub fn bitwise_xor_scalar<T>(src1: &Matrix<T>, scalar: Scalar<T>) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + BitXor<Output = T> + Default + 'static,
{
    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);
    binary_op_scalar!(&mut dst, src1, scalar, T, T, |d, s, sc| *d = s ^ sc);
    Ok(dst)
}

/// Inverts every bit of every array element.
///
/// dst = ~src
pub fn bitwise_not<T>(src: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + Not<Output = T> + Default + SimdElement + 'static,
{
    let mut dst = Matrix::<T>::new(src.rows, src.cols, src.channels);

    unary_op!(dst, src, T, T, |d, s| *d = !s, simd: simd_bitwise_not);

    Ok(dst)
}

/// Calculates the per-element minimum of two matrices.
pub fn min<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Default + SimdElement + 'static,
{
    if !src1.dims_match(src2) {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| {
        *d = if s1 < s2 { s1 } else { s2 };
    }, simd: simd_min);

    Ok(dst)
}

/// Calculates the per-element maximum of two matrices.
pub fn max<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Default + SimdElement + 'static,
{
    if !src1.dims_match(src2) {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| {
        *d = if s1 > s2 { s1 } else { s2 };
    }, simd: simd_max);

    Ok(dst)
}

/// Calculates the per-element absolute difference between two matrices.
pub fn abs_diff<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Default + SimdElement + 'static,
{
    if !src1.dims_match(src2) {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| {
        *d = if s1 > s2 { s1 - s2 } else { s2 - s1 };
    }, simd: simd_absdiff);

    Ok(dst)
}

/// Calculates the per-element minimum of a matrix and a scalar.
pub fn min_scalar<T>(src1: &Matrix<T>, scalar: Scalar<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Default + 'static,
{
    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);
    binary_op_scalar!(&mut dst, src1, scalar, T, T, |d, s, sc| {
        *d = if s < sc { s } else { sc };
    });
    Ok(dst)
}

/// Calculates the per-element maximum of a matrix and a scalar.
pub fn max_scalar<T>(src1: &Matrix<T>, scalar: Scalar<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Default + 'static,
{
    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);
    binary_op_scalar!(&mut dst, src1, scalar, T, T, |d, s, sc| {
        *d = if s > sc { s } else { sc };
    });
    Ok(dst)
}

/// Calculates the per-element absolute difference between a matrix and a scalar.
pub fn abs_diff_scalar<T>(src1: &Matrix<T>, scalar: Scalar<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Default + 'static,
{
    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);
    binary_op_scalar!(&mut dst, src1, scalar, T, T, |d, s, sc| {
        *d = if s > sc { s - sc } else { sc - s };
    });
    Ok(dst)
}

/// Performs per-element comparison of two matrices.
///
/// The function compares the corresponding elements of two matrices.
/// The result is a 8-bit single-channel image where elements are 255 (true) or 0 (false).
pub fn compare<T>(src1: &Matrix<T>, src2: &Matrix<T>, cmpop: CmpTypes) -> Result<Matrix<u8>>
where
    T: Copy + Send + Sync + PartialOrd + Default + 'static,
{
    if !src1.dims_match(src2) {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<u8>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, u8, T, |d, s1, s2| {
        let res = match cmpop {
            CmpTypes::Eq => s1 == s2,
            CmpTypes::Gt => s1 > s2,
            CmpTypes::Ge => s1 >= s2,
            CmpTypes::Lt => s1 < s2,
            CmpTypes::Le => s1 <= s2,
            CmpTypes::Ne => s1 != s2,
        };
        *d = if res { 255 } else { 0 };
    });

    Ok(dst)
}

/// Performs per-element comparison of a matrix and a scalar.
///
/// The function compares every element of the matrix with the corresponding scalar value.
/// The result is a 8-bit image where elements are 255 (true) or 0 (false).
pub fn compare_scalar<T>(src1: &Matrix<T>, scalar: Scalar<T>, cmpop: CmpTypes) -> Result<Matrix<u8>>
where
    T: Copy + Send + Sync + PartialOrd + Default + 'static,
{
    let mut dst = Matrix::<u8>::new(src1.rows, src1.cols, src1.channels);

    binary_op_scalar!(&mut dst, src1, scalar, u8, T, |d, s, sc| {
        let res = match cmpop {
            CmpTypes::Eq => s == sc,
            CmpTypes::Gt => s > sc,
            CmpTypes::Ge => s >= sc,
            CmpTypes::Lt => s < sc,
            CmpTypes::Le => s <= sc,
            CmpTypes::Ne => s != sc,
        };
        *d = if res { 255 } else { 0 };
    });

    Ok(dst)
}

/// Checks if array elements lie between the elements of two other arrays.
///
/// For multi-channel arrays, each channel is checked independently:
/// `dst(I) = lowerb(I)_0 <= src(I)_0 <= upperb(I)_0 && lowerb(I)_1 <= src(I)_1 <= upperb(I)_1 ...`
pub fn in_range<T>(
    src: &Matrix<T>,
    lowerb: &Matrix<T>,
    upperb: &Matrix<T>,
    dst: &mut Matrix<u8>,
) -> Result<()>
where
    T: DataType + PartialOrd + Default + Copy + Sync + Send,
{
    if src.rows != lowerb.rows
        || src.cols != lowerb.cols
        || src.channels != lowerb.channels
        || src.rows != upperb.rows
        || src.cols != upperb.cols
        || src.channels != upperb.channels
    {
        return Err(PureCvError::InvalidInput(
            "Size or channels mismatch".into(),
        ));
    }

    dst.create(src.rows, src.cols, 1);
    let s = src.as_slice();
    let l = lowerb.as_slice();
    let u = upperb.as_slice();
    let d = dst.as_mut_slice();
    let channels = src.channels;

    #[cfg(feature = "parallel")]
    {
        d.par_iter_mut().enumerate().for_each(|(i, mask_val)| {
            let offset = i * channels;
            let mut res = true;
            for c in 0..channels {
                let val = s[offset + c];
                if val < l[offset + c] || val > u[offset + c] {
                    res = false;
                    break;
                }
            }
            *mask_val = if res { 255 } else { 0 };
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (i, mask_val) in d.iter_mut().enumerate() {
            let offset = i * channels;
            let mut res = true;
            for c in 0..channels {
                let val = s[offset + c];
                if val < l[offset + c] || val > u[offset + c] {
                    res = false;
                    break;
                }
            }
            *mask_val = if res { 255 } else { 0 };
        }
    }
    Ok(())
}

/// Checks if array elements lie between two scalars.
///
/// For multi-channel arrays, each channel is checked independently and the
/// result is ANDed: `dst(I) = lowerb_0 <= src(I)_0 <= upperb_0 && lowerb_1 <= src(I)_1 <= upperb_1 ...`
pub fn in_range_scalar<T>(
    src: &Matrix<T>,
    lowerb: &[T],
    upperb: &[T],
    dst: &mut Matrix<u8>,
) -> Result<()>
where
    T: DataType + PartialOrd + Default + Copy + Sync + Send,
{
    if lowerb.len() < src.channels || upperb.len() < src.channels {
        return Err(PureCvError::InvalidInput(
            "Scalars must have at least as many elements as src channels".into(),
        ));
    }

    dst.create(src.rows, src.cols, 1);
    let channels = src.channels;
    let src_data = src.as_slice();
    let dst_data = dst.as_mut_slice();

    #[cfg(feature = "parallel")]
    {
        dst_data
            .par_iter_mut()
            .enumerate()
            .for_each(|(pixel_idx, d)| {
                let offset = pixel_idx * channels;
                let mut res = true;
                for c in 0..channels {
                    let s = src_data[offset + c];
                    let l = lowerb[c];
                    let u = upperb[c];
                    if s < l || s > u {
                        res = false;
                        break;
                    }
                }
                *d = if res { 255 } else { 0 };
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (pixel_idx, d) in dst_data.iter_mut().enumerate() {
            let offset = pixel_idx * channels;
            let mut res = true;
            for c in 0..channels {
                let s = src_data[offset + c];
                let l = lowerb[c];
                let u = upperb[c];
                if s < l || s > u {
                    res = false;
                    break;
                }
            }
            *d = if res { 255 } else { 0 };
        }
    }

    Ok(())
}

/// Calculates the per-element absolute difference between two matrices.
///
/// dst = |src1 - src2|
pub fn absdiff<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Sub<Output = T> + Default + SimdElement + 'static,
{
    if !src1.dims_match(src2) {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| {
        *d = if s1 > s2 { s1 - s2 } else { s2 - s1 };
    }, simd: simd_absdiff);

    Ok(dst)
}

/// Calculates the sum of array elements.
///
/// Returns a Scalar with the sum of each channel.
pub fn sum<T>(src: &Matrix<T>) -> Scalar<f64>
where
    T: Num + ToPrimitive + Copy + Send + Sync + SimdElement + 'static,
{
    // SIMD fast-path: single-channel and type supports SIMD
    #[cfg(feature = "simd")]
    {
        if src.channels == 1 && T::has_simd() {
            if let Some(s) = T::simd_sum(&src.data) {
                return Scalar::new(s, 0.0, 0.0, 0.0);
            }
        }
    }

    #[cfg(feature = "parallel")]
    {
        let sums = src
            .data
            .par_chunks_exact(src.channels)
            .fold(
                || [0.0f64; 4],
                |mut acc, pixel| {
                    for (i, &val) in pixel.iter().enumerate() {
                        if i < 4 {
                            acc[i] += val.to_f64().unwrap_or(0.0);
                        }
                    }
                    acc
                },
            )
            .reduce(
                || [0.0f64; 4],
                |mut a, b| {
                    for i in 0..4 {
                        a[i] += b[i];
                    }
                    a
                },
            );
        Scalar::new(sums[0], sums[1], sums[2], sums[3])
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut sums = [0.0f64; 4];
        for pixel in src.data.chunks_exact(src.channels) {
            for (i, &val) in pixel.iter().enumerate() {
                if i < 4 {
                    sums[i] += val.to_f64().unwrap_or(0.0);
                }
            }
        }
        Scalar::new(sums[0], sums[1], sums[2], sums[3])
    }
}

/// Calculates the mean of matrix elements.
///
/// Returns a Scalar with the mean of each channel.
pub fn mean<T>(src: &Matrix<T>) -> Scalar<f64>
where
    T: DataType + Num + ToPrimitive + Copy + Send + Sync + SimdElement + 'static,
{
    let s = sum(src);
    let total_pixels = (src.rows * src.cols) as f64;
    Scalar::new(
        s.v[0] / total_pixels,
        s.v[1] / total_pixels,
        s.v[2] / total_pixels,
        s.v[3] / total_pixels,
    )
}

/// Calculates a mean and standard deviation of matrix elements.
pub fn mean_std_dev<T>(src: &Matrix<T>) -> (Scalar<f64>, Scalar<f64>)
where
    T: DataType + Num + ToPrimitive + Copy + Send + Sync + SimdElement + 'static,
{
    let m = mean(src);
    let mut sq_sum = [0.0f64; 4];
    let total_pixels = (src.rows * src.cols) as f64;

    for pixel in src.data.chunks_exact(src.channels) {
        for (i, &val) in pixel.iter().enumerate() {
            if i < 4 {
                let v = val.to_f64().unwrap_or(0.0) - m.v[i];
                sq_sum[i] += v * v;
            }
        }
    }

    let std_dev = Scalar::new(
        (sq_sum[0] / total_pixels).sqrt(),
        (sq_sum[1] / total_pixels).sqrt(),
        (sq_sum[2] / total_pixels).sqrt(),
        (sq_sum[3] / total_pixels).sqrt(),
    );

    (m, std_dev)
}

/// Calculates an absolute array norm.
///
/// Returns the norm value as f64.
pub fn norm<T>(src: &Matrix<T>, norm_type: NormTypes, mask: Option<&Matrix<u8>>) -> Result<f64>
where
    T: DataType + ToPrimitive + Default + Copy + Sync + Send + SimdElement,
{
    match norm_type {
        NormTypes::Inf => {
            #[cfg(feature = "parallel")]
            {
                if let Some(m) = mask {
                    Ok(src
                        .data
                        .par_iter()
                        .zip(m.data.par_iter())
                        .filter(|(_, &mask_val)| mask_val != 0)
                        .map(|(&x, _)| x.to_f64().unwrap_or(0.0).abs())
                        .reduce(|| 0.0, f64::max))
                } else {
                    Ok(src
                        .data
                        .par_iter()
                        .map(|&x| x.to_f64().unwrap_or(0.0).abs())
                        .reduce(|| 0.0, f64::max))
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                if let Some(m) = mask {
                    Ok(src
                        .data
                        .iter()
                        .zip(m.data.iter())
                        .filter(|(_, &mask_val)| mask_val != 0)
                        .map(|(&x, _)| x.to_f64().unwrap_or(0.0).abs())
                        .fold(0.0, f64::max))
                } else {
                    Ok(src
                        .data
                        .iter()
                        .map(|&x| x.to_f64().unwrap_or(0.0).abs())
                        .fold(0.0, f64::max))
                }
            }
        }
        NormTypes::L1 => {
            #[cfg(feature = "parallel")]
            {
                if let Some(m) = mask {
                    Ok(src
                        .data
                        .par_iter()
                        .zip(m.data.par_iter())
                        .filter(|(_, &mask_val)| mask_val != 0)
                        .map(|(&x, _)| x.to_f64().unwrap_or(0.0).abs())
                        .sum::<f64>())
                } else {
                    Ok(src
                        .data
                        .par_iter()
                        .map(|&x| x.to_f64().unwrap_or(0.0).abs())
                        .sum::<f64>())
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                if let Some(m) = mask {
                    Ok(src
                        .data
                        .iter()
                        .zip(m.data.iter())
                        .filter(|(_, &mask_val)| mask_val != 0)
                        .map(|(&x, _)| x.to_f64().unwrap_or(0.0).abs())
                        .sum::<f64>())
                } else {
                    Ok(src
                        .data
                        .iter()
                        .map(|&x| x.to_f64().unwrap_or(0.0).abs())
                        .sum::<f64>())
                }
            }
        }
        NormTypes::L2 => {
            // SIMD fast-path for L2 norm without mask
            #[cfg(feature = "simd")]
            {
                if mask.is_none() && T::has_simd() {
                    if let Some(sq_sum) = T::simd_norm_l2_sq(&src.data) {
                        return Ok(sq_sum.sqrt());
                    }
                }
            }

            #[cfg(feature = "parallel")]
            {
                let sq_sum = if let Some(m) = mask {
                    src.data
                        .par_iter()
                        .zip(m.data.par_iter())
                        .filter(|(_, &mask_val)| mask_val != 0)
                        .map(|(&x, _)| {
                            let val = x.to_f64().unwrap_or(0.0);
                            val * val
                        })
                        .sum::<f64>()
                } else {
                    src.data
                        .par_iter()
                        .map(|&x| {
                            let val = x.to_f64().unwrap_or(0.0);
                            val * val
                        })
                        .sum::<f64>()
                };
                Ok(sq_sum.sqrt())
            }
            #[cfg(not(feature = "parallel"))]
            {
                let sq_sum = if let Some(m) = mask {
                    src.data
                        .iter()
                        .zip(m.data.iter())
                        .filter(|(_, &mask_val)| mask_val != 0)
                        .map(|(&x, _)| {
                            let val = x.to_f64().unwrap_or(0.0);
                            val * val
                        })
                        .sum::<f64>()
                } else {
                    src.data
                        .iter()
                        .map(|&x| {
                            let val = x.to_f64().unwrap_or(0.0);
                            val * val
                        })
                        .sum::<f64>()
                };
                Ok(sq_sum.sqrt())
            }
        }
        _ => Err(PureCvError::NotImplemented(format!(
            "Norm type {:?} is not implemented",
            norm_type
        ))),
    }
}

/// Normalizes the norm or value range of an array.
///
/// For MINMAX normalization, alpha is the lower bound and beta is the upper bound.
/// For other norms, alpha is the norm value to reach.
pub fn normalize<T>(
    src: &Matrix<T>,
    dst: &mut Matrix<T>,
    alpha: f64,
    beta: f64,
    norm_type: NormTypes,
    _dtype: i32,
    mask: Option<&Matrix<u8>>,
) -> Result<()>
where
    T: DataType + Send + Sync + FromPrimitive + Default + Copy + ToPrimitive + SimdElement,
{
    // If dtype is negative, we keep source type. Currently we don't support converting here yet,
    // so we just ignore it if it equals T's depth.

    match norm_type {
        NormTypes::MinMax => {
            let mut min_val = f64::MAX;
            let mut max_val = f64::MIN;

            if let Some(m) = mask {
                for (&val, &mask_val) in src.data.iter().zip(m.data.iter()) {
                    if mask_val != 0 {
                        let v = val.to_f64().unwrap_or(0.0);
                        if v < min_val {
                            min_val = v;
                        }
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
            } else {
                for &val in src.data.iter() {
                    let v = val.to_f64().unwrap_or(0.0);
                    if v < min_val {
                        min_val = v;
                    }
                    if v > max_val {
                        max_val = v;
                    }
                }
            }

            let scale = if max_val != min_val {
                (beta - alpha) / (max_val - min_val)
            } else {
                0.0
            };

            #[cfg(feature = "parallel")]
            {
                if let Some(m) = mask {
                    dst.data
                        .par_iter_mut()
                        .zip(src.data.par_iter())
                        .zip(m.data.par_iter())
                        .for_each(|((d, &s), &mask_val)| {
                            if mask_val != 0 {
                                let v = s.to_f64().unwrap_or(0.0);
                                let res = (v - min_val) * scale + alpha;
                                *d = T::from_f64(res).unwrap_or(T::default());
                            }
                        });
                } else {
                    dst.data
                        .par_iter_mut()
                        .zip(src.data.par_iter())
                        .for_each(|(d, &s)| {
                            let v = s.to_f64().unwrap_or(0.0);
                            let res = (v - min_val) * scale + alpha;
                            *d = T::from_f64(res).unwrap_or(T::default());
                        });
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                if let Some(m) = mask {
                    for (i, &mask_val) in m.data.iter().enumerate() {
                        if mask_val != 0 {
                            let v = src.data[i].to_f64().unwrap_or(0.0);
                            let res = (v - min_val) * scale + alpha;
                            dst.data[i] = T::from_f64(res).unwrap_or(T::default());
                        }
                    }
                } else {
                    for (d, s) in dst.data.iter_mut().zip(src.data.iter()) {
                        let v = s.to_f64().unwrap_or(0.0);
                        let res = (v - min_val) * scale + alpha;
                        *d = T::from_f64(res).unwrap_or(T::default());
                    }
                }
            }
        }
        NormTypes::L1 | NormTypes::L2 | NormTypes::Inf => {
            let n = norm(src, norm_type, mask)?;
            let scale = if n != 0.0 { alpha / n } else { 0.0 };

            #[cfg(feature = "parallel")]
            {
                if let Some(m) = mask {
                    dst.data
                        .par_iter_mut()
                        .zip(src.data.par_iter())
                        .zip(m.data.par_iter())
                        .for_each(|((d, &s), &mask_val)| {
                            if mask_val != 0 {
                                let res = s.to_f64().unwrap_or(0.0) * scale;
                                *d = T::from_f64(res).unwrap_or(T::default());
                            }
                        });
                } else {
                    dst.data
                        .par_iter_mut()
                        .zip(src.data.par_iter())
                        .for_each(|(d, &s)| {
                            let res = s.to_f64().unwrap_or(0.0) * scale;
                            *d = T::from_f64(res).unwrap_or(T::default());
                        });
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                if let Some(m) = mask {
                    for (i, &mask_val) in m.data.iter().enumerate() {
                        if mask_val != 0 {
                            let res = src.data[i].to_f64().unwrap_or(0.0) * scale;
                            dst.data[i] = T::from_f64(res).unwrap_or(T::default());
                        }
                    }
                } else {
                    for (d, s) in dst.data.iter_mut().zip(src.data.iter()) {
                        let res = s.to_f64().unwrap_or(0.0) * scale;
                        *d = T::from_f64(res).unwrap_or(T::default());
                    }
                }
            }
        }
        _ => {
            return Err(PureCvError::NotImplemented(
                "Requested normalization type is not implemented yet".to_string(),
            ))
        }
    }
    Ok(())
}

/// Reduces a matrix to a vector.
///
/// dim 0: reduced to 1 row.
/// dim 1: reduced to 1 column.
pub fn reduce<T>(src: &Matrix<T>, dim: i32, reduce_op: ReduceTypes) -> Result<Matrix<T>>
where
    T: DataType
        + Num
        + ToPrimitive
        + FromPrimitive
        + Copy
        + Send
        + Sync
        + PartialOrd
        + Default
        + 'static,
{
    let (rows, cols) = if dim == 0 {
        (1, src.cols)
    } else if dim == 1 {
        (src.rows, 1)
    } else {
        return Err(PureCvError::InvalidInput("dim must be 0 or 1".to_string()));
    };

    let mut dst_data = vec![T::default(); rows * cols * src.channels];
    let channels = src.channels;
    let src_rows = src.rows;
    let src_cols = src.cols;

    #[cfg(feature = "parallel")]
    {
        dst_data.par_iter_mut().enumerate().for_each(|(idx, d)| {
            let channel = idx % channels;
            let pos = idx / channels;

            let count = if dim == 0 { src_rows } else { src_cols };
            let mut accum: f64 = match reduce_op {
                ReduceTypes::Sum | ReduceTypes::Avg => 0.0,
                ReduceTypes::Max => f64::MIN,
                ReduceTypes::Min => f64::MAX,
            };

            for i in 0..count {
                let (r, c) = if dim == 0 { (i, pos) } else { (pos, i) };
                if let Some(val) = src.get(r, c, channel).and_then(|v| v.to_f64()) {
                    match reduce_op {
                        ReduceTypes::Sum | ReduceTypes::Avg => accum += val,
                        ReduceTypes::Max => {
                            if val > accum {
                                accum = val;
                            }
                        }
                        ReduceTypes::Min => {
                            if val < accum {
                                accum = val;
                            }
                        }
                    }
                }
            }

            let res = if reduce_op == ReduceTypes::Avg {
                accum / count as f64
            } else {
                accum
            };
            *d = T::from_f64(res).unwrap_or_default();
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (idx, d) in dst_data.iter_mut().enumerate() {
            let channel = idx % channels;
            let pos = idx / channels;

            let count = if dim == 0 { src_rows } else { src_cols };
            let mut accum: f64 = match reduce_op {
                ReduceTypes::Sum | ReduceTypes::Avg => 0.0,
                ReduceTypes::Max => f64::MIN,
                ReduceTypes::Min => f64::MAX,
            };

            for i in 0..count {
                let (r, c) = if dim == 0 { (i, pos) } else { (pos, i) };
                if let Some(val) = src.get(r, c, channel).and_then(|v| v.to_f64()) {
                    match reduce_op {
                        ReduceTypes::Sum | ReduceTypes::Avg => accum += val,
                        ReduceTypes::Max => {
                            if val > accum {
                                accum = val;
                            }
                        }
                        ReduceTypes::Min => {
                            if val < accum {
                                accum = val;
                            }
                        }
                    }
                }
            }

            let res = if reduce_op == ReduceTypes::Avg {
                accum / count as f64
            } else {
                accum
            };
            *d = T::from_f64(res).unwrap_or_default();
        }
    }

    Ok(Matrix {
        rows,
        cols,
        channels,
        data: dst_data,
    })
}

/// Counts non-zero array elements.
///
/// The input matrix must be single-channel.
pub fn count_non_zero<T>(src: &Matrix<T>) -> Result<i32>
where
    T: Num + Copy + Send + Sync + 'static,
{
    if src.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "countNonZero requires single-channel matrix".into(),
        ));
    }
    #[cfg(feature = "parallel")]
    {
        Ok(src.data.par_iter().filter(|&&x| !x.is_zero()).count() as i32)
    }
    #[cfg(not(feature = "parallel"))]
    {
        Ok(src.data.iter().filter(|&&x| !x.is_zero()).count() as i32)
    }
}

/// Finds the global minimum and maximum in an array and their locations.
///
/// Returns (min_val, max_val, min_loc, max_loc).
/// Note: for multi-channel arrays, it finds min/max across all channels, but locations are indices in the flat array.
pub fn min_max_loc<T>(src: &Matrix<T>) -> (f64, f64, (i32, i32), (i32, i32))
where
    T: Num + ToPrimitive + Copy + PartialOrd + Default + 'static,
{
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    let mut min_loc = (0, 0);
    let mut max_loc = (0, 0);

    for i in 0..src.rows {
        for j in 0..src.cols {
            for ch in 0..src.channels {
                let val = src.get(i, j, ch).and_then(|v| v.to_f64()).unwrap_or(0.0);
                if val < min_val {
                    min_val = val;
                    min_loc = (j as i32, i as i32);
                }
                if val > max_val {
                    max_val = val;
                    max_loc = (j as i32, i as i32);
                }
            }
        }
    }
    (min_val, max_val, min_loc, max_loc)
}

/// Calculates the weighted sum of two matrices.
///
/// dst = src1*alpha + src2*beta + gamma
pub fn add_weighted<T>(
    src1: &Matrix<T>,
    alpha: f64,
    src2: &Matrix<T>,
    beta: f64,
    gamma: f64,
) -> Result<Matrix<T>>
where
    T: Num
        + Copy
        + Send
        + Sync
        + PartialOrd
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + Default
        + SimdElement
        + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    #[cfg(feature = "simd")]
    {
        // Only f32/f64 have simd_add_weighted; u8 and others use the scalar fallback.
        if (std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
            || std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>())
            && <T as SimdElement>::has_simd()
        {
            <T>::simd_add_weighted(&mut dst.data, &src1.data, &src2.data, alpha, beta, gamma);
            return Ok(dst);
        }
    }

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| {
        let val = s1.to_f64().unwrap_or(0.0) * alpha + s2.to_f64().unwrap_or(0.0) * beta + gamma;
        *d = T::from_f64(val).unwrap_or(T::zero());
    });

    Ok(dst)
}
///
/// dst = sqrt(src)
pub fn sqrt<T>(src: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num
        + Copy
        + Send
        + Sync
        + PartialOrd
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + Default
        + SimdElement
        + 'static,
{
    let mut dst = Matrix::<T>::new(src.rows, src.cols, src.channels);

    unary_op!(dst, src, T, T, |d, s| {
        let val = s.to_f64().unwrap_or(0.0).sqrt();
        *d = T::from_f64(val).unwrap_or(T::zero());
    }, simd: simd_sqrt);

    Ok(dst)
}
///
/// dst = exp(src)
pub fn exp<T>(src: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num
        + Copy
        + Send
        + Sync
        + PartialOrd
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + Default
        + 'static,
{
    let mut dst = Matrix::<T>::new(src.rows, src.cols, src.channels);

    unary_op!(dst, src, T, T, |d, s| {
        let val = s.to_f64().unwrap_or(0.0).exp();
        *d = T::from_f64(val).unwrap_or(T::zero());
    });

    Ok(dst)
}

/// Calculates the natural logarithm of every matrix element.
///
/// dst = log(src)
pub fn log<T>(src: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num
        + Copy
        + Send
        + Sync
        + PartialOrd
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + Default
        + 'static,
{
    let mut dst = Matrix::<T>::new(src.rows, src.cols, src.channels);

    unary_op!(dst, src, T, T, |d, s| {
        let val = s.to_f64().unwrap_or(0.0).ln();
        *d = T::from_f64(val).unwrap_or(T::zero());
    });

    Ok(dst)
}

/// Raises every matrix element to a power.
///
/// dst = src^p
pub fn pow<T>(src: &Matrix<T>, p: f64) -> Result<Matrix<T>>
where
    T: Num
        + Copy
        + Send
        + Sync
        + PartialOrd
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + Default
        + 'static,
{
    let mut dst = Matrix::<T>::new(src.rows, src.cols, src.channels);

    unary_op!(dst, src, T, T, |d, s| {
        let val = s.to_f64().unwrap_or(0.0).powf(p);
        *d = T::from_f64(val).unwrap_or(T::zero());
    });

    Ok(dst)
}

/// Scales, calculates absolute values, and converts the result to 8-bit.
///
/// dst(I) = saturate_cast<u8>(|src(I)*alpha + beta|)
pub fn convert_scale_abs<T>(src: &Matrix<T>, alpha: f64, beta: f64) -> Result<Matrix<u8>>
where
    T: Num + Copy + Send + Sync + ToPrimitive + Default + SimdElement + 'static,
{
    let mut dst = Matrix::<u8>::new(src.rows, src.cols, src.channels);

    #[cfg(feature = "simd")]
    {
        // Only f32/f64 have simd_convert_scale_abs; others use the scalar fallback.
        if (std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
            || std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>())
            && <T as SimdElement>::has_simd()
        {
            <T>::simd_convert_scale_abs(&mut dst.data, &src.data, alpha, beta);
            return Ok(dst);
        }
    }

    unary_op!(dst, src, u8, T, |d, s| {
        let val = (s.to_f64().unwrap_or(0.0) * alpha + beta).abs();
        *d = val.clamp(0.0, 255.0).round() as u8;
    });

    Ok(dst)
}

pub const GEMM_1_T: i32 = 1;
pub const GEMM_2_T: i32 = 2;
pub const GEMM_3_T: i32 = 4;

/// Performs generalized matrix multiplication.
///
/// dst = alpha * op(src1) * op(src2) + beta * op(src3)
/// where op(X) is X or X^T based on flags.
pub fn gemm<T>(
    src1: &Matrix<T>,
    src2: &Matrix<T>,
    alpha: f64,
    src3: &Matrix<T>,
    beta: f64,
    flags: i32,
) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + ToPrimitive + FromPrimitive + Default + 'static,
{
    let trans1 = (flags & GEMM_1_T) != 0;
    let trans2 = (flags & GEMM_2_T) != 0;
    let trans3 = (flags & GEMM_3_T) != 0;

    let (m, k1) = if trans1 {
        (src1.cols, src1.rows)
    } else {
        (src1.rows, src1.cols)
    };
    let (k2, n) = if trans2 {
        (src2.cols, src2.rows)
    } else {
        (src2.rows, src2.cols)
    };

    if k1 != k2 {
        return Err(PureCvError::InvalidDimensions(format!(
            "Incompatible dimensions for GEMM: {}x{} and {}x{}",
            m, k1, k2, n
        )));
    }

    let k = k1;
    let mut dst = Matrix::<T>::new(m, n, src1.channels);

    #[cfg(feature = "parallel")]
    {
        dst.data
            .par_chunks_mut(n * src1.channels)
            .enumerate()
            .for_each(|(i, row_slice)| {
                for j in 0..n {
                    for c in 0..src1.channels {
                        let mut sum = 0.0;
                        for l in 0..k {
                            let idx1 = if trans1 {
                                (l * src1.cols + i) * src1.channels + c
                            } else {
                                (i * src1.cols + l) * src1.channels + c
                            };
                            let idx2 = if trans2 {
                                (j * src2.cols + l) * src2.channels + c
                            } else {
                                (l * src2.cols + j) * src2.channels + c
                            };

                            let v1 = src1.data[idx1].to_f64().unwrap_or(0.0);
                            let v2 = src2.data[idx2].to_f64().unwrap_or(0.0);
                            sum += v1 * v2;
                        }

                        let v3 = if beta != 0.0 && src3.rows > 0 {
                            let (r3, c3) = if trans3 { (j, i) } else { (i, j) };
                            let idx3 = (r3 * src3.cols + c3) * src3.channels + c;
                            src3.data[idx3].to_f64().unwrap_or(0.0)
                        } else {
                            0.0
                        };

                        let final_val = alpha * sum + beta * v3;
                        row_slice[j * src1.channels + c] =
                            T::from_f64(final_val).unwrap_or(T::zero());
                    }
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..m {
            for j in 0..n {
                for c in 0..src1.channels {
                    let mut sum = 0.0;
                    for l in 0..k {
                        let idx1 = if trans1 {
                            (l * src1.cols + i) * src1.channels + c
                        } else {
                            (i * src1.cols + l) * src1.channels + c
                        };
                        let idx2 = if trans2 {
                            (j * src2.cols + l) * src2.channels + c
                        } else {
                            (l * src2.cols + j) * src2.channels + c
                        };

                        let v1 = src1.data[idx1].to_f64().unwrap_or(0.0);
                        let v2 = src2.data[idx2].to_f64().unwrap_or(0.0);
                        sum += v1 * v2;
                    }

                    let v3 = if beta != 0.0 && src3.rows > 0 {
                        let (r3, c3) = if trans3 { (j, i) } else { (i, j) };
                        let idx3 = (r3 * src3.cols + c3) * src3.channels + c;
                        src3.data[idx3].to_f64().unwrap_or(0.0)
                    } else {
                        0.0
                    };

                    let final_val = alpha * sum + beta * v3;
                    dst.set(i, j, c, T::from_f64(final_val).unwrap_or(T::zero()));
                }
            }
        }
    }

    Ok(dst)
}

/// Sets the matrix elements to 0, except for the diagonal which is set to a given value.
///
/// Initializes a scaled identity matrix.
///
/// The function initializes a scaled identity matrix:
/// `dst(i,j) = s` if `i=j`, else `0`.
///
/// # Arguments
///
/// * `mtx` - Matrix to be initialized as an identity.
/// * `s` - Scalar value to be assigned to diagonal elements.
///
/// # Example
///
/// ```rust
/// use purecv::core::{Matrix, Scalar, set_identity};
///
/// let mut mat = Matrix::<f32>::new(3, 3, 1);
/// set_identity(&mut mat, Scalar::all(1.0));
/// // mat becomes a 3x3 identity matrix
/// ```
pub fn set_identity<T>(mtx: &mut Matrix<T>, s: Scalar<T>)
where
    T: Num + Copy + Send + Sync + Default + 'static,
{
    mtx.data.fill(T::zero());
    let n = std::cmp::min(mtx.rows, mtx.cols);
    let channels = mtx.channels;
    let cols = mtx.cols;

    for i in 0..n {
        let base_idx = (i * cols + i) * channels;
        for c in 0..channels {
            mtx.data[base_idx + c] = s.v[c];
        }
    }
}

/// Checks if array elements lie within a specified range.
///
/// The function checks if every element of the input matrix is within the range `[min_val, max_val]`.
///
/// # Arguments
///
/// * `src` - Input matrix.
/// * `min_val` - Lower boundary of the range (inclusive).
/// * `max_val` - Upper boundary of the range (inclusive).
///
/// # Example
///
/// ```rust
/// use purecv::core::{Matrix, check_range};
///
/// let mat = Matrix::from_vec(2, 2, 1, vec![10, 20, 30, 40]);
/// let ok = check_range(&mat, 0.0, 255.0);
/// assert!(ok);
/// ```
pub fn check_range<T>(src: &Matrix<T>, min_val: f64, max_val: f64) -> bool
where
    T: Num + Copy + Send + Sync + ToPrimitive + Default + 'static,
{
    src.data.iter().all(|&val| {
        let v = val.to_f64().unwrap_or(0.0);
        v >= min_val && v <= max_val
    })
}

/// Computes the scalar dot product of two matrices (vectors).
///
/// The function computes the dot product of two matrices. If the matrices are not vectors,
/// they are treated as vectors by iterating through all elements.
///
/// # Arguments
///
/// * `src1` - First source matrix.
/// * `src2` - Second source matrix.
///
/// # Example
///
/// ```rust
/// use purecv::core::{Matrix, dot};
///
/// let v1 = Matrix::from_vec(1, 3, 1, vec![1.0, 2.0, 3.0]);
/// let v2 = Matrix::from_vec(1, 3, 1, vec![4.0, 5.0, 6.0]);
/// let d = dot(&v1, &v2).unwrap();
/// assert_eq!(d, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
/// ```
pub fn dot<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<f64>
where
    T: Num + Copy + Send + Sync + ToPrimitive + Default + SimdElement + 'static,
{
    if src1.data.len() != src2.data.len() {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same number of elements".to_string(),
        ));
    }

    #[cfg(feature = "simd")]
    {
        // Only f32/f64 have simd_dot; others use the scalar fallback.
        if (std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
            || std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>())
            && <T as SimdElement>::has_simd()
        {
            if let Some(result) = <T>::simd_dot(&src1.data, &src2.data) {
                return Ok(result);
            }
        }
    }

    #[cfg(feature = "parallel")]
    {
        let sum: f64 = src1
            .data
            .par_iter()
            .zip(src2.data.par_iter())
            .map(|(&v1, &v2)| v1.to_f64().unwrap_or(0.0) * v2.to_f64().unwrap_or(0.0))
            .sum();
        Ok(sum)
    }

    #[cfg(not(feature = "parallel"))]
    {
        let sum: f64 = src1
            .data
            .iter()
            .zip(src2.data.iter())
            .map(|(&v1, &v2)| v1.to_f64().unwrap_or(0.0) * v2.to_f64().unwrap_or(0.0))
            .sum();
        Ok(sum)
    }
}

/// Computes the 3D cross product of two vectors.
///
/// The function computes the cross product of two 3-element vectors.
///
/// # Arguments
///
/// * `src1` - First 3-element vector.
/// * `src2` - Second 3-element vector.
///
/// # Example
///
/// ```rust
/// use purecv::core::{Matrix, cross};
///
/// let v1 = Matrix::from_vec(1, 3, 1, vec![1.0, 0.0, 0.0]);
/// let v2 = Matrix::from_vec(1, 3, 1, vec![0.0, 1.0, 0.0]);
/// let v3 = cross(&v1, &v2).unwrap();
/// // v3 should be [0.0, 0.0, 1.0]
/// ```
pub fn cross<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + Default + 'static,
{
    let len1 = src1.rows * src1.cols * src1.channels;
    let len2 = src2.rows * src2.cols * src2.channels;

    if len1 != 3 || len2 != 3 {
        return Err(PureCvError::InvalidDimensions(
            "Cross product requires 3-element vectors".to_string(),
        ));
    }

    let min_rows_cols = if src1.rows == 3 {
        (3, 1)
    } else if src1.cols == 3 {
        (1, 3)
    } else {
        (1, 1)
    };

    let v1 = [src1.data[0], src1.data[1], src1.data[2]];
    let v2 = [src2.data[0], src2.data[1], src2.data[2]];

    let mut dst = Matrix::<T>::new(min_rows_cols.0, min_rows_cols.1, src1.channels);
    dst.data[0] = v1[1] * v2[2] - v1[2] * v2[1];
    dst.data[1] = v1[2] * v2[0] - v1[0] * v2[2];
    dst.data[2] = v1[0] * v2[1] - v1[1] * v2[0];

    Ok(dst)
}

/// Returns the sum of diagonal elements of a matrix.
///
/// The function returns the sum of diagonal elements of a matrix (the trace).
///
/// # Arguments
///
/// * `src` - Input matrix.
///
/// # Example
///
/// ```rust
/// use purecv::core::{Matrix, trace};
///
/// let mat = Matrix::from_vec(2, 2, 1, vec![1.0, 2.0, 3.0, 4.0]);
/// let t = trace(&mat);
/// assert_eq!(t.v[0], 5.0); // 1 + 4 = 5
/// ```
pub fn trace<T>(src: &Matrix<T>) -> Scalar<f64>
where
    T: Num + Copy + Send + Sync + ToPrimitive + Default + 'static,
{
    let n = std::cmp::min(src.rows, src.cols);
    let channels = src.channels;
    let cols = src.cols;
    let mut sum = [0.0; 4];

    for i in 0..n {
        let base_idx = (i * cols + i) * channels;
        for (c, s) in sum.iter_mut().enumerate().take(channels) {
            *s += src.data[base_idx + c].to_f64().unwrap_or(0.0);
        }
    }

    Scalar { v: sum }
}

/// Returns the determinant of a square matrix.
///
/// The function returns the determinant of a square single-channel matrix.
/// For matrices larger than 3x3, LU decomposition is used.
///
/// # Arguments
///
/// * `src` - Input square single-channel matrix.
///
/// # Example
///
/// ```rust
/// use purecv::core::{Matrix, determinant};
///
/// let mat = Matrix::from_vec(2, 2, 1, vec![1.0, 2.0, 3.0, 4.0]);
/// let det = determinant(&mat);
/// assert_eq!(det, -2.0); // 1*4 - 2*3 = -2
/// ```
pub fn determinant<T>(src: &Matrix<T>) -> f64
where
    T: Num + Copy + Send + Sync + ToPrimitive + Default + 'static,
{
    if src.rows != src.cols || src.channels != 1 {
        return 0.0;
    }

    let n = src.rows;
    if n == 0 {
        return 0.0;
    }

    match n {
        1 => src.data[0].to_f64().unwrap_or(0.0),
        2 => {
            let m = &src.data;
            let a = m[0].to_f64().unwrap_or(0.0);
            let b = m[1].to_f64().unwrap_or(0.0);
            let c = m[2].to_f64().unwrap_or(0.0);
            let d = m[3].to_f64().unwrap_or(0.0);
            a * d - b * c
        }
        3 => {
            let m = &src.data;
            let a11 = m[0].to_f64().unwrap_or(0.0);
            let a12 = m[1].to_f64().unwrap_or(0.0);
            let a13 = m[2].to_f64().unwrap_or(0.0);
            let a21 = m[3].to_f64().unwrap_or(0.0);
            let a22 = m[4].to_f64().unwrap_or(0.0);
            let a23 = m[5].to_f64().unwrap_or(0.0);
            let a31 = m[6].to_f64().unwrap_or(0.0);
            let a32 = m[7].to_f64().unwrap_or(0.0);
            let a33 = m[8].to_f64().unwrap_or(0.0);

            a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31)
                + a13 * (a21 * a32 - a22 * a31)
        }
        _ => {
            // LU decomposition for n > 3
            let mut lu = Vec::with_capacity(n * n);
            for val in &src.data {
                lu.push(val.to_f64().unwrap_or(0.0));
            }

            let mut det = 1.0;
            for i in 0..n {
                // Find pivot
                let mut pivot = i;
                for j in (i + 1)..n {
                    if lu[j * n + i].abs() > lu[pivot * n + i].abs() {
                        pivot = j;
                    }
                }

                if lu[pivot * n + i].abs() < 1e-12 {
                    return 0.0;
                }

                if pivot != i {
                    for j in i..n {
                        lu.swap(i * n + j, pivot * n + j);
                    }
                    det = -det;
                }

                let p = lu[i * n + i];
                det *= p;

                for j in (i + 1)..n {
                    let factor = lu[j * n + i] / p;
                    for k in (i + 1)..n {
                        lu[j * n + k] -= factor * lu[i * n + k];
                    }
                }
            }
            det
        }
    }
}

/// Decomposition types for solve and invert.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DecompTypes {
    /// Gaussian elimination with optimal pivot element chosen.
    DECOMP_LU = 0,
    /// Singular value decomposition (SVD) method.
    DECOMP_SVD = 1,
    /// Eigenvalue decomposition method.
    DECOMP_EIG = 2,
    /// Cholesky decomposition; the matrix must be symmetrical and positively defined.
    DECOMP_CHOLESKY = 3,
    /// QR decomposition.
    DECOMP_QR = 4,
    /// While the other flags are mutually exclusive, this one can be combined with any of them.
    /// It means that the normal equations are solved.
    DECOMP_NORMAL = 16,
}

/// Finds the inverse of a matrix.
///
/// The function inverts the matrix `src` and stores the result in `dst`.
///
/// # Arguments
///
/// * `src` - Input square single-channel matrix.
/// * `dst` - Output matrix of the same size and type as `src`.
/// * `flags` - Inversion method (currently only DECOMP_LU is supported).
///
/// # Returns
///
/// Returns the determinant of `src`.
///
/// # Example
///
/// ```rust
/// use purecv::core::{Matrix, invert, DecompTypes};
///
/// let a = Matrix::from_vec(2, 2, 1, vec![4.0, 7.0, 2.0, 6.0]);
/// let mut inv_a = Matrix::<f64>::new(2, 2, 1);
/// let det = invert(&a, &mut inv_a, DecompTypes::DECOMP_LU).unwrap();
/// ```
pub fn invert<T>(src: &Matrix<T>, dst: &mut Matrix<f64>, flags: DecompTypes) -> Result<f64>
where
    T: Num + Copy + Send + Sync + ToPrimitive + Default + 'static,
{
    if src.rows != src.cols || src.channels != 1 {
        return Err(PureCvError::InvalidDimensions(
            "Inverse only supports single-channel square matrices".to_string(),
        ));
    }

    let n = src.rows;
    let mut identity = Matrix::<f64>::new(n, n, 1);
    set_identity(&mut identity, Scalar::new(1.0, 0.0, 0.0, 0.0));

    if solve(src, &identity, dst, flags)? {
        Ok(determinant(src))
    } else {
        dst.data.fill(0.0);
        Ok(0.0)
    }
}

/// Solves a linear system or least-squares problem.
///
/// The function solves the linear system `src1 * dst = src2`.
///
/// # Arguments
///
/// * `src1` - Input matrix A.
/// * `src2` - Input matrix B.
/// * `dst` - Output matrix X.
/// * `flags` - Solver method (currently only DECOMP_LU is supported).
///
/// # Returns
///
/// Returns `true` if the system was solved successfully.
///
/// # Example
///
/// ```rust
/// use purecv::core::{Matrix, solve, DecompTypes};
///
/// let a = Matrix::from_vec(2, 2, 1, vec![1.0, 1.0, 1.0, -1.0]);
/// let b = Matrix::from_vec(2, 1, 1, vec![2.0, 0.0]);
/// let mut x = Matrix::<f64>::new(2, 1, 1);
/// solve(&a, &b, &mut x, DecompTypes::DECOMP_LU).unwrap();
/// // x should be [1.0, 1.0] (x+y=2, x-y=0)
/// ```
pub fn solve<T, S>(
    src1: &Matrix<T>,
    src2: &Matrix<S>,
    dst: &mut Matrix<f64>,
    flags: DecompTypes,
) -> Result<bool>
where
    T: Num + Copy + Send + Sync + ToPrimitive + Default + 'static,
    S: Num + Copy + Send + Sync + ToPrimitive + Default + 'static,
{
    if src1.rows != src1.cols || src1.channels != 1 || src2.channels != 1 || src1.rows != src2.rows
    {
        return Err(PureCvError::InvalidDimensions(
            "Linear system solver requires compatible single-channel matrices".to_string(),
        ));
    }

    if flags != DecompTypes::DECOMP_LU {
        return Err(PureCvError::NotImplemented(
            "Only DECOMP_LU is currently supported".to_string(),
        ));
    }

    let n = src1.rows;
    let m = src2.cols;

    // Convert to f64 and copy to working buffers
    let mut a = Vec::with_capacity(n * n);
    for val in &src1.data {
        a.push(val.to_f64().unwrap_or(0.0));
    }

    let mut b = Vec::with_capacity(n * m);
    for val in &src2.data {
        b.push(val.to_f64().unwrap_or(0.0));
    }

    // LU decomposition with partial pivoting
    let mut p = (0..n).collect::<Vec<usize>>();
    for i in 0..n {
        let mut max_abs = 0.0;
        let mut pivot = i;
        for j in i..n {
            let abs_val = a[j * n + i].abs();
            if abs_val > max_abs {
                max_abs = abs_val;
                pivot = j;
            }
        }

        if max_abs < 1e-12 {
            return Ok(false); // Singular matrix
        }

        if pivot != i {
            for j in 0..n {
                a.swap(i * n + j, pivot * n + j);
            }
            p.swap(i, pivot);
        }

        for j in (i + 1)..n {
            a[j * n + i] /= a[i * n + i];
            for k in (i + 1)..n {
                a[j * n + k] -= a[j * n + i] * a[i * n + k];
            }
        }
    }

    // Forward and backward substitution for each column of B
    dst.rows = n;
    dst.cols = m;
    dst.channels = 1;
    dst.data.resize(n * m, 0.0);

    for col in 0..m {
        // Forward substitution (LY = PB)
        let mut y = vec![0.0; n];
        for i in 0..n {
            y[i] = b[p[i] * m + col];
            for j in 0..i {
                y[i] -= a[i * n + j] * y[j];
            }
        }

        // Backward substitution (UX = Y)
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = y[i];
            for j in (i + 1)..n {
                x[i] -= a[i * n + j] * x[j];
            }
            x[i] /= a[i * n + i];
        }

        for (i, val) in x.iter().enumerate().take(n) {
            dst.data[i * m + col] = *val;
        }
    }

    Ok(true)
}

/// Calculates the magnitude of 2D vectors.
///
/// The function calculates the magnitude of 2D vectors formed
/// from the corresponding elements of x and y arrays:
/// `dst(I) = sqrt(x(I)^2 + y(I)^2)`
///
/// @sa cartToPolar, polarToCart, phase, sqrt
pub fn magnitude<T>(x: &Matrix<T>, y: &Matrix<T>, dst: &mut Matrix<T>) -> Result<()>
where
    T: DataType
        + ToPrimitive
        + FromPrimitive
        + Default
        + Copy
        + Send
        + Sync
        + SimdElement
        + 'static,
{
    if !x.dims_match(y) {
        return Err(PureCvError::InvalidDimensions(
            "x and y must have the same dimensions".to_string(),
        ));
    }

    dst.create(x.rows, x.cols, x.channels);

    #[cfg(feature = "simd")]
    {
        // Only f32/f64 have simd_magnitude; others use the scalar fallback.
        if (std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
            || std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>())
            && <T as SimdElement>::has_simd()
        {
            <T>::simd_magnitude(&mut dst.data, &x.data, &y.data);
            return Ok(());
        }
    }

    #[cfg(feature = "parallel")]
    {
        dst.data
            .par_iter_mut()
            .zip(x.data.par_iter())
            .zip(y.data.par_iter())
            .for_each(|((d, &xv), &yv)| {
                let xf = xv.to_f64().unwrap_or(0.0);
                let yf = yv.to_f64().unwrap_or(0.0);
                *d = T::from_f64((xf * xf + yf * yf).sqrt()).unwrap_or_default();
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        dst.data
            .iter_mut()
            .zip(x.data.iter())
            .zip(y.data.iter())
            .for_each(|((d, &xv), &yv)| {
                let xf = xv.to_f64().unwrap_or(0.0);
                let yf = yv.to_f64().unwrap_or(0.0);
                *d = T::from_f64((xf * xf + yf * yf).sqrt()).unwrap_or_default();
            });
    }

    Ok(())
}

/// Calculates the rotation angle of 2D vectors.
///
/// The function calculates the rotation angle of each 2D vector that
/// is formed from the corresponding elements of x and y:
/// `angle(I) = atan2(y(I), x(I))`
///
/// When `angle_in_degrees` is true, the output angles are measured in
/// degrees, otherwise in radians.
///
/// @sa cartToPolar, magnitude
pub fn phase<T>(
    x: &Matrix<T>,
    y: &Matrix<T>,
    angle: &mut Matrix<T>,
    angle_in_degrees: bool,
) -> Result<()>
where
    T: DataType + ToPrimitive + FromPrimitive + Default + Copy + Send + Sync + 'static,
{
    if !x.dims_match(y) {
        return Err(PureCvError::InvalidDimensions(
            "x and y must have the same dimensions".to_string(),
        ));
    }

    angle.create(x.rows, x.cols, x.channels);

    #[cfg(feature = "parallel")]
    {
        angle
            .data
            .par_iter_mut()
            .zip(x.data.par_iter())
            .zip(y.data.par_iter())
            .for_each(|((d, &xv), &yv)| {
                let xf = xv.to_f64().unwrap_or(0.0);
                let yf = yv.to_f64().unwrap_or(0.0);
                let mut a = yf.atan2(xf);
                if a < 0.0 {
                    a += 2.0 * std::f64::consts::PI;
                }
                if angle_in_degrees {
                    a = a.to_degrees();
                }
                *d = T::from_f64(a).unwrap_or_default();
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        angle
            .data
            .iter_mut()
            .zip(x.data.iter())
            .zip(y.data.iter())
            .for_each(|((d, &xv), &yv)| {
                let xf = xv.to_f64().unwrap_or(0.0);
                let yf = yv.to_f64().unwrap_or(0.0);
                let mut a = yf.atan2(xf);
                if a < 0.0 {
                    a += 2.0 * std::f64::consts::PI;
                }
                if angle_in_degrees {
                    a = a.to_degrees();
                }
                *d = T::from_f64(a).unwrap_or_default();
            });
    }

    Ok(())
}

/// Calculates the magnitude and angle of 2D vectors.
///
/// The function calculates either the magnitude, angle, or both
/// for every 2D vector (x(I), y(I)):
/// `magnitude(I) = sqrt(x(I)^2 + y(I)^2)`
/// `angle(I) = atan2(y(I), x(I))`
///
/// When `angle_in_degrees` is true, the output angles are measured in
/// degrees, otherwise in radians.
///
/// @sa Sobel, Scharr
pub fn cart_to_polar<T>(
    x: &Matrix<T>,
    y: &Matrix<T>,
    mag: &mut Matrix<T>,
    ang: &mut Matrix<T>,
    angle_in_degrees: bool,
) -> Result<()>
where
    T: DataType + ToPrimitive + FromPrimitive + Default + Copy + Send + Sync + 'static,
{
    if !x.dims_match(y) {
        return Err(PureCvError::InvalidDimensions(
            "x and y must have the same dimensions".to_string(),
        ));
    }

    mag.create(x.rows, x.cols, x.channels);
    ang.create(x.rows, x.cols, x.channels);

    #[cfg(feature = "parallel")]
    {
        mag.data
            .par_iter_mut()
            .zip(ang.data.par_iter_mut())
            .zip(x.data.par_iter())
            .zip(y.data.par_iter())
            .for_each(|(((m, a), &xv), &yv)| {
                let xf = xv.to_f64().unwrap_or(0.0);
                let yf = yv.to_f64().unwrap_or(0.0);
                *m = T::from_f64((xf * xf + yf * yf).sqrt()).unwrap_or_default();
                let mut angle = yf.atan2(xf);
                if angle < 0.0 {
                    angle += 2.0 * std::f64::consts::PI;
                }
                if angle_in_degrees {
                    angle = angle.to_degrees();
                }
                *a = T::from_f64(angle).unwrap_or_default();
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        mag.data
            .iter_mut()
            .zip(ang.data.iter_mut())
            .zip(x.data.iter())
            .zip(y.data.iter())
            .for_each(|(((m, a), &xv), &yv)| {
                let xf = xv.to_f64().unwrap_or(0.0);
                let yf = yv.to_f64().unwrap_or(0.0);
                *m = T::from_f64((xf * xf + yf * yf).sqrt()).unwrap_or_default();
                let mut angle = yf.atan2(xf);
                if angle < 0.0 {
                    angle += 2.0 * std::f64::consts::PI;
                }
                if angle_in_degrees {
                    angle = angle.to_degrees();
                }
                *a = T::from_f64(angle).unwrap_or_default();
            });
    }

    Ok(())
}

/// Calculates x and y coordinates of 2D vectors from their magnitude and angle.
///
/// The function calculates the Cartesian coordinates of each 2D vector
/// represented by the corresponding elements of magnitude and angle:
/// `x(I) = magnitude(I) * cos(angle(I))`
/// `y(I) = magnitude(I) * sin(angle(I))`
///
/// When `angle_in_degrees` is true, the input angles are measured in
/// degrees, otherwise in radians.
///
/// @sa cartToPolar, magnitude, phase
pub fn polar_to_cart<T>(
    mag: &Matrix<T>,
    ang: &Matrix<T>,
    x: &mut Matrix<T>,
    y: &mut Matrix<T>,
    angle_in_degrees: bool,
) -> Result<()>
where
    T: DataType + ToPrimitive + FromPrimitive + Default + Copy + Send + Sync + 'static,
{
    if !mag.dims_match(ang) {
        return Err(PureCvError::InvalidDimensions(
            "magnitude and angle must have the same dimensions".to_string(),
        ));
    }

    x.create(mag.rows, mag.cols, mag.channels);
    y.create(mag.rows, mag.cols, mag.channels);

    #[cfg(feature = "parallel")]
    {
        x.data
            .par_iter_mut()
            .zip(y.data.par_iter_mut())
            .zip(mag.data.par_iter())
            .zip(ang.data.par_iter())
            .for_each(|(((xv, yv), &mv), &av)| {
                let mf = mv.to_f64().unwrap_or(0.0);
                let mut af = av.to_f64().unwrap_or(0.0);
                if angle_in_degrees {
                    af = af.to_radians();
                }
                *xv = T::from_f64(mf * af.cos()).unwrap_or_default();
                *yv = T::from_f64(mf * af.sin()).unwrap_or_default();
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        x.data
            .iter_mut()
            .zip(y.data.iter_mut())
            .zip(mag.data.iter())
            .zip(ang.data.iter())
            .for_each(|(((xv, yv), &mv), &av)| {
                let mf = mv.to_f64().unwrap_or(0.0);
                let mut af = av.to_f64().unwrap_or(0.0);
                if angle_in_degrees {
                    af = af.to_radians();
                }
                *xv = T::from_f64(mf * af.cos()).unwrap_or_default();
                *yv = T::from_f64(mf * af.sin()).unwrap_or_default();
            });
    }

    Ok(())
}

/// Performs the matrix transformation of every array element.
///
/// Each element of `src` is treated as an `scn`-element vector, where
/// `scn` is the number of channels. The transformation matrix `m` is
/// applied to each vector independently:
///
/// `dst(I) = m * src(I)`
///
/// When `m` has size `dcn × (scn + 1)`, the extra column is treated as a
/// translation vector (affine transform).
///
/// Mirrors OpenCV's `cv::transform`.
///
/// # Arguments
/// * `src` - Input matrix with `scn` channels.
/// * `dst` - Output matrix; will be (re)allocated with `dcn` channels.
/// * `m`   - Transformation matrix of size `dcn × scn` or `dcn × (scn + 1)`,
///   single-channel.
///
/// # Errors
/// Returns `PureCvError::InvalidDimensions` if `m` dimensions are
/// incompatible with the source channel count.
#[allow(clippy::needless_range_loop)]
pub fn transform<T>(src: &Matrix<T>, dst: &mut Matrix<T>, m: &Matrix<f64>) -> Result<()>
where
    T: Default + Clone + Copy + FromPrimitive + ToPrimitive + Send + Sync + 'static,
{
    if m.channels != 1 {
        return Err(PureCvError::InvalidDimensions(
            "transformation matrix must be single-channel".into(),
        ));
    }

    let scn = src.channels;
    let dcn = m.rows;
    let m_cols = m.cols;

    // m must be dcn × scn  OR  dcn × (scn + 1)  (affine)
    if m_cols != scn && m_cols != scn + 1 {
        return Err(PureCvError::InvalidDimensions(format!(
            "transformation matrix columns ({}) must equal src channels ({}) or src channels + 1 ({})",
            m_cols, scn, scn + 1
        )));
    }

    let affine = m_cols == scn + 1;

    dst.create(src.rows, src.cols, dcn);

    #[cfg(feature = "parallel")]
    {
        dst.data
            .par_chunks_exact_mut(dcn)
            .zip(src.data.par_chunks_exact(scn))
            .for_each(|(d_chunk, s_chunk)| {
                for r in 0..dcn {
                    let row_base = r * m_cols;
                    let mut val = 0.0;
                    for c in 0..scn {
                        val += m.data[row_base + c] * s_chunk[c].to_f64().unwrap_or(0.0);
                    }
                    if affine {
                        val += m.data[row_base + scn];
                    }
                    d_chunk[r] = T::from_f64(val).unwrap_or_default();
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let pixel_count = src.rows * src.cols;
        for p in 0..pixel_count {
            let s_off = p * scn;
            let d_off = p * dcn;
            for r in 0..dcn {
                let row_base = r * m_cols;
                let mut val = 0.0;
                for c in 0..scn {
                    val += m.data[row_base + c] * src.data[s_off + c].to_f64().unwrap_or(0.0);
                }
                if affine {
                    val += m.data[row_base + scn];
                }
                dst.data[d_off + r] = T::from_f64(val).unwrap_or_default();
            }
        }
    }

    Ok(())
}

/// Performs the perspective matrix transformation of vectors.
///
/// Every element of `src` (an `scn`-channel vector, where `scn` is 2 or 3)
/// is extended with a `1`, multiplied by the `(scn + 1) × (scn + 1)` matrix
/// `m`, and then the result is divided by its last component (perspective
/// divide).
///
/// Mirrors OpenCV's `cv::perspectiveTransform`.
///
/// # Arguments
/// * `src` - Input matrix with 2 or 3 channels.
/// * `dst` - Output matrix; will be (re)allocated with the same number of
///   channels as `src`.
/// * `m`   - `(scn + 1) × (scn + 1)` single-channel transformation matrix
///   (3×3 for 2-channel input, 4×4 for 3-channel input).
///
/// # Errors
/// Returns `PureCvError::InvalidDimensions` if channel count is not 2 or 3,
/// or if `m` has the wrong size.
pub fn perspective_transform<T>(src: &Matrix<T>, dst: &mut Matrix<T>, m: &Matrix<f64>) -> Result<()>
where
    T: Default + Clone + Copy + FromPrimitive + ToPrimitive + Send + Sync + 'static,
{
    let scn = src.channels;

    if scn != 2 && scn != 3 {
        return Err(PureCvError::InvalidDimensions(
            "perspectiveTransform requires 2- or 3-channel input".into(),
        ));
    }

    if m.channels != 1 {
        return Err(PureCvError::InvalidDimensions(
            "transformation matrix must be single-channel".into(),
        ));
    }

    let n = scn + 1; // homogeneous dimension
    if m.rows != n || m.cols != n {
        return Err(PureCvError::InvalidDimensions(format!(
            "transformation matrix must be {}x{} for {}-channel input, got {}x{}",
            n, n, scn, m.rows, m.cols
        )));
    }

    dst.create(src.rows, src.cols, scn);

    #[cfg(feature = "parallel")]
    {
        dst.data
            .par_chunks_exact_mut(scn)
            .zip(src.data.par_chunks_exact(scn))
            .for_each(|(d_chunk, s_chunk)| {
                perspective_transform_pixel(s_chunk, d_chunk, &m.data, scn, n);
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let pixel_count = src.rows * src.cols;
        for p in 0..pixel_count {
            let s_off = p * scn;
            let d_off = p * scn;
            perspective_transform_pixel(
                &src.data[s_off..s_off + scn],
                &mut dst.data[d_off..d_off + scn],
                &m.data,
                scn,
                n,
            );
        }
    }

    Ok(())
}

/// Applies perspective transformation to a single pixel vector.
#[inline]
#[allow(clippy::needless_range_loop)]
fn perspective_transform_pixel<T>(src: &[T], dst: &mut [T], m: &[f64], scn: usize, n: usize)
where
    T: Copy + FromPrimitive + ToPrimitive + Default,
{
    // Build the homogeneous result vector (length n).
    // result[r] = sum(m[r][c] * v[c], c=0..n)  where v = [src..., 1]
    let mut result = [0.0f64; 4]; // max n = 4
    for r in 0..n {
        let row_base = r * n;
        let mut val = 0.0;
        for c in 0..scn {
            val += m[row_base + c] * src[c].to_f64().unwrap_or(0.0);
        }
        // Last element of the homogeneous source vector is 1.
        val += m[row_base + scn];
        result[r] = val;
    }

    // Perspective divide: divide by the last component.
    let w = result[scn];
    let w_inv = if w.abs() > f64::EPSILON { 1.0 / w } else { 0.0 };

    for c in 0..scn {
        dst[c] = T::from_f64(result[c] * w_inv).unwrap_or_default();
    }
}

// ---------------------------------------------------------------------------
//  solvePoly — find polynomial roots (Durand–Kerner / Aberth companion)
// ---------------------------------------------------------------------------

/// Finds the real and complex roots of a polynomial equation.
///
/// The function `solve_poly` finds the real and complex roots of a polynomial
/// equation:
///
/// `coeffs[n]*x^n + coeffs[n-1]*x^(n-1) + ... + coeffs[1]*x + coeffs[0] = 0`
///
/// # Arguments
/// * `coeffs` - Array of polynomial coefficients (single-channel, N+1 elements).
///   The leading coefficient (highest degree) is `coeffs[n]`.
/// * `roots`  - Output (N×1) two-channel matrix where each element stores (real, imag).
/// * `max_iters` - Maximum number of iterations. If 0, a default of 300 is used.
///
/// # Returns
/// The maximum residual magnitude across all roots.
///
/// Mirrors OpenCV's `cv::solvePoly`.
pub fn solve_poly(coeffs: &Matrix<f64>, roots: &mut Matrix<f64>, max_iters: i32) -> Result<f64> {
    // Flatten coefficients into a slice
    let total = coeffs.rows * coeffs.cols * coeffs.channels;
    if total < 2 {
        return Err(PureCvError::InvalidInput(
            "solvePoly requires at least 2 coefficients (degree >= 1)".into(),
        ));
    }
    if coeffs.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "solvePoly requires single-channel coefficient matrix".into(),
        ));
    }

    let c = &coeffs.data;
    let n = total - 1; // polynomial degree

    // Normalise so that c[n] == 1 (monic)
    let lead = c[n];
    if lead.abs() < f64::EPSILON {
        return Err(PureCvError::InvalidInput(
            "Leading coefficient is zero".into(),
        ));
    }
    let a: Vec<f64> = c.iter().map(|&v| v / lead).collect();

    let iters = if max_iters <= 0 {
        300
    } else {
        max_iters as usize
    };

    // Durand-Kerner initial guesses: z_k = 0.4 + 0.9i)^k
    let mut zr = Vec::with_capacity(n);
    let mut zi = Vec::with_capacity(n);
    {
        let mut wr = 1.0f64;
        let mut wi = 0.0f64;
        let br = 0.4f64;
        let bi = 0.9f64;
        for _ in 0..n {
            zr.push(wr);
            zi.push(wi);
            let new_wr = wr * br - wi * bi;
            let new_wi = wr * bi + wi * br;
            wr = new_wr;
            wi = new_wi;
        }
    }

    // Durand-Kerner iteration
    for _ in 0..iters {
        let mut max_delta = 0.0f64;
        for k in 0..n {
            // Evaluate polynomial at z_k  (Horner)
            let mut pr = a[n]; // = 1.0 (monic)
            let mut pi = 0.0f64;
            for j in (0..n).rev() {
                // (pr + i*pi) * (zr[k] + i*zi[k]) + a[j]
                let new_pr = pr * zr[k] - pi * zi[k] + a[j];
                let new_pi = pr * zi[k] + pi * zr[k];
                pr = new_pr;
                pi = new_pi;
            }

            // Compute denominator: product_{j!=k} (z_k - z_j)
            let mut dr = 1.0f64;
            let mut di = 0.0f64;
            for j in 0..n {
                if j == k {
                    continue;
                }
                let diff_r = zr[k] - zr[j];
                let diff_i = zi[k] - zi[j];
                let new_dr = dr * diff_r - di * diff_i;
                let new_di = dr * diff_i + di * diff_r;
                dr = new_dr;
                di = new_di;
            }

            // delta = p(z_k) / denom
            let denom = dr * dr + di * di;
            if denom < f64::EPSILON * f64::EPSILON {
                continue;
            }
            let delta_r = (pr * dr + pi * di) / denom;
            let delta_i = (pi * dr - pr * di) / denom;
            zr[k] -= delta_r;
            zi[k] -= delta_i;

            max_delta = max_delta.max((delta_r * delta_r + delta_i * delta_i).sqrt());
        }
        if max_delta < 1e-15 {
            break;
        }
    }

    // Store roots as 2-channel (real, imag)
    roots.rows = n;
    roots.cols = 1;
    roots.channels = 2;
    roots.data = Vec::with_capacity(n * 2);
    let mut max_residual = 0.0f64;
    for k in 0..n {
        // Snap near-zero imaginary parts
        let im = if zi[k].abs() < 1e-12 { 0.0 } else { zi[k] };
        let re = if im == 0.0 && zr[k].abs() < 1e-12 {
            0.0
        } else {
            zr[k]
        };
        roots.data.push(re);
        roots.data.push(im);

        // Residual: |P(z_k)|
        let mut pr = 1.0f64;
        let mut pi_val = 0.0f64;
        for j in (0..n).rev() {
            let new_pr = pr * re - pi_val * im + a[j];
            let new_pi = pr * im + pi_val * re;
            pr = new_pr;
            pi_val = new_pi;
        }
        max_residual = max_residual.max((pr * pr + pi_val * pi_val).sqrt());
    }

    Ok(max_residual)
}

// ---------------------------------------------------------------------------
//  sort / sort_idx
// ---------------------------------------------------------------------------

/// Sorts each matrix row or each matrix column.
///
/// The function sorts each row or column of the input matrix in ascending
/// or descending order. The matrix must be single-channel.
///
/// # Arguments
/// * `src`   - Input single-channel matrix.
/// * `dst`   - Output matrix of the same size and type.
/// * `flags` - Operation flags: a combination of `SORT_EVERY_ROW` /
///   `SORT_EVERY_COLUMN` and `SORT_ASCENDING` / `SORT_DESCENDING`.
///
/// Mirrors OpenCV's `cv::sort`.
pub fn sort<T>(src: &Matrix<T>, dst: &mut Matrix<T>, flags: i32) -> Result<()>
where
    T: Default + Clone + Copy + PartialOrd + Send + Sync + 'static,
{
    if src.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "sort requires single-channel matrix".into(),
        ));
    }
    dst.rows = src.rows;
    dst.cols = src.cols;
    dst.channels = 1;
    dst.data = src.data.clone();

    let by_column = (flags & 1) != 0;
    let descending = (flags & 16) != 0;

    if !by_column {
        // Sort every row
        for r in 0..dst.rows {
            let start = r * dst.cols;
            let end = start + dst.cols;
            let row = &mut dst.data[start..end];
            row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if descending {
                row.reverse();
            }
        }
    } else {
        // Sort every column: extract column, sort, put back
        for c in 0..dst.cols {
            let mut col: Vec<T> = (0..dst.rows).map(|r| dst.data[r * dst.cols + c]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if descending {
                col.reverse();
            }
            for (r, val) in col.iter().enumerate() {
                dst.data[r * dst.cols + c] = *val;
            }
        }
    }
    Ok(())
}

/// Sorts each matrix row or column and returns the indices.
///
/// Similar to `sort`, but instead of rearranging the elements themselves,
/// it stores the indices of sorted elements in the output matrix.
///
/// Mirrors OpenCV's `cv::sortIdx`.
pub fn sort_idx<T>(src: &Matrix<T>, dst: &mut Matrix<i32>, flags: i32) -> Result<()>
where
    T: Default + Clone + Copy + PartialOrd + Send + Sync + 'static,
{
    if src.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "sortIdx requires single-channel matrix".into(),
        ));
    }
    dst.rows = src.rows;
    dst.cols = src.cols;
    dst.channels = 1;
    dst.data = vec![0i32; src.rows * src.cols];

    let by_column = (flags & 1) != 0;
    let descending = (flags & 16) != 0;

    if !by_column {
        for r in 0..src.rows {
            let start = r * src.cols;
            let mut indices: Vec<usize> = (0..src.cols).collect();
            indices.sort_by(|&a, &b| {
                src.data[start + a]
                    .partial_cmp(&src.data[start + b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if descending {
                indices.reverse();
            }
            for (i, &idx) in indices.iter().enumerate() {
                dst.data[start + i] = idx as i32;
            }
        }
    } else {
        for c in 0..src.cols {
            let mut indices: Vec<usize> = (0..src.rows).collect();
            indices.sort_by(|&a, &b| {
                src.data[a * src.cols + c]
                    .partial_cmp(&src.data[b * src.cols + c])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if descending {
                indices.reverse();
            }
            for (r, &idx) in indices.iter().enumerate() {
                dst.data[r * src.cols + c] = idx as i32;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
//  kmeans
// ---------------------------------------------------------------------------

/// Finds centers of clusters and groups input samples around the clusters.
///
/// The function implements the k-means clustering algorithm. It returns the
/// compactness measure (the sum of squared distances from each point to its
/// cluster center).
///
/// # Arguments
/// * `data`        - Floating-point matrix of input samples, one row per sample.
/// * `k`           - Number of clusters.
/// * `best_labels` - Output integer array storing the cluster index for each sample.
/// * `criteria`    - Termination criteria (`TermCriteria`).
/// * `attempts`    - Number of times the algorithm is executed using different
///   initial labellings; the best result (lowest compactness) is kept.
/// * `flags`       - Initialization flags: `KMEANS_RANDOM_CENTERS`,
///   `KMEANS_PP_CENTERS`, or `KMEANS_USE_INITIAL_LABELS`.
/// * `centers`     - Optional output matrix of cluster centers.
///
/// # Returns
/// The compactness of the best clustering.
///
/// Mirrors OpenCV's `cv::kmeans`.
pub fn kmeans(
    data: &Matrix<f32>,
    k: i32,
    best_labels: &mut Matrix<i32>,
    criteria: crate::core::types::TermCriteria,
    attempts: i32,
    flags: i32,
    centers: &mut Option<Matrix<f32>>,
) -> Result<f64> {
    use crate::core::types::{TermType, KMEANS_PP_CENTERS, KMEANS_USE_INITIAL_LABELS};

    let n = data.rows; // number of samples
    let dims = data.cols * data.channels; // dimensionality

    if k <= 0 || (k as usize) > n {
        return Err(PureCvError::InvalidInput(format!(
            "k ({}) must be in [1, {}]",
            k, n
        )));
    }
    let k = k as usize;

    let max_iter = if criteria.max_count > 0 {
        criteria.max_count as usize
    } else {
        100
    };
    let eps = if criteria.epsilon > 0.0 {
        criteria.epsilon
    } else {
        1e-6
    };
    let check_eps = matches!(criteria.type_, TermType::Eps | TermType::Both);
    let check_count = matches!(criteria.type_, TermType::Count | TermType::Both);

    let attempts = if attempts < 1 { 1 } else { attempts as usize };
    let use_initial = (flags & KMEANS_USE_INITIAL_LABELS) != 0;
    let pp_centers = (flags & KMEANS_PP_CENTERS) != 0;

    // Flatten data to f64 for computation
    let samples: Vec<f64> = data.data.iter().map(|&v| v as f64).collect();

    let mut best_compactness = f64::MAX;
    let mut best_lab: Vec<i32> = vec![0; n];
    let mut best_ctr: Vec<f64> = vec![0.0; k * dims];

    // Use the RNG from crate::core::rng
    // We use a simple thread-local seeded approach via the existing PRNG.
    // For each attempt we re-seed to get different initial centers.

    for attempt in 0..attempts {
        // ---- Initialise labels/centers ----
        let mut labels: Vec<i32> = vec![0; n];
        let mut ctr = vec![0.0f64; k * dims]; // k × dims

        if use_initial && attempt == 0 {
            // Use user-supplied labels (first attempt only)
            if best_labels.data.len() == n {
                labels.copy_from_slice(&best_labels.data);
                // Clamp to valid range
                for l in labels.iter_mut() {
                    if *l < 0 || (*l as usize) >= k {
                        *l = 0;
                    }
                }
            }
            // Compute centers from labels
            kmeans_compute_centers(&samples, &labels, &mut ctr, n, dims, k);
        } else if pp_centers {
            kmeans_pp_init(&samples, &mut ctr, n, dims, k, attempt as u64);
            // Assign labels from centers
            kmeans_assign(&samples, &ctr, &mut labels, n, dims, k);
        } else {
            // Random centers: pick k distinct sample indices
            kmeans_random_init(&samples, &mut ctr, n, dims, k, attempt as u64);
            kmeans_assign(&samples, &ctr, &mut labels, n, dims, k);
        }

        // ---- Lloyd iterations ----
        for iter in 0..max_iter {
            let old_ctr = ctr.clone();

            // Update centers
            kmeans_compute_centers(&samples, &labels, &mut ctr, n, dims, k);

            // Handle empty clusters: steal farthest point from largest cluster
            kmeans_handle_empty(&samples, &mut labels, &mut ctr, n, dims, k);

            // Re-assign
            kmeans_assign(&samples, &ctr, &mut labels, n, dims, k);

            // Check convergence (max center movement)
            if check_eps {
                let mut max_shift = 0.0f64;
                for i in 0..k * dims {
                    let d = ctr[i] - old_ctr[i];
                    max_shift = max_shift.max(d * d);
                }
                if max_shift.sqrt() < eps {
                    break;
                }
            }
            if check_count && iter + 1 >= max_iter {
                break;
            }
        }

        // Compute compactness
        let compactness = kmeans_compactness(&samples, &ctr, &labels, n, dims);

        if compactness < best_compactness {
            best_compactness = compactness;
            best_lab.copy_from_slice(&labels);
            best_ctr.copy_from_slice(&ctr);
        }
    }

    // Write outputs
    best_labels.rows = n;
    best_labels.cols = 1;
    best_labels.channels = 1;
    best_labels.data = best_lab;

    if let Some(ref mut c) = centers {
        c.rows = k;
        c.cols = data.cols;
        c.channels = data.channels;
        c.data = best_ctr.iter().map(|&v| v as f32).collect();
    }

    Ok(best_compactness)
}

/// Assign each sample to its nearest center.
fn kmeans_assign(
    samples: &[f64],
    centers: &[f64],
    labels: &mut [i32],
    n: usize,
    dims: usize,
    k: usize,
) {
    for i in 0..n {
        let s = &samples[i * dims..(i + 1) * dims];
        let mut best_dist = f64::MAX;
        let mut best_k = 0usize;
        for j in 0..k {
            let c = &centers[j * dims..(j + 1) * dims];
            let mut dist = 0.0f64;
            for d in 0..dims {
                let diff = s[d] - c[d];
                dist += diff * diff;
            }
            if dist < best_dist {
                best_dist = dist;
                best_k = j;
            }
        }
        labels[i] = best_k as i32;
    }
}

/// Compute cluster centers from current labels.
fn kmeans_compute_centers(
    samples: &[f64],
    labels: &[i32],
    centers: &mut [f64],
    n: usize,
    dims: usize,
    k: usize,
) {
    centers.iter_mut().for_each(|v| *v = 0.0);
    let mut counts = vec![0usize; k];
    for i in 0..n {
        let lbl = labels[i] as usize;
        counts[lbl] += 1;
        let s = &samples[i * dims..(i + 1) * dims];
        let c = &mut centers[lbl * dims..(lbl + 1) * dims];
        for d in 0..dims {
            c[d] += s[d];
        }
    }
    for j in 0..k {
        if counts[j] > 0 {
            let c = &mut centers[j * dims..(j + 1) * dims];
            let cnt = counts[j] as f64;
            for val in c.iter_mut().take(dims) {
                *val /= cnt;
            }
        }
    }
}

/// Handle empty clusters by stealing the farthest point from the largest cluster.
fn kmeans_handle_empty(
    samples: &[f64],
    labels: &mut [i32],
    centers: &mut [f64],
    n: usize,
    dims: usize,
    k: usize,
) {
    let mut counts = vec![0usize; k];
    for l in labels.iter() {
        counts[*l as usize] += 1;
    }

    for j in 0..k {
        if counts[j] == 0 {
            // Find the largest cluster
            let largest = counts
                .iter()
                .enumerate()
                .max_by_key(|&(_, &c)| c)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // Find farthest point in largest cluster from its center
            let mut farthest_idx = 0usize;
            let mut farthest_dist = 0.0f64;
            for i in 0..n {
                if labels[i] as usize != largest {
                    continue;
                }
                let s = &samples[i * dims..(i + 1) * dims];
                let c = &centers[largest * dims..(largest + 1) * dims];
                let mut dist = 0.0f64;
                for d in 0..dims {
                    let diff = s[d] - c[d];
                    dist += diff * diff;
                }
                if dist > farthest_dist {
                    farthest_dist = dist;
                    farthest_idx = i;
                }
            }

            // Move that sample to the empty cluster
            labels[farthest_idx] = j as i32;
            counts[j] += 1;
            counts[largest] -= 1;

            // Copy sample as new center
            let s = &samples[farthest_idx * dims..(farthest_idx + 1) * dims];
            centers[j * dims..(j + 1) * dims].copy_from_slice(s);

            // Recompute largest cluster center
            kmeans_compute_centers_single(samples, labels, centers, n, dims, largest);
        }
    }
}

/// Recompute a single cluster center.
fn kmeans_compute_centers_single(
    samples: &[f64],
    labels: &[i32],
    centers: &mut [f64],
    n: usize,
    dims: usize,
    cluster: usize,
) {
    let c = &mut centers[cluster * dims..(cluster + 1) * dims];
    c.iter_mut().for_each(|v| *v = 0.0);
    let mut cnt = 0usize;
    for i in 0..n {
        if labels[i] as usize == cluster {
            cnt += 1;
            let s = &samples[i * dims..(i + 1) * dims];
            for d in 0..dims {
                c[d] += s[d];
            }
        }
    }
    if cnt > 0 {
        let cnt_f = cnt as f64;
        for v in c.iter_mut() {
            *v /= cnt_f;
        }
    }
}

/// Compute compactness (sum of squared distances to assigned centers).
fn kmeans_compactness(
    samples: &[f64],
    centers: &[f64],
    labels: &[i32],
    n: usize,
    dims: usize,
) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..n {
        let lbl = labels[i] as usize;
        let s = &samples[i * dims..(i + 1) * dims];
        let c = &centers[lbl * dims..(lbl + 1) * dims];
        for d in 0..dims {
            let diff = s[d] - c[d];
            sum += diff * diff;
        }
    }
    sum
}

/// Random center initialization: pick k distinct sample indices.
fn kmeans_random_init(
    samples: &[f64],
    centers: &mut [f64],
    n: usize,
    dims: usize,
    k: usize,
    seed: u64,
) {
    // Simple Fisher-Yates-style selection using a basic LCG
    let mut rng_state: u64 = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = i + (rng_state as usize % (n - i));
        indices.swap(i, j);
        let idx = indices[i];
        centers[i * dims..(i + 1) * dims].copy_from_slice(&samples[idx * dims..(idx + 1) * dims]);
    }
}

/// K-means++ center initialization.
fn kmeans_pp_init(
    samples: &[f64],
    centers: &mut [f64],
    n: usize,
    dims: usize,
    k: usize,
    seed: u64,
) {
    let mut rng_state: u64 = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);

    // Pick first center uniformly at random
    rng_state = rng_state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let first = rng_state as usize % n;
    centers[0..dims].copy_from_slice(&samples[first * dims..(first + 1) * dims]);

    let mut min_dists = vec![f64::MAX; n];

    for c_idx in 1..k {
        // Update min distances to the latest center
        let prev = c_idx - 1;
        let prev_center = &centers[prev * dims..(prev + 1) * dims];
        let mut total_dist = 0.0f64;
        for i in 0..n {
            let s = &samples[i * dims..(i + 1) * dims];
            let mut dist = 0.0f64;
            for d in 0..dims {
                let diff = s[d] - prev_center[d];
                dist += diff * diff;
            }
            if dist < min_dists[i] {
                min_dists[i] = dist;
            }
            total_dist += min_dists[i];
        }

        // Weighted random selection proportional to D²
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let threshold = (rng_state as f64 / u64::MAX as f64) * total_dist;
        let mut cumulative = 0.0f64;
        let mut chosen = 0usize;
        for (i, &d) in min_dists.iter().enumerate().take(n) {
            cumulative += d;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centers[c_idx * dims..(c_idx + 1) * dims]
            .copy_from_slice(&samples[chosen * dims..(chosen + 1) * dims]);
    }
}

/// Performs a look-up table transform of a matrix.
///
/// The source matrix must have `u8` depth. The look-up table must contain
/// exactly 256 entries. If `lut_table` has one channel it is applied to every
/// channel of `src`; otherwise the number of channels must match.
pub fn lut<T>(src: &Matrix<u8>, lut_table: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Copy + Clone + Default + Send + Sync + 'static,
{
    let lut_entries = lut_table.rows * lut_table.cols;
    if lut_entries != 256 {
        return Err(PureCvError::InvalidInput(format!(
            "LUT must have exactly 256 entries, got {}",
            lut_entries
        )));
    }
    let lut_cn = lut_table.channels;
    let src_cn = src.channels;
    if lut_cn != 1 && lut_cn != src_cn {
        return Err(PureCvError::IncompatibleChannels(format!(
            "LUT channels ({}) must be 1 or match source channels ({})",
            lut_cn, src_cn
        )));
    }

    let total = src.rows * src.cols * src_cn;
    let mut dst_data = vec![T::default(); total];
    let lut_data = &lut_table.data;

    if lut_cn == 1 {
        // Broadcast: same LUT for all channels
        #[cfg(feature = "parallel")]
        {
            dst_data
                .par_iter_mut()
                .zip(src.data.par_iter())
                .for_each(|(d, &s)| {
                    *d = lut_data[s as usize];
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for (d, &s) in dst_data.iter_mut().zip(src.data.iter()) {
                *d = lut_data[s as usize];
            }
        }
    } else {
        // Per-channel LUT
        #[cfg(feature = "parallel")]
        {
            dst_data.par_iter_mut().enumerate().for_each(|(i, d)| {
                let ch = i % src_cn;
                let idx = src.data[i] as usize;
                *d = lut_data[idx * lut_cn + ch];
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for (i, d) in dst_data.iter_mut().enumerate() {
                let ch = i % src_cn;
                let idx = src.data[i] as usize;
                *d = lut_data[idx * lut_cn + ch];
            }
        }
    }

    Ok(Matrix {
        rows: src.rows,
        cols: src.cols,
        channels: src_cn,
        data: dst_data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinant() {
        let mut mat = Matrix::<f32>::new(2, 2, 1);
        mat.data = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(determinant(&mat), -2.0);

        let mut mat3 = Matrix::<f32>::new(3, 3, 1);
        mat3.data = vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0];
        // det = 1*(0-24) - 2*(0-20) + 3*(0-5) = -24 + 40 - 15 = 1.0
        assert_eq!(determinant(&mat3), 1.0);
    }

    #[test]
    fn test_solve() {
        let mut a = Matrix::<f32>::new(3, 3, 1);
        a.data = vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0];
        let mut b = Matrix::<f32>::new(3, 1, 1);
        b.data = vec![1.0, 2.0, 3.0];

        let mut x = Matrix::<f64>::new(0, 0, 0);
        assert!(solve(&a, &b, &mut x, DecompTypes::DECOMP_LU).unwrap());

        // Check solution: A*X = B
        // Manual calculation for this system: x=27, y=-22, z=6
        // Output for solving with b as 3x1 matrix with 1 column should be in x.data
        assert!(
            (x.data[0] - 27.0).abs() < 1e-10,
            "x failed: expected 27.0, got {}",
            x.data[0]
        );
        assert!(
            (x.data[1] - (-22.0)).abs() < 1e-10,
            "y failed: expected -22.0, got {}",
            x.data[1]
        );
        assert!(
            (x.data[2] - 6.0).abs() < 1e-10,
            "z failed: expected 6.0, got {}",
            x.data[2]
        );
    }

    #[test]
    fn test_invert() {
        let mut a = Matrix::<f32>::new(2, 2, 1);
        a.data = vec![4.0, 7.0, 2.0, 6.0];
        let mut inv_a = Matrix::<f64>::new(0, 0, 0);
        invert(&a, &mut inv_a, DecompTypes::DECOMP_LU).unwrap();

        // det = 4*6 - 7*2 = 24 - 14 = 10
        // inv = 1/10 * [6, -7; -2, 4] = [0.6, -0.7; -0.2, 0.4]
        assert!((inv_a.data[0] - 0.6).abs() < 1e-10);
        assert!((inv_a.data[1] - (-0.7)).abs() < 1e-10);
        assert!((inv_a.data[2] - (-0.2)).abs() < 1e-10);
        assert!((inv_a.data[3] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_magnitude() {
        let mut x = Matrix::<f64>::new(1, 4, 1);
        let mut y = Matrix::<f64>::new(1, 4, 1);
        x.as_mut_slice().copy_from_slice(&[3.0, 0.0, 1.0, 5.0]);
        y.as_mut_slice().copy_from_slice(&[4.0, 3.0, 1.0, 12.0]);
        let mut mag = Matrix::<f64>::new(0, 0, 0);
        magnitude(&x, &y, &mut mag).unwrap();
        let expected = [5.0, 3.0, 2.0f64.sqrt(), 13.0];
        for (i, &v) in mag.as_slice().iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_phase() {
        let mut x = Matrix::<f64>::new(1, 4, 1);
        let mut y = Matrix::<f64>::new(1, 4, 1);
        x.as_mut_slice().copy_from_slice(&[1.0, 0.0, -1.0, 0.0]);
        y.as_mut_slice().copy_from_slice(&[0.0, 1.0, 0.0, -1.0]);
        let mut ph = Matrix::<f64>::new(0, 0, 0);
        phase(&x, &y, &mut ph, true).unwrap();
        let expected = [0.0, 90.0, 180.0, 270.0];
        for (i, &v) in ph.as_slice().iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cart_polar_roundtrip() {
        let mut x = Matrix::<f64>::new(1, 4, 1);
        let mut y = Matrix::<f64>::new(1, 4, 1);
        x.as_mut_slice().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        y.as_mut_slice().copy_from_slice(&[4.0, 3.0, 2.0, 1.0]);

        let mut mag = Matrix::<f64>::new(0, 0, 0);
        let mut ang = Matrix::<f64>::new(0, 0, 0);
        cart_to_polar(&x, &y, &mut mag, &mut ang, true).unwrap();

        let mut x2 = Matrix::<f64>::new(0, 0, 0);
        let mut y2 = Matrix::<f64>::new(0, 0, 0);
        polar_to_cart(&mag, &ang, &mut x2, &mut y2, true).unwrap();

        for (i, &v) in x.as_slice().iter().enumerate() {
            assert!((v - x2.as_slice()[i]).abs() < 1e-6);
            assert!((y.as_slice()[i] - y2.as_slice()[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_min_max() {
        let mut src1 = Matrix::<f32>::new(1, 4, 1);
        let mut src2 = Matrix::<f32>::new(1, 4, 1);
        src1.data = vec![1.0, 5.0, 3.0, 7.0];
        src2.data = vec![2.0, 4.0, 6.0, 1.0];

        let min_res = min(&src1, &src2).unwrap();
        assert_eq!(min_res.data, vec![1.0, 4.0, 3.0, 1.0]);

        let max_res = max(&src1, &src2).unwrap();
        assert_eq!(max_res.data, vec![2.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_reduce() {
        let mut src = Matrix::<f32>::new(2, 2, 1);
        src.data = vec![1.0, 2.0, 3.0, 4.0];

        // Reduce to row (dim=0)
        let dst_sum = reduce(&src, 0, ReduceTypes::Sum).unwrap();
        assert_eq!(dst_sum.rows, 1);
        assert_eq!(dst_sum.cols, 2);
        assert_eq!(dst_sum.data, vec![4.0, 6.0]);

        let dst_avg = reduce(&src, 0, ReduceTypes::Avg).unwrap();
        assert_eq!(dst_avg.data, vec![2.0, 3.0]);

        let dst_min = reduce(&src, 0, ReduceTypes::Min).unwrap();
        assert_eq!(dst_min.data, vec![1.0, 2.0]);

        let dst_max = reduce(&src, 0, ReduceTypes::Max).unwrap();
        assert_eq!(dst_max.data, vec![3.0, 4.0]);

        // Reduce to column (dim=1)
        let dst_col_sum = reduce(&src, 1, ReduceTypes::Sum).unwrap();
        assert_eq!(dst_col_sum.rows, 2);
        assert_eq!(dst_col_sum.cols, 1);
        assert_eq!(dst_col_sum.data, vec![3.0, 7.0]);
    }
    #[test]
    fn test_in_range() {
        let mut src = Matrix::<u8>::new(1, 3, 3); // 1x3 3-channel
                                                  // Pixel 0: (10, 20, 30) - In range
                                                  // Pixel 1: (5, 20, 30)  - Out of range (ch 0 low)
                                                  // Pixel 2: (100, 100, 100) - Out of range (high)
        src.data = vec![10, 20, 30, 5, 20, 30, 100, 100, 100];

        let mut lower = Matrix::<u8>::new(1, 3, 3);
        lower.data = vec![10, 10, 10, 10, 10, 10, 10, 10, 10];

        let mut upper = Matrix::<u8>::new(1, 3, 3);
        upper.data = vec![50, 50, 50, 50, 50, 50, 50, 50, 50];

        let mut mask = Matrix::<u8>::new(0, 0, 0);
        in_range(&src, &lower, &upper, &mut mask).unwrap();
        assert_eq!(mask.channels, 1);
        assert_eq!(mask.data, vec![255, 0, 0]);
    }

    #[test]
    fn test_in_range_scalar() {
        let mut src = Matrix::<u8>::new(1, 3, 3);
        src.data = vec![10, 20, 30, 5, 20, 30, 100, 100, 100];

        let mut mask = Matrix::<u8>::new(0, 0, 0);
        in_range_scalar(&src, &[10, 10, 10], &[50, 50, 50], &mut mask).unwrap();
        assert_eq!(mask.channels, 1);
        assert_eq!(mask.data, vec![255, 0, 0]);
    }

    #[test]
    fn test_solve_poly_quadratic_real() {
        // x^2 - 1 = 0  → roots {-1, 1}
        // coeffs = [-1, 0, 1]  (c0=-1, c1=0, c2=1)
        let coeffs = Matrix::from_vec(1, 3, 1, vec![-1.0, 0.0, 1.0]);
        let mut roots = Matrix::<f64>::new(0, 0, 0);
        let residual = solve_poly(&coeffs, &mut roots, 300).unwrap();
        assert!(residual < 1e-6, "residual too large: {}", residual);

        assert_eq!(roots.rows, 2);
        assert_eq!(roots.channels, 2);

        // Collect real parts (imag should be ~0)
        let mut reals: Vec<f64> = Vec::new();
        for k in 0..roots.rows {
            assert!(
                roots.data[k * 2 + 1].abs() < 1e-6,
                "unexpected imaginary part"
            );
            reals.push(roots.data[k * 2]);
        }
        reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((reals[0] - (-1.0)).abs() < 1e-6);
        assert!((reals[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_solve_poly_quadratic_complex() {
        // x^2 + 1 = 0  → roots {i, -i}
        let coeffs = Matrix::from_vec(1, 3, 1, vec![1.0, 0.0, 1.0]);
        let mut roots = Matrix::<f64>::new(0, 0, 0);
        solve_poly(&coeffs, &mut roots, 300).unwrap();

        assert_eq!(roots.rows, 2);
        // Both roots should have real part ~0, imag part ~±1
        for k in 0..2 {
            let re = roots.data[k * 2];
            let im = roots.data[k * 2 + 1];
            assert!(re.abs() < 1e-6, "real part should be ~0, got {}", re);
            assert!(
                (im.abs() - 1.0).abs() < 1e-6,
                "imag part should be ~±1, got {}",
                im
            );
        }
    }

    #[test]
    fn test_sort_row_ascending() {
        let src = Matrix::from_vec(2, 3, 1, vec![3.0f32, 1.0, 2.0, 6.0, 4.0, 5.0]);
        let mut dst = Matrix::<f32>::new(0, 0, 0);
        sort(&src, &mut dst, 0).unwrap(); // SORT_EVERY_ROW | SORT_ASCENDING
        assert_eq!(dst.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_sort_row_descending() {
        let src = Matrix::from_vec(2, 3, 1, vec![3.0f32, 1.0, 2.0, 6.0, 4.0, 5.0]);
        let mut dst = Matrix::<f32>::new(0, 0, 0);
        sort(&src, &mut dst, 16).unwrap(); // SORT_EVERY_ROW | SORT_DESCENDING
        assert_eq!(dst.data, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn test_sort_column_ascending() {
        let src = Matrix::from_vec(3, 2, 1, vec![3.0f32, 6.0, 1.0, 4.0, 2.0, 5.0]);
        let mut dst = Matrix::<f32>::new(0, 0, 0);
        sort(&src, &mut dst, 1).unwrap(); // SORT_EVERY_COLUMN | SORT_ASCENDING
        assert_eq!(dst.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_sort_idx_row() {
        let src = Matrix::from_vec(1, 4, 1, vec![30.0f32, 10.0, 40.0, 20.0]);
        let mut dst = Matrix::<i32>::new(0, 0, 0);
        sort_idx(&src, &mut dst, 0).unwrap(); // ascending by row
        assert_eq!(dst.data, vec![1, 3, 0, 2]);
    }

    #[test]
    fn test_kmeans_two_clusters() {
        use crate::core::types::{TermCriteria, TermType, KMEANS_PP_CENTERS};

        // 10 samples: 5 near 0, 5 near 100
        let mut data = Matrix::<f32>::new(10, 1, 1);
        data.data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 96.0, 97.0, 98.0, 99.0, 100.0];

        let mut labels = Matrix::<i32>::new(0, 0, 0);
        let criteria = TermCriteria::new(TermType::Both, 100, 1e-6);
        let mut centers = Some(Matrix::<f32>::new(0, 0, 0));

        let compactness = kmeans(
            &data,
            2,
            &mut labels,
            criteria,
            3,
            KMEANS_PP_CENTERS,
            &mut centers,
        )
        .unwrap();

        assert!(compactness < 100.0, "compactness too high: {}", compactness);
        assert_eq!(labels.data.len(), 10);

        // All first-5 should share one label, all last-5 another
        let label_a = labels.data[0];
        let label_b = labels.data[5];
        assert_ne!(label_a, label_b);
        for i in 0..5 {
            assert_eq!(labels.data[i], label_a);
        }
        for i in 5..10 {
            assert_eq!(labels.data[i], label_b);
        }

        // Centers should be near 3.0 and 98.0
        let c = centers.unwrap();
        let mut c_vals: Vec<f32> = vec![c.data[0], c.data[1]];
        c_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((c_vals[0] - 3.0).abs() < 1.0);
        assert!((c_vals[1] - 98.0).abs() < 1.0);
    }
}
