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
 *  WebARKitLib-rs is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with WebARKitLib-rs.  If not, see <http://www.gnu.org/licenses/>.
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

use crate::core::Matrix;
use crate::core::error::{PureCvError, Result};
use num_traits::{Num, Bounded, ToPrimitive, FromPrimitive};
use std::ops::{BitAnd, BitOr, BitXor, Not};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "simd")]
use pulp::Arch;

/// Internal macro to handle feature-gated loop execution for binary operations.
/// Handles Parallel + SIMD, Parallel only, SIMD only, and Sequential.
macro_rules! binary_op {
    ($dst:expr, $src1:expr, $src2:expr, $t_dst:ty, $t_src:ty, |$d:ident, $s1:ident, $s2:ident| $body:expr) => {
        #[cfg(feature = "parallel")]
        {
            #[cfg(feature = "simd")]
            {
                let arch = Arch::new();
                $dst.data.par_chunks_mut(1024)
                    .zip($src1.data.par_chunks(1024))
                    .zip($src2.data.par_chunks(1024))
                    .for_each(|((d_raw, s1_raw), s2_raw)| {
                        arch.dispatch(|| {
                            for ((d_inner, &s1_inner), &s2_inner) in d_raw.iter_mut().zip(s1_raw).zip(s2_raw) {
                                let $d: &mut $t_dst = d_inner;
                                let $s1: $t_src = s1_inner;
                                let $s2: $t_src = s2_inner;
                                $body
                            }
                        });
                    });
            }
            #[cfg(not(feature = "simd"))]
            {
                $dst.data.par_iter_mut()
                    .zip($src1.data.par_iter())
                    .zip($src2.data.par_iter())
                    .for_each(|((d_raw, &s1_raw), &s2_raw)| {
                        let $d: &mut $t_dst = d_raw;
                        let $s1: $t_src = s1_raw;
                        let $s2: $t_src = s2_raw;
                        $body
                    });
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(feature = "simd")]
            {
                let arch = Arch::new();
                arch.dispatch(|| {
                    for ((d_raw, &s1_raw), &s2_raw) in $dst.data.iter_mut().zip(&$src1.data).zip(&$src2.data) {
                        let $d: &mut $t_dst = d_raw;
                        let $s1: $t_src = s1_raw;
                        let $s2: $t_src = s2_raw;
                        $body
                    }
                });
            }
            #[cfg(not(feature = "simd"))]
            {
                $dst.data.iter_mut()
                    .zip($src1.data.iter())
                    .zip($src2.data.iter())
                    .for_each(|((d_raw, &s1_raw), &s2_raw)| {
                        let $d: &mut $t_dst = d_raw;
                        let $s1: $t_src = s1_raw;
                        let $s2: $t_src = s2_raw;
                        $body
                    });
            }
        }
    };
}

/// Internal macro to handle feature-gated loop execution for unary operations.
macro_rules! unary_op {
    ($dst:expr, $src:expr, $t_dst:ty, $t_src:ty, |$d:ident, $s:ident| $body:expr) => {
        #[cfg(feature = "parallel")]
        {
            #[cfg(feature = "simd")]
            {
                let arch = Arch::new();
                $dst.data.par_chunks_mut(1024)
                    .zip($src.data.par_chunks(1024))
                    .for_each(|(d_raw, s_raw)| {
                        arch.dispatch(|| {
                            for (d_inner, &s_inner) in d_raw.iter_mut().zip(s_raw) {
                                let $d: &mut $t_dst = d_inner;
                                let $s: $t_src = s_inner;
                                $body
                            }
                        });
                    });
            }
            #[cfg(not(feature = "simd"))]
            {
                $dst.data.par_iter_mut()
                    .zip($src.data.par_iter())
                    .for_each(|(d_raw, &s_raw)| {
                        let $d: &mut $t_dst = d_raw;
                        let $s: $t_src = s_raw;
                        $body
                    });
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(feature = "simd")]
            {
                let arch = Arch::new();
                arch.dispatch(|| {
                    for (d_raw, &s_raw) in $dst.data.iter_mut().zip(&$src.data) {
                        let $d: &mut $t_dst = d_raw;
                        let $s: $t_src = s_raw;
                        $body
                    }
                });
            }
            #[cfg(not(feature = "simd"))]
            {
                $dst.data.iter_mut()
                    .zip($src.data.iter())
                    .for_each(|(d_raw, &s_raw)| {
                        let $d: &mut $t_dst = d_raw;
                        let $s: $t_src = s_raw;
                        $body
                    });
            }
        }
    };
}

/// Calculates the per-element sum of two matrices.
///
/// dst = src1 + src2
pub fn add<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + Default + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions("Matrices must have the same dimensions".to_string()));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 + s2);

    Ok(dst)
}

/// Calculates the per-element difference between two matrices.
///
/// dst = src1 - src2
pub fn subtract<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + Default + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions("Matrices must have the same dimensions".to_string()));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 - s2);

    Ok(dst)
}

/// Calculates the per-element absolute difference between two matrices.
///
/// dst = |src1 - src2|
pub fn absdiff<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + Default + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions("Matrices must have the same dimensions".to_string()));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = if s1 > s2 { s1 - s2 } else { s2 - s1 });

    Ok(dst)
}

/// Calculates the per-element product of two matrices.
///
/// dst = src1 * src2
pub fn multiply<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + Default + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions("Matrices must have the same dimensions".to_string()));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 * s2);

    Ok(dst)
}

/// Calculates the per-element quotient of two matrices.
///
/// dst = src1 / src2
pub fn divide<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + Default + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions("Matrices must have the same dimensions".to_string()));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| {
        if !s2.is_zero() {
            *d = s1 / s2;
        } else {
            *d = T::zero();
        }
    });

    Ok(dst)
}

/// Calculates the per-element bit-wise conjunction of two matrices.
///
/// dst = src1 & src2
pub fn bitwise_and<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + BitAnd<Output = T> + Default + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions("Matrices must have the same dimensions".to_string()));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 & s2);

    Ok(dst)
}

/// Calculates the per-element bit-wise disjunction of two matrices.
///
/// dst = src1 | src2
pub fn bitwise_or<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + BitOr<Output = T> + Default + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions("Matrices must have the same dimensions".to_string()));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 | s2);

    Ok(dst)
}

/// Calculates the per-element bit-wise "exclusive or" operation on two matrices.
///
/// dst = src1 ^ src2
pub fn bitwise_xor<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + BitXor<Output = T> + Default + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions("Matrices must have the same dimensions".to_string()));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 ^ s2);

    Ok(dst)
}

/// Inverts every bit of every array element.
///
/// dst = ~src
pub fn bitwise_not<T>(src: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + Not<Output = T> + Default + 'static,
{
    let mut dst = Matrix::<T>::new(src.rows, src.cols, src.channels);

    unary_op!(dst, src, T, T, |d, s| *d = !s);

    Ok(dst)
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
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + ToPrimitive + FromPrimitive + Default + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions("Matrices must have the same dimensions".to_string()));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| {
        let val = s1.to_f64().unwrap_or(0.0) * alpha 
                + s2.to_f64().unwrap_or(0.0) * beta 
                + gamma;
        *d = T::from_f64(val).unwrap_or(T::zero());
    });

    Ok(dst)
}

/// Calculates the square root of every matrix element.
///
/// dst = sqrt(src)
pub fn sqrt<T>(src: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + ToPrimitive + FromPrimitive + Default + 'static,
{
    let mut dst = Matrix::<T>::new(src.rows, src.cols, src.channels);

    unary_op!(dst, src, T, T, |d, s| {
        let val = s.to_f64().unwrap_or(0.0).sqrt();
        *d = T::from_f64(val).unwrap_or(T::zero());
    });

    Ok(dst)
}

/// Calculates the exponent of every matrix element.
///
/// dst = exp(src)
pub fn exp<T>(src: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + ToPrimitive + FromPrimitive + Default + 'static,
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
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + ToPrimitive + FromPrimitive + Default + 'static,
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
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + ToPrimitive + FromPrimitive + Default + 'static,
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
    T: Num + Copy + Send + Sync + ToPrimitive + Default + 'static,
{
    let mut dst = Matrix::<u8>::new(src.rows, src.cols, src.channels);

    unary_op!(dst, src, u8, T, |d, s| {
        let val = (s.to_f64().unwrap_or(0.0) * alpha + beta).abs();
        *d = val.clamp(0.0, 255.0).round() as u8;
    });

    Ok(dst)
}
