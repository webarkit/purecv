/*
 *  matrix.rs
 *  purecv
 *
 *  This file is part of purecv - WebARKit.
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
use crate::core::error::Result;
use crate::core::logging::tags;
use crate::core::saturate::SaturateCast;
use crate::core::types::Scalar;
use crate::cv_bail;
use alloc::{vec, vec::Vec};
use core::any::TypeId;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Converts one element with `saturate_cast` semantics, staying in integer
/// arithmetic when `x` is an integer value so that 64-bit values above 2⁵³
/// survive. `ToPrimitive` cannot tell an integer type from a float, so "is an
/// integer" is decided by value: `to_i64` / `to_u64` truncate, and the result
/// is used only when it converts back to the bit pattern of `x as f64`. That
/// holds for every integer and every integral float, where the integer path
/// gives the same result as the `f64` one, and fails for fractions, NaN,
/// infinities and `-0.0`, which take the `f64` path.
#[inline]
fn cast_element<T: num_traits::ToPrimitive, U: SaturateCast>(x: T) -> U {
    let Some(f) = x.to_f64() else {
        return U::saturate_from_f64(f64::NAN);
    };
    match x.to_i64() {
        Some(i) => {
            if (i as f64).to_bits() == f.to_bits() {
                return U::saturate_from_i64(i);
            }
        }
        None => {
            if let Some(u) = x.to_u64() {
                if (u as f64).to_bits() == f.to_bits() {
                    return U::saturate_from_u64(u);
                }
            }
        }
    }
    U::saturate_from_f64(f)
}

/// `src.map(f).collect()`, on the Rayon pool with the `parallel` feature.
#[cfg(feature = "parallel")]
#[inline]
fn map_elements<T: Copy + Sync, U: Send>(src: &[T], f: impl Fn(T) -> U + Sync) -> Vec<U> {
    src.par_iter().map(|&x| f(x)).collect()
}

/// `src.map(f).collect()`, sequentially: the fallback without the `parallel` feature.
#[cfg(not(feature = "parallel"))]
#[inline]
fn map_elements<T: Copy + Sync, U: Send>(src: &[T], f: impl Fn(T) -> U + Sync) -> Vec<U> {
    src.iter().map(|&x| f(x)).collect()
}

/// `dst[i] = f(src[i])` for every `i` that `selected` accepts, on the Rayon pool with the
/// `parallel` feature.
#[cfg(feature = "parallel")]
#[inline]
fn write_elements<T: Copy + Sync, U: Send>(
    src: &[T],
    dst: &mut [U],
    selected: impl Fn(usize) -> bool + Sync,
    f: impl Fn(T) -> U + Sync,
) {
    dst.par_iter_mut()
        .zip(src.par_iter())
        .enumerate()
        .for_each(|(i, (d, &s))| {
            if selected(i) {
                *d = f(s);
            }
        });
}

/// `dst[i] = f(src[i])` for every `i` that `selected` accepts, sequentially: the fallback
/// without the `parallel` feature.
#[cfg(not(feature = "parallel"))]
#[inline]
fn write_elements<T: Copy + Sync, U: Send>(
    src: &[T],
    dst: &mut [U],
    selected: impl Fn(usize) -> bool + Sync,
    f: impl Fn(T) -> U + Sync,
) {
    for (i, (d, &s)) in dst.iter_mut().zip(src).enumerate() {
        if selected(i) {
            *d = f(s);
        }
    }
}

/// The working precision of `convertTo` with a scale, see [`Matrix::convert_to_scaled`].
#[inline]
fn scales_in_f64<T: 'static, U: SaturateCast>() -> bool {
    let src_bytes = core::mem::size_of::<T>();
    let dst_bytes = core::mem::size_of::<U>();
    dst_bytes == 8
        || (dst_bytes == 4
            && if U::IS_FLOAT {
                src_bytes == 8
            } else {
                src_bytes == 8 || TypeId::of::<T>() == TypeId::of::<i32>()
            })
}

#[inline]
fn scale_in_f64<T: num_traits::ToPrimitive, U: SaturateCast>(x: T, alpha: f64, beta: f64) -> U {
    U::saturate_from_f64(x.to_f64().unwrap_or(f64::NAN) * alpha + beta)
}

#[inline]
fn scale_in_f32<T: num_traits::ToPrimitive, U: SaturateCast>(x: T, alpha: f32, beta: f32) -> U {
    U::saturate_from_f32(x.to_f32().unwrap_or(f32::NAN) * alpha + beta)
}

/// Matrix depth: number of bits per element and its signedness/type.
/// Follows OpenCV's depth conventions (CV_8U, CV_32F, etc.).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Depth {
    CV_8U = 0,
    CV_8S = 1,
    CV_16U = 2,
    CV_16S = 3,
    CV_32S = 4,
    CV_32F = 5,
    CV_64F = 6,
    CV_16F = 7, // float16 if needed
}

impl Depth {
    pub fn is_signed(&self) -> bool {
        !matches!(self, Depth::CV_8U | Depth::CV_16U)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Depth::CV_32F | Depth::CV_64F | Depth::CV_16F)
    }

    pub fn byte_size(&self) -> usize {
        match self {
            Depth::CV_8U | Depth::CV_8S => 1,
            Depth::CV_16U | Depth::CV_16S | Depth::CV_16F => 2,
            Depth::CV_32S | Depth::CV_32F => 4,
            Depth::CV_64F => 8,
        }
    }
}

/// Matrix type: combines Depth and number of channels.
/// Follows OpenCV's type conventions (CV_8UC1, CV_32FC3, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatType(pub i32);

impl MatType {
    pub const fn new(depth: Depth, channels: usize) -> Self {
        Self((depth as i32) + (((channels - 1) as i32) << 3))
    }

    pub fn depth(&self) -> Depth {
        match self.0 & 0x7 {
            0 => Depth::CV_8U,
            1 => Depth::CV_8S,
            2 => Depth::CV_16U,
            3 => Depth::CV_16S,
            4 => Depth::CV_32S,
            5 => Depth::CV_32F,
            6 => Depth::CV_64F,
            7 => Depth::CV_16F,
            _ => unreachable!(),
        }
    }

    pub fn channels(&self) -> usize {
        ((self.0 >> 3) + 1) as usize
    }

    pub fn to_int(&self) -> i32 {
        self.0
    }
}

/// Trait to map Rust types to OpenCV depth.
pub trait DataType {
    fn depth() -> Depth;
}

impl DataType for u8 {
    fn depth() -> Depth {
        Depth::CV_8U
    }
}
impl DataType for i8 {
    fn depth() -> Depth {
        Depth::CV_8S
    }
}
impl DataType for u16 {
    fn depth() -> Depth {
        Depth::CV_16U
    }
}
impl DataType for i16 {
    fn depth() -> Depth {
        Depth::CV_16S
    }
}
impl DataType for i32 {
    fn depth() -> Depth {
        Depth::CV_32S
    }
}
impl DataType for f32 {
    fn depth() -> Depth {
        Depth::CV_32F
    }
}
impl DataType for f64 {
    fn depth() -> Depth {
        Depth::CV_64F
    }
}
// Note: CV_16F would typically map to a half-precision float type like `f16` from a crate.

// OpenCV-style depth constants
pub const CV_8U: Depth = Depth::CV_8U;
pub const CV_8S: Depth = Depth::CV_8S;
pub const CV_16U: Depth = Depth::CV_16U;
pub const CV_16S: Depth = Depth::CV_16S;
pub const CV_32S: Depth = Depth::CV_32S;
pub const CV_32F: Depth = Depth::CV_32F;
pub const CV_64F: Depth = Depth::CV_64F;
pub const CV_16F: Depth = Depth::CV_16F;

// Common OpenCV-style type constants
pub const CV_8UC1: MatType = MatType::new(Depth::CV_8U, 1);
pub const CV_8UC2: MatType = MatType::new(Depth::CV_8U, 2);
pub const CV_8UC3: MatType = MatType::new(Depth::CV_8U, 3);
pub const CV_8UC4: MatType = MatType::new(Depth::CV_8U, 4);

pub const CV_8SC1: MatType = MatType::new(Depth::CV_8S, 1);
pub const CV_8SC2: MatType = MatType::new(Depth::CV_8S, 2);
pub const CV_8SC3: MatType = MatType::new(Depth::CV_8S, 3);
pub const CV_8SC4: MatType = MatType::new(Depth::CV_8S, 4);

pub const CV_16UC1: MatType = MatType::new(Depth::CV_16U, 1);
pub const CV_16UC2: MatType = MatType::new(Depth::CV_16U, 2);
pub const CV_16UC3: MatType = MatType::new(Depth::CV_16U, 3);
pub const CV_16UC4: MatType = MatType::new(Depth::CV_16U, 4);

pub const CV_16SC1: MatType = MatType::new(Depth::CV_16S, 1);
pub const CV_16SC2: MatType = MatType::new(Depth::CV_16S, 2);
pub const CV_16SC3: MatType = MatType::new(Depth::CV_16S, 3);
pub const CV_16SC4: MatType = MatType::new(Depth::CV_16S, 4);

pub const CV_32SC1: MatType = MatType::new(Depth::CV_32S, 1);
pub const CV_32SC2: MatType = MatType::new(Depth::CV_32S, 2);
pub const CV_32SC3: MatType = MatType::new(Depth::CV_32S, 3);
pub const CV_32SC4: MatType = MatType::new(Depth::CV_32S, 4);

pub const CV_32FC1: MatType = MatType::new(Depth::CV_32F, 1);
pub const CV_32FC2: MatType = MatType::new(Depth::CV_32F, 2);
pub const CV_32FC3: MatType = MatType::new(Depth::CV_32F, 3);
pub const CV_32FC4: MatType = MatType::new(Depth::CV_32F, 4);

pub const CV_64FC1: MatType = MatType::new(Depth::CV_64F, 1);
pub const CV_64FC2: MatType = MatType::new(Depth::CV_64F, 2);
pub const CV_64FC3: MatType = MatType::new(Depth::CV_64F, 3);
pub const CV_64FC4: MatType = MatType::new(Depth::CV_64F, 4);

/// A generic, memory-safe 2D matrix optimized for image processing.
/// Uses a contiguous row-major memory layout, making it suitable for
/// SIMD auto-vectorization and WebAssembly (WASM) targets.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub channels: usize,
    /// Contiguous data buffer storing the matrix elements.
    pub data: Vec<T>,
}

impl<T: Default + Clone> Matrix<T> {
    /// Creates a new `Matrix` initialized with the default value of `T`.
    /// E.g., for `u8`, it initializes a black image.
    pub fn new(rows: usize, cols: usize, channels: usize) -> Self {
        let capacity = rows * cols * channels;
        Self {
            rows,
            cols,
            channels,
            data: vec![T::default(); capacity],
        }
    }

    /// Creates a new `Matrix` from a Size
    pub fn from_size<U: Into<usize>>(size: crate::core::Size<U>, channels: usize) -> Self {
        Self::new(size.height.into(), size.width.into(), channels)
    }

    /// Creates a new `Matrix` from an existing `Vec<T>`.
    pub fn from_vec(rows: usize, cols: usize, channels: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), rows * cols * channels, "Data length mismatch");
        Self {
            rows,
            cols,
            channels,
            data,
        }
    }

    /// Checks if this matrix has the same dimensions and channels as another.
    pub fn dims_match<U>(&self, other: &Matrix<U>) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.channels == other.channels
    }

    /// Calculates the 1D flat index for a 2D coordinate and channel.
    /// Marked as `#[inline]` to ensure zero-cost abstraction in loops.
    #[inline(always)]
    pub fn flat_index(&self, row: usize, col: usize, channel: usize) -> usize {
        debug_assert!(
            row < self.rows && col < self.cols && channel < self.channels,
            "Index out of bounds"
        );
        (row * self.cols * self.channels) + (col * self.channels) + channel
    }

    /// Safely retrieves a reference to a specific pixel's channel value.
    #[inline]
    pub fn get(&self, row: usize, col: usize, channel: usize) -> Option<&T> {
        if row >= self.rows || col >= self.cols || channel >= self.channels {
            return None;
        }
        let idx = self.flat_index(row, col, channel);
        self.data.get(idx)
    }

    /// Safely retrieves a reference to a specific pixel's channel value using i32 indices.
    #[inline]
    pub fn at(&self, row: i32, col: i32, channel: usize) -> Option<&T> {
        if row < 0 || col < 0 {
            return None;
        }
        self.get(row as usize, col as usize, channel)
    }

    /// Safely retrieves a mutable reference to a specific pixel's channel value.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize, channel: usize) -> Option<&mut T> {
        if row >= self.rows || col >= self.cols || channel >= self.channels {
            return None;
        }
        let idx = self.flat_index(row, col, channel);
        self.data.get_mut(idx)
    }

    /// Safely retrieves a mutable reference to a specific pixel's channel value using i32 indices.
    #[inline]
    pub fn at_mut(&mut self, row: i32, col: i32, channel: usize) -> Option<&mut T> {
        if row < 0 || col < 0 {
            return None;
        }
        self.get_mut(row as usize, col as usize, channel)
    }

    /// Sets a pixel's channel value using usize indices.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, channel: usize, value: T) {
        if row >= self.rows || col >= self.cols || channel >= self.channels {
            return;
        }
        let idx = self.flat_index(row, col, channel);
        if let Some(p) = self.data.get_mut(idx) {
            *p = value;
        }
    }

    /// Returns the underlying buffer as an immutable slice.
    /// Perfect for Rayon's `par_iter()` or sequential iterators.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns the underlying buffer as a mutable slice.
    /// Ideal for `par_chunks_mut()` when writing algorithms.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns the matrix type (depth and channels).
    pub fn mat_type(&self) -> MatType
    where
        T: DataType,
    {
        MatType::new(T::depth(), self.channels)
    }

    /// Returns the matrix depth.
    pub fn depth(&self) -> Depth
    where
        T: DataType,
    {
        T::depth()
    }

    /// Creates a new `Matrix` with a specific `MatType`.
    ///
    /// # Errors
    /// Returns `PureCvError::InvalidInput` if the depth of `mat_type`
    /// does not match the element type `T`.
    pub fn new_with_type(rows: usize, cols: usize, mat_type: MatType) -> Result<Self>
    where
        T: Default + Clone + DataType,
    {
        if mat_type.depth() != T::depth() {
            cv_bail!(
                tags::CORE,
                InvalidInput,
                "new_with_type: MatType depth {:?} does not match element type {:?}",
                mat_type.depth(),
                T::depth()
            );
        }
        Ok(Self::new(rows, cols, mat_type.channels()))
    }

    /// Returns the number of channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Reallocates the matrix to the specified size and number of channels.
    /// If the current size and channels are already correct, this does nothing.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows.
    /// * `cols` - Number of columns.
    /// * `channels` - Number of channels.
    pub fn create(&mut self, rows: usize, cols: usize, channels: usize)
    where
        T: Default + Clone,
    {
        if self.rows == rows && self.cols == cols && self.channels == channels {
            return;
        }

        self.rows = rows;
        self.cols = cols;
        self.channels = channels;
        self.data = vec![T::default(); rows * cols * channels];
    }

    /// Reallocates the matrix to the specified size and `MatType`.
    ///
    /// # Errors
    /// Returns `PureCvError::InvalidInput` if the depth of `mat_type`
    /// does not match the element type `T`.
    pub fn create_with_type(&mut self, rows: usize, cols: usize, mat_type: MatType) -> Result<()>
    where
        T: Default + Clone + DataType,
    {
        if mat_type.depth() != T::depth() {
            cv_bail!(
                tags::CORE,
                InvalidInput,
                "create_with_type: MatType depth {:?} does not match element type {:?}",
                mat_type.depth(),
                T::depth()
            );
        }
        self.create(rows, cols, mat_type.channels());
        Ok(())
    }

    /// Creates a new `Matrix` with every pixel initialised to the given [`Scalar`].
    ///
    /// Each pixel stores `channels` values; channel `c` is taken from `s[c]`.
    /// For matrices with more than 4 channels, channels ≥ 4 are filled with
    /// `T::default()` (i.e. zero), matching OpenCV's behaviour.
    ///
    /// # Example
    /// ```
    /// use purecv::core::{Matrix, Scalar};
    /// // 2×3 single-channel matrix filled with 7.0
    /// let m = Matrix::<f32>::new_with_scalar(2, 3, 1, Scalar::all(7.0));
    /// assert_eq!(m.get(0, 0, 0), Some(&7.0_f32));
    ///
    /// // 1×1 BGR matrix filled with blue (255, 0, 0)
    /// let blue = Matrix::<u8>::new_with_scalar(1, 1, 3, Scalar::new(255, 0, 0, 0));
    /// assert_eq!(blue.get(0, 0, 0), Some(&255_u8));
    /// assert_eq!(blue.get(0, 0, 1), Some(&0_u8));
    /// ```
    pub fn new_with_scalar(rows: usize, cols: usize, channels: usize, s: Scalar<T>) -> Self {
        let mut data = Vec::with_capacity(rows * cols * channels);
        for _ in 0..rows * cols {
            for c in 0..channels {
                data.push(s.v.get(c).cloned().unwrap_or_default());
            }
        }
        Self {
            rows,
            cols,
            channels,
            data,
        }
    }

    /// Creates a new `Matrix` from a [`Size`] with every pixel initialised to `s`.
    ///
    /// Convenience wrapper around [`new_with_scalar`](Self::new_with_scalar) that
    /// accepts a `Size<U>` instead of separate `rows`/`cols` arguments.
    ///
    /// # Example
    /// ```
    /// use purecv::core::{Matrix, Scalar, Size};
    /// let size = Size::new(640_usize, 480_usize);
    /// let m = Matrix::<f32>::new_with_scalar_from_size(size, 1, Scalar::all(1.0));
    /// assert_eq!(m.rows, 480);
    /// assert_eq!(m.cols, 640);
    /// ```
    pub fn new_with_scalar_from_size<U: Into<usize>>(
        size: crate::core::Size<U>,
        channels: usize,
        s: Scalar<T>,
    ) -> Self {
        Self::new_with_scalar(size.height.into(), size.width.into(), channels, s)
    }

    /// Overwrites every pixel in this matrix with the channel values from `s`.
    ///
    /// For matrices with more than 4 channels, channels ≥ 4 are set to
    /// `T::default()` (zero).  Equivalent to OpenCV's `Mat::setTo(scalar)`.
    ///
    /// # Example
    /// ```
    /// use purecv::core::{Matrix, Scalar};
    /// let mut m = Matrix::<u8>::zeros(4, 4, 3);
    /// m.set_to(Scalar::new(255, 128, 0, 0)); // fill every pixel with (R=255, G=128, B=0)
    /// assert_eq!(m.get(2, 2, 0), Some(&255_u8));
    /// assert_eq!(m.get(2, 2, 1), Some(&128_u8));
    /// ```
    pub fn set_to(&mut self, s: Scalar<T>) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                for c in 0..self.channels {
                    self.set(row, col, c, s.v.get(c).cloned().unwrap_or_default());
                }
            }
        }
    }

    /// Overwrites pixels with `s` at every position where `mask` is non-zero.
    ///
    /// Pixels where `mask == 0` are left unchanged.  This mirrors
    /// OpenCV's `Mat::setTo(scalar, mask)`.
    ///
    /// Channels beyond 4 are set to `T::default()` (zero).
    ///
    /// # Errors
    /// Returns [`crate::core::error::PureCvError::InvalidDimensions`] if `mask` does not have the
    /// same number of rows and columns as `self`.
    ///
    /// # Example
    /// ```
    /// use purecv::core::{Matrix, Scalar};
    /// let mut m = Matrix::<u8>::zeros(2, 2, 1);
    /// // mask: only top-left pixel is set
    /// let mask = Matrix::from_vec(2, 2, 1, vec![1u8, 0, 0, 0]);
    /// m.set_to_masked(Scalar::all(42), &mask).unwrap();
    /// assert_eq!(m.get(0, 0, 0), Some(&42_u8));
    /// assert_eq!(m.get(0, 1, 0), Some(&0_u8));  // unchanged
    /// ```
    pub fn set_to_masked(&mut self, s: Scalar<T>, mask: &Matrix<u8>) -> Result<()> {
        if self.rows != mask.rows || self.cols != mask.cols {
            cv_bail!(
                tags::CORE,
                InvalidDimensions,
                "set_to_masked: mask {}×{} does not match matrix {}×{}",
                mask.rows,
                mask.cols,
                self.rows,
                self.cols
            );
        }
        for row in 0..self.rows {
            for col in 0..self.cols {
                if *mask.get(row, col, 0).unwrap_or(&0) != 0 {
                    for c in 0..self.channels {
                        self.set(row, col, c, s.v.get(c).cloned().unwrap_or_default());
                    }
                }
            }
        }
        Ok(())
    }

    /// Converts the matrix elements to a different type `U`, like OpenCV's
    /// [`Mat::convertTo`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/convert.dispatch.cpp#L248-L300).
    ///
    /// Integer targets round to nearest with ties to even and clamp to the range
    /// of `U`, and the conversion never fails on range (see [`SaturateCast`]);
    /// float targets are a plain cast. A same-type conversion and an integer
    /// source widened to a larger integer or a float are exact, including
    /// `i64` / `u64` values above 2⁵³. The scale and shift form is
    /// [`Matrix::convert_to_scaled`].
    ///
    /// # Example
    /// ```
    /// use purecv::core::Matrix;
    ///
    /// let m = Matrix::from_vec(1, 4, 1, vec![0.5f32, 2.5, 300.0, -1.0]);
    /// assert_eq!(m.convert_to::<u8>().unwrap().data, vec![0, 2, 255, 0]);
    /// ```
    pub fn convert_to<U>(&self) -> Result<Matrix<U>>
    where
        U: SaturateCast + Default + Clone,
        T: num_traits::ToPrimitive + Copy,
    {
        let out_data: Vec<U> = self.data.iter().map(|&x| cast_element(x)).collect();
        Ok(Matrix::from_vec(
            self.rows,
            self.cols,
            self.channels,
            out_data,
        ))
    }

    /// Converts the matrix elements to `U` as `saturate_cast<U>(x * alpha + beta)`,
    /// like OpenCV's
    /// [`Mat::convertTo(m, rtype, alpha, beta)`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1248).
    ///
    /// `alpha = 1, beta = 0` is [`Matrix::convert_to`]. Otherwise the arithmetic
    /// runs in the working precision OpenCV selects in
    /// [`convert_scale.simd.hpp`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/convert_scale.simd.hpp#L228-L295):
    /// `f64` when the target is `f64`, when the target is `f32` and the source
    /// `f64`, or when the target is `i32` and the source `i32` or `f64`;
    /// `f32` for every other pair (so `f64 -> u8` scales in single precision).
    /// The multiply and the add are separate roundings; OpenCV's SIMD loops
    /// fuse them, which can differ in the last bit of a tie. A row narrower
    /// than one vector runs OpenCV's scalar tail instead, which is not fused
    /// and agrees with this path, so that difference needs a wider row.
    ///
    /// # Example
    /// ```
    /// use purecv::core::Matrix;
    ///
    /// let m = Matrix::from_vec(1, 3, 1, vec![1u8, 2, 255]);
    /// let out = m.convert_to_scaled::<u8>(0.5, 0.25).unwrap();
    /// assert_eq!(out.data, vec![1, 1, 128]); // 0.75, 1.25, 127.75
    /// ```
    pub fn convert_to_scaled<U>(&self, alpha: f64, beta: f64) -> Result<Matrix<U>>
    where
        U: SaturateCast + Default + Clone + Send,
        T: num_traits::ToPrimitive + Copy + Sync + 'static,
    {
        if alpha == 1.0 && beta == 0.0 {
            return self.convert_to();
        }
        let out_data: Vec<U> = if scales_in_f64::<T, U>() {
            map_elements(&self.data, |x| scale_in_f64(x, alpha, beta))
        } else {
            let (a, b) = (alpha as f32, beta as f32);
            map_elements(&self.data, |x| scale_in_f32(x, a, b))
        };
        Ok(Matrix::from_vec(
            self.rows,
            self.cols,
            self.channels,
            out_data,
        ))
    }

    /// [`Matrix::convert_to_scaled`] into an existing buffer of the same length, writing only
    /// the elements `selected` accepts. It allocates nothing, which is what keeps a repeated
    /// `normalize` into the same destination fast.
    pub(crate) fn convert_scaled_into<U>(
        &self,
        dst: &mut [U],
        alpha: f64,
        beta: f64,
        selected: impl Fn(usize) -> bool + Sync,
    ) where
        U: SaturateCast + Send,
        T: num_traits::ToPrimitive + Copy + Sync + 'static,
    {
        if alpha == 1.0 && beta == 0.0 {
            write_elements(&self.data, dst, selected, cast_element);
        } else if scales_in_f64::<T, U>() {
            write_elements(&self.data, dst, selected, |x| scale_in_f64(x, alpha, beta));
        } else {
            let (a, b) = (alpha as f32, beta as f32);
            write_elements(&self.data, dst, selected, |x| scale_in_f32(x, a, b));
        }
    }

    /// Swaps the underlying buffer array contents and metadata layout matching dimensions directly with another matrix.
    ///
    /// This relies efficiently on memory switching rather than deep copies mimicking memory
    /// layout transfer bindings traditionally encountered via native pointers on OpenCV C++ bindings natively.
    ///
    /// # Arguments
    ///
    /// * `other` - The matrix with which to swap references logic correctly.
    pub fn swap(&mut self, other: &mut Self) {
        core::mem::swap(self, other);
    }

    /// Returns a raw immutable pointer to the start of the underlying data
    /// buffer, cast to `*const u8`.
    ///
    /// This is useful for zero-copy interop with WASM host buffers and
    /// low-level SIMD intrinsics that need a stable byte address.
    ///
    /// # Safety (for callers)
    /// Dereferencing the returned pointer requires `unsafe`. The pointer
    /// is valid as long as `self` is alive and no reallocation occurs.
    #[inline]
    pub fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr() as *const u8
    }

    /// Returns a raw mutable pointer to the start of the underlying data
    /// buffer, cast to `*mut u8`.
    ///
    /// # Safety (for callers)
    /// Dereferencing the returned pointer requires `unsafe`. The pointer
    /// is valid as long as `self` is alive and no reallocation occurs.
    #[inline]
    pub fn data_ptr_mut(&mut self) -> *mut u8 {
        self.data.as_mut_ptr() as *mut u8
    }

    /// Deep-copies the contents of `self` into `dst`.
    ///
    /// After the call, `dst` is an independent copy of `self` with
    /// identical dimensions, channel count, and data. If `dst` already
    /// has the correct shape, only the data is overwritten (no allocation).
    /// If the shape differs, `dst` is reallocated to match `self`.
    ///
    /// # Example
    /// ```
    /// use purecv::core::Matrix;
    /// let src = Matrix::<u8>::from_vec(2, 2, 1, vec![1, 2, 3, 4]);
    /// let mut dst = Matrix::<u8>::new(0, 0, 0);
    /// src.copy_to(&mut dst).unwrap();
    /// assert_eq!(src.data, dst.data);
    /// assert_eq!(dst.rows, 2);
    /// ```
    pub fn copy_to(&self, dst: &mut Matrix<T>) -> Result<()> {
        if self.rows != dst.rows || self.cols != dst.cols || self.channels != dst.channels {
            #[cfg(debug_assertions)]
            log::warn!(
                "[purecv::copy_to] dst resized from {}x{}x{} to {}x{}x{}",
                dst.rows,
                dst.cols,
                dst.channels,
                self.rows,
                self.cols,
                self.channels
            );
            dst.rows = self.rows;
            dst.cols = self.cols;
            dst.channels = self.channels;
            dst.data = self.data.clone();
        } else {
            dst.data.clone_from(&self.data);
        }
        Ok(())
    }
}

impl<T: num_traits::Zero + num_traits::One + Default + Clone> Matrix<T> {
    /// Returns a zero array of the specified size and type.
    pub fn zeros(rows: usize, cols: usize, channels: usize) -> Self {
        Self {
            rows,
            cols,
            channels,
            data: vec![T::zero(); rows * cols * channels],
        }
    }

    /// Returns a zero array of the specified size and type.
    pub fn zeros_from_size<U: Into<usize>>(size: crate::core::Size<U>, channels: usize) -> Self {
        Self::zeros(size.height.into(), size.width.into(), channels)
    }

    /// Returns an array of all 1's of the specified size and type.
    pub fn ones(rows: usize, cols: usize, channels: usize) -> Self {
        Self {
            rows,
            cols,
            channels,
            data: vec![T::one(); rows * cols * channels],
        }
    }

    /// Returns an array of all 1's of the specified size and type.
    pub fn ones_from_size<U: Into<usize>>(size: crate::core::Size<U>, channels: usize) -> Self {
        Self::ones(size.height.into(), size.width.into(), channels)
    }

    /// Returns a zero array of the specified size and `MatType`.
    pub fn zeros_with_type(rows: usize, cols: usize, mat_type: MatType) -> Result<Self>
    where
        T: DataType,
    {
        if mat_type.depth() != T::depth() {
            cv_bail!(
                tags::CORE,
                InvalidInput,
                "zeros_with_type: MatType depth {:?} does not match element type {:?}",
                mat_type.depth(),
                T::depth()
            );
        }
        Ok(Self::zeros(rows, cols, mat_type.channels()))
    }

    /// Returns an array of all 1's of the specified size and `MatType`.
    pub fn ones_with_type(rows: usize, cols: usize, mat_type: MatType) -> Result<Self>
    where
        T: DataType,
    {
        if mat_type.depth() != T::depth() {
            cv_bail!(
                tags::CORE,
                InvalidInput,
                "ones_with_type: MatType depth {:?} does not match element type {:?}",
                mat_type.depth(),
                T::depth()
            );
        }
        Ok(Self::ones(rows, cols, mat_type.channels()))
    }

    /// Returns an identity matrix of the specified size and type.
    /// Following OpenCV, the diagonal has value 1 and others are 0.
    pub fn eye(rows: usize, cols: usize, channels: usize) -> Self {
        let mut mat = Self::zeros(rows, cols, channels);
        let min_dim = core::cmp::min(rows, cols);
        for i in 0..min_dim {
            for c in 0..channels {
                mat.set(i, i, c, T::one());
            }
        }
        mat
    }

    /// Returns a diagonal matrix from a 1D slice.
    pub fn diag(diagonal: &[T]) -> Self {
        let n = diagonal.len();
        let mut mat = Self::zeros(n, n, 1);
        for (i, val) in diagonal.iter().enumerate().take(n) {
            mat.set(i, i, 0, val.clone());
        }
        mat
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Scalar constructors that require DataType (fallible, typed)
// ──────────────────────────────────────────────────────────────────────────────

impl<T: Default + Clone + DataType> Matrix<T> {
    /// Creates a new `Matrix` from a [`Size`] and [`MatType`] with every pixel set to `s`.
    ///
    /// The number of channels is taken from `mat_type.channels()`.  This is the
    /// type-safe variant: the element depth encoded in `mat_type` must match the
    /// concrete element type `T`; otherwise an error is returned so that
    /// mismatches are caught at runtime rather than producing silent garbage.
    ///
    /// # Errors
    /// Returns [`crate::core::error::PureCvError::InvalidInput`] when `mat_type.depth() != T::depth()`.
    ///
    /// # Example
    /// ```
    /// use purecv::core::{Matrix, Scalar, Size, CV_32FC3};
    /// let size = Size::new(320_usize, 240_usize);
    /// let m = Matrix::<f32>::new_with_scalar_typed_from_size(
    ///     size,
    ///     CV_32FC3,
    ///     Scalar::new(1.0, 0.5, 0.0, 0.0),
    /// )
    /// .unwrap();
    /// assert_eq!(m.channels, 3);
    /// ```
    pub fn new_with_scalar_typed_from_size<U: Into<usize>>(
        size: crate::core::Size<U>,
        mat_type: MatType,
        s: Scalar<T>,
    ) -> Result<Self> {
        if mat_type.depth() != T::depth() {
            cv_bail!(
                tags::CORE,
                InvalidInput,
                "new_with_scalar_typed_from_size: MatType depth {:?} does not match element type depth {:?}",
                mat_type.depth(),
                T::depth()
            );
        }
        let rows = size.height.into();
        let cols = size.width.into();
        Ok(Self::new_with_scalar(rows, cols, mat_type.channels(), s))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ndarray interoperability (behind the "ndarray" feature flag)
// ──────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "ndarray")]
impl<T: Default + Clone> Matrix<T> {
    /// Returns a zero-cost, immutable 3D ndarray view (rows × cols × channels)
    /// over the underlying contiguous data buffer.
    ///
    /// # Panics
    /// Panics if the data length does not match `rows * cols * channels`.
    pub fn as_ndarray_view(&self) -> ndarray::ArrayView3<'_, T> {
        ndarray::ArrayView3::from_shape((self.rows, self.cols, self.channels), &self.data)
            .expect("Matrix data length must equal rows * cols * channels")
    }

    /// Returns a zero-cost, mutable 3D ndarray view (rows × cols × channels)
    /// over the underlying contiguous data buffer.
    ///
    /// # Panics
    /// Panics if the data length does not match `rows * cols * channels`.
    pub fn as_ndarray_view_mut(&mut self) -> ndarray::ArrayViewMut3<'_, T> {
        ndarray::ArrayViewMut3::from_shape((self.rows, self.cols, self.channels), &mut self.data)
            .expect("Matrix data length must equal rows * cols * channels")
    }

    /// Consumes the `Matrix` and returns an owned 3D ndarray
    /// (rows × cols × channels), transferring ownership of the data buffer.
    ///
    /// # Panics
    /// Panics if the internal data length does not match `rows * cols * channels`.
    pub fn into_ndarray(self) -> ndarray::Array3<T> {
        ndarray::Array3::from_shape_vec((self.rows, self.cols, self.channels), self.data)
            .expect("Matrix data length must equal rows * cols * channels")
    }

    /// Creates a `Matrix` from an owned 3D ndarray (rows × cols × channels).
    ///
    /// The incoming array is converted to standard (C-contiguous, row-major)
    /// layout before extracting its raw `Vec`, ensuring the flat data buffer
    /// is always strictly contiguous — a requirement for SIMD/WASM targets.
    pub fn from_ndarray(arr: ndarray::Array3<T>) -> Self {
        let shape = arr.shape();
        let rows = shape[0];
        let cols = shape[1];
        let channels = shape[2];
        let data = arr.as_standard_layout().into_owned().into_raw_vec();
        Self {
            rows,
            cols,
            channels,
            data,
        }
    }
}

#[cfg(feature = "ndarray")]
impl<T: Default + Clone> From<ndarray::Array3<T>> for Matrix<T> {
    fn from(arr: ndarray::Array3<T>) -> Self {
        Self::from_ndarray(arr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depth_logic() {
        assert!(!Depth::CV_8U.is_signed());
        assert!(Depth::CV_8S.is_signed());
        assert!(Depth::CV_32F.is_float());
        assert_eq!(Depth::CV_8U.byte_size(), 1);
        assert_eq!(Depth::CV_32F.byte_size(), 4);
    }

    #[test]
    fn test_mat_type() {
        // Test DataType trait depth mapping
        assert_eq!(u8::depth(), Depth::CV_8U);
        assert_eq!(f32::depth(), Depth::CV_32F);
        let ty = CV_8UC3;
        assert_eq!(ty.depth(), Depth::CV_8U);
        assert_eq!(ty.channels(), 3);
        assert_eq!(ty.to_int(), 16); // 2 << 3 | 0
    }

    #[test]
    fn test_matrix_type_retrieval() {
        let m8u = Matrix::<u8>::zeros(10, 10, 3);
        assert_eq!(m8u.depth(), Depth::CV_8U);
        assert_eq!(m8u.mat_type(), CV_8UC3);

        let m32f = Matrix::<f32>::zeros(10, 10, 1);
        assert_eq!(m32f.depth(), Depth::CV_32F);
        assert_eq!(m32f.mat_type(), CV_32FC1);
    }

    #[test]
    fn test_matrix_create() {
        let mut mat = Matrix::<u8>::zeros(1, 1, 1);
        mat.create(10, 20, 3);
        assert_eq!(mat.rows, 10);
        assert_eq!(mat.cols, 20);
        assert_eq!(mat.channels, 3);
        assert_eq!(mat.data.len(), 10 * 20 * 3);
    }

    #[cfg(feature = "ndarray")]
    mod ndarray_tests {
        use super::*;

        #[test]
        fn test_as_ndarray_view() {
            let mat = Matrix::<f32>::from_vec(2, 3, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            let view = mat.as_ndarray_view();
            assert_eq!(view.shape(), &[2, 3, 1]);
            assert_eq!(view[[0, 0, 0]], 1.0);
            assert_eq!(view[[1, 2, 0]], 6.0);
        }

        #[test]
        fn test_as_ndarray_view_mut() {
            let mut mat = Matrix::<u8>::from_vec(2, 2, 1, vec![1, 2, 3, 4]);
            {
                let mut view = mat.as_ndarray_view_mut();
                view[[0, 1, 0]] = 42;
            }
            assert_eq!(*mat.get(0, 1, 0).unwrap(), 42);
        }

        #[test]
        fn test_into_ndarray() {
            let mat = Matrix::<u8>::from_vec(2, 2, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
            let arr = mat.into_ndarray();
            assert_eq!(arr.shape(), &[2, 2, 3]);
            assert_eq!(arr[[0, 0, 0]], 1);
            assert_eq!(arr[[1, 1, 2]], 12);
        }

        #[test]
        fn test_from_ndarray() {
            let arr = ndarray::Array3::<f64>::from_shape_vec(
                (2, 3, 1),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )
            .unwrap();
            let mat = Matrix::from_ndarray(arr);
            assert_eq!(mat.rows, 2);
            assert_eq!(mat.cols, 3);
            assert_eq!(mat.channels, 1);
            assert_eq!(mat.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }

        #[test]
        fn test_from_trait_ndarray() {
            let arr =
                ndarray::Array3::<u8>::from_shape_vec((1, 2, 3), vec![10, 20, 30, 40, 50, 60])
                    .unwrap();
            let mat: Matrix<u8> = Matrix::from(arr);
            assert_eq!(mat.rows, 1);
            assert_eq!(mat.cols, 2);
            assert_eq!(mat.channels, 3);
            assert_eq!(mat.data, vec![10, 20, 30, 40, 50, 60]);
        }

        #[test]
        fn test_roundtrip_matrix_to_ndarray_and_back() {
            let original = Matrix::<f32>::from_vec(3, 4, 2, (0..24).map(|i| i as f32).collect());
            let arr = original.clone().into_ndarray();
            let recovered = Matrix::from_ndarray(arr);
            assert_eq!(original, recovered);
        }

        #[test]
        fn test_from_ndarray_non_contiguous() {
            // Create a transposed (Fortran-order) array to verify
            // that from_ndarray produces contiguous C-order data.
            let arr =
                ndarray::Array3::<u8>::from_shape_vec((2, 3, 1), vec![1, 2, 3, 4, 5, 6]).unwrap();
            let transposed = arr.reversed_axes(); // now shape (1, 3, 2), Fortran order
            let mat = Matrix::from_ndarray(transposed.into_owned());
            assert_eq!(mat.rows, 1);
            assert_eq!(mat.cols, 3);
            assert_eq!(mat.channels, 2);
            // Data must be C-contiguous (row-major)
            assert_eq!(mat.data.len(), 1 * 3 * 2);
        }
    }
}
