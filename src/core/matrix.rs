/*
 *  matrix.rs
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
use crate::core::types::*;

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
        match self {
            Depth::CV_8U | Depth::CV_16U => false,
            _ => true,
        }
    }

    pub fn is_float(&self) -> bool {
        match self {
            Depth::CV_32F | Depth::CV_64F | Depth::CV_16F => true,
            _ => false,
        }
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
    /// # Panics
    /// Panics if the depth of `mat_type` does not match `T`.
    pub fn new_with_type(rows: usize, cols: usize, mat_type: MatType) -> Self
    where
        T: Default + Clone + DataType,
    {
        assert_eq!(
            mat_type.depth(),
            T::depth(),
            "MatType depth must match matrix element type"
        );
        Self::new(rows, cols, mat_type.channels())
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
    /// # Panics
    /// Panics if the depth of `mat_type` does not match `T`.
    pub fn create_with_type(&mut self, rows: usize, cols: usize, mat_type: MatType)
    where
        T: Default + Clone + DataType,
    {
        assert_eq!(
            mat_type.depth(),
            T::depth(),
            "MatType depth must match matrix element type"
        );
        self.create(rows, cols, mat_type.channels());
    }

    /// Converts the matrix elements to a different type `U`.
    /// Similar to OpenCV's `Mat::convertTo`.
    ///
    /// Note: This version currently performs basic casting.
    pub fn convert_to<U>(&self) -> Result<Matrix<U>>
    where
        U: Default + Clone + num_traits::NumCast,
        T: num_traits::ToPrimitive + Copy,
    {
        let out_data: Vec<U> = self
            .data
            .iter()
            .map(|&x| {
                U::from(x).ok_or_else(|| PureCvError::InvalidInput("Conversion failed".into()))
            })
            .collect::<Result<Vec<U>>>()?;

        Ok(Matrix::from_vec(
            self.rows,
            self.cols,
            self.channels,
            out_data,
        ))
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
    pub fn zeros_with_type(rows: usize, cols: usize, mat_type: MatType) -> Self
    where
        T: DataType,
    {
        assert_eq!(
            mat_type.depth(),
            T::depth(),
            "MatType depth must match element type"
        );
        Self::zeros(rows, cols, mat_type.channels())
    }

    /// Returns an array of all 1's of the specified size and `MatType`.
    pub fn ones_with_type(rows: usize, cols: usize, mat_type: MatType) -> Self
    where
        T: DataType,
    {
        assert_eq!(
            mat_type.depth(),
            T::depth(),
            "MatType depth must match element type"
        );
        Self::ones(rows, cols, mat_type.channels())
    }

    /// Returns an identity matrix of the specified size and type.
    /// Following OpenCV, the diagonal has value 1 and others are 0.
    pub fn eye(rows: usize, cols: usize, channels: usize) -> Self {
        let mut mat = Self::zeros(rows, cols, channels);
        let min_dim = std::cmp::min(rows, cols);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depth_logic() {
        assert_eq!(Depth::CV_8U.is_signed(), false);
        assert_eq!(Depth::CV_8S.is_signed(), true);
        assert_eq!(Depth::CV_32F.is_float(), true);
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
}
