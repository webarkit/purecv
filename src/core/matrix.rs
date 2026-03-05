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
        debug_assert!(row < self.rows && col < self.cols && channel < self.channels, "Index out of bounds");
        (row * self.cols * self.channels) + (col * self.channels) + channel
    }

    /// Safely retrieves a reference to a specific pixel's channel value.
    #[inline]
    pub fn get(&self, row: usize, col: usize, channel: usize) -> Option<&T> {
        let idx = self.flat_index(row, col, channel);
        self.data.get(idx)
    }

    /// Safely retrieves a mutable reference to a specific pixel's channel value.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize, channel: usize) -> Option<&mut T> {
        let idx = self.flat_index(row, col, channel);
        self.data.get_mut(idx)
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
}