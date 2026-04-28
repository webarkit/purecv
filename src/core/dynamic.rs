/*
 *  dynamic.rs
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
 *  Author(s): Walter Perdan <https://github.com/kalwalt>
 *
 */

use crate::core::error::{PureCvError, Result};
use crate::core::matrix::{Depth, MatType, Matrix};

/// An enum bridging type-erased dynamic usage to strongly typed generic `Matrix<T>`.
#[derive(Debug, Clone, PartialEq)]
pub enum DynamicData {
    U8(Matrix<u8>),
    I8(Matrix<i8>),
    U16(Matrix<u16>),
    I16(Matrix<i16>),
    I32(Matrix<i32>),
    F32(Matrix<f32>),
    F64(Matrix<f64>),
}

/// A type-erased matrix that holds any dynamic OpenCV-like depth.
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicMatrix {
    pub data: DynamicData,
}

/// Macro for dispatching dynamic implementations to their underlying generic variants.
#[macro_export]
macro_rules! dispatch_dynamic {
    ($data:expr, $mat:pat => $body:expr) => {
        match $data {
            $crate::core::dynamic::DynamicData::U8($mat) => $body,
            $crate::core::dynamic::DynamicData::I8($mat) => $body,
            $crate::core::dynamic::DynamicData::U16($mat) => $body,
            $crate::core::dynamic::DynamicData::I16($mat) => $body,
            $crate::core::dynamic::DynamicData::I32($mat) => $body,
            $crate::core::dynamic::DynamicData::F32($mat) => $body,
            $crate::core::dynamic::DynamicData::F64($mat) => $body,
        }
    };
}

impl DynamicMatrix {
    // -- OpenCV-style constructors (MatType as single type argument) ----------

    /// Creates a new zero-filled matrix using an OpenCV-style `MatType`.
    ///
    /// This is the primary constructor, mirroring `cv::Mat m(rows, cols, CV_8UC3)`.
    /// `MatType` encodes both depth and channels in a single `i32` value.
    ///
    /// # Errors
    /// Returns `PureCvError::InvalidInput` if `CV_16F` is requested (not yet supported).
    ///
    /// # Example
    /// ```rust
    /// use purecv::core::DynamicMatrix;
    /// use purecv::core::matrix::CV_8UC3;
    ///
    /// let frame = DynamicMatrix::new(480, 640, CV_8UC3).unwrap();
    /// ```
    pub fn new(rows: usize, cols: usize, mat_type: MatType) -> Result<Self> {
        let ch = mat_type.channels();
        let n = rows * cols * ch;
        let data = match mat_type.depth() {
            Depth::CV_8U => DynamicData::U8(Matrix::from_vec(rows, cols, ch, vec![0u8; n])),
            Depth::CV_8S => DynamicData::I8(Matrix::from_vec(rows, cols, ch, vec![0i8; n])),
            Depth::CV_16U => DynamicData::U16(Matrix::from_vec(rows, cols, ch, vec![0u16; n])),
            Depth::CV_16S => DynamicData::I16(Matrix::from_vec(rows, cols, ch, vec![0i16; n])),
            Depth::CV_32S => DynamicData::I32(Matrix::from_vec(rows, cols, ch, vec![0i32; n])),
            Depth::CV_32F => DynamicData::F32(Matrix::from_vec(rows, cols, ch, vec![0f32; n])),
            Depth::CV_64F => DynamicData::F64(Matrix::from_vec(rows, cols, ch, vec![0f64; n])),
            Depth::CV_16F => {
                return Err(PureCvError::InvalidInput(
                    "CV_16F is not yet supported".into(),
                ))
            }
        };
        Ok(Self { data })
    }

    /// Creates a zero-filled matrix using a `MatType`.
    /// Explicit alias for [`new`] — mirrors OpenCV's `Mat::zeros(rows, cols, type)`.
    pub fn zeros(rows: usize, cols: usize, mat_type: MatType) -> Result<Self> {
        Self::new(rows, cols, mat_type)
    }

    /// Creates a matrix filled with ones using a `MatType`.
    /// Mirrors OpenCV's `Mat::ones(rows, cols, type)`.
    ///
    /// # Errors
    /// Returns `PureCvError::InvalidInput` if `CV_16F` is requested (not yet supported).
    pub fn ones(rows: usize, cols: usize, mat_type: MatType) -> Result<Self> {
        let ch = mat_type.channels();
        let n = rows * cols * ch;
        let data = match mat_type.depth() {
            Depth::CV_8U => DynamicData::U8(Matrix::from_vec(rows, cols, ch, vec![1u8; n])),
            Depth::CV_8S => DynamicData::I8(Matrix::from_vec(rows, cols, ch, vec![1i8; n])),
            Depth::CV_16U => DynamicData::U16(Matrix::from_vec(rows, cols, ch, vec![1u16; n])),
            Depth::CV_16S => DynamicData::I16(Matrix::from_vec(rows, cols, ch, vec![1i16; n])),
            Depth::CV_32S => DynamicData::I32(Matrix::from_vec(rows, cols, ch, vec![1i32; n])),
            Depth::CV_32F => DynamicData::F32(Matrix::from_vec(rows, cols, ch, vec![1f32; n])),
            Depth::CV_64F => DynamicData::F64(Matrix::from_vec(rows, cols, ch, vec![1f64; n])),
            Depth::CV_16F => {
                return Err(PureCvError::InvalidInput(
                    "CV_16F is not yet supported".into(),
                ))
            }
        };
        Ok(Self { data })
    }

    // -- Typed constructors (from existing Vec<T>) ----------------------------

    /// Creates a `DynamicMatrix` from an existing `Vec<u8>`.
    ///
    /// Use this when you already have pixel data in hand (e.g. from a decoder or
    /// a JS `Uint8Array`). The data is not copied — ownership is transferred.
    ///
    /// # Errors
    /// Returns `PureCvError::InvalidInput` if `data.len() != rows * cols * channels`.
    pub fn new_u8(rows: usize, cols: usize, channels: usize, data: Vec<u8>) -> Result<Self> {
        let expected = rows * cols * channels;
        if data.len() != expected {
            return Err(PureCvError::InvalidInput(format!(
                "Data length {} does not match {}×{}×{} = {}",
                data.len(),
                rows,
                cols,
                channels,
                expected
            )));
        }
        Ok(Self {
            data: DynamicData::U8(Matrix::from_vec(rows, cols, channels, data)),
        })
    }

    /// Creates a `DynamicMatrix` from an existing `Vec<i8>`.
    ///
    /// # Errors
    /// Returns `PureCvError::InvalidInput` if `data.len() != rows * cols * channels`.
    pub fn new_i8(rows: usize, cols: usize, channels: usize, data: Vec<i8>) -> Result<Self> {
        let expected = rows * cols * channels;
        if data.len() != expected {
            return Err(PureCvError::InvalidInput(format!(
                "Data length {} does not match {}×{}×{} = {}",
                data.len(),
                rows,
                cols,
                channels,
                expected
            )));
        }
        Ok(Self {
            data: DynamicData::I8(Matrix::from_vec(rows, cols, channels, data)),
        })
    }

    /// Creates a `DynamicMatrix` from an existing `Vec<u16>`.
    ///
    /// # Errors
    /// Returns `PureCvError::InvalidInput` if `data.len() != rows * cols * channels`.
    pub fn new_u16(rows: usize, cols: usize, channels: usize, data: Vec<u16>) -> Result<Self> {
        let expected = rows * cols * channels;
        if data.len() != expected {
            return Err(PureCvError::InvalidInput(format!(
                "Data length {} does not match {}×{}×{} = {}",
                data.len(),
                rows,
                cols,
                channels,
                expected
            )));
        }
        Ok(Self {
            data: DynamicData::U16(Matrix::from_vec(rows, cols, channels, data)),
        })
    }

    /// Creates a `DynamicMatrix` from an existing `Vec<i16>`.
    ///
    /// # Errors
    /// Returns `PureCvError::InvalidInput` if `data.len() != rows * cols * channels`.
    pub fn new_i16(rows: usize, cols: usize, channels: usize, data: Vec<i16>) -> Result<Self> {
        let expected = rows * cols * channels;
        if data.len() != expected {
            return Err(PureCvError::InvalidInput(format!(
                "Data length {} does not match {}×{}×{} = {}",
                data.len(),
                rows,
                cols,
                channels,
                expected
            )));
        }
        Ok(Self {
            data: DynamicData::I16(Matrix::from_vec(rows, cols, channels, data)),
        })
    }

    /// Creates a `DynamicMatrix` from an existing `Vec<i32>`.
    ///
    /// # Errors
    /// Returns `PureCvError::InvalidInput` if `data.len() != rows * cols * channels`.
    pub fn new_i32(rows: usize, cols: usize, channels: usize, data: Vec<i32>) -> Result<Self> {
        let expected = rows * cols * channels;
        if data.len() != expected {
            return Err(PureCvError::InvalidInput(format!(
                "Data length {} does not match {}×{}×{} = {}",
                data.len(),
                rows,
                cols,
                channels,
                expected
            )));
        }
        Ok(Self {
            data: DynamicData::I32(Matrix::from_vec(rows, cols, channels, data)),
        })
    }

    /// Creates a `DynamicMatrix` from an existing `Vec<f32>`.
    ///
    /// # Errors
    /// Returns `PureCvError::InvalidInput` if `data.len() != rows * cols * channels`.
    pub fn new_f32(rows: usize, cols: usize, channels: usize, data: Vec<f32>) -> Result<Self> {
        let expected = rows * cols * channels;
        if data.len() != expected {
            return Err(PureCvError::InvalidInput(format!(
                "Data length {} does not match {}×{}×{} = {}",
                data.len(),
                rows,
                cols,
                channels,
                expected
            )));
        }
        Ok(Self {
            data: DynamicData::F32(Matrix::from_vec(rows, cols, channels, data)),
        })
    }

    /// Creates a `DynamicMatrix` from an existing `Vec<f64>`.
    ///
    /// # Errors
    /// Returns `PureCvError::InvalidInput` if `data.len() != rows * cols * channels`.
    pub fn new_f64(rows: usize, cols: usize, channels: usize, data: Vec<f64>) -> Result<Self> {
        let expected = rows * cols * channels;
        if data.len() != expected {
            return Err(PureCvError::InvalidInput(format!(
                "Data length {} does not match {}×{}×{} = {}",
                data.len(),
                rows,
                cols,
                channels,
                expected
            )));
        }
        Ok(Self {
            data: DynamicData::F64(Matrix::from_vec(rows, cols, channels, data)),
        })
    }

    // -- Dimension accessors --------------------------------------------------

    pub fn rows(&self) -> usize {
        dispatch_dynamic!(&self.data, mat => mat.rows)
    }

    pub fn cols(&self) -> usize {
        dispatch_dynamic!(&self.data, mat => mat.cols)
    }

    pub fn channels(&self) -> usize {
        dispatch_dynamic!(&self.data, mat => mat.channels)
    }

    /// Returns the `MatType` of this matrix (encodes depth + channels).
    pub fn mat_type(&self) -> MatType {
        match &self.data {
            DynamicData::U8(m) => MatType::new(Depth::CV_8U, m.channels),
            DynamicData::I8(m) => MatType::new(Depth::CV_8S, m.channels),
            DynamicData::U16(m) => MatType::new(Depth::CV_16U, m.channels),
            DynamicData::I16(m) => MatType::new(Depth::CV_16S, m.channels),
            DynamicData::I32(m) => MatType::new(Depth::CV_32S, m.channels),
            DynamicData::F32(m) => MatType::new(Depth::CV_32F, m.channels),
            DynamicData::F64(m) => MatType::new(Depth::CV_64F, m.channels),
        }
    }

    /// Returns a human-readable name of the element depth (e.g. `"u8"`, `"f32"`).
    pub fn depth_name(&self) -> &str {
        match &self.data {
            DynamicData::U8(_) => "u8",
            DynamicData::I8(_) => "i8",
            DynamicData::U16(_) => "u16",
            DynamicData::I16(_) => "i16",
            DynamicData::I32(_) => "i32",
            DynamicData::F32(_) => "f32",
            DynamicData::F64(_) => "f64",
        }
    }

    /// Total number of elements (rows × cols × channels).
    pub fn total(&self) -> usize {
        dispatch_dynamic!(&self.data, mat => mat.data.len())
    }

    // -- Typed data accessors ------------------------------------------------

    pub fn data_u8(&self) -> Option<&[u8]> {
        match &self.data {
            DynamicData::U8(m) => Some(&m.data),
            _ => None,
        }
    }

    pub fn data_f32(&self) -> Option<&[f32]> {
        match &self.data {
            DynamicData::F32(m) => Some(&m.data),
            _ => None,
        }
    }

    pub fn data_f64(&self) -> Option<&[f64]> {
        match &self.data {
            DynamicData::F64(m) => Some(&m.data),
            _ => None,
        }
    }

    /// Returns a raw pointer to the underlying buffer data.
    pub fn data_ptr(&self) -> *const u8 {
        match &self.data {
            DynamicData::U8(m) => m.data_ptr() as *const u8,
            DynamicData::I8(m) => m.data_ptr() as *const u8,
            DynamicData::U16(m) => m.data_ptr() as *const u8,
            DynamicData::I16(m) => m.data_ptr() as *const u8,
            DynamicData::I32(m) => m.data_ptr() as *const u8,
            DynamicData::F32(m) => m.data_ptr() as *const u8,
            DynamicData::F64(m) => m.data_ptr() as *const u8,
        }
    }

    /// Returns a mutable raw pointer to the underlying buffer data.
    pub fn data_ptr_mut(&mut self) -> *mut u8 {
        match &mut self.data {
            DynamicData::U8(m) => m.data_ptr_mut() as *mut u8,
            DynamicData::I8(m) => m.data_ptr_mut() as *mut u8,
            DynamicData::U16(m) => m.data_ptr_mut() as *mut u8,
            DynamicData::I16(m) => m.data_ptr_mut() as *mut u8,
            DynamicData::I32(m) => m.data_ptr_mut() as *mut u8,
            DynamicData::F32(m) => m.data_ptr_mut() as *mut u8,
            DynamicData::F64(m) => m.data_ptr_mut() as *mut u8,
        }
    }

    // -- Typed matrix borrow -------------------------------------------------

    pub fn as_matrix_u8(&self) -> Option<&Matrix<u8>> {
        match &self.data {
            DynamicData::U8(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_matrix_f32(&self) -> Option<&Matrix<f32>> {
        match &self.data {
            DynamicData::F32(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_matrix_f64(&self) -> Option<&Matrix<f64>> {
        match &self.data {
            DynamicData::F64(m) => Some(m),
            _ => None,
        }
    }

    // -- Read a single element as f64 (for JS interop) -----------------------

    /// Returns the element at `(row, col, channel)` cast to `f64`.
    pub fn at_f64(&self, row: i32, col: i32, channel: usize) -> Option<f64> {
        match &self.data {
            DynamicData::U8(m) => m.at(row, col, channel).map(|v| *v as f64),
            DynamicData::I8(m) => m.at(row, col, channel).map(|v| *v as f64),
            DynamicData::U16(m) => m.at(row, col, channel).map(|v| *v as f64),
            DynamicData::I16(m) => m.at(row, col, channel).map(|v| *v as f64),
            DynamicData::I32(m) => m.at(row, col, channel).map(|v| *v as f64),
            DynamicData::F32(m) => m.at(row, col, channel).map(|v| *v as f64),
            DynamicData::F64(m) => m.at(row, col, channel).copied(),
        }
    }

    // -- Type conversion -----------------------------------------------------

    /// Creates a new `DynamicMatrix` with a different element depth.
    ///
    /// `depth` is a string: `"u8"`, `"i8"`, `"u16"`, `"i16"`, `"i32"`, `"f32"`, `"f64"`.
    pub fn convert_to(&self, depth: &str) -> Result<DynamicMatrix> {
        macro_rules! convert_inner {
            ($src_mat:expr, $depth:expr) => {
                match $depth {
                    "u8" => Ok(DynamicMatrix {
                        data: DynamicData::U8($src_mat.convert_to::<u8>()?),
                    }),
                    "i8" => Ok(DynamicMatrix {
                        data: DynamicData::I8($src_mat.convert_to::<i8>()?),
                    }),
                    "u16" => Ok(DynamicMatrix {
                        data: DynamicData::U16($src_mat.convert_to::<u16>()?),
                    }),
                    "i16" => Ok(DynamicMatrix {
                        data: DynamicData::I16($src_mat.convert_to::<i16>()?),
                    }),
                    "i32" => Ok(DynamicMatrix {
                        data: DynamicData::I32($src_mat.convert_to::<i32>()?),
                    }),
                    "f32" => Ok(DynamicMatrix {
                        data: DynamicData::F32($src_mat.convert_to::<f32>()?),
                    }),
                    "f64" => Ok(DynamicMatrix {
                        data: DynamicData::F64($src_mat.convert_to::<f64>()?),
                    }),
                    other => Err(crate::core::error::PureCvError::InvalidInput(format!(
                        "Unknown depth: {other}"
                    ))),
                }
            };
        }
        match &self.data {
            DynamicData::U8(m) => convert_inner!(m, depth),
            DynamicData::I8(m) => convert_inner!(m, depth),
            DynamicData::U16(m) => convert_inner!(m, depth),
            DynamicData::I16(m) => convert_inner!(m, depth),
            DynamicData::I32(m) => convert_inner!(m, depth),
            DynamicData::F32(m) => convert_inner!(m, depth),
            DynamicData::F64(m) => convert_inner!(m, depth),
        }
    }

    /// Deep copies the matrix data into `dst`. Resizes `dst` if necessary.
    /// Fails if `dst` depth does not match.
    pub fn copy_to(&self, dst: &mut DynamicMatrix) -> Result<()> {
        match (&self.data, &mut dst.data) {
            (DynamicData::U8(s), DynamicData::U8(d)) => s.copy_to(d),
            (DynamicData::I8(s), DynamicData::I8(d)) => s.copy_to(d),
            (DynamicData::U16(s), DynamicData::U16(d)) => s.copy_to(d),
            (DynamicData::I16(s), DynamicData::I16(d)) => s.copy_to(d),
            (DynamicData::I32(s), DynamicData::I32(d)) => s.copy_to(d),
            (DynamicData::F32(s), DynamicData::F32(d)) => s.copy_to(d),
            (DynamicData::F64(s), DynamicData::F64(d)) => s.copy_to(d),
            _ => Err(crate::core::error::PureCvError::InvalidInput(format!(
                "Cannot copy_to because depth mismatch: src={} dst={}",
                self.depth_name(),
                dst.depth_name()
            ))),
        }
    }
}
