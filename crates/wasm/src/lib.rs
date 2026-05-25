/*
 *  lib.rs
 *  purecv-wasm
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

use wasm_bindgen::prelude::*;

use purecv::core::arithm;
use purecv::core::dynamic::{DynamicData, DynamicMatrix};
use purecv::core::matrix::{Depth, MatType};
use purecv::core::structural;
use purecv::core::types::{BorderTypes, Point2i, Scalar as CoreScalar, Size2i};
use purecv::core::Matrix;
use purecv::imgproc::color::{cvt_color, ColorConversionCode};
use purecv::imgproc::derivatives;
use purecv::imgproc::edge;
use purecv::imgproc::feature;
use purecv::imgproc::filter;
use purecv::imgproc::hough;
use purecv::imgproc::morph::{self, MorphShapes, MorphTypes};
use purecv::imgproc::pyramid;
use purecv::imgproc::threshold::{threshold, ThresholdTypes};
use purecv::version;
use purecv::video::optical_flow;

use purecv::features2d::{
    FastFeatureDetector as CoreFastFeatureDetector, FastType as CoreFastType,
    KeyPoint as CoreKeyPoint, Orb as CoreOrb, ScoreType as CoreScoreType,
};

// ---------------------------------------------------------------------------
//  Initialization helpers
// ---------------------------------------------------------------------------

/// Returns the current version string of the library.
#[wasm_bindgen]
pub fn get_version() -> String {
    version::get_version().to_string()
}

/// Logs the library name and version to the browser console.
#[wasm_bindgen]
pub fn print_version() {
    let msg = format!("purecv v{}", version::get_version());
    web_sys::console::log_1(&msg.into());
}

/// Initializes the panic hook for better Rust panic messages in the browser console.
#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Initializes the WASM module: installs the panic hook and logs the library version.
#[wasm_bindgen]
pub fn init_purecv() {
    console_error_panic_hook::set_once();
    print_version();
}

// ---------------------------------------------------------------------------
//  Mat — unified opaque wrapper around DynamicMatrix
// ---------------------------------------------------------------------------

/// An opaque wrapper around `DynamicMatrix`, exposed as `Mat` in JavaScript.
///
/// This is the single matrix type used across the entire WASM API.
/// The element depth and number of channels are selected at construction time
/// via an OpenCV-style `MatType` integer (e.g. `CV_8UC3()`, `CV_32FC1()`).
///
/// ```js
/// // Equivalent to cv::Mat m(480, 640, CV_8UC3) in C++ OpenCV
/// const frame = new Mat(480, 640, CV_8UC3());
/// const hdr   = new Mat(4,   4,   CV_32FC1());
/// ```
#[wasm_bindgen(js_name = "Mat")]
pub struct Mat {
    inner: DynamicMatrix,
}

/// Internal helper: require f32 depth, return `&Matrix<f32>` or a JsError.
fn require_f32<'a>(mat: &'a Mat, op_name: &str) -> Result<&'a Matrix<f32>, JsError> {
    mat.inner.as_matrix_f32().ok_or_else(|| {
        JsError::new(&format!(
            "{op_name} requires f32 depth, but Mat has depth '{}'",
            mat.inner.depth_name()
        ))
    })
}

/// Internal helper: require u8 depth, return `&Matrix<u8>` or a JsError.
fn require_u8<'a>(mat: &'a Mat, op_name: &str) -> Result<&'a Matrix<u8>, JsError> {
    mat.inner.as_matrix_u8().ok_or_else(|| {
        JsError::new(&format!(
            "{op_name} requires u8 depth, but Mat has depth '{}'",
            mat.inner.depth_name()
        ))
    })
}

/// Internal helper: require f64 depth, return `&Matrix<f64>` or a JsError.
fn require_f64<'a>(mat: &'a Mat, op_name: &str) -> Result<&'a Matrix<f64>, JsError> {
    mat.inner.as_matrix_f64().ok_or_else(|| {
        JsError::new(&format!(
            "{op_name} requires f64 depth, but Mat has depth '{}'",
            mat.inner.depth_name()
        ))
    })
}

/// Internal helper: require mutable f64 depth, return `&mut Matrix<f64>` or a JsError.
fn require_f64_mut<'a>(mat: &'a mut Mat, op_name: &str) -> Result<&'a mut Matrix<f64>, JsError> {
    let depth_name = mat.inner.depth_name().to_string();
    mat.inner.as_matrix_f64_mut().ok_or_else(move || {
        JsError::new(&format!(
            "{op_name} requires f64 depth, but Mat has depth '{}'",
            depth_name
        ))
    })
}

#[wasm_bindgen]
impl Mat {
    // -- Constructors -------------------------------------------------------

    /// Creates a new zero-filled matrix.
    ///
    /// * `rows`     – Number of rows (height).
    /// * `cols`     – Number of columns (width).
    /// * `mat_type` – OpenCV-style type integer encoding depth + channels.
    ///               Use the exported constants: `CV_8UC1()`, `CV_8UC3()`,
    ///               `CV_32FC1()`, etc.
    ///
    /// ```js
    /// const frame = new Mat(480, 640, CV_8UC3());   // u8, 3 channels
    /// const mask  = new Mat(480, 640, CV_8UC1());   // u8, 1 channel
    /// const hdr   = new Mat(480, 640, CV_32FC3());  // f32, 3 channels
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(rows: usize, cols: usize, mat_type: i32) -> Result<Mat, JsError> {
        DynamicMatrix::new(rows, cols, MatType(mat_type))
            .map(|d| Mat { inner: d })
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Creates a Mat from a `Uint8Array` (CV_8U depth).
    ///
    /// ```js
    /// const mat = Mat.fromU8Data(2, 2, 3, new Uint8Array([r,g,b, r,g,b, r,g,b, r,g,b]));
    /// ```
    #[wasm_bindgen(js_name = "fromU8Data")]
    pub fn from_u8_data(
        rows: usize,
        cols: usize,
        channels: usize,
        data: &[u8],
    ) -> Result<Mat, JsError> {
        let dm = DynamicMatrix::new_u8(rows, cols, channels, data.to_vec())
            .map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(Mat { inner: dm })
    }

    /// Creates a Mat from a `Float32Array` (CV_32F depth).
    #[wasm_bindgen(js_name = "fromF32Data")]
    pub fn from_f32_data(
        rows: usize,
        cols: usize,
        channels: usize,
        data: &[f32],
    ) -> Result<Mat, JsError> {
        let dm = DynamicMatrix::new_f32(rows, cols, channels, data.to_vec())
            .map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(Mat { inner: dm })
    }

    /// Creates a Mat from a `Float64Array` (CV_64F depth).
    #[wasm_bindgen(js_name = "fromF64Data")]
    pub fn from_f64_data(
        rows: usize,
        cols: usize,
        channels: usize,
        data: &[f64],
    ) -> Result<Mat, JsError> {
        let dm = DynamicMatrix::new_f64(rows, cols, channels, data.to_vec())
            .map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(Mat { inner: dm })
    }

    // -- Accessors ----------------------------------------------------------

    /// Returns the number of rows (height).
    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.inner.rows()
    }

    /// Returns the number of columns (width).
    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> usize {
        self.inner.cols()
    }

    /// Returns the number of channels.
    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> usize {
        self.inner.channels()
    }

    /// Returns the total number of elements (rows × cols × channels).
    #[wasm_bindgen(getter, js_name = "length")]
    pub fn length(&self) -> usize {
        self.inner.total()
    }

    /// Returns the OpenCV-style type integer (encodes depth + channels).
    /// This value matches the constants `CV_8UC3()`, `CV_32FC1()`, etc.
    #[wasm_bindgen(getter, js_name = "type")]
    pub fn mat_type(&self) -> i32 {
        self.inner.mat_type().to_int()
    }

    /// Returns the element depth as a string (e.g. `"u8"`, `"f32"`).
    #[wasm_bindgen(getter)]
    pub fn depth(&self) -> String {
        self.inner.depth_name().to_string()
    }

    // -- Data access --------------------------------------------------------

    /// Returns a copy of the underlying data as a `Uint8Array`.
    /// Errors if this Mat is not of depth `u8`.
    #[wasm_bindgen(js_name = "dataU8")]
    pub fn data_u8(&self) -> Result<Vec<u8>, JsError> {
        self.inner.data_u8().map(|s| s.to_vec()).ok_or_else(|| {
            JsError::new(&format!(
                "dataU8() requires u8 depth, but Mat has depth '{}'",
                self.inner.depth_name()
            ))
        })
    }

    /// Returns a copy of the underlying data as a `Float32Array`.
    /// Errors if this Mat is not of depth `f32`.
    #[wasm_bindgen(js_name = "dataF32")]
    pub fn data_f32(&self) -> Result<Vec<f32>, JsError> {
        self.inner.data_f32().map(|s| s.to_vec()).ok_or_else(|| {
            JsError::new(&format!(
                "dataF32() requires f32 depth, but Mat has depth '{}'",
                self.inner.depth_name()
            ))
        })
    }

    /// Returns a copy of the underlying data as a `Float64Array`.
    /// Errors if this Mat is not of depth `f64`.
    #[wasm_bindgen(js_name = "dataF64")]
    pub fn data_f64(&self) -> Result<Vec<f64>, JsError> {
        self.inner.data_f64().map(|s| s.to_vec()).ok_or_else(|| {
            JsError::new(&format!(
                "dataF64() requires f64 depth, but Mat has depth '{}'",
                self.inner.depth_name()
            ))
        })
    }

    /// Returns a pointer to the underlying buffer data.
    /// This allows zero-copy interoperability with WASM memory.
    #[wasm_bindgen(js_name = "dataPtr")]
    pub fn data_ptr(&self) -> usize {
        self.inner.data_ptr() as usize
    }

    /// Returns a mutable pointer to the underlying buffer data.
    /// This allows zero-copy interoperability with WASM memory.
    #[wasm_bindgen(js_name = "dataPtrMut")]
    pub fn data_ptr_mut(&mut self) -> usize {
        self.inner.data_ptr_mut() as usize
    }

    /// Deep copies the matrix data into `dst`. Resizes `dst` if necessary.
    /// Errors if the destination matrix does not have the same depth.
    #[wasm_bindgen(js_name = "copyTo")]
    pub fn copy_to(&self, dst: &mut Mat) -> Result<(), JsError> {
        self.inner
            .copy_to(&mut dst.inner)
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Sets the underlying data from a `Uint8Array`. Errors if depth is not `u8`.
    #[wasm_bindgen(js_name = "setDataU8")]
    pub fn set_data_u8(&mut self, data: &[u8]) -> Result<(), JsError> {
        match &mut self.inner.data {
            DynamicData::U8(m) => {
                if data.len() != m.data.len() {
                    return Err(JsError::new(&format!(
                        "Data length {} does not match matrix length {}",
                        data.len(),
                        m.data.len()
                    )));
                }
                m.data.copy_from_slice(data);
                Ok(())
            }
            _ => Err(JsError::new(&format!(
                "setDataU8() requires u8 depth, but Mat has depth '{}'",
                self.inner.depth_name()
            ))),
        }
    }

    /// Sets the underlying data from a `Float32Array`. Errors if depth is not `f32`.
    #[wasm_bindgen(js_name = "setDataF32")]
    pub fn set_data_f32(&mut self, data: &[f32]) -> Result<(), JsError> {
        match &mut self.inner.data {
            DynamicData::F32(m) => {
                if data.len() != m.data.len() {
                    return Err(JsError::new(&format!(
                        "Data length {} does not match matrix length {}",
                        data.len(),
                        m.data.len()
                    )));
                }
                m.data.copy_from_slice(data);
                Ok(())
            }
            _ => Err(JsError::new(&format!(
                "setDataF32() requires f32 depth, but Mat has depth '{}'",
                self.inner.depth_name()
            ))),
        }
    }

    /// Sets the underlying data from a `Float64Array`. Errors if depth is not `f64`.
    #[wasm_bindgen(js_name = "setDataF64")]
    pub fn set_data_f64(&mut self, data: &[f64]) -> Result<(), JsError> {
        match &mut self.inner.data {
            DynamicData::F64(m) => {
                if data.len() != m.data.len() {
                    return Err(JsError::new(&format!(
                        "Data length {} does not match matrix length {}",
                        data.len(),
                        m.data.len()
                    )));
                }
                m.data.copy_from_slice(data);
                Ok(())
            }
            _ => Err(JsError::new(&format!(
                "setDataF64() requires f64 depth, but Mat has depth '{}'",
                self.inner.depth_name()
            ))),
        }
    }

    /// Returns the value at (row, col, channel) cast to `f64`.
    /// Returns `undefined` if the coordinates are out of bounds.
    #[wasm_bindgen(js_name = "at")]
    pub fn at(&self, row: i32, col: i32, channel: usize) -> Option<f64> {
        self.inner.at_f64(row, col, channel)
    }

    // -- Type conversion ----------------------------------------------------

    /// Converts this Mat to a new Mat with a different element depth.
    ///
    /// * `depth` – Target depth: `"u8"`, `"i8"`, `"u16"`, `"i16"`, `"i32"`, `"f32"`, `"f64"`.
    #[wasm_bindgen(js_name = "convertTo")]
    pub fn convert_to(&self, depth: &str) -> Result<Mat, JsError> {
        let dm = self
            .inner
            .convert_to(depth)
            .map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(Mat { inner: dm })
    }

    // -- Scalar constructors / fill -----------------------------------------

    /// Creates a new Mat with every pixel set to the given `Scalar`.
    ///
    /// Channel values in `s` are cast to the element depth encoded in `mat_type`.
    ///
    /// ```js
    /// // 480×640 BGR u8 image filled with blue (255, 0, 0)
    /// const blue = new Scalar(255, 0, 0, 0);
    /// const img  = Mat.newWithScalar(480, 640, CV_8UC3(), blue);
    /// ```
    #[wasm_bindgen(js_name = "newWithScalar")]
    pub fn new_with_scalar(
        rows: usize,
        cols: usize,
        mat_type: i32,
        s: &Scalar,
    ) -> Result<Mat, JsError> {
        let mt = MatType(mat_type);
        let ch = mt.channels();
        let data = match mt.depth() {
            Depth::CV_8U => DynamicData::U8(Matrix::new_with_scalar(
                rows,
                cols,
                ch,
                CoreScalar::new(s.v0 as u8, s.v1 as u8, s.v2 as u8, s.v3 as u8),
            )),
            Depth::CV_8S => DynamicData::I8(Matrix::new_with_scalar(
                rows,
                cols,
                ch,
                CoreScalar::new(s.v0 as i8, s.v1 as i8, s.v2 as i8, s.v3 as i8),
            )),
            Depth::CV_16U => DynamicData::U16(Matrix::new_with_scalar(
                rows,
                cols,
                ch,
                CoreScalar::new(s.v0 as u16, s.v1 as u16, s.v2 as u16, s.v3 as u16),
            )),
            Depth::CV_16S => DynamicData::I16(Matrix::new_with_scalar(
                rows,
                cols,
                ch,
                CoreScalar::new(s.v0 as i16, s.v1 as i16, s.v2 as i16, s.v3 as i16),
            )),
            Depth::CV_32S => DynamicData::I32(Matrix::new_with_scalar(
                rows,
                cols,
                ch,
                CoreScalar::new(s.v0 as i32, s.v1 as i32, s.v2 as i32, s.v3 as i32),
            )),
            Depth::CV_32F => DynamicData::F32(Matrix::new_with_scalar(
                rows,
                cols,
                ch,
                CoreScalar::new(s.v0 as f32, s.v1 as f32, s.v2 as f32, s.v3 as f32),
            )),
            Depth::CV_64F => DynamicData::F64(Matrix::new_with_scalar(
                rows,
                cols,
                ch,
                CoreScalar::new(s.v0, s.v1, s.v2, s.v3),
            )),
            Depth::CV_16F => return Err(JsError::new("CV_16F is not yet supported")),
        };
        Ok(Mat {
            inner: DynamicMatrix { data },
        })
    }

    /// Fills every pixel of this Mat with the given `Scalar`.
    ///
    /// Channel values are cast to the element depth of this Mat.
    ///
    /// ```js
    /// const gray = new Scalar(128, 128, 128, 255);
    /// mat.setTo(gray);
    /// ```
    #[wasm_bindgen(js_name = "setTo")]
    pub fn set_to(&mut self, s: &Scalar) {
        match &mut self.inner.data {
            DynamicData::U8(m) => m.set_to(CoreScalar::new(
                s.v0 as u8, s.v1 as u8, s.v2 as u8, s.v3 as u8,
            )),
            DynamicData::I8(m) => m.set_to(CoreScalar::new(
                s.v0 as i8, s.v1 as i8, s.v2 as i8, s.v3 as i8,
            )),
            DynamicData::U16(m) => m.set_to(CoreScalar::new(
                s.v0 as u16,
                s.v1 as u16,
                s.v2 as u16,
                s.v3 as u16,
            )),
            DynamicData::I16(m) => m.set_to(CoreScalar::new(
                s.v0 as i16,
                s.v1 as i16,
                s.v2 as i16,
                s.v3 as i16,
            )),
            DynamicData::I32(m) => m.set_to(CoreScalar::new(
                s.v0 as i32,
                s.v1 as i32,
                s.v2 as i32,
                s.v3 as i32,
            )),
            DynamicData::F32(m) => m.set_to(CoreScalar::new(
                s.v0 as f32,
                s.v1 as f32,
                s.v2 as f32,
                s.v3 as f32,
            )),
            DynamicData::F64(m) => m.set_to(CoreScalar::new(s.v0, s.v1, s.v2, s.v3)),
        }
    }

    /// Fills pixels with the given `Scalar` where `mask` is non-zero.
    ///
    /// `mask` must be a single-channel `u8` Mat with the same dimensions as `self`.
    ///
    /// ```js
    /// const white = new Scalar(255, 255, 255, 255);
    /// mat.setToMasked(white, mask);  // only pixels where mask != 0 are filled
    /// ```
    #[wasm_bindgen(js_name = "setToMasked")]
    pub fn set_to_masked(&mut self, s: &Scalar, mask: &Mat) -> Result<(), JsError> {
        let mask_m = require_u8(mask, "setToMasked")?;
        match &mut self.inner.data {
            DynamicData::U8(m) => m
                .set_to_masked(
                    CoreScalar::new(s.v0 as u8, s.v1 as u8, s.v2 as u8, s.v3 as u8),
                    mask_m,
                )
                .map_err(|e| JsError::new(&format!("{e}")))?,
            DynamicData::I8(m) => m
                .set_to_masked(
                    CoreScalar::new(s.v0 as i8, s.v1 as i8, s.v2 as i8, s.v3 as i8),
                    mask_m,
                )
                .map_err(|e| JsError::new(&format!("{e}")))?,
            DynamicData::U16(m) => m
                .set_to_masked(
                    CoreScalar::new(s.v0 as u16, s.v1 as u16, s.v2 as u16, s.v3 as u16),
                    mask_m,
                )
                .map_err(|e| JsError::new(&format!("{e}")))?,
            DynamicData::I16(m) => m
                .set_to_masked(
                    CoreScalar::new(s.v0 as i16, s.v1 as i16, s.v2 as i16, s.v3 as i16),
                    mask_m,
                )
                .map_err(|e| JsError::new(&format!("{e}")))?,
            DynamicData::I32(m) => m
                .set_to_masked(
                    CoreScalar::new(s.v0 as i32, s.v1 as i32, s.v2 as i32, s.v3 as i32),
                    mask_m,
                )
                .map_err(|e| JsError::new(&format!("{e}")))?,
            DynamicData::F32(m) => m
                .set_to_masked(
                    CoreScalar::new(s.v0 as f32, s.v1 as f32, s.v2 as f32, s.v3 as f32),
                    mask_m,
                )
                .map_err(|e| JsError::new(&format!("{e}")))?,
            DynamicData::F64(m) => m
                .set_to_masked(CoreScalar::new(s.v0, s.v1, s.v2, s.v3), mask_m)
                .map_err(|e| JsError::new(&format!("{e}")))?,
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
//  Scalar — 4-channel value type
// ---------------------------------------------------------------------------

/// A 4-channel value for initialising or filling matrices.
///
/// Channel values are `f64` and are cast to the element depth of the target
/// `Mat` when passed to `newWithScalar`, `setTo`, or `setToMasked`.
///
/// ```js
/// const blue  = new Scalar(255, 0, 0, 0);      // BGR blue
/// const gray  = new Scalar(128, 128, 128, 255); // RGBA mid-gray
/// const white = Scalar.all(255);                // broadcast
/// ```
#[wasm_bindgen(js_name = "Scalar")]
pub struct Scalar {
    pub v0: f64,
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
}

#[wasm_bindgen]
impl Scalar {
    /// Creates a Scalar from four channel values.
    #[wasm_bindgen(constructor)]
    pub fn new(v0: f64, v1: f64, v2: f64, v3: f64) -> Scalar {
        Scalar { v0, v1, v2, v3 }
    }

    /// Creates a Scalar with the same value broadcast to all four channels.
    ///
    /// ```js
    /// const white = Scalar.all(255); // [255, 255, 255, 255]
    /// ```
    pub fn all(v: f64) -> Scalar {
        Scalar {
            v0: v,
            v1: v,
            v2: v,
            v3: v,
        }
    }

    /// Creates a Scalar with `v` in channel 0 and zero in channels 1–3.
    /// Mirrors OpenCV's `cv::Scalar(v)`.
    ///
    /// ```js
    /// const luma = Scalar.fromValue(128); // [128, 0, 0, 0]
    /// ```
    #[wasm_bindgen(js_name = "fromValue")]
    pub fn from_value(v: f64) -> Scalar {
        Scalar {
            v0: v,
            v1: 0.0,
            v2: 0.0,
            v3: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
//  Vec types
// ---------------------------------------------------------------------------

#[wasm_bindgen(js_name = "Vec2")]
pub struct Vec2 {
    pub v0: f64,
    pub v1: f64,
}

#[wasm_bindgen]
impl Vec2 {
    #[wasm_bindgen(constructor)]
    pub fn new(v0: f64, v1: f64) -> Vec2 {
        Vec2 { v0, v1 }
    }
}

#[wasm_bindgen(js_name = "Vec3")]
pub struct Vec3 {
    pub v0: f64,
    pub v1: f64,
    pub v2: f64,
}

#[wasm_bindgen]
impl Vec3 {
    #[wasm_bindgen(constructor)]
    pub fn new(v0: f64, v1: f64, v2: f64) -> Vec3 {
        Vec3 { v0, v1, v2 }
    }
}

#[wasm_bindgen(js_name = "Vec4")]
pub struct Vec4 {
    pub v0: f64,
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
}

#[wasm_bindgen]
impl Vec4 {
    #[wasm_bindgen(constructor)]
    pub fn new(v0: f64, v1: f64, v2: f64, v3: f64) -> Vec4 {
        Vec4 { v0, v1, v2, v3 }
    }
}

// ---------------------------------------------------------------------------
//  Arithmetic operations (require f32 depth)
// ---------------------------------------------------------------------------

/// Per-element addition: `dst = a + b`.
#[wasm_bindgen(js_name = "add")]
pub fn add(a: &Mat, b: &Mat) -> Result<Mat, JsError> {
    let ma = require_f32(a, "add")?;
    let mb = require_f32(b, "add")?;
    let result = arithm::add(ma, mb).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Per-element subtraction: `dst = a - b`.
#[wasm_bindgen(js_name = "subtract")]
pub fn subtract(a: &Mat, b: &Mat) -> Result<Mat, JsError> {
    let ma = require_f32(a, "subtract")?;
    let mb = require_f32(b, "subtract")?;
    let result = arithm::subtract(ma, mb).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Per-element multiplication: `dst = a * b`.
#[wasm_bindgen(js_name = "multiply")]
pub fn multiply(a: &Mat, b: &Mat) -> Result<Mat, JsError> {
    let ma = require_f32(a, "multiply")?;
    let mb = require_f32(b, "multiply")?;
    let result = arithm::multiply(ma, mb).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Per-element division: `dst = a / b`.
#[wasm_bindgen(js_name = "divide")]
pub fn divide(a: &Mat, b: &Mat) -> Result<Mat, JsError> {
    let ma = require_f32(a, "divide")?;
    let mb = require_f32(b, "divide")?;
    let result = arithm::divide(ma, mb).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Per-element absolute difference: `dst = |a - b|`.
#[wasm_bindgen(js_name = "absDiff")]
pub fn abs_diff(a: &Mat, b: &Mat) -> Result<Mat, JsError> {
    let ma = require_f32(a, "absDiff")?;
    let mb = require_f32(b, "absDiff")?;
    let result = arithm::abs_diff(ma, mb).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Per-element minimum: `dst(i) = min(a(i), b(i))`.
#[wasm_bindgen(js_name = "min")]
pub fn min(a: &Mat, b: &Mat) -> Result<Mat, JsError> {
    let ma = require_f32(a, "min")?;
    let mb = require_f32(b, "min")?;
    let result = arithm::min(ma, mb).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Per-element maximum: `dst(i) = max(a(i), b(i))`.
#[wasm_bindgen(js_name = "max")]
pub fn max(a: &Mat, b: &Mat) -> Result<Mat, JsError> {
    let ma = require_f32(a, "max")?;
    let mb = require_f32(b, "max")?;
    let result = arithm::max(ma, mb).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

// ---------------------------------------------------------------------------
//  Structural operations (dispatch across all depths)
// ---------------------------------------------------------------------------

/// Flips a matrix around vertical (0), horizontal (1), or both axes (-1).
#[wasm_bindgen(js_name = "flip")]
pub fn flip(src: &Mat, flip_code: i32) -> Result<Mat, JsError> {
    let data = match &src.inner.data {
        DynamicData::U8(m) => DynamicData::U8(
            structural::flip(m, flip_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::I8(m) => DynamicData::I8(
            structural::flip(m, flip_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::U16(m) => DynamicData::U16(
            structural::flip(m, flip_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::I16(m) => DynamicData::I16(
            structural::flip(m, flip_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::I32(m) => DynamicData::I32(
            structural::flip(m, flip_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::F32(m) => DynamicData::F32(
            structural::flip(m, flip_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::F64(m) => DynamicData::F64(
            structural::flip(m, flip_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
    };
    Ok(Mat {
        inner: DynamicMatrix { data },
    })
}

/// Transposes a matrix (swaps rows and columns).
#[wasm_bindgen(js_name = "transpose")]
pub fn transpose(src: &Mat) -> Result<Mat, JsError> {
    let data = match &src.inner.data {
        DynamicData::U8(m) => {
            DynamicData::U8(structural::transpose(m).map_err(|e| JsError::new(&format!("{e}")))?)
        }
        DynamicData::I8(m) => {
            DynamicData::I8(structural::transpose(m).map_err(|e| JsError::new(&format!("{e}")))?)
        }
        DynamicData::U16(m) => {
            DynamicData::U16(structural::transpose(m).map_err(|e| JsError::new(&format!("{e}")))?)
        }
        DynamicData::I16(m) => {
            DynamicData::I16(structural::transpose(m).map_err(|e| JsError::new(&format!("{e}")))?)
        }
        DynamicData::I32(m) => {
            DynamicData::I32(structural::transpose(m).map_err(|e| JsError::new(&format!("{e}")))?)
        }
        DynamicData::F32(m) => {
            DynamicData::F32(structural::transpose(m).map_err(|e| JsError::new(&format!("{e}")))?)
        }
        DynamicData::F64(m) => {
            DynamicData::F64(structural::transpose(m).map_err(|e| JsError::new(&format!("{e}")))?)
        }
    };
    Ok(Mat {
        inner: DynamicMatrix { data },
    })
}

/// Rotates a matrix: 0 = 90° CW, 1 = 180°, 2 = 90° CCW.
#[wasm_bindgen(js_name = "rotate")]
pub fn rotate(src: &Mat, rotate_code: i32) -> Result<Mat, JsError> {
    let data = match &src.inner.data {
        DynamicData::U8(m) => DynamicData::U8(
            structural::rotate(m, rotate_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::I8(m) => DynamicData::I8(
            structural::rotate(m, rotate_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::U16(m) => DynamicData::U16(
            structural::rotate(m, rotate_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::I16(m) => DynamicData::I16(
            structural::rotate(m, rotate_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::I32(m) => DynamicData::I32(
            structural::rotate(m, rotate_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::F32(m) => DynamicData::F32(
            structural::rotate(m, rotate_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::F64(m) => DynamicData::F64(
            structural::rotate(m, rotate_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
    };
    Ok(Mat {
        inner: DynamicMatrix { data },
    })
}

// ---------------------------------------------------------------------------
//  Color conversion (requires u8 depth)
// ---------------------------------------------------------------------------

/// Helper to convert a JS integer into a `ColorConversionCode`.
fn color_code_from_i32(code: i32) -> Result<ColorConversionCode, JsError> {
    match code {
        0 => Ok(ColorConversionCode::COLOR_BGR2GRAY),
        1 => Ok(ColorConversionCode::COLOR_RGB2GRAY),
        2 => Ok(ColorConversionCode::COLOR_BGRA2GRAY),
        3 => Ok(ColorConversionCode::COLOR_RGBA2GRAY),
        4 => Ok(ColorConversionCode::COLOR_GRAY2RGB),
        5 => Ok(ColorConversionCode::COLOR_GRAY2BGR),
        6 => Ok(ColorConversionCode::COLOR_GRAY2RGBA),
        7 => Ok(ColorConversionCode::COLOR_GRAY2BGRA),
        _ => Err(JsError::new(&format!(
            "Unknown color conversion code: {code}"
        ))),
    }
}

/// Converts an 8-bit image from one colour space to another.
///
/// Codes (integer):
///   0 = BGR2GRAY, 1 = RGB2GRAY, 2 = BGRA2GRAY, 3 = RGBA2GRAY,
///   4 = GRAY2RGB, 5 = GRAY2BGR, 6 = GRAY2RGBA, 7 = GRAY2BGRA.
#[wasm_bindgen(js_name = "cvtColor")]
pub fn convert_color(src: &Mat, code: i32) -> Result<Mat, JsError> {
    let m = require_u8(src, "cvtColor")?;
    let cc = color_code_from_i32(code)?;
    let result = cvt_color(m, cc).map_err(JsError::new)?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::U8(result),
        },
    })
}

// ---------------------------------------------------------------------------
//  Threshold (requires f32 depth)
// ---------------------------------------------------------------------------

/// Helper to convert a JS integer into a `ThresholdTypes`.
fn thresh_type_from_i32(t: i32) -> Result<ThresholdTypes, JsError> {
    match t {
        0 => Ok(ThresholdTypes::THRESH_BINARY),
        1 => Ok(ThresholdTypes::THRESH_BINARY_INV),
        2 => Ok(ThresholdTypes::THRESH_TRUNC),
        3 => Ok(ThresholdTypes::THRESH_TOZERO),
        4 => Ok(ThresholdTypes::THRESH_TOZERO_INV),
        _ => Err(JsError::new(&format!("Unknown threshold type: {t}"))),
    }
}

/// Result of the threshold operation, containing the computed threshold
/// value and the output matrix.
#[wasm_bindgen]
pub struct ThresholdResult {
    thresh_val: f64,
    matrix: Mat,
}

#[wasm_bindgen]
impl ThresholdResult {
    /// The threshold value that was used (relevant for Otsu / Triangle).
    #[wasm_bindgen(getter, js_name = "threshVal")]
    pub fn thresh_val(&self) -> f64 {
        self.thresh_val
    }

    /// The output (thresholded) matrix.  Consumes the result.
    #[wasm_bindgen(js_name = "getMatrix")]
    pub fn get_matrix(self) -> Mat {
        self.matrix
    }
}

/// Applies a fixed-level threshold to every element.
///
/// * `threshold_type`: 0 = BINARY, 1 = BINARY_INV, 2 = TRUNC, 3 = TOZERO, 4 = TOZERO_INV.
#[wasm_bindgen(js_name = "threshold")]
pub fn apply_threshold(
    src: &Mat,
    thresh: f64,
    maxval: f64,
    threshold_type: i32,
) -> Result<ThresholdResult, JsError> {
    let m = require_f32(src, "threshold")?;
    let tt = thresh_type_from_i32(threshold_type)?;
    let (tv, mat) = threshold(m, thresh, maxval, tt).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(ThresholdResult {
        thresh_val: tv,
        matrix: Mat {
            inner: DynamicMatrix {
                data: DynamicData::F32(mat),
            },
        },
    })
}

// ---------------------------------------------------------------------------
//  Edge detection (requires f32 depth)
// ---------------------------------------------------------------------------

/// Helper to convert a JS integer into a `BorderTypes`.
fn border_type_from_i32(bt: i32) -> Result<BorderTypes, JsError> {
    match bt {
        0 => Ok(BorderTypes::Constant),
        1 => Ok(BorderTypes::Replicate),
        2 => Ok(BorderTypes::Reflect),
        3 => Ok(BorderTypes::Wrap),
        4 => Ok(BorderTypes::Reflect101),
        5 => Ok(BorderTypes::Transparent),
        16 => Ok(BorderTypes::Isolated),
        _ => Err(JsError::new(&format!("Unknown border type: {bt}"))),
    }
}

/// Canny edge detection.  Input must be single-channel f32.
/// Returns a u8 edge map.
///
/// * `aperture_size` – Sobel kernel size (default: 3).
/// * `l2_gradient`   – Use L₂ norm (true) or L₁ norm (false).
#[wasm_bindgen(js_name = "canny")]
pub fn canny(
    src: &Mat,
    threshold1: f64,
    threshold2: f64,
    aperture_size: i32,
    l2_gradient: bool,
) -> Result<Mat, JsError> {
    let m = require_f32(src, "canny")?;
    let result = edge::canny(m, threshold1, threshold2, aperture_size, l2_gradient)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::U8(result),
        },
    })
}

/// Sobel derivative filter.
///
/// * `dx`, `dy`       – Order of the derivative in x / y.
/// * `ksize`          – Aperture size (1, 3, 5, or 7; -1 = Scharr).
/// * `scale`, `delta` – Scale factor and optional offset.
/// * `border_type`    – Border interpolation (integer, see `BorderTypes`).
#[wasm_bindgen(js_name = "sobel")]
pub fn sobel(
    src: &Mat,
    dx: i32,
    dy: i32,
    ksize: i32,
    scale: f64,
    delta: f64,
    border_type: i32,
) -> Result<Mat, JsError> {
    let m = require_f32(src, "sobel")?;
    let bt = border_type_from_i32(border_type)?;
    let result = derivatives::sobel(m, dx, dy, ksize, scale, delta, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Scharr derivative filter (equivalent to Sobel with ksize = -1).
#[wasm_bindgen(js_name = "scharr")]
pub fn scharr(
    src: &Mat,
    dx: i32,
    dy: i32,
    scale: f64,
    delta: f64,
    border_type: i32,
) -> Result<Mat, JsError> {
    let m = require_f32(src, "scharr")?;
    let bt = border_type_from_i32(border_type)?;
    let result = derivatives::scharr(m, dx, dy, scale, delta, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Laplacian of an image.
#[wasm_bindgen(js_name = "laplacian")]
pub fn laplacian(
    src: &Mat,
    ksize: i32,
    scale: f64,
    delta: f64,
    border_type: i32,
) -> Result<Mat, JsError> {
    let m = require_f32(src, "laplacian")?;
    let bt = border_type_from_i32(border_type)?;
    let result = derivatives::laplacian(m, ksize, scale, delta, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

// ---------------------------------------------------------------------------
//  Blur / filter operations (require f32 depth)
// ---------------------------------------------------------------------------

/// Box blur (mean filter).
///
/// * `ksize_w`, `ksize_h` – Kernel width and height.
/// * `border_type`        – Border interpolation (integer, see `BorderTypes`).
#[wasm_bindgen(js_name = "blur")]
pub fn blur(src: &Mat, ksize_w: i32, ksize_h: i32, border_type: i32) -> Result<Mat, JsError> {
    let m = require_f32(src, "blur")?;
    let bt = border_type_from_i32(border_type)?;
    let ksize = Size2i::new(ksize_w, ksize_h);
    let anchor = Point2i::new(-1, -1);
    let result = filter::blur(m, ksize, anchor, bt).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Gaussian blur.
///
/// * `ksize_w`, `ksize_h` – Kernel width and height (must be odd).
/// * `sigma1`             – Gaussian σ in X direction (0 = auto).
/// * `sigma2`             – Gaussian σ in Y direction (0 = same as σ₁).
/// * `border_type`        – Border interpolation (integer).
#[wasm_bindgen(js_name = "gaussianBlur")]
pub fn gaussian_blur(
    src: &Mat,
    ksize_w: i32,
    ksize_h: i32,
    sigma1: f64,
    sigma2: f64,
    border_type: i32,
) -> Result<Mat, JsError> {
    let m = require_f32(src, "gaussianBlur")?;
    let bt = border_type_from_i32(border_type)?;
    let ksize = Size2i::new(ksize_w, ksize_h);
    let result = filter::gaussian_blur(m, ksize, sigma1, sigma2, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Median blur.
///
/// * `ksize` – Aperture size (must be odd and > 1).
#[wasm_bindgen(js_name = "medianBlur")]
pub fn median_blur(src: &Mat, ksize: i32) -> Result<Mat, JsError> {
    let m = require_f32(src, "medianBlur")?;
    let result = filter::median_blur(m, ksize).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Bilateral filter.
///
/// * `d`           – Diameter of pixel neighbourhood (-1 = auto from sigma_space).
/// * `sigma_color` – Filter sigma in the colour space.
/// * `sigma_space` – Filter sigma in the coordinate space.
/// * `border_type` – Border interpolation (integer).
#[wasm_bindgen(js_name = "bilateralFilter")]
pub fn bilateral_filter(
    src: &Mat,
    d: i32,
    sigma_color: f64,
    sigma_space: f64,
    border_type: i32,
) -> Result<Mat, JsError> {
    let m = require_f32(src, "bilateralFilter")?;
    let bt = border_type_from_i32(border_type)?;
    let result = filter::bilateral_filter(m, d, sigma_color, sigma_space, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

// ---------------------------------------------------------------------------
//  JS-side enum constants
// ---------------------------------------------------------------------------

// Color conversion codes
#[wasm_bindgen(js_name = "COLOR_BGR2GRAY")]
pub fn color_bgr2gray() -> i32 {
    0
}
#[wasm_bindgen(js_name = "COLOR_RGB2GRAY")]
pub fn color_rgb2gray() -> i32 {
    1
}
#[wasm_bindgen(js_name = "COLOR_BGRA2GRAY")]
pub fn color_bgra2gray() -> i32 {
    2
}
#[wasm_bindgen(js_name = "COLOR_RGBA2GRAY")]
pub fn color_rgba2gray() -> i32 {
    3
}
#[wasm_bindgen(js_name = "COLOR_GRAY2RGB")]
pub fn color_gray2rgb() -> i32 {
    4
}
#[wasm_bindgen(js_name = "COLOR_GRAY2BGR")]
pub fn color_gray2bgr() -> i32 {
    5
}
#[wasm_bindgen(js_name = "COLOR_GRAY2RGBA")]
pub fn color_gray2rgba() -> i32 {
    6
}
#[wasm_bindgen(js_name = "COLOR_GRAY2BGRA")]
pub fn color_gray2bgra() -> i32 {
    7
}

// -- Threshold types --------------------------------------------------------

#[wasm_bindgen(js_name = "THRESH_BINARY")]
pub fn thresh_binary() -> i32 {
    0
}
#[wasm_bindgen(js_name = "THRESH_BINARY_INV")]
pub fn thresh_binary_inv() -> i32 {
    1
}
#[wasm_bindgen(js_name = "THRESH_TRUNC")]
pub fn thresh_trunc() -> i32 {
    2
}
#[wasm_bindgen(js_name = "THRESH_TOZERO")]
pub fn thresh_tozero() -> i32 {
    3
}
#[wasm_bindgen(js_name = "THRESH_TOZERO_INV")]
pub fn thresh_tozero_inv() -> i32 {
    4
}

// -- Border types -----------------------------------------------------------

#[wasm_bindgen(js_name = "BORDER_CONSTANT")]
pub fn border_constant() -> i32 {
    0
}
#[wasm_bindgen(js_name = "BORDER_REPLICATE")]
pub fn border_replicate() -> i32 {
    1
}
#[wasm_bindgen(js_name = "BORDER_REFLECT")]
pub fn border_reflect() -> i32 {
    2
}
#[wasm_bindgen(js_name = "BORDER_WRAP")]
pub fn border_wrap() -> i32 {
    3
}
#[wasm_bindgen(js_name = "BORDER_REFLECT_101")]
pub fn border_reflect_101() -> i32 {
    4
}
#[wasm_bindgen(js_name = "BORDER_DEFAULT")]
pub fn border_default() -> i32 {
    4
}

// -- Flip codes -------------------------------------------------------------

#[wasm_bindgen(js_name = "FLIP_VERTICAL")]
pub fn flip_vertical() -> i32 {
    0
}
#[wasm_bindgen(js_name = "FLIP_HORIZONTAL")]
pub fn flip_horizontal() -> i32 {
    1
}
#[wasm_bindgen(js_name = "FLIP_BOTH")]
pub fn flip_both() -> i32 {
    -1
}

// -- Rotate codes -----------------------------------------------------------

#[wasm_bindgen(js_name = "ROTATE_90_CLOCKWISE")]
pub fn rotate_90_clockwise() -> i32 {
    0
}
#[wasm_bindgen(js_name = "ROTATE_180")]
pub fn rotate_180() -> i32 {
    1
}
#[wasm_bindgen(js_name = "ROTATE_90_COUNTERCLOCKWISE")]
pub fn rotate_90_counterclockwise() -> i32 {
    2
}

// ---------------------------------------------------------------------------
//  Morphological operations (require u8 depth)
// ---------------------------------------------------------------------------

/// Helper to convert a JS integer into a `MorphShapes`.
fn morph_shape_from_i32(s: i32) -> Result<MorphShapes, JsError> {
    match s {
        0 => Ok(MorphShapes::Rect),
        1 => Ok(MorphShapes::Cross),
        2 => Ok(MorphShapes::Ellipse),
        _ => Err(JsError::new(&format!("Unknown morph shape: {s}"))),
    }
}

/// Helper to convert a JS integer into a `MorphTypes`.
fn morph_type_from_i32(t: i32) -> Result<MorphTypes, JsError> {
    match t {
        0 => Ok(MorphTypes::Erode),
        1 => Ok(MorphTypes::Dilate),
        2 => Ok(MorphTypes::Open),
        3 => Ok(MorphTypes::Close),
        4 => Ok(MorphTypes::Gradient),
        5 => Ok(MorphTypes::TopHat),
        6 => Ok(MorphTypes::BlackHat),
        _ => Err(JsError::new(&format!("Unknown morph type: {t}"))),
    }
}

/// Creates a structuring element for morphological operations.
///
/// * `shape`   – 0 = RECT, 1 = CROSS, 2 = ELLIPSE.
/// * `ksize_w` – Kernel width.
/// * `ksize_h` – Kernel height.
///
/// Returns a single-channel u8 Mat (0/1 values).
#[wasm_bindgen(js_name = "getStructuringElement")]
pub fn get_structuring_element(shape: i32, ksize_w: usize, ksize_h: usize) -> Result<Mat, JsError> {
    let s = morph_shape_from_i32(shape)?;
    let ksize = purecv::core::types::Size::new(ksize_w, ksize_h);
    let anchor = purecv::core::types::Point::new(-1_i32, -1_i32);
    let kernel = morph::get_structuring_element(s, ksize, anchor)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::U8(kernel),
        },
    })
}

/// Erodes an image using a structuring element.
///
/// * `src`         – Input image (u8 depth).
/// * `kernel`      – Structuring element (from `getStructuringElement`).
/// * `iterations`  – Number of times erosion is applied.
/// * `border_type` – Border interpolation (integer).
#[wasm_bindgen(js_name = "erode")]
pub fn wasm_erode(
    src: &Mat,
    kernel: &Mat,
    iterations: usize,
    border_type: i32,
) -> Result<Mat, JsError> {
    let m = require_u8(src, "erode")?;
    let k = require_u8(kernel, "erode (kernel)")?;
    let bt = border_type_from_i32(border_type)?;
    let anchor = purecv::core::types::Point::new(-1_i32, -1_i32);
    let result =
        morph::erode(m, k, anchor, iterations, bt).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::U8(result),
        },
    })
}

/// Dilates an image using a structuring element.
///
/// * `src`         – Input image (u8 depth).
/// * `kernel`      – Structuring element (from `getStructuringElement`).
/// * `iterations`  – Number of times dilation is applied.
/// * `border_type` – Border interpolation (integer).
#[wasm_bindgen(js_name = "dilate")]
pub fn wasm_dilate(
    src: &Mat,
    kernel: &Mat,
    iterations: usize,
    border_type: i32,
) -> Result<Mat, JsError> {
    let m = require_u8(src, "dilate")?;
    let k = require_u8(kernel, "dilate (kernel)")?;
    let bt = border_type_from_i32(border_type)?;
    let anchor = purecv::core::types::Point::new(-1_i32, -1_i32);
    let result =
        morph::dilate(m, k, anchor, iterations, bt).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::U8(result),
        },
    })
}

/// Performs advanced morphological transformations.
///
/// * `src`         – Input image (u8 depth).
/// * `op`          – 0=ERODE, 1=DILATE, 2=OPEN, 3=CLOSE,
///                   4=GRADIENT, 5=TOPHAT, 6=BLACKHAT.
/// * `kernel`      – Structuring element.
/// * `iterations`  – Number of times the base operation is applied.
/// * `border_type` – Border interpolation (integer).
#[wasm_bindgen(js_name = "morphologyEx")]
pub fn wasm_morphology_ex(
    src: &Mat,
    op: i32,
    kernel: &Mat,
    iterations: usize,
    border_type: i32,
) -> Result<Mat, JsError> {
    let m = require_u8(src, "morphologyEx")?;
    let k = require_u8(kernel, "morphologyEx (kernel)")?;
    let mt = morph_type_from_i32(op)?;
    let bt = border_type_from_i32(border_type)?;
    let anchor = purecv::core::types::Point::new(-1_i32, -1_i32);
    let result = morph::morphology_ex(m, mt, k, anchor, iterations, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::U8(result),
        },
    })
}

// ---------------------------------------------------------------------------
//  Pyramid operations (require u8 depth)
// ---------------------------------------------------------------------------

/// Downsamples an image (Gaussian pyramid).
///
/// Output size is `((cols+1)/2, (rows+1)/2)`.
///
/// * `src`         – Input image (u8 depth).
/// * `border_type` – Border interpolation (integer).
#[wasm_bindgen(js_name = "pyrDown")]
pub fn wasm_pyr_down(src: &Mat, border_type: i32) -> Result<Mat, JsError> {
    let m = require_u8(src, "pyrDown")?;
    let bt = border_type_from_i32(border_type)?;
    let result = pyramid::pyr_down(m, None, bt).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::U8(result),
        },
    })
}

/// Upsamples an image (Gaussian pyramid).
///
/// Output size is `(cols*2, rows*2)`.
///
/// * `src`         – Input image (u8 depth).
/// * `border_type` – Border interpolation (integer).
#[wasm_bindgen(js_name = "pyrUp")]
pub fn wasm_pyr_up(src: &Mat, border_type: i32) -> Result<Mat, JsError> {
    let m = require_u8(src, "pyrUp")?;
    let bt = border_type_from_i32(border_type)?;
    let result = pyramid::pyr_up(m, None, bt).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::U8(result),
        },
    })
}

/// Constructs a Gaussian pyramid for an image.
///
/// * `src`         – Input image (u8 depth).
/// * `max_level`   – Maximum level (0-based).
/// * `border_type` – Border interpolation (integer).
///
/// Returns an array of Mats.
#[wasm_bindgen(js_name = "buildPyramid")]
pub fn wasm_build_pyramid(
    src: &Mat,
    max_level: usize,
    border_type: i32,
) -> Result<Vec<Mat>, JsError> {
    let m = require_u8(src, "buildPyramid")?;
    let bt = border_type_from_i32(border_type)?;
    let levels =
        pyramid::build_pyramid(m, max_level, bt).map_err(|e| JsError::new(&format!("{e}")))?;

    Ok(levels
        .into_iter()
        .map(|lvl| Mat {
            inner: DynamicMatrix {
                data: DynamicData::U8(lvl),
            },
        })
        .collect())
}

// ---------------------------------------------------------------------------
//  Hough Transform operations (require u8 depth)
// ---------------------------------------------------------------------------

/// Standard Hough Transform for line detection.
///
/// Input must be a single-channel u8 binary image (e.g. Canny output).
///
/// * `rho`       – Distance resolution of the accumulator in pixels.
/// * `theta`     – Angle resolution of the accumulator in radians.
/// * `threshold` – Accumulator threshold; only lines with enough votes are returned.
/// * `min_theta` – Minimum angle to check for lines (radians).
/// * `max_theta` – Maximum angle to check for lines (radians).
///
/// Returns a `Float32Array` of flattened `[rho, theta, rho, theta, ...]` pairs.
#[wasm_bindgen(js_name = "houghLines")]
pub fn wasm_hough_lines(
    src: &Mat,
    rho: f64,
    theta: f64,
    threshold: i32,
    min_theta: f64,
    max_theta: f64,
) -> Result<Vec<f32>, JsError> {
    let m = require_u8(src, "houghLines")?;
    let lines = hough::hough_lines(m, rho, theta, threshold, min_theta, max_theta)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    // Flatten Vec<[f32; 2]> → Vec<f32>
    Ok(lines.into_iter().flat_map(|l| l.into_iter()).collect())
}

/// Probabilistic Hough Transform for line segment detection.
///
/// Input must be a single-channel u8 binary image (e.g. Canny output).
///
/// * `rho`             – Distance resolution of the accumulator in pixels.
/// * `theta`           – Angle resolution of the accumulator in radians.
/// * `threshold`       – Accumulator threshold.
/// * `min_line_length` – Minimum line length; shorter segments are rejected.
/// * `max_line_gap`    – Maximum allowed gap between points on the same line.
///
/// Returns an `Int32Array` of flattened `[x1, y1, x2, y2, ...]` quadruples.
#[wasm_bindgen(js_name = "houghLinesP")]
pub fn wasm_hough_lines_p(
    src: &Mat,
    rho: f64,
    theta: f64,
    threshold: i32,
    min_line_length: f64,
    max_line_gap: f64,
) -> Result<Vec<i32>, JsError> {
    let m = require_u8(src, "houghLinesP")?;
    let segments = hough::hough_lines_p(m, rho, theta, threshold, min_line_length, max_line_gap)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    // Flatten Vec<[i32; 4]> → Vec<i32>
    Ok(segments.into_iter().flat_map(|s| s.into_iter()).collect())
}

/// Hough Circle Transform (gradient method).
///
/// Input must be a single-channel u8 grayscale image.
///
/// * `dp`         – Inverse ratio of the accumulator resolution to the image resolution.
/// * `min_dist`   – Minimum distance between the centres of detected circles.
/// * `param1`     – Higher Canny threshold (gradient magnitude threshold).
/// * `param2`     – Accumulator threshold for circle centres.
/// * `min_radius` – Minimum circle radius.
/// * `max_radius` – Maximum circle radius.
///
/// Returns a `Float32Array` of flattened `[cx, cy, r, cx, cy, r, ...]` triples.
#[wasm_bindgen(js_name = "houghCircles")]
pub fn wasm_hough_circles(
    src: &Mat,
    dp: f64,
    min_dist: f64,
    param1: f64,
    param2: f64,
    min_radius: i32,
    max_radius: i32,
) -> Result<Vec<f32>, JsError> {
    let m = require_u8(src, "houghCircles")?;
    let circles = hough::hough_circles(m, dp, min_dist, param1, param2, min_radius, max_radius)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    // Flatten Vec<[f32; 3]> → Vec<f32>
    Ok(circles.into_iter().flat_map(|c| c.into_iter()).collect())
}

// ---------------------------------------------------------------------------
//  Feature detection operations (require f32 depth)
// ---------------------------------------------------------------------------

/// Calculates eigenvalues and eigenvectors of image blocks for corner detection.
///
/// Returns a 6-channel f32 Mat: (λ1, λ2, x1, y1, x2, y2) per pixel.
///
/// * `block_size`  – Neighbourhood size (positive odd integer).
/// * `ksize`       – Aperture size for the Sobel operator (3, 5, or −1 for Scharr).
/// * `border_type` – Border interpolation (integer).
#[wasm_bindgen(js_name = "cornerEigenValsAndVecs")]
pub fn wasm_corner_eigen_vals_and_vecs(
    src: &Mat,
    block_size: i32,
    ksize: i32,
    border_type: i32,
) -> Result<Mat, JsError> {
    let m = require_f32(src, "cornerEigenValsAndVecs")?;
    let bt = border_type_from_i32(border_type)?;
    let result = feature::corner_eigen_vals_and_vecs(m, block_size, ksize, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Calculates the minimal eigenvalue of gradient covariance matrices (Shi-Tomasi).
///
/// Returns a single-channel f32 Mat.
///
/// * `block_size`  – Neighbourhood size (positive odd integer).
/// * `ksize`       – Aperture size for the Sobel operator.
/// * `border_type` – Border interpolation (integer).
#[wasm_bindgen(js_name = "cornerMinEigenVal")]
pub fn wasm_corner_min_eigen_val(
    src: &Mat,
    block_size: i32,
    ksize: i32,
    border_type: i32,
) -> Result<Mat, JsError> {
    let m = require_f32(src, "cornerMinEigenVal")?;
    let bt = border_type_from_i32(border_type)?;
    let result = feature::corner_min_eigen_val(m, block_size, ksize, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Harris corner detector.
///
/// Returns a single-channel f32 Mat with the Harris response.
///
/// * `block_size`  – Neighbourhood size (positive odd integer).
/// * `ksize`       – Aperture size for the Sobel operator.
/// * `k`           – Harris detector free parameter (typically 0.04–0.06).
/// * `border_type` – Border interpolation (integer).
#[wasm_bindgen(js_name = "cornerHarris")]
pub fn wasm_corner_harris(
    src: &Mat,
    block_size: i32,
    ksize: i32,
    k: f64,
    border_type: i32,
) -> Result<Mat, JsError> {
    let m = require_f32(src, "cornerHarris")?;
    let bt = border_type_from_i32(border_type)?;
    let result = feature::corner_harris(m, block_size, ksize, k, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Determines strong corners on an image (Shi-Tomasi / Harris).
///
/// Returns a `Float32Array` of flattened `[x, y, x, y, ...]` pairs.
///
/// * `max_corners`          – Maximum number of corners (≤ 0 = unlimited).
/// * `quality_level`        – Fraction of best response below which corners are rejected.
/// * `min_distance`         – Minimum Euclidean distance between returned corners.
/// * `block_size`           – Neighbourhood size for the structure tensor.
/// * `use_harris_detector`  – Use Harris (true) or min eigenvalue (false).
/// * `harris_k`             – Harris free parameter (only used when `use_harris_detector`).
#[wasm_bindgen(js_name = "goodFeaturesToTrack")]
pub fn wasm_good_features_to_track(
    src: &Mat,
    max_corners: i32,
    quality_level: f64,
    min_distance: f64,
    block_size: i32,
    use_harris_detector: bool,
    harris_k: f64,
) -> Result<Vec<f32>, JsError> {
    let m = require_f32(src, "goodFeaturesToTrack")?;
    let corners = feature::good_features_to_track(
        m,
        max_corners,
        quality_level,
        min_distance,
        block_size,
        use_harris_detector,
        harris_k,
    )
    .map_err(|e| JsError::new(&format!("{e}")))?;
    // Flatten Vec<Point2f> → Vec<f32> [x, y, x, y, ...]
    Ok(corners.iter().flat_map(|p| [p.x, p.y]).collect())
}

/// Refines corner locations to sub-pixel accuracy.
///
/// Takes a `Float32Array` of corner coordinates `[x, y, x, y, ...]` and returns
/// the refined coordinates in the same format.
///
/// * `corners`    – Flattened corner coordinates (must have even length).
/// * `win_w`      – Half-width of the search window.
/// * `win_h`      – Half-height of the search window.
/// * `zero_w`     – Half-width of the dead zone (−1 to disable).
/// * `zero_h`     – Half-height of the dead zone (−1 to disable).
/// * `max_count`  – Maximum number of iterations.
/// * `epsilon`    – Convergence threshold.
#[wasm_bindgen(js_name = "cornerSubPix")]
#[allow(clippy::too_many_arguments)]
pub fn wasm_corner_sub_pix(
    src: &Mat,
    corners: &[f32],
    win_w: i32,
    win_h: i32,
    zero_w: i32,
    zero_h: i32,
    max_count: i32,
    epsilon: f64,
) -> Result<Vec<f32>, JsError> {
    use purecv::core::types::{Point2f, Size2i, TermCriteria, TermType};

    let m = require_f32(src, "cornerSubPix")?;
    if !corners.len().is_multiple_of(2) {
        return Err(JsError::new(
            "corners array must have even length (x, y pairs)",
        ));
    }

    // Reconstruct Vec<Point2f> from flattened array.
    let mut pts: Vec<Point2f> = corners
        .chunks_exact(2)
        .map(|c| Point2f::new(c[0], c[1]))
        .collect();

    let win_size = Size2i::new(win_w, win_h);
    let zero_zone = Size2i::new(zero_w, zero_h);
    let criteria = TermCriteria {
        type_: TermType::Both,
        max_count,
        epsilon,
    };

    feature::corner_sub_pix(m, &mut pts, win_size, zero_zone, criteria)
        .map_err(|e| JsError::new(&format!("{e}")))?;

    // Flatten back to [x, y, x, y, ...]
    Ok(pts.iter().flat_map(|p| [p.x, p.y]).collect())
}

/// Calculates a feature map for corner detection.
///
/// Returns a single-channel f32 Mat.
///
/// * `ksize`       – Aperture size for the Sobel operator.
/// * `border_type` – Border interpolation (integer).
#[wasm_bindgen(js_name = "preCornerDetect")]
pub fn wasm_pre_corner_detect(src: &Mat, ksize: i32, border_type: i32) -> Result<Mat, JsError> {
    let m = require_f32(src, "preCornerDetect")?;
    let bt = border_type_from_i32(border_type)?;
    let result =
        feature::pre_corner_detect(m, ksize, bt).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

// ---------------------------------------------------------------------------
//  JS-side enum constants: Morph shapes
// ---------------------------------------------------------------------------

#[wasm_bindgen(js_name = "MORPH_RECT")]
pub fn morph_rect() -> i32 {
    0
}
#[wasm_bindgen(js_name = "MORPH_CROSS")]
pub fn morph_cross() -> i32 {
    1
}
#[wasm_bindgen(js_name = "MORPH_ELLIPSE")]
pub fn morph_ellipse() -> i32 {
    2
}

// ---------------------------------------------------------------------------
//  JS-side enum constants: Morph types
// ---------------------------------------------------------------------------

#[wasm_bindgen(js_name = "MORPH_ERODE")]
pub fn morph_erode() -> i32 {
    0
}
#[wasm_bindgen(js_name = "MORPH_DILATE")]
pub fn morph_dilate_op() -> i32 {
    1
}
#[wasm_bindgen(js_name = "MORPH_OPEN")]
pub fn morph_open() -> i32 {
    2
}
#[wasm_bindgen(js_name = "MORPH_CLOSE")]
pub fn morph_close() -> i32 {
    3
}
#[wasm_bindgen(js_name = "MORPH_GRADIENT")]
pub fn morph_gradient() -> i32 {
    4
}
#[wasm_bindgen(js_name = "MORPH_TOPHAT")]
pub fn morph_tophat() -> i32 {
    5
}
#[wasm_bindgen(js_name = "MORPH_BLACKHAT")]
pub fn morph_blackhat() -> i32 {
    6
}

// ---------------------------------------------------------------------------
//  Video — Optical flow
// ---------------------------------------------------------------------------

/// Builds a Gaussian image pyramid suitable for Lucas-Kanade optical flow.
///
/// The input Mat must be single-channel `u8` (CV_8UC1).  Returns a new `Mat`
/// containing the **flattened** pyramid levels concatenated row-wise into a
/// single-channel `f32` Mat.  A companion `Float32Array` metadata is returned
/// via `buildOpticalFlowPyramidInfo()` so JS can slice the blob back into
/// individual levels.
///
/// In practice you rarely call this directly — `calcOpticalFlowPyrLK` builds
/// pyramids internally.  This is exposed for advanced use-cases (e.g. pyramid
/// visualisation or caching).
///
/// * `win_w`, `win_h` – Tracking window size.
/// * `max_level`      – Maximum number of additional pyramid levels.
/// * `with_derivatives`– Compute Sobel derivatives alongside the pyramid.
/// * `pyr_border`     – Border interpolation for downsampling.
/// * `deriv_border`   – Border interpolation for derivatives.
///
/// Returns a JS object `{ levelCount, rows[], cols[] }` via `serde`.
#[wasm_bindgen(js_name = "buildOpticalFlowPyramid")]
pub fn wasm_build_optical_flow_pyramid(
    src: &Mat,
    win_w: i32,
    win_h: i32,
    max_level: usize,
    with_derivatives: bool,
    pyr_border: i32,
    deriv_border: i32,
) -> Result<JsValue, JsError> {
    let m = require_u8(src, "buildOpticalFlowPyramid")?;
    let pb = border_type_from_i32(pyr_border)?;
    let db = border_type_from_i32(deriv_border)?;
    let pyr = optical_flow::build_optical_flow_pyramid(
        m,
        Size2i::new(win_w, win_h),
        max_level,
        with_derivatives,
        pb,
        db,
    )
    .map_err(|e| JsError::new(&format!("{e}")))?;

    // Serialize metadata so JS knows the level dimensions.
    let level_count = pyr.levels.len();
    let rows: Vec<usize> = pyr.levels.iter().map(|l| l.rows).collect();
    let cols: Vec<usize> = pyr.levels.iter().map(|l| l.cols).collect();

    // Build a simple JS object { levelCount, rows, cols }.
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("levelCount"),
        &JsValue::from(level_count as u32),
    )
    .map_err(|_| JsError::new("Failed to set levelCount"))?;
    let js_rows = js_sys::Array::new();
    for r in &rows {
        js_rows.push(&JsValue::from(*r as u32));
    }
    js_sys::Reflect::set(&obj, &JsValue::from_str("rows"), &js_rows)
        .map_err(|_| JsError::new("Failed to set rows"))?;
    let js_cols = js_sys::Array::new();
    for c in &cols {
        js_cols.push(&JsValue::from(*c as u32));
    }
    js_sys::Reflect::set(&obj, &JsValue::from_str("cols"), &js_cols)
        .map_err(|_| JsError::new("Failed to set cols"))?;

    Ok(obj.into())
}

/// Calculates the sparse optical flow using the iterative pyramidal
/// Lucas-Kanade method.
///
/// Both `prev` and `next` must be single-channel `u8` Mats (CV_8UC1) of the
/// same dimensions.  `prev_pts` is a `Float32Array` of `[x, y, x, y, ...]`
/// pairs — the same format returned by `goodFeaturesToTrack()`.
///
/// Returns a JS object `{ nextPts: Float32Array, status: Uint8Array, err: Float32Array }`.
///
/// * `win_w`, `win_h` – Search window size at each pyramid level.
/// * `max_level`      – Pyramid depth (0 = original only).
/// * `max_count`      – Max iterations of the LK solver.
/// * `epsilon`        – Convergence threshold.
/// * `flags`          – Combine `OPTFLOW_USE_INITIAL_FLOW()` and/or
///   `OPTFLOW_LK_GET_MIN_EIGENVALS()`.
/// * `min_eigen_threshold` – Min eigenvalue below which a point is lost.
///
/// ```js
/// const gray0 = Mat.fromU8Data(h, w, 1, frameData0);
/// const gray1 = Mat.fromU8Data(h, w, 1, frameData1);
/// const pts   = goodFeaturesToTrack(gray0, 100, 0.01, 10, 3, false, 0.04);
/// const result = calcOpticalFlowPyrLK(
///     gray0, gray1, pts, win_w, win_h, 3, 30, 0.01,
///     OPTFLOW_LK_GET_MIN_EIGENVALS(), 1e-4,
/// );
/// // result.nextPts — Float32Array [x,y,x,y,...]
/// // result.status  — Uint8Array  [1,1,0,...]
/// // result.err     — Float32Array
/// ```
#[wasm_bindgen(js_name = "calcOpticalFlowPyrLK")]
#[allow(clippy::too_many_arguments)]
pub fn wasm_calc_optical_flow_pyr_lk(
    prev: &Mat,
    next: &Mat,
    prev_pts: &[f32],
    win_w: i32,
    win_h: i32,
    max_level: i32,
    max_count: i32,
    epsilon: f64,
    flags: i32,
    min_eigen_threshold: f64,
) -> Result<JsValue, JsError> {
    use purecv::core::types::{Point2f, TermCriteria, TermType};

    let prev_m = require_u8(prev, "calcOpticalFlowPyrLK")?;
    let next_m = require_u8(next, "calcOpticalFlowPyrLK")?;

    if !prev_pts.len().is_multiple_of(2) {
        return Err(JsError::new("prev_pts must have even length (x, y pairs)"));
    }

    // Reconstruct Vec<Point2f> from the flattened array.
    let pts: Vec<Point2f> = prev_pts
        .chunks_exact(2)
        .map(|c| Point2f::new(c[0], c[1]))
        .collect();

    let criteria = TermCriteria {
        type_: TermType::Both,
        max_count,
        epsilon,
    };

    let (next_pts, status, err) = optical_flow::calc_optical_flow_pyramid_lk(
        prev_m,
        next_m,
        &pts,
        None, // initial_next_pts — TODO: expose when OPTFLOW_USE_INITIAL_FLOW is needed
        Size2i::new(win_w, win_h),
        max_level,
        criteria,
        flags,
        min_eigen_threshold,
    )
    .map_err(|e| JsError::new(&format!("{e}")))?;

    // Build the JS result object { nextPts, status, err }.
    let obj = js_sys::Object::new();

    // Flatten next_pts → Float32Array
    let flat_pts: Vec<f32> = next_pts.iter().flat_map(|p| [p.x, p.y]).collect();
    let js_pts = js_sys::Float32Array::from(flat_pts.as_slice());
    js_sys::Reflect::set(&obj, &JsValue::from_str("nextPts"), &js_pts)
        .map_err(|_| JsError::new("Failed to set nextPts"))?;

    // status → Uint8Array
    let js_status = js_sys::Uint8Array::from(status.as_slice());
    js_sys::Reflect::set(&obj, &JsValue::from_str("status"), &js_status)
        .map_err(|_| JsError::new("Failed to set status"))?;

    // err → Float32Array
    let js_err = js_sys::Float32Array::from(err.as_slice());
    js_sys::Reflect::set(&obj, &JsValue::from_str("err"), &js_err)
        .map_err(|_| JsError::new("Failed to set err"))?;

    Ok(obj.into())
}

// ---------------------------------------------------------------------------
//  JS-side enum constants: Optical flow flags
// ---------------------------------------------------------------------------

/// Use initial estimates supplied via `initial_next_pts`.
#[wasm_bindgen(js_name = "OPTFLOW_USE_INITIAL_FLOW")]
pub fn optflow_use_initial_flow() -> i32 {
    optical_flow::OPTFLOW_USE_INITIAL_FLOW
}

/// Return minimum eigenvalue instead of mean-absolute-error in the `err` array.
#[wasm_bindgen(js_name = "OPTFLOW_LK_GET_MIN_EIGENVALS")]
pub fn optflow_lk_get_min_eigenvals() -> i32 {
    optical_flow::OPTFLOW_LK_GET_MIN_EIGENVALS
}

// ---------------------------------------------------------------------------
//  Calib3d (Pose Estimation & Homography)
// ---------------------------------------------------------------------------

use purecv::calib3d::geometry::rodrigues;
use purecv::calib3d::homography::{find_homography, HomographyMethod};
use purecv::calib3d::pose::{solve_pnp, solve_pnp_ransac, SolvePnPMethod};

#[wasm_bindgen]
pub struct Point2fVector {
    pub(crate) inner: Vec<purecv::core::types::Point2f>,
}

#[wasm_bindgen]
impl Point2fVector {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    pub fn push(&mut self, x: f32, y: f32) {
        self.inner.push(purecv::core::types::Point2f::new(x, y));
    }

    pub fn size(&self) -> usize {
        self.inner.len()
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl Default for Point2fVector {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
pub struct Point3fVector {
    pub(crate) inner: Vec<purecv::core::types::Point3f>,
}

#[wasm_bindgen]
impl Point3fVector {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    pub fn push(&mut self, x: f32, y: f32, z: f32) {
        self.inner.push(purecv::core::types::Point3f::new(x, y, z));
    }

    pub fn size(&self) -> usize {
        self.inner.len()
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl Default for Point3fVector {
    fn default() -> Self {
        Self::new()
    }
}

fn map_solve_pnp_method(val: i32) -> SolvePnPMethod {
    match val {
        1 => SolvePnPMethod::EPnP,
        2 => SolvePnPMethod::P3P,
        5 => SolvePnPMethod::AP3P,
        6 => SolvePnPMethod::SqPnP,
        _ => SolvePnPMethod::Iterative,
    }
}

fn map_homography_method(val: i32) -> HomographyMethod {
    match val {
        4 => HomographyMethod::LMedS,
        8 => HomographyMethod::Ransac,
        16 => HomographyMethod::Rho,
        _ => HomographyMethod::None,
    }
}

#[wasm_bindgen(js_name = "solvePnP")]
#[allow(clippy::too_many_arguments)]
pub fn solve_pnp_wasm(
    object_points: &Point3fVector,
    image_points: &Point2fVector,
    camera_matrix: &Mat,
    dist_coeffs: Option<Vec<f64>>,
    rvec: &mut Mat,
    tvec: &mut Mat,
    use_extrinsic_guess: bool,
    flags: i32,
) -> Result<bool, JsError> {
    let cam_mat = require_f64(camera_matrix, "solvePnP")?;
    let r_mat = require_f64_mut(rvec, "solvePnP (rvec)")?;
    let t_mat = require_f64_mut(tvec, "solvePnP (tvec)")?;

    let dist_ref = dist_coeffs.as_deref();

    solve_pnp(
        &object_points.inner,
        &image_points.inner,
        cam_mat,
        dist_ref,
        r_mat,
        t_mat,
        use_extrinsic_guess,
        map_solve_pnp_method(flags),
    )
    .map_err(|e| JsError::new(&format!("{e}")))
}

#[wasm_bindgen(js_name = "solvePnPRansac")]
#[allow(clippy::too_many_arguments)]
pub fn solve_pnp_ransac_wasm(
    object_points: &Point3fVector,
    image_points: &Point2fVector,
    camera_matrix: &Mat,
    dist_coeffs: Option<Vec<f64>>,
    rvec: &mut Mat,
    tvec: &mut Mat,
    use_extrinsic_guess: bool,
    iterations_count: i32,
    reproj_threshold: f32,
    confidence: f64,
    flags: i32,
) -> Result<JsValue, JsError> {
    let cam_mat = require_f64(camera_matrix, "solvePnPRansac")?;
    let r_mat = require_f64_mut(rvec, "solvePnPRansac (rvec)")?;
    let t_mat = require_f64_mut(tvec, "solvePnPRansac (tvec)")?;

    let dist_ref = dist_coeffs.as_deref();
    let mut inliers_vec = Vec::new();

    let success = solve_pnp_ransac(
        &object_points.inner,
        &image_points.inner,
        cam_mat,
        dist_ref,
        r_mat,
        t_mat,
        use_extrinsic_guess,
        iterations_count,
        reproj_threshold,
        confidence,
        Some(&mut inliers_vec),
        map_solve_pnp_method(flags),
    )
    .map_err(|e| JsError::new(&format!("{e}")))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("success"),
        &JsValue::from_bool(success),
    )
    .map_err(|_| JsError::new("Failed to set success"))?;

    let js_inliers = js_sys::Int32Array::from(inliers_vec.as_slice());
    js_sys::Reflect::set(&obj, &JsValue::from_str("inliers"), &js_inliers)
        .map_err(|_| JsError::new("Failed to set inliers"))?;

    Ok(obj.into())
}

#[wasm_bindgen(js_name = "findHomography")]
pub fn find_homography_wasm(
    src_points: &Point2fVector,
    dst_points: &Point2fVector,
    method: i32,
    ransac_reproj_threshold: f64,
) -> Result<JsValue, JsError> {
    let mut mask_vec = Vec::new();
    let h_mat = find_homography(
        &src_points.inner,
        &dst_points.inner,
        map_homography_method(method),
        ransac_reproj_threshold,
        Some(&mut mask_vec),
    )
    .map_err(|e| JsError::new(&format!("{e}")))?;

    let obj = js_sys::Object::new();
    let js_h = Mat {
        inner: purecv::core::dynamic::DynamicMatrix {
            data: purecv::core::dynamic::DynamicData::F64(h_mat),
        },
    };
    js_sys::Reflect::set(&obj, &JsValue::from_str("homography"), &JsValue::from(js_h))
        .map_err(|_| JsError::new("Failed to set homography"))?;

    let js_mask = js_sys::Uint8Array::from(mask_vec.as_slice());
    js_sys::Reflect::set(&obj, &JsValue::from_str("mask"), &js_mask)
        .map_err(|_| JsError::new("Failed to set mask"))?;

    Ok(obj.into())
}

#[wasm_bindgen(js_name = "rodrigues")]
pub fn rodrigues_wasm(src: &Mat, dst: &mut Mat) -> Result<(), JsError> {
    let src_mat = require_f64(src, "rodrigues (src)")?;
    let dst_mat = require_f64_mut(dst, "rodrigues (dst)")?;
    rodrigues(src_mat, dst_mat).map_err(|e| JsError::new(&format!("{e}")))
}

// ---------------------------------------------------------------------------
//  features2d — KeyPoint, FAST, and ORB bindings
// ---------------------------------------------------------------------------

fn fast_type_from_i32(val: i32) -> Result<CoreFastType, JsError> {
    match val {
        0 => Ok(CoreFastType::Type5_8),
        1 => Ok(CoreFastType::Type7_12),
        2 => Ok(CoreFastType::Type9_16),
        _ => Err(JsError::new(&format!("Invalid FAST type value: {val}"))),
    }
}

#[wasm_bindgen(js_name = "FAST_TYPE_5_8")]
pub fn fast_type_5_8() -> i32 {
    0
}

#[wasm_bindgen(js_name = "FAST_TYPE_7_12")]
pub fn fast_type_7_12() -> i32 {
    1
}

#[wasm_bindgen(js_name = "FAST_TYPE_9_16")]
pub fn fast_type_9_16() -> i32 {
    2
}

fn score_type_from_i32(val: i32) -> Result<CoreScoreType, JsError> {
    match val {
        0 => Ok(CoreScoreType::Harris),
        1 => Ok(CoreScoreType::Fast),
        _ => Err(JsError::new(&format!("Invalid ScoreType value: {val}"))),
    }
}

#[wasm_bindgen(js_name = "ORB_SCORE_HARRIS")]
pub fn orb_score_harris() -> i32 {
    0
}

#[wasm_bindgen(js_name = "ORB_SCORE_FAST")]
pub fn orb_score_fast() -> i32 {
    1
}

/// JavaScript-facing KeyPoint struct.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub struct KeyPoint {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: i32,
    pub class_id: i32,
}

#[wasm_bindgen]
impl KeyPoint {
    #[wasm_bindgen(constructor)]
    pub fn new(
        x: f32,
        y: f32,
        size: f32,
        angle: f32,
        response: f32,
        octave: i32,
        class_id: i32,
    ) -> Self {
        Self {
            x,
            y,
            size,
            angle,
            response,
            octave,
            class_id,
        }
    }
}

impl From<CoreKeyPoint> for KeyPoint {
    fn from(kp: CoreKeyPoint) -> Self {
        Self {
            x: kp.pt.x,
            y: kp.pt.y,
            size: kp.size,
            angle: kp.angle,
            response: kp.response,
            octave: kp.octave,
            class_id: kp.class_id,
        }
    }
}

impl From<KeyPoint> for CoreKeyPoint {
    fn from(kp: KeyPoint) -> Self {
        Self {
            pt: purecv::core::types::Point2f::new(kp.x, kp.y),
            size: kp.size,
            angle: kp.angle,
            response: kp.response,
            octave: kp.octave,
            class_id: kp.class_id,
        }
    }
}

/// Managed vector of KeyPoints exposed to JavaScript.
#[wasm_bindgen]
pub struct KeyPointVector {
    pub(crate) inner: Vec<CoreKeyPoint>,
}

#[wasm_bindgen]
impl KeyPointVector {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    pub fn push(&mut self, kp: &KeyPoint) {
        self.inner.push((*kp).into());
    }

    pub fn size(&self) -> usize {
        self.inner.len()
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }

    pub fn get(&self, idx: usize) -> Option<KeyPoint> {
        self.inner.get(idx).cloned().map(|kp| kp.into())
    }
}

impl Default for KeyPointVector {
    fn default() -> Self {
        Self::new()
    }
}

/// Standalone FAST corner detection.
#[wasm_bindgen(js_name = "FAST")]
pub fn fast(
    image: &Mat,
    threshold: i32,
    nonmax_suppression: bool,
    type_val: i32,
) -> Result<KeyPointVector, JsError> {
    let img = require_u8(image, "FAST")?;
    let f_type = fast_type_from_i32(type_val)?;
    let detector = CoreFastFeatureDetector::new(threshold as u8, nonmax_suppression, f_type);
    let kpts = detector
        .detect(img)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(KeyPointVector { inner: kpts })
}

/// Class wrapper for FAST feature detection.
#[wasm_bindgen]
pub struct FastFeatureDetector {
    inner: CoreFastFeatureDetector,
}

#[wasm_bindgen]
impl FastFeatureDetector {
    #[wasm_bindgen(constructor)]
    pub fn new(
        threshold: i32,
        nonmax_suppression: bool,
        type_val: i32,
    ) -> Result<FastFeatureDetector, JsError> {
        let f_type = fast_type_from_i32(type_val)?;
        Ok(Self {
            inner: CoreFastFeatureDetector::new(threshold as u8, nonmax_suppression, f_type),
        })
    }

    pub fn detect(&self, image: &Mat) -> Result<KeyPointVector, JsError> {
        let img = require_u8(image, "FastFeatureDetector::detect")?;
        let kpts = self
            .inner
            .detect(img)
            .map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(KeyPointVector { inner: kpts })
    }
}

/// Oriented FAST and Rotated BRIEF (ORB) detector and descriptor extractor.
#[wasm_bindgen]
pub struct ORB {
    inner: CoreOrb,
}

#[wasm_bindgen]
impl ORB {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nfeatures: usize,
        scale_factor: f32,
        nlevels: usize,
        edge_threshold: i32,
        first_level: usize,
        wta_k: usize,
        score_type_val: i32,
        patch_size: usize,
        fast_threshold: i32,
    ) -> Result<ORB, JsError> {
        let s_type = score_type_from_i32(score_type_val)?;
        Ok(Self {
            inner: CoreOrb::new(
                nfeatures,
                scale_factor,
                nlevels,
                edge_threshold,
                first_level,
                wta_k,
                s_type,
                patch_size,
                fast_threshold as u8,
            ),
        })
    }

    /// Creates an ORB instance with default parameters.
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self {
            inner: CoreOrb::default(),
        }
    }

    pub fn detect(&self, image: &Mat) -> Result<KeyPointVector, JsError> {
        let img = require_u8(image, "ORB::detect")?;
        let kpts = self
            .inner
            .detect(img)
            .map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(KeyPointVector { inner: kpts })
    }

    pub fn compute(&self, image: &Mat, keypoints: &KeyPointVector) -> Result<Mat, JsError> {
        let img = require_u8(image, "ORB::compute")?;
        let descs = self
            .inner
            .compute(img, &keypoints.inner)
            .map_err(|e| JsError::new(&format!("{e}")))?;

        Ok(Mat {
            inner: DynamicMatrix {
                data: DynamicData::U8(descs),
            },
        })
    }

    /// Detects keypoints and computes their descriptors in one pass, returning
    /// a JavaScript object: `{ keypoints: KeyPointVector, descriptors: Mat }`.
    #[wasm_bindgen(js_name = "detectAndCompute")]
    pub fn detect_and_compute(&self, image: &Mat) -> Result<JsValue, JsError> {
        let img = require_u8(image, "ORB::detectAndCompute")?;
        let (kpts, descs) = self
            .inner
            .detect_and_compute(img)
            .map_err(|e| JsError::new(&format!("{e}")))?;

        let kpts_vector = KeyPointVector { inner: kpts };
        let descs_mat = Mat {
            inner: DynamicMatrix {
                data: DynamicData::U8(descs),
            },
        };

        let obj = js_sys::Object::new();
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("keypoints"),
            &JsValue::from(kpts_vector),
        )
        .map_err(|_| JsError::new("Failed to set keypoints"))?;

        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("descriptors"),
            &JsValue::from(descs_mat),
        )
        .map_err(|_| JsError::new("Failed to set descriptors"))?;

        Ok(obj.into())
    }
}
