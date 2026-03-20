/*
 *  lib.rs
 *  purecv-wasm
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

use wasm_bindgen::prelude::*;

use purecv::core::arithm;
use purecv::core::structural;
use purecv::core::types::{BorderTypes, Point2i, Size2i};
use purecv::core::Matrix;
use purecv::imgproc::color::{cvt_color, ColorConversionCode};
use purecv::imgproc::derivatives;
use purecv::imgproc::edge;
use purecv::imgproc::filter;
use purecv::imgproc::threshold::{threshold, ThresholdTypes};
use purecv::version;

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
//  PureCvMatrixF32 — opaque wrapper around Matrix<f32>
// ---------------------------------------------------------------------------

/// An opaque wrapper around `Matrix<f32>`.
///
/// For the browser, `f32` is the natural numeric type and a good middle
/// ground between precision and performance for image‑processing operations.
/// Data flows JS → WASM as `Float32Array`, gets processed, and comes back
/// the same way.
#[wasm_bindgen]
pub struct PureCvMatrixF32 {
    inner: Matrix<f32>,
}

#[wasm_bindgen]
impl PureCvMatrixF32 {
    // -- Constructors -------------------------------------------------------

    /// Creates a new zero-filled matrix.
    ///
    /// * `rows`     – Number of rows (height).
    /// * `cols`     – Number of columns (width).
    /// * `channels` – Number of channels (e.g. 1 for gray, 3 for RGB, 4 for RGBA).
    #[wasm_bindgen(constructor)]
    pub fn new(rows: usize, cols: usize, channels: usize) -> PureCvMatrixF32 {
        PureCvMatrixF32 {
            inner: Matrix::<f32>::new(rows, cols, channels),
        }
    }

    /// Creates a matrix from a `Float32Array`.
    ///
    /// The array length **must** equal `rows × cols × channels`.
    #[wasm_bindgen(js_name = "fromData")]
    pub fn from_data(
        rows: usize,
        cols: usize,
        channels: usize,
        data: &[f32],
    ) -> Result<PureCvMatrixF32, JsError> {
        let expected = rows * cols * channels;
        if data.len() != expected {
            return Err(JsError::new(&format!(
                "Data length {} does not match {}×{}×{} = {}",
                data.len(),
                rows,
                cols,
                channels,
                expected
            )));
        }
        let mut mat = Matrix::<f32>::new(rows, cols, channels);
        mat.data.copy_from_slice(data);
        Ok(PureCvMatrixF32 { inner: mat })
    }

    // -- Accessors ----------------------------------------------------------

    /// Returns the number of rows (height).
    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.inner.rows
    }

    /// Returns the number of columns (width).
    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> usize {
        self.inner.cols
    }

    /// Returns the number of channels.
    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> usize {
        self.inner.channels
    }

    /// Returns the total number of elements (rows × cols × channels).
    #[wasm_bindgen(getter, js_name = "length")]
    pub fn length(&self) -> usize {
        self.inner.data.len()
    }

    /// Returns a copy of the underlying data as a `Float32Array`.
    #[wasm_bindgen(js_name = "data")]
    pub fn data(&self) -> Vec<f32> {
        self.inner.data.clone()
    }

    /// Sets the underlying data from a `Float32Array`.
    #[wasm_bindgen(js_name = "setData")]
    pub fn set_data(&mut self, data: &[f32]) -> Result<(), JsError> {
        if data.len() != self.inner.data.len() {
            return Err(JsError::new(&format!(
                "Data length {} does not match matrix length {}",
                data.len(),
                self.inner.data.len()
            )));
        }
        self.inner.data.copy_from_slice(data);
        Ok(())
    }

    /// Returns the value at (row, col, channel).
    #[wasm_bindgen(js_name = "at")]
    pub fn at(&self, row: i32, col: i32, channel: usize) -> Option<f32> {
        self.inner.at(row, col, channel).copied()
    }
}

// ---------------------------------------------------------------------------
//  PureCvMatrixU8 — opaque wrapper around Matrix<u8>
// ---------------------------------------------------------------------------

/// An opaque wrapper around `Matrix<u8>`.
///
/// Used for operations that operate on or produce 8-bit images: colour
/// conversions, Canny edge detection, thresholding of byte images, etc.
/// Data flows JS → WASM as `Uint8Array` / `Uint8ClampedArray`.
#[wasm_bindgen]
pub struct PureCvMatrixU8 {
    inner: Matrix<u8>,
}

#[wasm_bindgen]
impl PureCvMatrixU8 {
    // -- Constructors -------------------------------------------------------

    /// Creates a new zero-filled u8 matrix.
    #[wasm_bindgen(constructor)]
    pub fn new(rows: usize, cols: usize, channels: usize) -> PureCvMatrixU8 {
        PureCvMatrixU8 {
            inner: Matrix::<u8>::new(rows, cols, channels),
        }
    }

    /// Creates a u8 matrix from a `Uint8Array`.
    #[wasm_bindgen(js_name = "fromData")]
    pub fn from_data(
        rows: usize,
        cols: usize,
        channels: usize,
        data: &[u8],
    ) -> Result<PureCvMatrixU8, JsError> {
        let expected = rows * cols * channels;
        if data.len() != expected {
            return Err(JsError::new(&format!(
                "Data length {} does not match {}×{}×{} = {}",
                data.len(),
                rows,
                cols,
                channels,
                expected
            )));
        }
        let mut mat = Matrix::<u8>::new(rows, cols, channels);
        mat.data.copy_from_slice(data);
        Ok(PureCvMatrixU8 { inner: mat })
    }

    // -- Accessors ----------------------------------------------------------

    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.inner.rows
    }

    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> usize {
        self.inner.cols
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> usize {
        self.inner.channels
    }

    #[wasm_bindgen(getter, js_name = "length")]
    pub fn length(&self) -> usize {
        self.inner.data.len()
    }

    /// Returns a copy of the underlying data as a `Uint8Array`.
    #[wasm_bindgen(js_name = "data")]
    pub fn data(&self) -> Vec<u8> {
        self.inner.data.clone()
    }

    /// Sets the underlying data from a `Uint8Array`.
    #[wasm_bindgen(js_name = "setData")]
    pub fn set_data(&mut self, data: &[u8]) -> Result<(), JsError> {
        if data.len() != self.inner.data.len() {
            return Err(JsError::new(&format!(
                "Data length {} does not match matrix length {}",
                data.len(),
                self.inner.data.len()
            )));
        }
        self.inner.data.copy_from_slice(data);
        Ok(())
    }

    /// Returns the value at (row, col, channel).
    #[wasm_bindgen(js_name = "at")]
    pub fn at(&self, row: i32, col: i32, channel: usize) -> Option<u8> {
        self.inner.at(row, col, channel).copied()
    }
}

// ---------------------------------------------------------------------------
//  Type conversion helpers
// ---------------------------------------------------------------------------

/// Converts a `PureCvMatrixU8` to a `PureCvMatrixF32` (u8 → f32).
#[wasm_bindgen(js_name = "convertU8ToF32")]
pub fn convert_u8_to_f32(src: &PureCvMatrixU8) -> Result<PureCvMatrixF32, JsError> {
    let result = src
        .inner
        .convert_to::<f32>()
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

/// Converts a `PureCvMatrixF32` to a `PureCvMatrixU8` (f32 → u8, values clamped to 0–255).
#[wasm_bindgen(js_name = "convertF32ToU8")]
pub fn convert_f32_to_u8(src: &PureCvMatrixF32) -> Result<PureCvMatrixU8, JsError> {
    let result = src
        .inner
        .convert_to::<u8>()
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixU8 { inner: result })
}

// ---------------------------------------------------------------------------
//  Arithmetic operations (f32)
// ---------------------------------------------------------------------------

/// Per-element addition: `dst = a + b`.
#[wasm_bindgen(js_name = "add")]
pub fn add(a: &PureCvMatrixF32, b: &PureCvMatrixF32) -> Result<PureCvMatrixF32, JsError> {
    let result = arithm::add(&a.inner, &b.inner).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

/// Per-element subtraction: `dst = a - b`.
#[wasm_bindgen(js_name = "subtract")]
pub fn subtract(a: &PureCvMatrixF32, b: &PureCvMatrixF32) -> Result<PureCvMatrixF32, JsError> {
    let result =
        arithm::subtract(&a.inner, &b.inner).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

/// Per-element multiplication: `dst = a * b`.
#[wasm_bindgen(js_name = "multiply")]
pub fn multiply(a: &PureCvMatrixF32, b: &PureCvMatrixF32) -> Result<PureCvMatrixF32, JsError> {
    let result =
        arithm::multiply(&a.inner, &b.inner).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

/// Per-element division: `dst = a / b`.
#[wasm_bindgen(js_name = "divide")]
pub fn divide(a: &PureCvMatrixF32, b: &PureCvMatrixF32) -> Result<PureCvMatrixF32, JsError> {
    let result =
        arithm::divide(&a.inner, &b.inner).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

/// Per-element absolute difference: `dst = |a - b|`.
#[wasm_bindgen(js_name = "absDiff")]
pub fn abs_diff(a: &PureCvMatrixF32, b: &PureCvMatrixF32) -> Result<PureCvMatrixF32, JsError> {
    let result =
        arithm::abs_diff(&a.inner, &b.inner).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

/// Per-element minimum: `dst(i) = min(a(i), b(i))`.
#[wasm_bindgen(js_name = "min")]
pub fn min(a: &PureCvMatrixF32, b: &PureCvMatrixF32) -> Result<PureCvMatrixF32, JsError> {
    let result = arithm::min(&a.inner, &b.inner).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

/// Per-element maximum: `dst(i) = max(a(i), b(i))`.
#[wasm_bindgen(js_name = "max")]
pub fn max(a: &PureCvMatrixF32, b: &PureCvMatrixF32) -> Result<PureCvMatrixF32, JsError> {
    let result = arithm::max(&a.inner, &b.inner).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

// ---------------------------------------------------------------------------
//  Structural operations (f32)
// ---------------------------------------------------------------------------

/// Flips a matrix around vertical (0), horizontal (1), or both axes (-1).
#[wasm_bindgen(js_name = "flip")]
pub fn flip(src: &PureCvMatrixF32, flip_code: i32) -> Result<PureCvMatrixF32, JsError> {
    let result =
        structural::flip(&src.inner, flip_code).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

/// Transposes a matrix (swaps rows and columns).
#[wasm_bindgen(js_name = "transpose")]
pub fn transpose(src: &PureCvMatrixF32) -> Result<PureCvMatrixF32, JsError> {
    let result =
        structural::transpose(&src.inner).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

/// Rotates a matrix: 0 = 90° CW, 1 = 180°, 2 = 90° CCW.
#[wasm_bindgen(js_name = "rotate")]
pub fn rotate(src: &PureCvMatrixF32, rotate_code: i32) -> Result<PureCvMatrixF32, JsError> {
    let result = structural::rotate(&src.inner, rotate_code)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

// ---------------------------------------------------------------------------
//  Color conversion (u8)
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
        _ => Err(JsError::new(&format!("Unknown color conversion code: {code}"))),
    }
}

/// Converts an 8-bit image from one colour space to another.
///
/// Codes (integer):
///   0 = BGR2GRAY, 1 = RGB2GRAY, 2 = BGRA2GRAY, 3 = RGBA2GRAY,
///   4 = GRAY2RGB, 5 = GRAY2BGR, 6 = GRAY2RGBA, 7 = GRAY2BGRA.
#[wasm_bindgen(js_name = "cvtColor")]
pub fn convert_color(src: &PureCvMatrixU8, code: i32) -> Result<PureCvMatrixU8, JsError> {
    let cc = color_code_from_i32(code)?;
    let result = cvt_color(&src.inner, cc).map_err(|e| JsError::new(e))?;
    Ok(PureCvMatrixU8 { inner: result })
}

// ---------------------------------------------------------------------------
//  Threshold (f32)
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
    matrix: PureCvMatrixF32,
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
    pub fn get_matrix(self) -> PureCvMatrixF32 {
        self.matrix
    }
}

/// Applies a fixed-level threshold to every element.
///
/// * `threshold_type`: 0 = BINARY, 1 = BINARY_INV, 2 = TRUNC, 3 = TOZERO, 4 = TOZERO_INV.
#[wasm_bindgen(js_name = "threshold")]
pub fn apply_threshold(
    src: &PureCvMatrixF32,
    thresh: f64,
    maxval: f64,
    threshold_type: i32,
) -> Result<ThresholdResult, JsError> {
    let tt = thresh_type_from_i32(threshold_type)?;
    let (tv, mat) =
        threshold(&src.inner, thresh, maxval, tt).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(ThresholdResult {
        thresh_val: tv,
        matrix: PureCvMatrixF32 { inner: mat },
    })
}

// ---------------------------------------------------------------------------
//  Edge detection (f32 → u8 for Canny, f32 → f32 for Sobel/Scharr/Laplacian)
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
/// Returns an 8-bit edge map.
///
/// * `aperture_size` – Sobel kernel size (default: 3).
/// * `l2_gradient`   – Use L₂ norm (true) or L₁ norm (false).
#[wasm_bindgen(js_name = "canny")]
pub fn canny(
    src: &PureCvMatrixF32,
    threshold1: f64,
    threshold2: f64,
    aperture_size: i32,
    l2_gradient: bool,
) -> Result<PureCvMatrixU8, JsError> {
    let result = edge::canny(&src.inner, threshold1, threshold2, aperture_size, l2_gradient)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixU8 { inner: result })
}

/// Sobel derivative filter.
///
/// * `dx`, `dy`      – Order of the derivative in x / y.
/// * `ksize`         – Aperture size (1, 3, 5, or 7; -1 = Scharr).
/// * `scale`, `delta` – Scale factor and optional offset.
/// * `border_type`   – Border interpolation (integer, see `BorderTypes`).
#[wasm_bindgen(js_name = "sobel")]
pub fn sobel(
    src: &PureCvMatrixF32,
    dx: i32,
    dy: i32,
    ksize: i32,
    scale: f64,
    delta: f64,
    border_type: i32,
) -> Result<PureCvMatrixF32, JsError> {
    let bt = border_type_from_i32(border_type)?;
    let result = derivatives::sobel(&src.inner, dx, dy, ksize, scale, delta, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

/// Scharr derivative filter (equivalent to Sobel with ksize = -1).
#[wasm_bindgen(js_name = "scharr")]
pub fn scharr(
    src: &PureCvMatrixF32,
    dx: i32,
    dy: i32,
    scale: f64,
    delta: f64,
    border_type: i32,
) -> Result<PureCvMatrixF32, JsError> {
    let bt = border_type_from_i32(border_type)?;
    let result = derivatives::scharr(&src.inner, dx, dy, scale, delta, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

/// Laplacian of an image.
#[wasm_bindgen(js_name = "laplacian")]
pub fn laplacian(
    src: &PureCvMatrixF32,
    ksize: i32,
    scale: f64,
    delta: f64,
    border_type: i32,
) -> Result<PureCvMatrixF32, JsError> {
    let bt = border_type_from_i32(border_type)?;
    let result = derivatives::laplacian(&src.inner, ksize, scale, delta, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

// ---------------------------------------------------------------------------
//  Blur / filter operations (f32)
// ---------------------------------------------------------------------------

/// Box blur (mean filter).
///
/// * `ksize_w`, `ksize_h` – Kernel width and height.
/// * `border_type`        – Border interpolation (integer, see `BorderTypes`).
#[wasm_bindgen(js_name = "blur")]
pub fn blur(
    src: &PureCvMatrixF32,
    ksize_w: i32,
    ksize_h: i32,
    border_type: i32,
) -> Result<PureCvMatrixF32, JsError> {
    let bt = border_type_from_i32(border_type)?;
    let ksize = Size2i::new(ksize_w, ksize_h);
    let anchor = Point2i::new(-1, -1);
    let result =
        filter::blur(&src.inner, ksize, anchor, bt).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

/// Gaussian blur.
///
/// * `ksize_w`, `ksize_h` – Kernel width and height (must be odd).
/// * `sigma1`             – Gaussian σ in X direction (0 = auto).
/// * `sigma2`             – Gaussian σ in Y direction (0 = same as σ₁).
/// * `border_type`        – Border interpolation (integer).
#[wasm_bindgen(js_name = "gaussianBlur")]
pub fn gaussian_blur(
    src: &PureCvMatrixF32,
    ksize_w: i32,
    ksize_h: i32,
    sigma1: f64,
    sigma2: f64,
    border_type: i32,
) -> Result<PureCvMatrixF32, JsError> {
    let bt = border_type_from_i32(border_type)?;
    let ksize = Size2i::new(ksize_w, ksize_h);
    let result = filter::gaussian_blur(&src.inner, ksize, sigma1, sigma2, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

/// Median blur.
///
/// * `ksize` – Aperture size (must be odd and > 1).
#[wasm_bindgen(js_name = "medianBlur")]
pub fn median_blur(src: &PureCvMatrixF32, ksize: i32) -> Result<PureCvMatrixF32, JsError> {
    let result =
        filter::median_blur(&src.inner, ksize).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

/// Bilateral filter.
///
/// * `d`           – Diameter of pixel neighbourhood (-1 = auto from sigma_space).
/// * `sigma_color` – Filter sigma in the colour space.
/// * `sigma_space` – Filter sigma in the coordinate space.
/// * `border_type` – Border interpolation (integer).
#[wasm_bindgen(js_name = "bilateralFilter")]
pub fn bilateral_filter(
    src: &PureCvMatrixF32,
    d: i32,
    sigma_color: f64,
    sigma_space: f64,
    border_type: i32,
) -> Result<PureCvMatrixF32, JsError> {
    let bt = border_type_from_i32(border_type)?;
    let result = filter::bilateral_filter(&src.inner, d, sigma_color, sigma_space, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(PureCvMatrixF32 { inner: result })
}

// ---------------------------------------------------------------------------
//  JS-side enum constants (exposed as getter functions)
// ---------------------------------------------------------------------------

// Color conversion codes
#[wasm_bindgen(js_name = "COLOR_BGR2GRAY")]
pub fn color_bgr2gray() -> i32 { 0 }
#[wasm_bindgen(js_name = "COLOR_RGB2GRAY")]
pub fn color_rgb2gray() -> i32 { 1 }
#[wasm_bindgen(js_name = "COLOR_BGRA2GRAY")]
pub fn color_bgra2gray() -> i32 { 2 }
#[wasm_bindgen(js_name = "COLOR_RGBA2GRAY")]
pub fn color_rgba2gray() -> i32 { 3 }
#[wasm_bindgen(js_name = "COLOR_GRAY2RGB")]
pub fn color_gray2rgb() -> i32 { 4 }
#[wasm_bindgen(js_name = "COLOR_GRAY2BGR")]
pub fn color_gray2bgr() -> i32 { 5 }
#[wasm_bindgen(js_name = "COLOR_GRAY2RGBA")]
pub fn color_gray2rgba() -> i32 { 6 }
#[wasm_bindgen(js_name = "COLOR_GRAY2BGRA")]
pub fn color_gray2bgra() -> i32 { 7 }

// Threshold types
#[wasm_bindgen(js_name = "THRESH_BINARY")]
pub fn thresh_binary() -> i32 { 0 }
#[wasm_bindgen(js_name = "THRESH_BINARY_INV")]
pub fn thresh_binary_inv() -> i32 { 1 }
#[wasm_bindgen(js_name = "THRESH_TRUNC")]
pub fn thresh_trunc() -> i32 { 2 }
#[wasm_bindgen(js_name = "THRESH_TOZERO")]
pub fn thresh_tozero() -> i32 { 3 }
#[wasm_bindgen(js_name = "THRESH_TOZERO_INV")]
pub fn thresh_tozero_inv() -> i32 { 4 }

// Border types
#[wasm_bindgen(js_name = "BORDER_CONSTANT")]
pub fn border_constant() -> i32 { 0 }
#[wasm_bindgen(js_name = "BORDER_REPLICATE")]
pub fn border_replicate() -> i32 { 1 }
#[wasm_bindgen(js_name = "BORDER_REFLECT")]
pub fn border_reflect() -> i32 { 2 }
#[wasm_bindgen(js_name = "BORDER_WRAP")]
pub fn border_wrap() -> i32 { 3 }
#[wasm_bindgen(js_name = "BORDER_REFLECT_101")]
pub fn border_reflect_101() -> i32 { 4 }

// Flip codes
#[wasm_bindgen(js_name = "FLIP_VERTICAL")]
pub fn flip_vertical() -> i32 { 0 }
#[wasm_bindgen(js_name = "FLIP_HORIZONTAL")]
pub fn flip_horizontal() -> i32 { 1 }
#[wasm_bindgen(js_name = "FLIP_BOTH")]
pub fn flip_both() -> i32 { -1 }

// Rotate codes
#[wasm_bindgen(js_name = "ROTATE_90_CLOCKWISE")]
pub fn rotate_90_clockwise() -> i32 { 0 }
#[wasm_bindgen(js_name = "ROTATE_180")]
pub fn rotate_180() -> i32 { 1 }
#[wasm_bindgen(js_name = "ROTATE_90_COUNTERCLOCKWISE")]
pub fn rotate_90_counterclockwise() -> i32 { 2 }