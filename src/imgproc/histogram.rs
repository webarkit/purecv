/*
 *  histogram.rs
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
 *  Author(s): XiaoPengYouCode https://github.com/XiaoPengYouCode
 *
 */

use core::cmp::Ordering;

use alloc::{vec, vec::Vec};
#[allow(unused_imports)]
use num_traits::Float;
use num_traits::ToPrimitive;

use crate::core::error::Result;
use crate::core::logging::tags;
use crate::core::types::{BorderTypes, Size2i};
use crate::core::utils::border_interpolate;
use crate::core::Matrix;
use crate::cv_bail;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ---------------------------------------------------------------------------
//  Enums
// ---------------------------------------------------------------------------

/// Comparison method for `compare_hist`.
///
/// Mirrors OpenCV's `HistCompMethods`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistCompMethods {
    Correl = 0,
    ChiSqr = 1,
    ChiSqrAlt = 2,
    Intersection = 3,
    Bhattacharyya = 4,
    KullbackLeibler = 5,
}

/// The range specification for a single histogram dimension.
#[derive(Debug, Clone)]
pub enum RangeSpec {
    Uniform(f32, f32),
    NonUniform(Vec<f32>),
}

// ---------------------------------------------------------------------------
//  Multi-image channel resolution
// ---------------------------------------------------------------------------

/// Resolves a global channel index across multiple images into
/// `(image_index, local_channel_index)`.
///
/// OpenCV semantics: channels are numbered sequentially across images.
fn resolve_channel<T>(global_ch: usize, images: &[&Matrix<T>]) -> Result<(usize, usize)> {
    let mut remaining = global_ch;
    for (img_idx, img) in images.iter().enumerate() {
        if remaining < img.channels {
            return Ok((img_idx, remaining));
        }
        remaining -= img.channels;
    }
    cv_bail!(
        tags::IMGPROC,
        InvalidInput,
        "resolve_channel: global channel {} exceeds total channels {}",
        global_ch,
        images.iter().map(|i| i.channels).sum::<usize>()
    );
}

/// Reads a single pixel value from multi-image channel indexing and converts to `f32`.
#[inline(always)]
fn read_pixel_f32<T: ToPrimitive + Clone + Default>(
    images: &[&Matrix<T>],
    global_ch: usize,
    y: usize,
    x: usize,
) -> Option<f32> {
    let (img_idx, local_ch) = resolve_channel(global_ch, images).ok()?;
    images[img_idx].get(y, x, local_ch)?.to_f32()
}

// ---------------------------------------------------------------------------
//  Bin mapping helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn uniform_bin(val: f32, range_lo: f32, range_hi: f32, hist_size: usize) -> Option<usize> {
    if hist_size == 0 {
        return None;
    }
    if val < range_lo || val >= range_hi {
        return None;
    }
    let idx = ((val - range_lo) * (hist_size as f32) / (range_hi - range_lo)) as i32;
    Some(idx.clamp(0, hist_size as i32 - 1) as usize)
}

#[inline(always)]
fn nonuniform_bin(val: f32, boundaries: &[f32], hist_size: usize) -> Option<usize> {
    if hist_size == 0 || boundaries.len() <= hist_size {
        return None;
    }
    if val < boundaries[0] || val >= boundaries[hist_size] {
        return None;
    }
    // Find the first boundary that is strictly greater than val,
    // then subtract 1 to get the bin index.
    match boundaries[..=hist_size]
        .binary_search_by(|b| b.partial_cmp(&val).unwrap_or(Ordering::Less))
    {
        Ok(idx) => {
            // val == boundaries[idx]: bin is idx (left-inclusive)
            // bin i = [boundaries[i], boundaries[i+1])
            // If val == b1, it belongs to bin 1: [b1, b2)
            // binary_search finds idx where b[idx] == val
            // So bin = idx. But if idx == 0 and val == b0, bin = 0.
            if idx <= hist_size {
                Some(idx)
            } else {
                None
            }
        }
        Err(idx) => {
            // boundaries[idx-1] <= val < boundaries[idx]
            if idx > 0 && idx <= hist_size {
                Some(idx - 1)
            } else {
                None
            }
        }
    }
}

fn map_bin(val: f32, range: &RangeSpec, hist_size: usize) -> Option<usize> {
    match range {
        RangeSpec::Uniform(lo, hi) => uniform_bin(val, *lo, *hi, hist_size),
        RangeSpec::NonUniform(boundaries) => nonuniform_bin(val, boundaries, hist_size),
    }
}

// ---------------------------------------------------------------------------
//  calc_hist
// ---------------------------------------------------------------------------

/// Calculates a multi-dimensional histogram of a set of images.
///
/// * `images` - Slice of input images (may have multiple channels).
/// * `channels` - Global channel indices used for histogram computation.
///   Channel numbering spans across images: if `images[0]` has 3 channels
///   and `images[1]` has 1 channel, then channel 3 refers to `images[1]`'s channel 0.
/// * `mask` - Optional mask. Must have the same size as `images[0]`.
/// * `hist_size` - Number of bins per dimension.
/// * `ranges` - One `RangeSpec` per dimension. `Uniform(lo, hi)` for uniform bins,
///   `NonUniform(boundaries)` for explicit boundaries.
/// * `accumulate` - If true, adds to the returned histogram rather than clearing.
///   On first call, pass `None` for `hist` or create a zero histogram.
///
/// Returns a flattened `Matrix<f32>` histogram of size `(product(hist_size), 1, 1)`.
pub fn calc_hist<T: ToPrimitive + Clone + Default + Send + Sync>(
    images: &[&Matrix<T>],
    channels: &[usize],
    mask: Option<&Matrix<u8>>,
    hist_size: &[usize],
    ranges: &[RangeSpec],
    accumulate: bool,
    hist: Option<&Matrix<f32>>,
) -> Result<Matrix<f32>> {
    let dims = hist_size.len();
    if dims == 0 {
        cv_bail!(
            tags::IMGPROC,
            InvalidInput,
            "calc_hist: hist_size must not be empty"
        );
    }
    if images.is_empty() {
        cv_bail!(
            tags::IMGPROC,
            InvalidInput,
            "calc_hist: images must not be empty"
        );
    }
    if channels.len() != dims {
        cv_bail!(
            tags::IMGPROC,
            InvalidInput,
            "calc_hist: channels length ({}) must match hist_size length ({})",
            channels.len(),
            dims
        );
    }
    if ranges.len() != dims {
        cv_bail!(
            tags::IMGPROC,
            InvalidInput,
            "calc_hist: ranges length ({}) must match hist_size length ({})",
            ranges.len(),
            dims
        );
    }

    let rows = images[0].rows;
    let cols = images[0].cols;

    for img in images.iter() {
        if img.rows != rows || img.cols != cols {
            cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "calc_hist: all images must have the same size"
            );
        }
    }

    if let Some(m) = mask {
        if m.rows != rows || m.cols != cols {
            cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "calc_hist: mask must have the same size as images"
            );
        }
    }

    // Validate channel indices
    for &ch in channels {
        let _ = resolve_channel(ch, images)?;
    }

    // Validate hist_size and ranges (prevents panics in bin mapping)
    for (d, &sz) in hist_size.iter().enumerate() {
        if sz == 0 {
            cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "calc_hist: hist_size[{}] must be > 0 (got 0)",
                d
            );
        }
        match &ranges[d] {
            RangeSpec::Uniform(lo, hi) => {
                if lo.partial_cmp(hi) != Some(Ordering::Less) {
                    cv_bail!(
                        tags::IMGPROC,
                        InvalidInput,
                        "calc_hist: Uniform range [{}] must satisfy lo < hi (got {} >= {})",
                        d,
                        lo,
                        hi
                    );
                }
            }
            RangeSpec::NonUniform(boundaries) => {
                if boundaries.len() != sz + 1 {
                    cv_bail!(
                        tags::IMGPROC,
                        InvalidInput,
                        "calc_hist: NonUniform boundaries[{}] length {} must be hist_size[{}]+1 ({})",
                        d,
                        boundaries.len(),
                        d,
                        sz + 1
                    );
                }
                for k in 0..boundaries.len() - 1 {
                    if boundaries[k].partial_cmp(&boundaries[k + 1]) != Some(Ordering::Less) {
                        cv_bail!(
                            tags::IMGPROC,
                            InvalidInput,
                            "calc_hist: NonUniform boundaries[{}][{}] ({}) must be < boundaries[{}][{}] ({})",
                            d,
                            k,
                            boundaries[k],
                            d,
                            k + 1,
                            boundaries[k + 1]
                        );
                    }
                }
            }
        }
    }

    let total_bins: usize = hist_size.iter().product();

    // Validate accumulate histogram size to prevent OOB
    if accumulate {
        if let Some(h) = hist {
            let hist_len = h.data.len();
            if hist_len != total_bins {
                cv_bail!(
                    tags::IMGPROC,
                    InvalidInput,
                    "calc_hist: accumulate hist length {} does not match product(hist_size) {}",
                    hist_len,
                    total_bins
                );
            }
        }
    }

    let mut hist_data = if accumulate {
        hist.map(|h| h.data.clone())
            .unwrap_or_else(|| vec![0.0f32; total_bins])
    } else {
        vec![0.0f32; total_bins]
    };

    // Strides (row-major, last dim varies fastest)
    let mut strides = vec![1usize; dims];
    for i in (0..dims - 1).rev() {
        strides[i] = strides[i + 1] * hist_size[i + 1];
    }

    let accumulate_row = |local: &mut [f32], y: usize| {
        for x in 0..cols {
            if let Some(m) = mask {
                if let Some(&v) = m.get(y, x, 0) {
                    if v == 0 {
                        continue;
                    }
                }
            }

            let mut bin_idx = 0usize;
            let mut out_of_range = false;

            for d in 0..dims {
                let val = read_pixel_f32(images, channels[d], y, x).unwrap_or(0.0);
                match map_bin(val, &ranges[d], hist_size[d]) {
                    Some(b) => bin_idx += b * strides[d],
                    None => {
                        out_of_range = true;
                        break;
                    }
                }
            }

            if !out_of_range {
                local[bin_idx] += 1.0;
            }
        }
    };

    #[cfg(feature = "parallel")]
    {
        let partial = (0..rows)
            .into_par_iter()
            .fold(
                || vec![0.0f32; total_bins],
                |mut local, y| {
                    accumulate_row(&mut local, y);
                    local
                },
            )
            .reduce(
                || vec![0.0f32; total_bins],
                |mut a, b| {
                    for (av, bv) in a.iter_mut().zip(b.iter()) {
                        *av += bv;
                    }
                    a
                },
            );
        for (hv, pv) in hist_data.iter_mut().zip(partial.iter()) {
            *hv += pv;
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        for y in 0..rows {
            accumulate_row(&mut hist_data, y);
        }
    }

    Ok(Matrix::from_vec(total_bins, 1, 1, hist_data))
}

// ---------------------------------------------------------------------------
//  calc_back_project
// ---------------------------------------------------------------------------

/// Calculates the back projection of a histogram.
///
/// * `images` - Slice of input images.
/// * `channels` - Global channel indices (same semantics as `calc_hist`).
/// * `hist` - Input histogram (`f32`, flattened multi-dimensional).
/// * `ranges` - One `RangeSpec` per dimension.
/// * `scale` - Scale factor for the output values.
///
/// Returns a single-channel `Matrix<f32>` of the same size as `images[0]`.
/// Values are scaled and clamped to `[0.0, 255.0]` matching OpenCV's u8 output range.
pub fn calc_back_project<T: ToPrimitive + Clone + Default + Send + Sync>(
    images: &[&Matrix<T>],
    channels: &[usize],
    hist: &Matrix<f32>,
    ranges: &[RangeSpec],
    scale: f32,
) -> Result<Matrix<f32>> {
    let dims = channels.len();
    if dims == 0 {
        cv_bail!(
            tags::IMGPROC,
            InvalidInput,
            "calc_back_project: channels must not be empty"
        );
    }
    if images.is_empty() {
        cv_bail!(
            tags::IMGPROC,
            InvalidInput,
            "calc_back_project: images must not be empty"
        );
    }
    if ranges.len() != dims {
        cv_bail!(
            tags::IMGPROC,
            InvalidInput,
            "calc_back_project: ranges length ({}) must match channels length ({})",
            ranges.len(),
            dims
        );
    }

    let rows = images[0].rows;
    let cols = images[0].cols;

    let total_bins = hist.rows * hist.cols * hist.channels;
    if total_bins == 0 {
        cv_bail!(
            tags::IMGPROC,
            InvalidInput,
            "calc_back_project: hist must not be empty"
        );
    }
    let hist_size = infer_hist_size(total_bins, dims);

    let mut strides = vec![1usize; dims];
    for i in (0..dims - 1).rev() {
        strides[i] = strides[i + 1] * hist_size[i + 1];
    }

    let mut dst = Matrix::<f32>::new(rows, cols, 1);

    let process_row = |y: usize, dst_row: &mut [f32]| {
        for (x, out_pixel) in dst_row.iter_mut().enumerate() {
            let mut bin_idx = 0usize;
            let mut out_of_range = false;

            for d in 0..dims {
                let val = read_pixel_f32(images, channels[d], y, x).unwrap_or(0.0);
                match map_bin(val, &ranges[d], hist_size[d]) {
                    Some(b) => bin_idx += b * strides[d],
                    None => {
                        out_of_range = true;
                        break;
                    }
                }
            }

            *out_pixel = if out_of_range {
                0.0f32
            } else {
                (hist.data[bin_idx] * scale).clamp(0.0, 255.0)
            };
        }
    };

    #[cfg(feature = "parallel")]
    {
        dst.data
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(y, dst_row)| {
                process_row(y, dst_row);
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (y, dst_row) in dst.data.chunks_mut(cols).enumerate() {
            process_row(y, dst_row);
        }
    }

    Ok(dst)
}

fn infer_hist_size(total_bins: usize, dims: usize) -> Vec<usize> {
    if dims == 1 {
        return vec![total_bins];
    }
    let approx = (total_bins as f64).powf(1.0 / dims as f64).round() as usize;
    if approx.pow(dims as u32) == total_bins {
        return vec![approx; dims];
    }
    let mut sizes = vec![1usize; dims];
    sizes[dims - 1] = total_bins;
    sizes
}

// ---------------------------------------------------------------------------
//  compare_hist
// ---------------------------------------------------------------------------

/// Compares two dense histograms using the specified method.
///
/// Both histograms must be single-channel `f32` with the same size.
pub fn compare_hist(h1: &Matrix<f32>, h2: &Matrix<f32>, method: HistCompMethods) -> Result<f64> {
    if h1.channels != 1 || h2.channels != 1 {
        cv_bail!(
            tags::IMGPROC,
            InvalidInput,
            "compare_hist: histograms must be single-channel"
        );
    }

    let len = h1.data.len();
    if len != h2.data.len() {
        cv_bail!(
            tags::IMGPROC,
            InvalidInput,
            "compare_hist: histograms must have the same size ({} vs {})",
            len,
            h2.data.len()
        );
    }
    if len == 0 {
        cv_bail!(
            tags::IMGPROC,
            InvalidInput,
            "compare_hist: histograms must not be empty"
        );
    }

    let n = len as f64;

    match method {
        HistCompMethods::Correl => {
            let mut s1 = 0.0f64;
            let mut s2 = 0.0f64;
            let mut s11 = 0.0f64;
            let mut s12 = 0.0f64;
            let mut s22 = 0.0f64;

            for i in 0..len {
                let a = h1.data[i] as f64;
                let b = h2.data[i] as f64;
                s1 += a;
                s2 += b;
                s11 += a * a;
                s22 += b * b;
                s12 += a * b;
            }

            let scale = 1.0 / n;
            let num = s12 - s1 * s2 * scale;
            let denom2 = (s11 - s1 * s1 * scale) * (s22 - s2 * s2 * scale);
            Ok(if denom2.abs() > f64::EPSILON {
                num / denom2.sqrt()
            } else {
                1.0
            })
        }
        HistCompMethods::ChiSqr => {
            let mut result = 0.0f64;
            for i in 0..len {
                let a = h1.data[i] as f64;
                let b = h2.data[i] as f64;
                if a.abs() > f64::EPSILON {
                    let diff = a - b;
                    result += diff * diff / a;
                }
            }
            Ok(result)
        }
        HistCompMethods::ChiSqrAlt => {
            let mut result = 0.0f64;
            for i in 0..len {
                let a = h1.data[i] as f64;
                let b = h2.data[i] as f64;
                let sum = a + b;
                if sum.abs() > f64::EPSILON {
                    let diff = a - b;
                    result += diff * diff / sum;
                }
            }
            Ok(result * 2.0)
        }
        HistCompMethods::Intersection => {
            let mut result = 0.0f64;
            for i in 0..len {
                let a = h1.data[i] as f64;
                let b = h2.data[i] as f64;
                result += a.min(b);
            }
            Ok(result)
        }
        HistCompMethods::Bhattacharyya => {
            let mut s1 = 0.0f64;
            let mut s2 = 0.0f64;
            let mut bc = 0.0f64;

            for i in 0..len {
                let a = h1.data[i] as f64;
                let b = h2.data[i] as f64;
                s1 += a;
                s2 += b;
                bc += (a * b).sqrt();
            }

            let norm = s1 * s2;
            let norm_factor = if norm.abs() > f64::EPSILON {
                1.0 / norm.sqrt()
            } else {
                1.0
            };

            Ok(((1.0 - bc * norm_factor).max(0.0)).sqrt())
        }
        HistCompMethods::KullbackLeibler => {
            let mut result = 0.0f64;
            for i in 0..len {
                let p = h1.data[i] as f64;
                let q = h2.data[i] as f64;
                if p.abs() > f64::EPSILON {
                    let q_adj = if q.abs() <= f64::EPSILON { 1e-10 } else { q };
                    result += p * (p / q_adj).ln();
                }
            }
            Ok(result)
        }
    }
}

// ---------------------------------------------------------------------------
//  equalize_hist
// ---------------------------------------------------------------------------

/// Equalizes the histogram of a grayscale image.
///
/// * `src` - Source 8-bit single-channel image.
pub fn equalize_hist(src: &Matrix<u8>) -> Result<Matrix<u8>> {
    if src.channels != 1 {
        cv_bail!(
            tags::IMGPROC,
            InvalidInput,
            "equalize_hist: source must be single-channel (got {})",
            src.channels
        );
    }

    const HIST_SZ: usize = 256;
    let mut hist = [0u32; HIST_SZ];

    for &val in src.data.iter() {
        hist[val as usize] += 1;
    }

    let mut i = 0;
    while i < HIST_SZ && hist[i] == 0 {
        i += 1;
    }

    if i == HIST_SZ {
        return Ok(Matrix::new(src.rows, src.cols, 1));
    }

    let total = src.rows * src.cols;
    if hist[i] == total as u32 {
        let mut dst = Matrix::<u8>::new(src.rows, src.cols, 1);
        for pixel in dst.data.iter_mut() {
            *pixel = i as u8;
        }
        return Ok(dst);
    }

    let scale = (HIST_SZ as f64 - 1.0) / (total as f64 - hist[i] as f64);
    let mut lut = [0u8; HIST_SZ];
    let mut sum = 0u32;

    lut[i] = 0;
    i += 1;
    for j in i..HIST_SZ {
        sum += hist[j];
        lut[j] = (sum as f64 * scale).round().clamp(0.0, 255.0) as u8;
    }

    let mut dst = Matrix::<u8>::new(src.rows, src.cols, 1);

    #[cfg(feature = "parallel")]
    {
        dst.data
            .par_iter_mut()
            .zip(src.data.par_iter())
            .for_each(|(d, &s)| {
                *d = lut[s as usize];
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (d, &s) in dst.data.iter_mut().zip(src.data.iter()) {
            *d = lut[s as usize];
        }
    }

    Ok(dst)
}

// ---------------------------------------------------------------------------
//  CLAHE
// ---------------------------------------------------------------------------

/// Contrast Limited Adaptive Histogram Equalization.
#[derive(Debug, Clone)]
pub struct Clahe {
    clip_limit: f64,
    tiles_x: usize,
    tiles_y: usize,
    bit_shift: i32,
}

impl Clahe {
    pub fn new(clip_limit: f64, tile_grid_size: Size2i) -> Self {
        let tiles_x = tile_grid_size.width.max(0) as usize;
        let tiles_y = tile_grid_size.height.max(0) as usize;
        Self {
            clip_limit,
            tiles_x,
            tiles_y,
            bit_shift: 0,
        }
    }

    pub fn set_clip_limit(&mut self, clip_limit: f64) {
        self.clip_limit = clip_limit;
    }

    pub fn get_clip_limit(&self) -> f64 {
        self.clip_limit
    }

    pub fn set_tiles_grid_size(&mut self, tile_grid_size: Size2i) {
        self.tiles_x = tile_grid_size.width.max(0) as usize;
        self.tiles_y = tile_grid_size.height.max(0) as usize;
    }

    pub fn get_tiles_grid_size(&self) -> Size2i {
        Size2i::new(self.tiles_x as i32, self.tiles_y as i32)
    }

    pub fn set_bit_shift(&mut self, bit_shift: i32) {
        self.bit_shift = bit_shift;
    }

    pub fn get_bit_shift(&self) -> i32 {
        self.bit_shift
    }

    /// Applies CLAHE to a single-channel `u8` or `u16` image.
    ///
    /// The `src` must be `Matrix<u8>` (CV_8UC1) or `Matrix<u16>` (CV_16UC1).
    /// For u16, the function uses bit_shift to reduce the histogram size.
    pub fn apply_u8(&self, src: &Matrix<u8>) -> Result<Matrix<u8>> {
        if src.channels != 1 {
            cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "Clahe::apply_u8: source must be single-channel (got {})",
                src.channels
            );
        }
        self.apply_impl_u8(src)
    }

    pub fn apply_u16(&self, src: &Matrix<u16>) -> Result<Matrix<u16>> {
        if src.channels != 1 {
            cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "Clahe::apply_u16: source must be single-channel (got {})",
                src.channels
            );
        }
        self.apply_impl_u16(src)
    }

    fn apply_impl_u8(&self, src: &Matrix<u8>) -> Result<Matrix<u8>> {
        if self.tiles_x == 0 || self.tiles_y == 0 {
            cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "Clahe::apply_u8: tiles_x and tiles_y must be > 0 (got {}x{})",
                self.tiles_x,
                self.tiles_y
            );
        }
        if self.bit_shift < 0 || self.bit_shift > 7 {
            cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "Clahe::apply_u8: bit_shift must be in 0..=7 for u8 (got {})",
                self.bit_shift
            );
        }
        if src.rows == 0 || src.cols == 0 {
            cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "Clahe::apply_u8: source must not be empty"
            );
        }
        let hist_size = 256usize >> self.bit_shift;
        // Defensive: unreachable since bit_shift is validated to 0..=7 above.
        if hist_size == 0 {
            cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "Clahe::apply_u8: invalid hist_size 0 (bit_shift {})",
                self.bit_shift
            );
        }

        // OpenCV parity: pad bottom/right with BORDER_REFLECT_101 so dimensions become divisible
        let pad_bottom = (self.tiles_y - src.rows % self.tiles_y) % self.tiles_y;
        let pad_right = (self.tiles_x - src.cols % self.tiles_x) % self.tiles_x;
        let need_pad = pad_bottom > 0 || pad_right > 0;

        let padded = if need_pad {
            pad_reflect101(src, pad_bottom, pad_right)
        } else {
            src.clone()
        };

        let tile_rows = padded.rows / self.tiles_y;
        let tile_cols = padded.cols / self.tiles_x;
        // Defensive: unreachable given a non-empty src and validated tiles,
        // since padding above makes the padded dims divisible by the grid.
        if tile_rows == 0 || tile_cols == 0 {
            cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "Clahe::apply_u8: tile size must be > 0 (padded {}x{} / tiles {}x{} -> tile {}x{})",
                padded.rows,
                padded.cols,
                self.tiles_x,
                self.tiles_y,
                tile_cols,
                tile_rows
            );
        }
        let tile_size_total = tile_rows * tile_cols;

        let lut_scale = (hist_size as f64 - 1.0) / tile_size_total as f64;

        let mut clip_limit = 0i32;
        if self.clip_limit > 0.0 {
            clip_limit = (self.clip_limit * tile_size_total as f64 / hist_size as f64) as i32;
            clip_limit = clip_limit.max(1);
        }

        let num_tiles = match self.tiles_x.checked_mul(self.tiles_y) {
            Some(v) => v,
            None => cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "Clahe::apply_u8: tiles_x * tiles_y overflow"
            ),
        };
        let lut_len = match num_tiles.checked_mul(hist_size) {
            Some(v) => v,
            None => cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "Clahe::apply_u8: lut size overflow"
            ),
        };
        let mut lut = vec![0u8; lut_len];

        let build_tile_lut = |tile_idx: usize, tile_lut: &mut [u8]| {
            let ty = tile_idx / self.tiles_x;
            let tx = tile_idx % self.tiles_x;
            let y0 = ty * tile_rows;
            let x0 = tx * tile_cols;

            let mut tile_hist = vec![0i32; hist_size];
            for dy in 0..tile_rows {
                for dx in 0..tile_cols {
                    let val = *padded.get(y0 + dy, x0 + dx, 0).unwrap_or(&0) as usize;
                    let bin = val >> self.bit_shift;
                    if bin < hist_size {
                        tile_hist[bin] += 1;
                    }
                }
            }

            clip_and_redistribute(&mut tile_hist, clip_limit, hist_size);

            let mut sum = 0i32;
            for bin in 0..hist_size {
                sum += tile_hist[bin];
                tile_lut[bin] = (sum as f64 * lut_scale).round().clamp(0.0, 255.0) as u8;
            }
        };

        #[cfg(feature = "parallel")]
        {
            lut.par_chunks_mut(hist_size)
                .enumerate()
                .for_each(|(tile_idx, tile_lut)| {
                    build_tile_lut(tile_idx, tile_lut);
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for (tile_idx, tile_lut) in lut.chunks_mut(hist_size).enumerate() {
                build_tile_lut(tile_idx, tile_lut);
            }
        }

        let mut dst = Matrix::<u8>::new(src.rows, src.cols, 1);
        interpolate_tiles_u8(
            src,
            &mut dst,
            &lut,
            tile_rows,
            tile_cols,
            self.bit_shift,
            self.tiles_x,
            self.tiles_y,
            hist_size,
        );
        Ok(dst)
    }

    fn apply_impl_u16(&self, src: &Matrix<u16>) -> Result<Matrix<u16>> {
        if self.tiles_x == 0 || self.tiles_y == 0 {
            cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "Clahe::apply_u16: tiles_x and tiles_y must be > 0 (got {}x{})",
                self.tiles_x,
                self.tiles_y
            );
        }
        if self.bit_shift < 0 || self.bit_shift > 15 {
            cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "Clahe::apply_u16: bit_shift must be in 0..=15 for u16 (got {})",
                self.bit_shift
            );
        }
        if src.rows == 0 || src.cols == 0 {
            cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "Clahe::apply_u16: source must not be empty"
            );
        }
        let hist_size = 65536usize >> self.bit_shift;
        // Defensive: unreachable since bit_shift is validated to 0..=15 above.
        if hist_size == 0 {
            cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "Clahe::apply_u16: invalid hist_size 0 (bit_shift {})",
                self.bit_shift
            );
        }

        let pad_bottom = (self.tiles_y - src.rows % self.tiles_y) % self.tiles_y;
        let pad_right = (self.tiles_x - src.cols % self.tiles_x) % self.tiles_x;
        let need_pad = pad_bottom > 0 || pad_right > 0;

        let padded = if need_pad {
            pad_reflect101(src, pad_bottom, pad_right)
        } else {
            src.clone()
        };

        let tile_rows = padded.rows / self.tiles_y;
        let tile_cols = padded.cols / self.tiles_x;
        // Defensive: unreachable given a non-empty src and validated tiles,
        // since padding above makes the padded dims divisible by the grid.
        if tile_rows == 0 || tile_cols == 0 {
            cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "Clahe::apply_u16: tile size must be > 0 (padded {}x{} / tiles {}x{} -> tile {}x{})",
                padded.rows,
                padded.cols,
                self.tiles_x,
                self.tiles_y,
                tile_cols,
                tile_rows
            );
        }
        let tile_size_total = tile_rows * tile_cols;

        let lut_scale = (hist_size as f64 - 1.0) / tile_size_total as f64;

        let mut clip_limit = 0i32;
        if self.clip_limit > 0.0 {
            clip_limit = (self.clip_limit * tile_size_total as f64 / hist_size as f64) as i32;
            clip_limit = clip_limit.max(1);
        }

        let num_tiles = match self.tiles_x.checked_mul(self.tiles_y) {
            Some(v) => v,
            None => cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "Clahe::apply_u16: tiles_x * tiles_y overflow"
            ),
        };
        let lut_len = match num_tiles.checked_mul(hist_size) {
            Some(v) => v,
            None => cv_bail!(
                tags::IMGPROC,
                InvalidInput,
                "Clahe::apply_u16: lut size overflow"
            ),
        };
        let mut lut = vec![0u16; lut_len];

        let build_tile_lut = |tile_idx: usize, tile_lut: &mut [u16]| {
            let ty = tile_idx / self.tiles_x;
            let tx = tile_idx % self.tiles_x;
            let y0 = ty * tile_rows;
            let x0 = tx * tile_cols;

            let mut tile_hist = vec![0i32; hist_size];
            for dy in 0..tile_rows {
                for dx in 0..tile_cols {
                    let val = *padded.get(y0 + dy, x0 + dx, 0).unwrap_or(&0) as usize;
                    let bin = val >> self.bit_shift;
                    if bin < hist_size {
                        tile_hist[bin] += 1;
                    }
                }
            }

            clip_and_redistribute(&mut tile_hist, clip_limit, hist_size);

            let mut sum = 0i32;
            for bin in 0..hist_size {
                sum += tile_hist[bin];
                tile_lut[bin] = (sum as f64 * lut_scale).round().clamp(0.0, 65535.0) as u16;
            }
        };

        #[cfg(feature = "parallel")]
        {
            lut.par_chunks_mut(hist_size)
                .enumerate()
                .for_each(|(tile_idx, tile_lut)| {
                    build_tile_lut(tile_idx, tile_lut);
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for (tile_idx, tile_lut) in lut.chunks_mut(hist_size).enumerate() {
                build_tile_lut(tile_idx, tile_lut);
            }
        }

        let mut dst = Matrix::<u16>::new(src.rows, src.cols, 1);
        interpolate_tiles_u16(
            src,
            &mut dst,
            &lut,
            tile_rows,
            tile_cols,
            self.bit_shift,
            self.tiles_x,
            self.tiles_y,
            hist_size,
        );
        Ok(dst)
    }
}

fn clip_and_redistribute(tile_hist: &mut [i32], clip_limit: i32, hist_size: usize) {
    if clip_limit <= 0 {
        return;
    }
    // Exact port of OpenCV's CLAHE_CalcLut_Body redistribution (clahe.cpp):
    // uniform batch plus fixed-step residual.
    let mut clipped = 0i32;
    for bin in tile_hist.iter_mut() {
        if *bin > clip_limit {
            clipped += *bin - clip_limit;
            *bin = clip_limit;
        }
    }

    let redist_batch = clipped / hist_size as i32;
    let residual = clipped - redist_batch * hist_size as i32;

    for bin in tile_hist.iter_mut() {
        *bin += redist_batch;
    }

    if residual > 0 {
        let step = (hist_size as i32 / residual).max(1);
        let mut i = 0i32;
        let mut rem = residual;
        while i < hist_size as i32 && rem > 0 {
            tile_hist[i as usize] += 1;
            i += step;
            rem -= 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn interpolate_tiles_u8(
    src: &Matrix<u8>,
    dst: &mut Matrix<u8>,
    lut: &[u8],
    tile_rows: usize,
    tile_cols: usize,
    bit_shift: i32,
    tiles_x: usize,
    tiles_y: usize,
    hist_size: usize,
) {
    let inv_tw = 1.0f64 / tile_cols as f64;
    let inv_th = 1.0f64 / tile_rows as f64;
    let lut_idx =
        |ty: usize, tx: usize, bin: usize| ty * tiles_x * hist_size + tx * hist_size + bin;
    let cols = src.cols;

    let process_row = |y: usize, dst_row: &mut [u8]| {
        let tyf = y as f64 * inv_th - 0.5;
        let ty1 = (tyf.floor() as i32).max(0);
        let ty2 = (ty1 + 1).min(tiles_y as i32 - 1);
        let ya = tyf - ty1 as f64;
        let ya1 = 1.0 - ya;
        let ty1 = ty1 as usize;
        let ty2 = ty2 as usize;

        for (x, out_pixel) in dst_row.iter_mut().enumerate() {
            let txf = x as f64 * inv_tw - 0.5;
            let tx1 = (txf.floor() as i32).max(0);
            let tx2 = (tx1 + 1).min(tiles_x as i32 - 1);
            let xa = txf - tx1 as f64;
            let xa1 = 1.0 - xa;
            let tx1 = tx1 as usize;
            let tx2 = tx2 as usize;

            let src_val = *src.get(y, x, 0).unwrap_or(&0) as usize;
            let bin = (src_val >> bit_shift).min(hist_size - 1);

            let v00 = lut[lut_idx(ty1, tx1, bin)] as f64;
            let v01 = lut[lut_idx(ty1, tx2, bin)] as f64;
            let v10 = lut[lut_idx(ty2, tx1, bin)] as f64;
            let v11 = lut[lut_idx(ty2, tx2, bin)] as f64;

            let val = (v00 * xa1 + v01 * xa) * ya1 + (v10 * xa1 + v11 * xa) * ya;
            *out_pixel = (val.round() as u8) << bit_shift;
        }
    };

    #[cfg(feature = "parallel")]
    {
        dst.data
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(y, dst_row)| {
                process_row(y, dst_row);
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (y, dst_row) in dst.data.chunks_mut(cols).enumerate() {
            process_row(y, dst_row);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn interpolate_tiles_u16(
    src: &Matrix<u16>,
    dst: &mut Matrix<u16>,
    lut: &[u16],
    tile_rows: usize,
    tile_cols: usize,
    bit_shift: i32,
    tiles_x: usize,
    tiles_y: usize,
    hist_size: usize,
) {
    let inv_tw = 1.0f64 / tile_cols as f64;
    let inv_th = 1.0f64 / tile_rows as f64;
    let lut_idx =
        |ty: usize, tx: usize, bin: usize| ty * tiles_x * hist_size + tx * hist_size + bin;
    let cols = src.cols;

    let process_row = |y: usize, dst_row: &mut [u16]| {
        let tyf = y as f64 * inv_th - 0.5;
        let ty1 = (tyf.floor() as i32).max(0);
        let ty2 = (ty1 + 1).min(tiles_y as i32 - 1);
        let ya = tyf - ty1 as f64;
        let ya1 = 1.0 - ya;
        let ty1 = ty1 as usize;
        let ty2 = ty2 as usize;

        for (x, out_pixel) in dst_row.iter_mut().enumerate() {
            let txf = x as f64 * inv_tw - 0.5;
            let tx1 = (txf.floor() as i32).max(0);
            let tx2 = (tx1 + 1).min(tiles_x as i32 - 1);
            let xa = txf - tx1 as f64;
            let xa1 = 1.0 - xa;
            let tx1 = tx1 as usize;
            let tx2 = tx2 as usize;

            let src_val = *src.get(y, x, 0).unwrap_or(&0) as usize;
            let bin = (src_val >> bit_shift).min(hist_size - 1);

            let v00 = lut[lut_idx(ty1, tx1, bin)] as f64;
            let v01 = lut[lut_idx(ty1, tx2, bin)] as f64;
            let v10 = lut[lut_idx(ty2, tx1, bin)] as f64;
            let v11 = lut[lut_idx(ty2, tx2, bin)] as f64;

            let val = (v00 * xa1 + v01 * xa) * ya1 + (v10 * xa1 + v11 * xa) * ya;
            *out_pixel = (val.round() as u16) << bit_shift;
        }
    };

    #[cfg(feature = "parallel")]
    {
        dst.data
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(y, dst_row)| {
                process_row(y, dst_row);
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (y, dst_row) in dst.data.chunks_mut(cols).enumerate() {
            process_row(y, dst_row);
        }
    }
}

fn pad_reflect101<T: Copy + Default>(
    src: &Matrix<T>,
    pad_bottom: usize,
    pad_right: usize,
) -> Matrix<T> {
    let new_rows = src.rows + pad_bottom;
    let new_cols = src.cols + pad_right;
    let mut dst = Matrix::<T>::new(new_rows, new_cols, 1);
    for y in 0..new_rows {
        let sy = border_interpolate(y as i32, src.rows as i32, BorderTypes::Reflect101) as usize;
        for x in 0..new_cols {
            let sx =
                border_interpolate(x as i32, src.cols as i32, BorderTypes::Reflect101) as usize;
            dst.set(y, x, 0, *src.get(sy, sx, 0).unwrap_or(&T::default()));
        }
    }
    dst
}

pub fn create_clahe(clip_limit: f64, tile_grid_size: Size2i) -> Clahe {
    Clahe::new(clip_limit, tile_grid_size)
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_hist_uniform_1d() {
        let data: Vec<u8> = (0..16).collect();
        let img = Matrix::from_vec(4, 4, 1, data);
        let hist = calc_hist(
            &[&img],
            &[0],
            None,
            &[16],
            &[RangeSpec::Uniform(0.0, 16.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist.data.len(), 16);
        for &v in hist.data.iter() {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn test_calc_hist_uniform_1d_fewer_bins() {
        let data: Vec<u8> = (0..16).collect();
        let img = Matrix::from_vec(4, 4, 1, data);
        let hist = calc_hist(
            &[&img],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 16.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist.data.len(), 4);
        for &v in hist.data.iter() {
            assert_eq!(v, 4.0);
        }
    }

    #[test]
    fn test_calc_hist_mask() {
        let img = Matrix::from_vec(3, 4, 1, vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        // Partial mask: checkerboard
        let mask = Matrix::from_vec(3, 4, 1, vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
        let h = calc_hist(
            &[&img],
            &[0],
            Some(&mask),
            &[4],
            &[RangeSpec::Uniform(0.0, 12.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(h.data, vec![1.0, 2.0, 1.0, 2.0]);

        // All-zero mask: nothing counted
        let img2 = Matrix::from_vec(2, 2, 1, vec![0u8, 1, 2, 3]);
        let mask_zero = Matrix::from_vec(2, 2, 1, vec![0u8; 4]);
        let h = calc_hist(
            &[&img2],
            &[0],
            Some(&mask_zero),
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(h.data, vec![0.0, 0.0, 0.0, 0.0]);

        // All-one mask: all counted
        let mask_one = Matrix::from_vec(2, 2, 1, vec![1u8; 4]);
        let h = calc_hist(
            &[&img2],
            &[0],
            Some(&mask_one),
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(h.data, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_calc_hist_multi_image_channel_indexing() {
        // images[0] has 2 channels, images[1] has 1 channel
        let img0 = Matrix::from_vec(2, 1, 2, vec![10, 20, 30, 40]);
        let img1 = Matrix::from_vec(2, 1, 1, vec![100, 200]);

        let hist = calc_hist(
            &[&img0, &img1],
            &[0, 2],
            None,
            &[2, 2],
            &[
                RangeSpec::Uniform(0.0, 50.0),
                RangeSpec::Uniform(0.0, 250.0),
            ],
            false,
            None,
        )
        .unwrap();

        // pixel(0,0): ch0=10->bin0, ch2=100->bin0 -> idx=0
        // pixel(1,0): ch0=30->bin1, ch2=200->bin1 -> idx=3
        assert_eq!(hist.data[0], 1.0);
        assert_eq!(hist.data[1], 0.0);
        assert_eq!(hist.data[2], 0.0);
        assert_eq!(hist.data[3], 1.0);
    }

    #[test]
    fn test_calc_hist_nonuniform() {
        let data: Vec<u8> = (0..20).collect();
        let img = Matrix::from_vec(4, 5, 1, data);
        // Non-uniform: boundaries [0, 5, 20] -> 2 bins: [0,5) and [5,20)
        let hist = calc_hist(
            &[&img],
            &[0],
            None,
            &[2],
            &[RangeSpec::NonUniform(vec![0.0, 5.0, 20.0])],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist.data[0], 5.0); // values 0..4
        assert_eq!(hist.data[1], 15.0); // values 5..19
    }

    #[test]
    fn test_calc_back_project() {
        let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let img = Matrix::from_vec(2, 4, 1, data);
        let hist = Matrix::from_vec(4, 1, 1, vec![10.0, 20.0, 30.0, 40.0]);
        let bp =
            calc_back_project(&[&img], &[0], &hist, &[RangeSpec::Uniform(0.0, 8.0)], 1.0).unwrap();
        assert_eq!(
            bp.data,
            vec![10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0]
        );
    }

    #[test]
    fn test_compare_hist_correl_identical() {
        let h1 = Matrix::from_vec(4, 1, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let h2 = Matrix::from_vec(4, 1, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let corr = compare_hist(&h1, &h2, HistCompMethods::Correl).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compare_hist_correl_opposite() {
        let h1 = Matrix::from_vec(4, 1, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let h2 = Matrix::from_vec(4, 1, 1, vec![4.0, 3.0, 2.0, 1.0]);
        let corr = compare_hist(&h1, &h2, HistCompMethods::Correl).unwrap();
        assert!((corr - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_compare_hist_intersect() {
        let h1 = Matrix::from_vec(4, 1, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let h2 = Matrix::from_vec(4, 1, 1, vec![4.0, 3.0, 2.0, 1.0]);
        let inter = compare_hist(&h1, &h2, HistCompMethods::Intersection).unwrap();
        assert!((inter - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_compare_hist_chi_sqr() {
        let h1 = Matrix::from_vec(3, 1, 1, vec![1.0, 2.0, 3.0]);
        let h2 = Matrix::from_vec(3, 1, 1, vec![1.0, 2.0, 3.0]);
        let chi = compare_hist(&h1, &h2, HistCompMethods::ChiSqr).unwrap();
        assert!((chi - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compare_hist_bhattacharyya_identical() {
        let h1 = Matrix::from_vec(4, 1, 1, vec![0.25, 0.25, 0.25, 0.25]);
        let h2 = Matrix::from_vec(4, 1, 1, vec![0.25, 0.25, 0.25, 0.25]);
        let bc = compare_hist(&h1, &h2, HistCompMethods::Bhattacharyya).unwrap();
        assert!(bc.abs() < 1e-10);
    }

    #[test]
    fn test_compare_hist_kl_divergence() {
        let h1 = Matrix::from_vec(3, 1, 1, vec![0.5, 0.3, 0.2]);
        let h2 = Matrix::from_vec(3, 1, 1, vec![0.5, 0.3, 0.2]);
        let kl = compare_hist(&h1, &h2, HistCompMethods::KullbackLeibler).unwrap();
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_equalize_hist_uniform() {
        let img = Matrix::from_vec(4, 4, 1, vec![128u8; 16]);
        let dst = equalize_hist(&img).unwrap();
        for &v in dst.data.iter() {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_equalize_hist_gradient() {
        let data: Vec<u8> = (0..=255).collect();
        let img = Matrix::from_vec(16, 16, 1, data);
        let dst = equalize_hist(&img).unwrap();
        assert_eq!(dst.rows, 16);
        assert_eq!(dst.cols, 16);
        assert_eq!(dst.channels, 1);
        let min_val = *dst.data.iter().min().unwrap();
        let max_val = *dst.data.iter().max().unwrap();
        assert_eq!(min_val, 0);
        assert_eq!(max_val, 255);
    }

    #[test]
    fn test_clahe_basic() {
        let data: Vec<u8> = (0..=255).collect();
        let img = Matrix::from_vec(16, 16, 1, data);
        let clahe = create_clahe(40.0, Size2i::new(4, 4));
        let dst = clahe.apply_u8(&img).unwrap();
        assert_eq!(dst.rows, 16);
        assert_eq!(dst.cols, 16);
        assert_eq!(dst.channels, 1);
        assert_eq!(dst.data.len(), 256);
    }

    #[test]
    fn test_clahe_u16() {
        let data: Vec<u16> = (0..=1023).collect();
        let img = Matrix::from_vec(32, 32, 1, data);
        let clahe = create_clahe(40.0, Size2i::new(4, 4));
        let dst = clahe.apply_u16(&img).unwrap();
        assert_eq!(dst.rows, 32);
        assert_eq!(dst.cols, 32);
        assert_eq!(dst.channels, 1);
    }

    #[test]
    fn test_calc_hist_2d() {
        let img = Matrix::from_vec(3, 3, 1, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        let hist = calc_hist(
            &[&img, &img],
            &[0, 0],
            None,
            &[3, 3],
            &[RangeSpec::Uniform(0.0, 9.0), RangeSpec::Uniform(0.0, 9.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist.data.len(), 9);
        assert_eq!(hist.data[0], 3.0);
        assert_eq!(hist.data[1], 0.0);
        assert_eq!(hist.data[2], 0.0);
        assert_eq!(hist.data[3], 0.0);
        assert_eq!(hist.data[4], 3.0);
        assert_eq!(hist.data[5], 0.0);
        assert_eq!(hist.data[6], 0.0);
        assert_eq!(hist.data[7], 0.0);
        assert_eq!(hist.data[8], 3.0);
    }

    #[test]
    fn test_calc_hist_empty_images_error() {
        assert!(calc_hist::<u8>(
            &[],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .is_err());
    }

    #[test]
    fn test_calc_hist_mismatched_channels_error() {
        let img = Matrix::from_vec(2, 2, 1, vec![0u8; 4]);
        assert!(calc_hist(
            &[&img],
            &[0, 1],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .is_err());
    }

    #[test]
    fn test_calc_back_project_empty_channels_error() {
        let img = Matrix::from_vec(2, 2, 1, vec![0u8; 4]);
        let hist = Matrix::from_vec(4, 1, 1, vec![0.0; 4]);
        assert!(
            calc_back_project::<u8>(&[&img], &[], &hist, &[RangeSpec::Uniform(0.0, 4.0)], 1.0,)
                .is_err()
        );
    }

    #[test]
    fn test_calc_hist_u16_input() {
        let data: Vec<u16> = vec![0, 100, 200, 300, 400, 500];
        let img = Matrix::from_vec(2, 3, 1, data);
        let hist = calc_hist(
            &[&img],
            &[0],
            None,
            &[3],
            &[RangeSpec::Uniform(0.0, 600.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist.data, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_calc_hist_f32_input() {
        let data: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5];
        let img = Matrix::from_vec(2, 2, 1, data);
        let hist = calc_hist(
            &[&img],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist.data, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_calc_hist_boundary_exclusion() {
        let img = Matrix::from_vec(1, 4, 1, vec![0u8, 4, 8, 12]);
        let hist = calc_hist(
            &[&img],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 16.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist.data, vec![1.0, 1.0, 1.0, 1.0]);

        // Value at exact hi boundary should be excluded
        let img2 = Matrix::from_vec(1, 2, 1, vec![4u8, 4]);
        let hist2 = calc_hist(
            &[&img2],
            &[0],
            None,
            &[2],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist2.data, vec![0.0, 0.0]);

        // Value just below hi should be included
        let img3 = Matrix::from_vec(1, 1, 1, vec![3u8]);
        let hist3 = calc_hist(
            &[&img3],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist3.data[3], 1.0);
    }

    #[test]
    fn test_calc_hist_accumulate() {
        let img = Matrix::from_vec(1, 4, 1, vec![0u8, 1, 2, 3]);
        let h1 = calc_hist(
            &[&img],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(h1.data, vec![1.0, 1.0, 1.0, 1.0]);

        let h2 = calc_hist(
            &[&img],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            true,
            Some(&h1),
        )
        .unwrap();
        assert_eq!(h2.data, vec![2.0, 2.0, 2.0, 2.0]);

        let h3 = calc_hist(
            &[&img],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            true,
            Some(&h2),
        )
        .unwrap();
        assert_eq!(h3.data, vec![3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_compare_hist_edge_cases() {
        let z = Matrix::from_vec(3, 1, 1, vec![0.0, 0.0, 0.0]);
        assert!((compare_hist(&z, &z, HistCompMethods::Correl).unwrap() - 1.0).abs() < 1e-10);
        assert!((compare_hist(&z, &z, HistCompMethods::Intersection).unwrap()).abs() < 1e-10);
        assert!(
            (compare_hist(&z, &z, HistCompMethods::Bhattacharyya).unwrap() - 1.0).abs() < 1e-10
        );

        let p1 = Matrix::from_vec(4, 1, 1, vec![0.5, 0.25, 0.125, 0.125]);
        let p2 = Matrix::from_vec(4, 1, 1, vec![0.5, 0.25, 0.125, 0.125]);
        assert!((compare_hist(&p1, &p2, HistCompMethods::Correl).unwrap() - 1.0).abs() < 1e-10);
        assert!((compare_hist(&p1, &p2, HistCompMethods::ChiSqr).unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_calc_back_project_scale() {
        let img = Matrix::from_vec(1, 4, 1, vec![0u8, 1, 2, 3]);
        let hist = Matrix::from_vec(4, 1, 1, vec![10.0, 20.0, 30.0, 40.0]);
        let bp =
            calc_back_project(&[&img], &[0], &hist, &[RangeSpec::Uniform(0.0, 4.0)], 2.0).unwrap();
        assert_eq!(bp.data, vec![20.0, 40.0, 60.0, 80.0]);
    }

    #[test]
    fn test_calc_hist_errors() {
        let img = Matrix::from_vec(2, 2, 1, vec![0u8; 4]);
        assert!(calc_hist::<u8>(
            &[],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .is_err());
        assert!(calc_hist(
            &[&img],
            &[0, 1],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .is_err());
        let img2 = Matrix::from_vec(3, 3, 1, vec![0u8; 9]);
        assert!(calc_hist(
            &[&img, &img2],
            &[0, 0],
            None,
            &[2, 2],
            &[RangeSpec::Uniform(0.0, 2.0), RangeSpec::Uniform(0.0, 2.0)],
            false,
            None,
        )
        .is_err());
        let hist = Matrix::from_vec(4, 1, 1, vec![0.0; 4]);
        assert!(
            calc_back_project::<u8>(&[&img], &[], &hist, &[RangeSpec::Uniform(0.0, 4.0)], 1.0,)
                .is_err()
        );
    }

    #[test]
    fn test_calc_hist_multichannel_select() {
        let img = Matrix::from_vec(
            1,
            3,
            3,
            vec![
                10, 100, 200, // pixel 0
                20, 150, 250, // pixel 1
                30, 50, 100, // pixel 2
            ],
        );
        let hist = calc_hist(
            &[&img],
            &[1],
            None,
            &[3],
            &[RangeSpec::Uniform(0.0, 200.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist.data, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_clahe_no_clip() {
        let data: Vec<u8> = (0..=255).collect();
        let img = Matrix::from_vec(16, 16, 1, data);
        let clahe = create_clahe(0.0, Size2i::new(4, 4));
        let dst = clahe.apply_u8(&img).unwrap();
        assert_eq!(dst.data.len(), 256);
    }
}
