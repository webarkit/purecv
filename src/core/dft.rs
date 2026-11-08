/*
 *  dft.rs
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

use alloc::vec;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::core::error::Result;
use crate::core::logging::tags;
use crate::core::matrix::Matrix;
use crate::cv_bail;

// ---------------------------------------------------------------------------
// DFT flag constants (matching OpenCV values)
// ---------------------------------------------------------------------------

/// Compute the inverse DFT.
pub const DFT_INVERSE: i32 = 1;
/// Scale the result by `1 / (rows * cols)`.
pub const DFT_SCALE: i32 = 2;
/// Process each row independently (1-D transform per row).
pub const DFT_ROWS: i32 = 4;
/// Force complex (2-channel) output even for real input.
pub const DFT_COMPLEX_OUTPUT: i32 = 16;
/// Return only the real part of the result (1-channel output).
pub const DFT_REAL_OUTPUT: i32 = 32;

// ---------------------------------------------------------------------------
// Optimal DFT size lookup table
// ---------------------------------------------------------------------------

/// Pre-computed table of numbers of the form 2^a * 3^b * 5^c, sorted.
/// These are the sizes for which FFT algorithms are most efficient.
const OPTIMAL_DFT_SIZES: &[i32] = &[
    1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60,
    64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128, 135, 144, 150, 160, 162, 180, 192, 200,
    216, 225, 240, 243, 250, 256, 270, 288, 300, 320, 324, 360, 375, 384, 400, 405, 432, 450, 480,
    486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 729, 750, 768, 800, 810, 864, 900, 960,
    972, 1000, 1024, 1080, 1125, 1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536,
    1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048, 2160, 2187, 2250, 2304, 2400, 2430,
    2500, 2560, 2592, 2700, 2880, 2916, 3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750,
    3840, 3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800, 4860, 5000, 5120, 5184, 5400, 5625,
    5760, 5832, 6000, 6075, 6144, 6250, 6400, 6480, 6561, 6750, 6912, 7200, 7290, 7500, 7680, 7776,
    8000, 8100, 8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000, 10125, 10240, 10368, 10800,
    11250, 11520, 11664, 12000, 12150, 12288, 12500, 12800, 12960, 13122, 13500, 13824, 14400,
    14580, 15000, 15360, 15552, 15625, 16000, 16200, 16384, 16875, 17280, 17496, 18000, 18225,
    18432, 18750, 19200, 19440, 19683, 20000, 20250, 20480, 20736, 21600, 21870, 22500, 23040,
    23328, 24000, 24300, 24576, 25000, 25600, 25920, 26244, 27000, 27648, 28125, 28800, 29160,
    30000, 30375, 30720, 31104, 31250, 32000, 32400, 32768, 33750, 34560, 34992, 36000, 36450,
    36864, 37500, 38400, 38880, 39366, 40000, 40500, 40960, 41472, 43200, 43740, 45000, 46080,
    46656, 46875, 48000, 48600, 49152, 50000, 50625, 51200, 51840, 52488, 54000, 54675, 55296,
    56250, 57600, 58320, 59049, 60000, 60750, 61440, 62208, 62500, 64000, 64800, 65536,
];

/// Returns the optimal DFT size greater than or equal to a given value.
///
/// Optimal sizes are of the form `2^a * 3^b * 5^c` which allow the FFT
/// algorithm to run most efficiently.
///
/// Matches OpenCV's
/// [`getOptimalDFTSize`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/dxt.cpp#L4625-L4641)
/// for every `size` up to 65536 and at the edges: it returns `1` for `0` and `-1` for a negative
/// `size` and for `size >= 2125764000`, the last entry of OpenCV's table.
///
/// Above 65536 it returns the true smallest `2^a * 3^b * 5^c`, while OpenCV's table omits 41 of
/// those numbers (`2 * 3^15` among them), so the two can differ: for `28473477` OpenCV gives
/// `28800000` and this function `28697814`.
pub fn get_optimal_dft_size(size: i32) -> i32 {
    /// The largest size in OpenCV's `optimalDFTSizeTab`; the reference compares as unsigned,
    /// so negative sizes land here too.
    const OPENCV_TABLE_MAX: u32 = 2_125_764_000;
    if size as u32 >= OPENCV_TABLE_MAX {
        return -1;
    }
    if size <= 1 {
        return 1;
    }
    match OPTIMAL_DFT_SIZES.iter().position(|&s| s >= size) {
        Some(idx) => OPTIMAL_DFT_SIZES[idx],
        None => {
            // Fall back to computing: find smallest 2^a * 3^b * 5^c >= size
            compute_optimal_size(size)
        }
    }
}

/// Compute the smallest number >= `size` that is a product of 2, 3, and 5.
fn compute_optimal_size(size: i32) -> i32 {
    let mut best = i32::MAX;
    let mut p5 = 1i64;
    while p5 < size as i64 * 2 {
        let mut p35 = p5;
        while p35 < size as i64 * 2 {
            // Find smallest power of 2 such that p2 * p35 >= size
            let mut p = p35;
            while p < size as i64 {
                p *= 2;
            }
            if p < best as i64 {
                best = p as i32;
            }
            p35 *= 3;
        }
        p5 *= 5;
    }
    best
}

// ---------------------------------------------------------------------------
// DftFloat trait (sealed)
// ---------------------------------------------------------------------------

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Trait for floating-point types that support DFT operations.
///
/// This trait is sealed and only implemented for `f32` and `f64`.
pub trait DftFloat:
    sealed::Sealed + rustfft::FftNum + Default + Copy + Send + Sync + 'static
{
}

impl DftFloat for f32 {}
impl DftFloat for f64 {}

// ---------------------------------------------------------------------------
// DFT implementation
// ---------------------------------------------------------------------------

/// Performs a 1-D or 2-D Discrete Fourier Transform, like OpenCV's
/// [`cv::dft`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/dxt.cpp#L3486-L3534).
///
/// The input matrix must be 1-channel (real) or 2-channel (complex, where
/// channel 0 is real and channel 1 is imaginary).
///
/// # Output layout
/// * Forward transform of real input: the 1-channel **CCS** packing of the spectrum, unless
///   `DFT_COMPLEX_OUTPUT` asks for the full 2-channel spectrum. `DFT_REAL_OUTPUT` changes
///   nothing here.
/// * Inverse transform of 1-channel input: the input is read as CCS and the output is 1-channel
///   real, whatever `DFT_REAL_OUTPUT` or `DFT_COMPLEX_OUTPUT` say.
/// * Complex input: 2-channel output, or its real part with `DFT_REAL_OUTPUT`.
///
/// CCS keeps the non-redundant half of the conjugate-symmetric spectrum `Y` of an `M`-row,
/// `N`-column real matrix, as in the `dft` entry of the
/// [OpenCV docs](https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html):
/// columns `2k-1` and `2k` hold `Re Y(r, k)` and `Im Y(r, k)` for `k = 1 ..= (N-1)/2`;
/// column 0, and column `N-1` when `N` is even, hold the spectrum columns `0` and `N/2` packed
/// down the rows as `Re Y(0)`, `Re Y(1)`, `Im Y(1)`, …, plus `Re Y(M/2)` when `M` is even.
/// A 1×N or N×1 matrix, or each row under `DFT_ROWS`, is packed as a 1-D sequence.
///
/// # Arguments
/// * `src` - Input matrix of type `f32` or `f64`.
/// * `flags` - Combination of `DFT_*` constants (e.g. `DFT_INVERSE | DFT_SCALE`).
/// * `nonzero_rows` - When non-zero, a forward transform reads only the first `nonzero_rows`
///   rows of real input (the rest count as zero), and an inverse transform of CCS input writes
///   only the first `nonzero_rows` output rows (the rest are zero). Pass `0` to process all
///   rows.
///
/// # Returns
/// A matrix containing the DFT result, laid out as above.
pub fn dft<T: DftFloat>(src: &Matrix<T>, flags: i32, nonzero_rows: usize) -> Result<Matrix<T>> {
    let inverse = (flags & DFT_INVERSE) != 0;
    let scale = (flags & DFT_SCALE) != 0;
    let rows_only = (flags & DFT_ROWS) != 0;
    let want_complex_output = (flags & DFT_COMPLEX_OUTPUT) != 0;
    let want_real_output = (flags & DFT_REAL_OUTPUT) != 0;

    if src.channels != 1 && src.channels != 2 {
        cv_bail!(
            tags::CORE,
            InvalidInput,
            "dft: requires 1-channel (real) or 2-channel (complex) input, got {} channels",
            src.channels
        );
    }

    if want_real_output && want_complex_output {
        cv_bail!(
            tags::CORE,
            InvalidInput,
            "dft: DFT_REAL_OUTPUT and DFT_COMPLEX_OUTPUT are mutually exclusive"
        );
    }

    let is_complex_input = src.channels == 2;
    let rows = src.rows;
    let cols = src.cols;

    if rows == 0 || cols == 0 {
        cv_bail!(
            tags::CORE,
            InvalidDimensions,
            "dft: input must not be empty ({}×{})",
            rows,
            cols
        );
    }

    let active_rows = if nonzero_rows > 0 && nonzero_rows < rows {
        nonzero_rows
    } else {
        rows
    };
    // DFT_SCALE divides by the element count, or by the row length with DFT_ROWS.
    let inv_n = if scale {
        let n = if rows_only { cols } else { rows * cols };
        T::from_usize(n).map(|n| T::one() / n)
    } else {
        None
    };
    // The length of the column transforms; 1 when every row is transformed on its own.
    let col_len = if rows_only { 1 } else { rows };

    let mut planner = FftPlanner::<T>::new();

    if inverse && !is_complex_input {
        return Ok(ccs_inverse(src, col_len, active_rows, inv_n, &mut planner));
    }

    // Real input gives CCS unless DFT_COMPLEX_OUTPUT asks for the full spectrum.
    let ccs_output = !is_complex_input && !inverse && !want_complex_output;

    // Build the complex buffer from source
    let mut buf = vec![Complex::<T>::default(); rows * cols];
    let cn = src.channels;

    for r in 0..rows {
        for c in 0..cols {
            if is_complex_input {
                let base = (r * cols + c) * cn;
                let re = src.data[base];
                let im = src.data[base + 1];
                buf[r * cols + c] = Complex::new(re, im);
            } else if r < active_rows {
                let re = src.data[r * cols + c];
                buf[r * cols + c] = Complex::new(re, T::default());
            }
            // Rows beyond active_rows are already zero-initialized
        }
    }

    // Row-wise FFTs; zero rows of real input stay zero.
    let row_fft = if inverse {
        planner.plan_fft_inverse(cols)
    } else {
        planner.plan_fft_forward(cols)
    };
    let fft_rows = if is_complex_input { rows } else { active_rows };
    for row in buf[..fft_rows * cols].chunks_exact_mut(cols) {
        row_fft.process(row);
    }

    // Column-wise FFTs (unless DFT_ROWS). CCS reads only the columns 0 ..= cols/2.
    if col_len > 1 {
        let col_fft = if inverse {
            planner.plan_fft_inverse(rows)
        } else {
            planner.plan_fft_forward(rows)
        };
        let fft_cols = if ccs_output { cols / 2 + 1 } else { cols };
        column_ffts(&mut buf, rows, cols, fft_cols, &*col_fft);
    }

    if ccs_output {
        let mut dst_data = vec![T::default(); rows * cols];
        let block = col_len * cols;
        for (spec, out) in buf
            .chunks_exact(block)
            .zip(dst_data.chunks_exact_mut(block))
        {
            pack_ccs(spec, col_len, cols, out);
        }
        if let Some(inv_n) = inv_n {
            for v in &mut dst_data {
                *v = *v * inv_n;
            }
        }
        return Ok(Matrix {
            rows,
            cols,
            channels: 1,
            data: dst_data,
        });
    }

    // Scale if requested
    if let Some(inv_n) = inv_n {
        for v in &mut buf {
            v.re = v.re * inv_n;
            v.im = v.im * inv_n;
        }
    }

    // Build output matrix
    if want_real_output {
        let mut dst_data = vec![T::default(); rows * cols];
        for (i, v) in buf.iter().enumerate() {
            dst_data[i] = v.re;
        }
        return Ok(Matrix {
            rows,
            cols,
            channels: 1,
            data: dst_data,
        });
    }

    // 2-channel complex output (complex input, or DFT_COMPLEX_OUTPUT)
    let mut dst_data = vec![T::default(); rows * cols * 2];
    for (i, v) in buf.iter().enumerate() {
        dst_data[i * 2] = v.re;
        dst_data[i * 2 + 1] = v.im;
    }
    Ok(Matrix {
        rows,
        cols,
        channels: 2,
        data: dst_data,
    })
}

/// Runs `fft` down each of the first `fft_cols` columns of the row-major `rows × cols` `buf`.
fn column_ffts<T: DftFloat>(
    buf: &mut [Complex<T>],
    rows: usize,
    cols: usize,
    fft_cols: usize,
    fft: &dyn rustfft::Fft<T>,
) {
    let mut col_buf = vec![Complex::<T>::default(); rows];
    for c in 0..fft_cols {
        for (r, v) in col_buf.iter_mut().enumerate() {
            *v = buf[r * cols + c];
        }
        fft.process(&mut col_buf);
        for (r, v) in col_buf.iter().enumerate() {
            buf[r * cols + c] = *v;
        }
    }
}

/// Packs spectrum column `k` of `spec` down column `d` of `out` in 1-D CCS order:
/// `Re Y(0)`, `Re Y(1)`, `Im Y(1)`, …, then `Re Y(rows/2)` when `rows` is even.
fn pack_ccs_column<T: DftFloat>(
    spec: &[Complex<T>],
    rows: usize,
    cols: usize,
    k: usize,
    d: usize,
    out: &mut [T],
) {
    out[d] = spec[k].re;
    for m in 1..=(rows - 1) / 2 {
        let v = spec[m * cols + k];
        out[(2 * m - 1) * cols + d] = v.re;
        out[2 * m * cols + d] = v.im;
    }
    if rows.is_multiple_of(2) {
        out[(rows - 1) * cols + d] = spec[(rows / 2) * cols + k].re;
    }
}

/// Packs the `rows × cols` spectrum `spec` of a real matrix into OpenCV's CCS layout (see
/// [`dft`]). With `rows == 1` this is the 1-D packing of a single row.
fn pack_ccs<T: DftFloat>(spec: &[Complex<T>], rows: usize, cols: usize, out: &mut [T]) {
    for (row, dst) in spec.chunks_exact(cols).zip(out.chunks_exact_mut(cols)) {
        for k in 1..=(cols - 1) / 2 {
            dst[2 * k - 1] = row[k].re;
            dst[2 * k] = row[k].im;
        }
    }
    pack_ccs_column(spec, rows, cols, 0, 0, out);
    if cols.is_multiple_of(2) {
        pack_ccs_column(spec, rows, cols, cols / 2, cols - 1, out);
    }
}

/// The inverse of [`pack_ccs_column`]: fills spectrum column `k` of `spec` from column `d` of
/// the CCS `src`, completing it by conjugate symmetry `Y(rows - m) = conj Y(m)`.
fn unpack_ccs_column<T: DftFloat>(
    src: &[T],
    rows: usize,
    cols: usize,
    d: usize,
    k: usize,
    spec: &mut [Complex<T>],
) {
    spec[k] = Complex::new(src[d], T::zero());
    for m in 1..=(rows - 1) / 2 {
        let v = Complex::new(src[(2 * m - 1) * cols + d], src[2 * m * cols + d]);
        spec[m * cols + k] = v;
        spec[(rows - m) * cols + k] = v.conj();
    }
    if rows.is_multiple_of(2) {
        spec[(rows / 2) * cols + k] = Complex::new(src[(rows - 1) * cols + d], T::zero());
    }
}

/// Unpacks the CCS `src` into the spectrum columns `0 ..= cols/2` of `spec`. The other columns
/// follow from conjugate symmetry along each row once the column transforms are done.
fn unpack_ccs<T: DftFloat>(src: &[T], rows: usize, cols: usize, spec: &mut [Complex<T>]) {
    for (row, s) in spec.chunks_exact_mut(cols).zip(src.chunks_exact(cols)) {
        for k in 1..=(cols - 1) / 2 {
            row[k] = Complex::new(s[2 * k - 1], s[2 * k]);
        }
    }
    unpack_ccs_column(src, rows, cols, 0, 0, spec);
    if cols.is_multiple_of(2) {
        unpack_ccs_column(src, rows, cols, cols - 1, cols / 2, spec);
    }
}

/// Inverse transform of 1-channel CCS input to 1-channel real output, as OpenCV's
/// `InvCCSToReal`: the column transforms read every row, then only the first `active_rows`
/// output rows are computed and the rest stay zero. `col_len` is 1 under `DFT_ROWS`.
fn ccs_inverse<T: DftFloat>(
    src: &Matrix<T>,
    col_len: usize,
    active_rows: usize,
    inv_n: Option<T>,
    planner: &mut FftPlanner<T>,
) -> Matrix<T> {
    let (rows, cols) = (src.rows, src.cols);
    let mut buf = vec![Complex::<T>::default(); rows * cols];
    for (spec, s) in buf
        .chunks_exact_mut(col_len * cols)
        .zip(src.data.chunks_exact(col_len * cols))
    {
        unpack_ccs(s, col_len, cols, spec);
    }
    if col_len > 1 {
        let col_fft = planner.plan_fft_inverse(rows);
        column_ffts(&mut buf, rows, cols, cols / 2 + 1, &*col_fft);
    }

    let row_fft = planner.plan_fft_inverse(cols);
    let scale = inv_n.unwrap_or_else(T::one);
    let mut dst_data = vec![T::default(); rows * cols];
    for (row, dst) in buf
        .chunks_exact_mut(cols)
        .zip(dst_data.chunks_exact_mut(cols))
        .take(active_rows)
    {
        // Each row is now the spectrum of a real row: Y(cols - k) = conj Y(k).
        for k in cols / 2 + 1..cols {
            row[k] = row[cols - k].conj();
        }
        row_fft.process(row);
        for (d, v) in dst.iter_mut().zip(row.iter()) {
            *d = v.re * scale;
        }
    }
    Matrix {
        rows,
        cols,
        channels: 1,
        data: dst_data,
    }
}

/// Computes the Inverse Discrete Fourier Transform of a 1D or 2D array.
///
/// This is `dft(src, flags | DFT_INVERSE, nonzero_rows)`, as in OpenCV. The output is **not**
/// scaled unless `DFT_SCALE` is passed in `flags`; `DFT_SCALE` divides by the element count
/// (by the row length with `DFT_ROWS`). A 1-channel `src` is read as the CCS packing that
/// [`dft`] returns for real input, and the output is 1-channel real.
pub fn idft<T: DftFloat>(src: &Matrix<T>, flags: i32, nonzero_rows: usize) -> Result<Matrix<T>> {
    dft(src, flags | DFT_INVERSE, nonzero_rows)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_optimal_dft_size_exact() {
        assert_eq!(get_optimal_dft_size(1), 1);
        assert_eq!(get_optimal_dft_size(2), 2);
        assert_eq!(get_optimal_dft_size(8), 8);
        assert_eq!(get_optimal_dft_size(256), 256);
    }

    #[test]
    fn test_get_optimal_dft_size_round_up() {
        assert_eq!(get_optimal_dft_size(7), 8);
        assert_eq!(get_optimal_dft_size(13), 15);
        assert_eq!(get_optimal_dft_size(17), 18);
        assert_eq!(get_optimal_dft_size(97), 100);
    }

    #[test]
    fn test_get_optimal_dft_size_zero_negative() {
        assert_eq!(get_optimal_dft_size(0), 1);
        assert_eq!(get_optimal_dft_size(-5), -1);
    }

    #[test]
    fn test_get_optimal_dft_size_large() {
        // Should compute beyond table
        let s = get_optimal_dft_size(70000);
        assert!(s >= 70000);
        // Verify it's a product of 2, 3, 5
        let mut n = s;
        while n % 2 == 0 {
            n /= 2;
        }
        while n % 3 == 0 {
            n /= 3;
        }
        while n % 5 == 0 {
            n /= 5;
        }
        assert_eq!(n, 1, "Optimal size {} is not a product of 2,3,5", s);
    }

    #[test]
    fn test_dft_inverse_roundtrip_f32() {
        // Create a small real matrix
        let src = Matrix::from_vec(2, 4, 1, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Forward DFT
        let freq = dft(&src, DFT_COMPLEX_OUTPUT, 0).unwrap();
        assert_eq!(freq.channels, 2);

        // Inverse DFT with scale
        let back = dft(&freq, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT, 0).unwrap();
        assert_eq!(back.channels, 1);
        assert_eq!(back.rows, 2);
        assert_eq!(back.cols, 4);

        // Should recover original values within tolerance
        for i in 0..src.data.len() {
            assert!(
                (back.data[i] - src.data[i]).abs() < 1e-4,
                "Mismatch at index {}: got {} expected {}",
                i,
                back.data[i],
                src.data[i]
            );
        }
    }

    #[test]
    fn test_dft_inverse_roundtrip_f64() {
        let src = Matrix::from_vec(4, 4, 1, (1..=16).map(|x| x as f64).collect());

        let freq = dft(&src, DFT_COMPLEX_OUTPUT, 0).unwrap();
        let back = dft(&freq, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT, 0).unwrap();

        for i in 0..src.data.len() {
            assert!(
                (back.data[i] - src.data[i]).abs() < 1e-10,
                "Mismatch at index {}: got {} expected {}",
                i,
                back.data[i],
                src.data[i]
            );
        }
    }

    #[test]
    fn test_dft_dc_component() {
        // DFT of constant signal: DC = N * value, all others = 0
        let val = 3.0f64;
        let n = 4;
        let src = Matrix::from_vec(1, n, 1, vec![val; n]);

        let freq = dft(&src, DFT_COMPLEX_OUTPUT, 0).unwrap();
        // DC component (0,0) real part should be n * val
        // freq is 2-channel: data layout is [re0, im0, re1, im1, ...]
        let dc_re = freq.data[0]; // re of (0,0)
        assert!(
            (dc_re - (n as f64 * val)).abs() < 1e-10,
            "DC component: got {} expected {}",
            dc_re,
            n as f64 * val
        );
        // All other components should be zero
        for c in 1..n {
            let re = freq.data[c * 2];
            let im = freq.data[c * 2 + 1];
            assert!(re.abs() < 1e-10, "Non-zero real at col {}: {}", c, re);
            assert!(im.abs() < 1e-10, "Non-zero imag at col {}: {}", c, im);
        }
    }

    #[test]
    fn test_dft_rows_flag() {
        // With DFT_ROWS, each row is transformed independently
        let src = Matrix::from_vec(2, 4, 1, vec![1.0f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);

        let freq_rows = dft(&src, DFT_ROWS | DFT_COMPLEX_OUTPUT, 0).unwrap();
        let freq_2d = dft(&src, DFT_COMPLEX_OUTPUT, 0).unwrap();

        // Row-wise result should differ from full 2D result
        let mut any_diff = false;
        for i in 0..freq_rows.data.len() {
            if (freq_rows.data[i] - freq_2d.data[i]).abs() > 1e-10 {
                any_diff = true;
                break;
            }
        }
        assert!(
            any_diff,
            "DFT_ROWS should produce different result than 2D DFT"
        );
    }

    #[test]
    fn test_dft_scale_flag() {
        let src = Matrix::from_vec(2, 2, 1, vec![1.0f64, 2.0, 3.0, 4.0]);

        let unscaled = dft(&src, DFT_COMPLEX_OUTPUT, 0).unwrap();
        let scaled = dft(&src, DFT_COMPLEX_OUTPUT | DFT_SCALE, 0).unwrap();

        let n = (src.rows * src.cols) as f64;
        for i in 0..unscaled.data.len() {
            assert!(
                (scaled.data[i] - unscaled.data[i] / n).abs() < 1e-10,
                "Scale mismatch at index {}",
                i
            );
        }
    }

    /// CCS and its inverse agree with the full complex spectrum on larger shapes than the
    /// parity fixtures: every packed value appears in the `DFT_COMPLEX_OUTPUT` spectrum, and
    /// `idft(ccs, DFT_SCALE)` returns the input.
    #[test]
    fn test_ccs_matches_complex_spectrum_and_round_trips() {
        for (rows, cols) in [(7, 9), (8, 10), (8, 9), (7, 10), (16, 1), (1, 15), (2, 2)] {
            let data = (0..rows * cols)
                .map(|i| ((i * 37 + 11) % 23) as f64 - 11.0)
                .collect();
            let src = Matrix::from_vec(rows, cols, 1, data);
            let ccs = dft(&src, 0, 0).unwrap();
            let full = dft(&src, DFT_COMPLEX_OUTPUT, 0).unwrap();
            let spec: alloc::vec::Vec<Complex<f64>> = full
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[re, im]| Complex::new(re, im))
                .collect();
            let mut packed = vec![0.0f64; rows * cols];
            pack_ccs(&spec, rows, cols, &mut packed);
            for (i, (a, b)) in ccs.data.iter().zip(&packed).enumerate() {
                assert!((a - b).abs() < 1e-9, "{rows}x{cols} CCS[{i}]: {a} vs {b}");
            }
            let back = idft(&ccs, DFT_SCALE, 0).unwrap();
            assert_eq!(back.channels, 1);
            for (i, (a, b)) in back.data.iter().zip(&src.data).enumerate() {
                assert!((a - b).abs() < 1e-9, "{rows}x{cols} back[{i}]: {a} vs {b}");
            }
        }
    }

    #[test]
    fn test_dft_multichannel_error() {
        let src = Matrix::from_vec(2, 2, 3, vec![0.0f32; 12]);
        assert!(dft(&src, 0, 0).is_err());
    }

    #[test]
    fn test_dft_empty_error() {
        let src = Matrix::<f32>::new(0, 0, 1);
        assert!(dft(&src, 0, 0).is_err());
    }

    #[test]
    fn test_dft_complex_input() {
        // 2-channel (complex) input
        let src = Matrix::from_vec(1, 4, 2, vec![1.0f64, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0]);

        let freq = dft(&src, 0, 0).unwrap();
        assert_eq!(freq.channels, 2);

        // Inverse should recover original
        let back = dft(&freq, DFT_INVERSE | DFT_SCALE, 0).unwrap();
        for i in 0..src.data.len() {
            assert!(
                (back.data[i] - src.data[i]).abs() < 1e-10,
                "Complex roundtrip mismatch at {}",
                i
            );
        }
    }
}
