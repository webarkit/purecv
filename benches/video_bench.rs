/*
 *  video_bench.rs
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

//! Video module benchmarks.
//!
//! Run with one of the following commands to measure parallel / SIMD gains:
//!
//! ```sh
//! # Standard (sequential, no SIMD)
//! cargo bench --bench video_bench --no-default-features
//!
//! # SIMD only (sequential + pulp auto-vectorisation)
//! RUSTFLAGS="-C target-cpu=native" cargo bench --bench video_bench \
//!     --no-default-features --features simd
//!
//! # Parallel only (rayon multi-threading)
//! cargo bench --bench video_bench --features parallel
//!
//! # Parallel + SIMD (maximum throughput)
//! RUSTFLAGS="-C target-cpu=native" cargo bench --bench video_bench \
//!     --features parallel,simd
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use purecv::core::types::{BorderTypes, Point2f, Size2i, TermCriteria, TermType};
use purecv::core::Matrix;
use purecv::video::optical_flow::{build_optical_flow_pyramid, calc_optical_flow_pyramid_lk};
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a smooth Gaussian-blob grayscale image of `size × size`.
fn make_blob_image(size: usize, sigma: f32) -> Vec<u8> {
    let cx = size as f32 / 2.0;
    let cy = size as f32 / 2.0;
    let mut data = vec![0u8; size * size];
    for r in 0..size {
        for c in 0..size {
            let dx = c as f32 - cx;
            let dy = r as f32 - cy;
            let val = (200.0 * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()).round() as u8;
            data[r * size + c] = val;
        }
    }
    data
}

/// Shift an image by `shift` pixels to the right (column-wise copy).
fn shift_image(src: &[u8], size: usize, shift: usize) -> Vec<u8> {
    let mut dst = vec![0u8; size * size];
    for r in 0..size {
        for c in 0..size {
            let src_c = c.saturating_sub(shift);
            dst[r * size + c] = src[r * size + src_c];
        }
    }
    dst
}

/// Build a regular grid of feature points on a `size × size` image.
fn make_grid_points(size: usize, step: usize) -> Vec<Point2f> {
    let mut pts = Vec::new();
    for r in (step..size - step).step_by(step) {
        for c in (step..size - step).step_by(step) {
            pts.push(Point2f::new(c as f32, r as f32));
        }
    }
    pts
}

// ---------------------------------------------------------------------------
// build_optical_flow_pyramid
// ---------------------------------------------------------------------------

/// Measures `build_optical_flow_pyramid` with and without spatial derivatives.
///
/// Benchmark names:
/// * `build_optical_flow_pyramid/no_deriv` — pyramid only
/// * `build_optical_flow_pyramid/with_deriv` — pyramid + Sobel Ix, Iy per level
///   (this is where the `parallel` feature gives the most gain in this function)
fn bench_build_pyramid(c: &mut Criterion) {
    let size = 512usize;
    let data = make_blob_image(size, 60.0);
    let img = Matrix::<u8>::from_vec(size, size, 1, data);

    let win_size = Size2i::new(21, 21);
    let max_level = 3i32 as usize;
    let pyr_border = BorderTypes::Reflect101;
    let deriv_border = BorderTypes::Constant;

    let mut group = c.benchmark_group("build_optical_flow_pyramid");

    group.bench_with_input(
        BenchmarkId::new("no_deriv", format!("{size}x{size}")),
        &img,
        |b, img| {
            b.iter(|| {
                build_optical_flow_pyramid(
                    black_box(img),
                    win_size,
                    max_level,
                    false,
                    pyr_border,
                    deriv_border,
                )
                .unwrap()
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("with_deriv", format!("{size}x{size}")),
        &img,
        |b, img| {
            b.iter(|| {
                build_optical_flow_pyramid(
                    black_box(img),
                    win_size,
                    max_level,
                    true,
                    pyr_border,
                    deriv_border,
                )
                .unwrap()
            })
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// calc_optical_flow_pyramid_lk
// ---------------------------------------------------------------------------

/// Measures `calc_optical_flow_pyramid_lk` at different point-set sizes.
///
/// The parallel feature parallelises the outer `for i in 0..n_pts` loop,
/// so speedup scales with the number of tracked points.
///
/// Benchmark names (examples):
/// * `calc_optical_flow_pyr_lk/512x512_pts_49`
/// * `calc_optical_flow_pyr_lk/512x512_pts_196`
fn bench_optical_flow(c: &mut Criterion) {
    const SIZE: usize = 512;
    const SHIFT: usize = 3;

    let prev_data = make_blob_image(SIZE, 40.0);
    let next_data = shift_image(&prev_data, SIZE, SHIFT);

    let prev = Matrix::<u8>::from_vec(SIZE, SIZE, 1, prev_data);
    let next = Matrix::<u8>::from_vec(SIZE, SIZE, 1, next_data);

    let criteria = TermCriteria::new(TermType::Both, 30, 0.01);
    let win_size = Size2i::new(21, 21);
    let max_level = 3;

    let mut group = c.benchmark_group("calc_optical_flow_pyr_lk");

    // 7×7 = 49 points (baseline — original benchmark)
    let pts_49 = make_grid_points(SIZE, SIZE / 8);
    group.bench_with_input(
        BenchmarkId::new(format!("{SIZE}x{SIZE}"), "pts_49"),
        &pts_49,
        |b, pts| {
            b.iter(|| {
                calc_optical_flow_pyramid_lk(
                    black_box(&prev),
                    black_box(&next),
                    black_box(pts),
                    None,
                    win_size,
                    max_level,
                    criteria,
                    0,
                    1e-4,
                )
                .unwrap()
            })
        },
    );

    // 14×14 ≈ 196 points (shows parallel scaling)
    let pts_196 = make_grid_points(SIZE, SIZE / 16);
    group.bench_with_input(
        BenchmarkId::new(format!("{SIZE}x{SIZE}"), "pts_196"),
        &pts_196,
        |b, pts| {
            b.iter(|| {
                calc_optical_flow_pyramid_lk(
                    black_box(&prev),
                    black_box(&next),
                    black_box(pts),
                    None,
                    win_size,
                    max_level,
                    criteria,
                    0,
                    1e-4,
                )
                .unwrap()
            })
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// calc_optical_flow_pyramid_lk — window-size sensitivity
// ---------------------------------------------------------------------------

/// Measures `calc_optical_flow_pyramid_lk` for different window sizes.
///
/// Larger windows give the SIMD accumulation kernels more work per call,
/// which improves the SIMD amortisation ratio.
fn bench_optical_flow_win_size(c: &mut Criterion) {
    const SIZE: usize = 512;
    const SHIFT: usize = 3;

    let prev_data = make_blob_image(SIZE, 40.0);
    let next_data = shift_image(&prev_data, SIZE, SHIFT);

    let prev = Matrix::<u8>::from_vec(SIZE, SIZE, 1, prev_data);
    let next = Matrix::<u8>::from_vec(SIZE, SIZE, 1, next_data);

    let criteria = TermCriteria::new(TermType::Both, 30, 0.01);
    let max_level = 3;
    let pts = make_grid_points(SIZE, SIZE / 8); // 49 points

    let mut group = c.benchmark_group("calc_optical_flow_pyr_lk_win_size");

    for &win in &[11u32, 21, 31] {
        let win_size = Size2i::new(win as i32, win as i32);
        group.bench_with_input(BenchmarkId::new("win", win), &win_size, |b, &ws| {
            b.iter(|| {
                calc_optical_flow_pyramid_lk(
                    black_box(&prev),
                    black_box(&next),
                    black_box(&pts),
                    None,
                    ws,
                    max_level,
                    criteria,
                    0,
                    1e-4,
                )
                .unwrap()
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_build_pyramid,
    bench_optical_flow,
    bench_optical_flow_win_size
);
criterion_main!(benches);
