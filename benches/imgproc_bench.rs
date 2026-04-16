/*
 *  imgproc_bench.rs
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

use criterion::{criterion_group, criterion_main, Criterion};
use purecv::core::types::BorderTypes;
use purecv::core::{Matrix, Point2i, Size2i};
use purecv::imgproc::derivatives::{laplacian, scharr, sobel};
use purecv::imgproc::edge::canny;
use purecv::imgproc::filter::{bilateral_filter, box_filter, gaussian_blur};
use purecv::imgproc::threshold::{threshold, ThresholdTypes};
use purecv::imgproc::{cvt_color, ColorConversionCode};
use std::hint::black_box;

fn bench_imgproc(c: &mut Criterion) {
    let size = 1024;

    // cvt_color_rgb_to_gray benchmark setup
    let img_rgb = Matrix::<u8>::new(size, size, 3);

    c.bench_function("cvt_color_rgb2gray_1024x1024", |b| {
        b.iter(|| cvt_color(black_box(&img_rgb), ColorConversionCode::COLOR_RGB2GRAY).unwrap())
    });

    // box_filter benchmark setup
    let img_gray = Matrix::<f32>::new(size, size, 1);

    c.bench_function("box_filter_3x3_1024x1024", |b| {
        b.iter(|| {
            box_filter(
                black_box(&img_gray),
                Size2i::new(3, 3),
                Point2i::new(-1, -1),
                true,
                BorderTypes::default(),
            )
            .unwrap()
        })
    });

    // sobel benchmark setup
    c.bench_function("sobel_3x3_1024x1024", |b| {
        b.iter(|| {
            sobel(
                black_box(&img_gray),
                1,
                1,
                3,
                1.0,
                0.0,
                BorderTypes::default(),
            )
            .unwrap()
        })
    });

    // scharr benchmark setup
    c.bench_function("scharr_x_1024x1024", |b| {
        b.iter(|| scharr(black_box(&img_gray), 1, 0, 1.0, 0.0, BorderTypes::default()).unwrap())
    });

    // --- New benchmarks ---

    // threshold binary (u8)
    let mut img_gray_u8 = Matrix::<u8>::new(size, size, 1);
    for (i, v) in img_gray_u8.data.iter_mut().enumerate() {
        *v = (i % 256) as u8;
    }

    c.bench_function("threshold_binary_1024x1024_u8", |b| {
        b.iter(|| {
            threshold(
                black_box(&img_gray_u8),
                128.0,
                255.0,
                ThresholdTypes::THRESH_BINARY,
            )
            .unwrap()
        })
    });

    // cvt_color BGR to gray
    let img_bgr = Matrix::<u8>::new(size, size, 3);

    c.bench_function("cvt_color_bgr2gray_1024x1024", |b| {
        b.iter(|| cvt_color(black_box(&img_bgr), ColorConversionCode::COLOR_BGR2GRAY).unwrap())
    });

    // cvt_color RGBA to gray
    let img_rgba = Matrix::<u8>::new(size, size, 4);

    c.bench_function("cvt_color_rgba2gray_1024x1024", |b| {
        b.iter(|| cvt_color(black_box(&img_rgba), ColorConversionCode::COLOR_RGBA2GRAY).unwrap())
    });

    // gaussian_blur 3x3
    c.bench_function("gaussian_blur_3x3_1024x1024", |b| {
        b.iter(|| {
            gaussian_blur(
                black_box(&img_gray),
                Size2i::new(3, 3),
                1.0,
                0.0,
                BorderTypes::default(),
            )
            .unwrap()
        })
    });

    // gaussian_blur 5x5
    c.bench_function("gaussian_blur_5x5_1024x1024", |b| {
        b.iter(|| {
            gaussian_blur(
                black_box(&img_gray),
                Size2i::new(5, 5),
                1.0,
                0.0,
                BorderTypes::default(),
            )
            .unwrap()
        })
    });

    // laplacian 3x3
    c.bench_function("laplacian_3x3_1024x1024", |b| {
        b.iter(|| laplacian(black_box(&img_gray), 3, 1.0, 0.0, BorderTypes::default()).unwrap())
    });

    // canny edge detection (u8 single-channel)
    c.bench_function("canny_1024x1024", |b| {
        b.iter(|| canny(black_box(&img_gray_u8), 50.0, 150.0, 3, false).unwrap())
    });

    // bilateral_filter
    let img_512 = Matrix::<u8>::new(512, 512, 1);
    c.bench_function("bilateral_filter_512x512_u8", |b| {
        b.iter(|| {
            bilateral_filter(black_box(&img_512), -1, 25.0, 10.0, BorderTypes::default()).unwrap()
        })
    });

    // Sobel f32 with realistic data — exercises the SIMD fast-path in fast_deriv_3x3
    let mut img_f32 = Matrix::<f32>::new(size, size, 1);
    for (i, v) in img_f32.data.iter_mut().enumerate() {
        // Gradient-like pattern so the kernel produces non-trivial values
        let row = (i / size) as f32;
        let col = (i % size) as f32;
        *v = (row * 0.25 + col * 0.1).sin() * 128.0 + 128.0;
    }

    c.bench_function("sobel_3x3_f32_dx_1024x1024", |b| {
        b.iter(|| {
            sobel(
                black_box(&img_f32),
                1,
                0,
                3,
                1.0,
                0.0,
                BorderTypes::default(),
            )
            .unwrap()
        })
    });

    c.bench_function("sobel_3x3_f32_dy_1024x1024", |b| {
        b.iter(|| {
            sobel(
                black_box(&img_f32),
                0,
                1,
                3,
                1.0,
                0.0,
                BorderTypes::default(),
            )
            .unwrap()
        })
    });
}

criterion_group!(benches, bench_imgproc);
criterion_main!(benches);
