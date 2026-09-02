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
use purecv::core::{Matrix, Point2i, Size2i, CV_PI};
use purecv::imgproc::derivatives::{laplacian, scharr, sobel};
use purecv::imgproc::edge::canny;
use purecv::imgproc::feature::corner_harris;
use purecv::imgproc::filter::{bilateral_filter, box_filter, gaussian_blur};
use purecv::imgproc::histogram::{calc_back_project, calc_hist, equalize_hist, Clahe, RangeSpec};
use purecv::imgproc::hough::{hough_circles, hough_lines, hough_lines_p};
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

    // corner_harris
    c.bench_function("corner_harris_1024x1024_f32", |b| {
        b.iter(|| corner_harris(black_box(&img_f32), 3, 3, 0.04, BorderTypes::default()).unwrap())
    });

    // hough_lines (Standard) - Using a smaller size for standard hough as it is slower
    let size_hough = 512;
    let mut img_hough = Matrix::<u8>::new(size_hough, size_hough, 1);
    // Add some "lines" to the image
    for i in 0..size_hough {
        if let Some(v) = img_hough.at_mut(i as i32, i as i32, 0) {
            *v = 255;
        }
        if let Some(v) = img_hough.at_mut(i as i32, (size_hough - 1 - i) as i32, 0) {
            *v = 255;
        }
    }

    c.bench_function("hough_lines_512x512", |b| {
        b.iter(|| hough_lines(black_box(&img_hough), 1.0, CV_PI / 180.0, 50, 0.0, CV_PI).unwrap())
    });

    // hough_lines_p (Probabilistic)
    c.bench_function("hough_lines_p_512x512", |b| {
        b.iter(|| hough_lines_p(black_box(&img_hough), 1.0, CV_PI / 180.0, 50, 50.0, 10.0).unwrap())
    });

    // hough_circles
    c.bench_function("hough_circles_512x512", |b| {
        b.iter(|| {
            hough_circles(
                black_box(&img_hough),
                1.0,   // dp
                20.0,  // min_dist
                100.0, // param1
                30.0,  // param2
                0,     // min_radius
                0,     // max_radius
            )
            .unwrap()
        })
    });

    // calc_hist benchmark setup
    let mut img_hist = Matrix::<u8>::new(size, size, 1);
    for (i, p) in img_hist.data.iter_mut().enumerate() {
        *p = (i % 256) as u8;
    }
    let hist_ranges = [RangeSpec::Uniform(0.0, 256.0)];

    c.bench_function("calc_hist_1024x1024", |b| {
        b.iter(|| {
            calc_hist(
                black_box(&[&img_hist]),
                &[0],
                None,
                &[256],
                &hist_ranges,
                false,
                None,
            )
            .unwrap()
        })
    });

    // calc_back_project benchmark setup
    let hist_for_backproj =
        calc_hist(&[&img_hist], &[0], None, &[256], &hist_ranges, false, None).unwrap();

    c.bench_function("calc_back_project_1024x1024", |b| {
        b.iter(|| {
            calc_back_project(
                black_box(&[&img_hist]),
                &[0],
                &[256],
                &hist_for_backproj,
                &hist_ranges,
                1.0,
            )
            .unwrap()
        })
    });

    // equalize_hist benchmark setup
    c.bench_function("equalize_hist_1024x1024", |b| {
        b.iter(|| equalize_hist(black_box(&img_hist)).unwrap())
    });

    // Clahe::apply_u8 benchmark setup
    let clahe = Clahe::new(2.0, Size2i::new(8, 8));

    c.bench_function("clahe_apply_u8_1024x1024", |b| {
        b.iter(|| clahe.apply_u8(black_box(&img_hist)).unwrap())
    });
    // compare_hist benchmark setup
    let mut hist1 = Matrix::<f32>::new(256, 1, 1);
    let mut hist2 = Matrix::<f32>::new(256, 1, 1);
    for (i, p) in hist1.data.iter_mut().enumerate() {
        *p = (i as f32 * 0.1).sin().abs();
    }
    for (i, p) in hist2.data.iter_mut().enumerate() {
        *p = (i as f32 * 0.1).cos().abs();
    }

    c.bench_function("compare_hist_correl_256", |b| {
        b.iter(|| {
            purecv::imgproc::histogram::compare_hist(
                black_box(&hist1),
                black_box(&hist2),
                purecv::imgproc::histogram::HistCompMethods::Correl,
            )
            .unwrap()
        })
    });

    c.bench_function("compare_hist_intersection_256", |b| {
        b.iter(|| {
            purecv::imgproc::histogram::compare_hist(
                black_box(&hist1),
                black_box(&hist2),
                purecv::imgproc::histogram::HistCompMethods::Intersection,
            )
            .unwrap()
        })
    });

    c.bench_function("compare_hist_kullback_256", |b| {
        b.iter(|| {
            purecv::imgproc::histogram::compare_hist(
                black_box(&hist1),
                black_box(&hist2),
                purecv::imgproc::histogram::HistCompMethods::KullbackLeibler,
            )
            .unwrap()
        })
    });
}

criterion_group!(benches, bench_imgproc);
criterion_main!(benches);
