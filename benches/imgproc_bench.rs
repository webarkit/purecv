/*
 *  imgproc_bench.rs
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

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use purecv::core::{Matrix, Size2i, Point2i};
use purecv::core::types::BorderTypes;
use purecv::imgproc::{cvt_color, ColorConversionCode};
use purecv::imgproc::filter::box_filter;
use purecv::imgproc::derivatives::{sobel, scharr};

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
        b.iter(|| box_filter(black_box(&img_gray), Size2i::new(3, 3), Point2i::new(-1, -1), true, BorderTypes::default()).unwrap())
    });

    // sobel benchmark setup
    c.bench_function("sobel_3x3_1024x1024", |b| {
        b.iter(|| sobel(black_box(&img_gray), 1, 1, 3, 1.0, 0.0, BorderTypes::default()).unwrap())
    });

    // scharr benchmark setup
    c.bench_function("scharr_x_1024x1024", |b| {
        b.iter(|| scharr(black_box(&img_gray), 1, 0, 1.0, 0.0, BorderTypes::default()).unwrap())
    });
}

criterion_group!(benches, bench_imgproc);
criterion_main!(benches);
