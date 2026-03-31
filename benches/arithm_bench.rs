/*
 *  arithm_bench.rs
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
use purecv::core::arithm::{
    absdiff, add, add_weighted, bitwise_and, convert_scale_abs, count_non_zero, divide, dot, gemm,
    lut, magnitude, mean, min, multiply, norm, normalize, sqrt, subtract, sum,
};
use purecv::core::types::NormTypes;
use purecv::core::Matrix;

fn bench_arithm(c: &mut Criterion) {
    let size = 1024;
    let mut m1 = Matrix::<f32>::new(size, size, 3);
    let mut m2 = Matrix::<f32>::new(size, size, 3);

    // Fill with some data
    m1.data.fill(2.0);
    m2.data.fill(1.5);

    c.bench_function("matrix_add_1024x1024x3_f32", |b| {
        b.iter(|| add(black_box(&m1), black_box(&m2)).unwrap())
    });

    c.bench_function("matrix_sub_1024x1024x3_f32", |b| {
        b.iter(|| subtract(black_box(&m1), black_box(&m2)).unwrap())
    });

    c.bench_function("matrix_mul_1024x1024x3_f32", |b| {
        b.iter(|| multiply(black_box(&m1), black_box(&m2)).unwrap())
    });

    c.bench_function("matrix_div_1024x1024x3_f32", |b| {
        b.iter(|| divide(black_box(&m1), black_box(&m2)).unwrap())
    });

    // --- New benchmarks ---

    // dot product (1-channel for reduction benchmark)
    let mut v1 = Matrix::<f32>::new(size, size, 1);
    let mut v2 = Matrix::<f32>::new(size, size, 1);
    v1.data.fill(1.5);
    v2.data.fill(2.5);

    c.bench_function("dot_1024x1024_f32", |b| {
        b.iter(|| dot(black_box(&v1), black_box(&v2)).unwrap())
    });

    // magnitude: sqrt(x^2 + y^2)
    c.bench_function("magnitude_1024x1024_f32", |b| {
        let mut dst = Matrix::<f32>::new(size, size, 1);
        b.iter(|| magnitude(black_box(&v1), black_box(&v2), &mut dst).unwrap())
    });

    // norm L2
    c.bench_function("norm_l2_1024x1024_f32", |b| {
        b.iter(|| norm(black_box(&v1), NormTypes::L2, None).unwrap())
    });

    // add_weighted (FMA: dst = a*alpha + b*beta + gamma)
    c.bench_function("add_weighted_1024x1024_f32", |b| {
        b.iter(|| add_weighted(black_box(&m1), 0.7, black_box(&m2), 0.3, 1.0).unwrap())
    });

    // convert_scale_abs: dst = |src * alpha + beta|, clamped to u8
    c.bench_function("convert_scale_abs_1024x1024_f32", |b| {
        b.iter(|| convert_scale_abs(black_box(&v1), 10.0, 5.0).unwrap())
    });

    // sum (reduction)
    c.bench_function("sum_1024x1024_f32", |b| b.iter(|| sum(black_box(&v1))));

    // mean (sum + divide)
    c.bench_function("mean_1024x1024_f32", |b| b.iter(|| mean(black_box(&v1))));

    // bitwise_and (u8)
    let mut u1 = Matrix::<u8>::new(size, size, 1);
    let mut u2 = Matrix::<u8>::new(size, size, 1);
    u1.data.fill(0xAA);
    u2.data.fill(0x55);

    c.bench_function("bitwise_and_1024x1024_u8", |b| {
        b.iter(|| bitwise_and(black_box(&u1), black_box(&u2)).unwrap())
    });

    // min / max (element-wise)
    c.bench_function("min_1024x1024x3_f32", |b| {
        b.iter(|| min(black_box(&m1), black_box(&m2)).unwrap())
    });

    // absdiff
    c.bench_function("absdiff_1024x1024x3_f32", |b| {
        b.iter(|| absdiff(black_box(&m1), black_box(&m2)).unwrap())
    });

    // normalize (MINMAX)
    c.bench_function("normalize_1024x1024_f32", |b| {
        let mut dst = Matrix::<f32>::new(size, size, 1);
        b.iter(|| {
            normalize(
                black_box(&v1),
                &mut dst,
                0.0,
                1.0,
                NormTypes::MinMax,
                -1,
                None,
            )
            .unwrap()
        })
    });

    // sqrt (unary math)
    c.bench_function("sqrt_1024x1024_f32", |b| {
        b.iter(|| sqrt(black_box(&v1)).unwrap())
    });

    // gemm (smaller size: 256x256)
    let gemm_size = 256;
    let mut gm1 = Matrix::<f32>::new(gemm_size, gemm_size, 1);
    let mut gm2 = Matrix::<f32>::new(gemm_size, gemm_size, 1);
    let gm3 = Matrix::<f32>::new(gemm_size, gemm_size, 1);
    gm1.data.fill(1.0);
    gm2.data.fill(1.0);

    c.bench_function("gemm_256x256_f32", |b| {
        b.iter(|| {
            gemm(
                black_box(&gm1),
                black_box(&gm2),
                1.0,
                black_box(&gm3),
                0.0,
                0,
            )
            .unwrap()
        })
    });

    // --- count_non_zero ---
    let mut v1_i32 = Matrix::<i32>::new(size, size, 1);
    for (i, v) in v1_i32.data.iter_mut().enumerate() {
        *v = if i % 3 == 0 { 0 } else { (i as i32) + 1 };
    }

    c.bench_function("count_non_zero_1024x1024_i32", |b| {
        b.iter(|| count_non_zero(black_box(&v1_i32)).unwrap())
    });

    // --- LUT ---
    // Build a gamma-correction-style LUT: out = (in/255)^2.2 * 255
    let lut_data: Vec<u8> = (0..256)
        .map(|i| ((i as f64 / 255.0).powf(2.2) * 255.0) as u8)
        .collect();
    let lut_table = Matrix::from_vec(1, 256, 1, lut_data);

    // Single-channel LUT
    c.bench_function("lut_1024x1024_u8", |b| {
        b.iter(|| lut(black_box(&u1), black_box(&lut_table)).unwrap())
    });

    // 3-channel LUT (broadcast)
    let mut u3 = Matrix::<u8>::new(size, size, 3);
    u3.data.fill(128);

    c.bench_function("lut_1024x1024x3_u8_broadcast", |b| {
        b.iter(|| lut(black_box(&u3), black_box(&lut_table)).unwrap())
    });
}

criterion_group!(benches, bench_arithm);
criterion_main!(benches);
