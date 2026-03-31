/*
 *  dft_bench.rs
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
use purecv::core::dft::{dft, get_optimal_dft_size, DFT_COMPLEX_OUTPUT, DFT_INVERSE, DFT_SCALE};
use purecv::core::Matrix;

fn bench_dft(c: &mut Criterion) {
    // --- get_optimal_dft_size (pure computation, very fast) ---
    c.bench_function("get_optimal_dft_size", |b| {
        b.iter(|| get_optimal_dft_size(black_box(1000)))
    });

    // --- DFT forward, 256×256 f32 (real input) ---
    let size_small = 256;
    let mut src_256 = Matrix::<f32>::new(size_small, size_small, 1);
    for (i, v) in src_256.data.iter_mut().enumerate() {
        *v = ((i % size_small) as f32 * 0.1).sin();
    }

    c.bench_function("dft_forward_256x256_f32", |b| {
        b.iter(|| dft(black_box(&src_256), DFT_COMPLEX_OUTPUT, 0).unwrap())
    });

    // --- DFT inverse, 256×256 f32 ---
    let freq_256 = dft(&src_256, DFT_COMPLEX_OUTPUT, 0).unwrap();
    c.bench_function("dft_inverse_256x256_f32", |b| {
        b.iter(|| dft(black_box(&freq_256), DFT_INVERSE | DFT_SCALE, 0).unwrap())
    });

    // --- DFT forward, 512×512 f32 ---
    let size_med = 512;
    let mut src_512 = Matrix::<f32>::new(size_med, size_med, 1);
    for (i, v) in src_512.data.iter_mut().enumerate() {
        *v = ((i % size_med) as f32 * 0.1).sin();
    }

    c.bench_function("dft_forward_512x512_f32", |b| {
        b.iter(|| dft(black_box(&src_512), DFT_COMPLEX_OUTPUT, 0).unwrap())
    });

    // --- DFT forward, 1024×1024 f64 ---
    let size_lg = 1024;
    let mut src_1024 = Matrix::<f64>::new(size_lg, size_lg, 1);
    for (i, v) in src_1024.data.iter_mut().enumerate() {
        *v = ((i % size_lg) as f64 * 0.1).sin();
    }

    c.bench_function("dft_forward_1024x1024_f64", |b| {
        b.iter(|| dft(black_box(&src_1024), DFT_COMPLEX_OUTPUT, 0).unwrap())
    });
}

criterion_group!(benches, bench_dft);
criterion_main!(benches);
