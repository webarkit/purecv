/*
 *  structural_bench.rs
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

use criterion::{criterion_group, criterion_main, Criterion};
use purecv::core::structural::{flip, merge, split, transpose};
use purecv::core::Matrix;
use std::hint::black_box;

fn bench_structural(c: &mut Criterion) {
    let size = 1024;

    // flip horizontal (3-channel u8)
    let mut img_u8 = Matrix::<u8>::new(size, size, 3);
    for (i, v) in img_u8.data.iter_mut().enumerate() {
        *v = (i % 256) as u8;
    }

    c.bench_function("flip_horiz_1024x1024x3_u8", |b| {
        b.iter(|| flip(black_box(&img_u8), 1).unwrap())
    });

    // transpose (1-channel f32)
    let mut mat_f32 = Matrix::<f32>::new(size, size, 1);
    mat_f32.data.fill(1.0);

    c.bench_function("transpose_1024x1024_f32", |b| {
        b.iter(|| transpose(black_box(&mat_f32)).unwrap())
    });

    // split 3-channel → 3 single-channel
    c.bench_function("split_1024x1024x3_u8", |b| {
        b.iter(|| split(black_box(&img_u8)).unwrap())
    });

    // merge 3 single-channel → 3-channel
    let channels = split(&img_u8).unwrap();

    c.bench_function("merge_1024x1024x3_u8", |b| {
        b.iter(|| merge(black_box(&channels)).unwrap())
    });
}

criterion_group!(benches, bench_structural);
criterion_main!(benches);
