/*
 *  features2d_bench.rs
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
use purecv::core::Matrix;
use purecv::features2d::{FastFeatureDetector, FastType, Orb};
use std::hint::black_box;

fn bench_features2d(c: &mut Criterion) {
    let size = 512;

    // Create a synthetic grayscale image with gradients and structures to ensure keypoints are detected
    let mut img = Matrix::<u8>::new(size, size, 1);
    for y in 0..size {
        for x in 0..size {
            let row_pattern = (y as f32 * 0.1).sin() * 128.0 + 128.0;
            let col_pattern = (x as f32 * 0.1).cos() * 128.0 + 128.0;
            img.set(y, x, 0, ((row_pattern + col_pattern) / 2.0) as u8);
        }
    }

    // Benchmark FAST detector
    let fast = FastFeatureDetector::new(20, true, FastType::Type9_16);
    c.bench_function("fast_detect_512x512", |b| {
        b.iter(|| fast.detect(black_box(&img)).unwrap())
    });

    // Benchmark ORB detect_and_compute
    let orb = Orb::default();
    c.bench_function("orb_detect_and_compute_512x512", |b| {
        b.iter(|| orb.detect_and_compute(black_box(&img)).unwrap())
    });
}

criterion_group!(benches, bench_features2d);
criterion_main!(benches);
