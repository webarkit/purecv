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
use purecv::features2d::{build_orb_pyramid, FastFeatureDetector, FastType, Orb, ScoreType};
use std::hint::black_box;

// The following `Lcg` struct and `lcg_textured` function are copied verbatim from
// `src/features2d/tests.rs` because benches cannot see test-only helpers.

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }
}
/// Mid-gray background, 500 filled rectangles (side 4..=63, value 0..=255), then ±6 noise.
fn lcg_textured(rows: usize, cols: usize, seed: u64) -> Matrix<u8> {
    let mut rng = Lcg(seed);
    let mut buf = vec![128i32; rows * cols];
    for _ in 0..500 {
        let w = 4 + rng.below(60) as usize;
        let h = 4 + rng.below(60) as usize;
        let x0 = rng.below(cols as u32) as usize;
        let y0 = rng.below(rows as u32) as usize;
        let v = rng.below(256) as i32;
        for y in y0..(y0 + h).min(rows) {
            for x in x0..(x0 + w).min(cols) {
                buf[y * cols + x] = v;
            }
        }
    }
    let mut img = Matrix::<u8>::new(rows, cols, 1);
    for (d, s) in img.data.iter_mut().zip(buf.iter()) {
        *d = (s + rng.below(13) as i32 - 6).clamp(0, 255) as u8;
    }
    img
}

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

    // Benchmark ORB detect_and_compute. This sinusoid yields only 67 keypoints, all in octaves
    // 6-7, so it barely exercises the scale pyramid. Kept for continuity with prior baselines;
    // see `bench_orb_textured` below for a workload that spreads keypoints across all octaves.
    let orb = Orb::default();
    c.bench_function("orb_detect_and_compute_512x512", |b| {
        b.iter(|| orb.detect_and_compute(black_box(&img)).unwrap())
    });
}

// purecv#123/#124/#125: ORB on a textured 640x480 image, built from 500 random rectangles plus
// noise (`lcg_textured`) so keypoints spread across pyramid octaves, unlike the sinusoid above.
fn bench_orb_textured(c: &mut Criterion) {
    let img = lcg_textured(480, 640, 42);

    let orb = Orb::default();
    c.bench_function("orb_detect_and_compute_640x480_textured", |b| {
        b.iter(|| orb.detect_and_compute(black_box(&img)).unwrap())
    });

    // #125: Harris is now scored only at FAST keypoints, not over the whole level.
    let orb_harris = Orb::default();
    c.bench_function("orb_detect_harris_640x480", |b| {
        b.iter(|| orb_harris.detect(black_box(&img)).unwrap())
    });

    // A lower bound: skips the Harris response pass entirely.
    let mut orb_fast = Orb::default();
    orb_fast.set_score_type(ScoreType::Fast);
    c.bench_function("orb_detect_fast_640x480", |b| {
        b.iter(|| orb_fast.detect(black_box(&img)).unwrap())
    });

    // #123/#124: detect the keypoints once outside the timed loop, so `compute` alone (including
    // the per-level blur added by #124) is isolated from pyramid-build and detection cost.
    let kps = orb.detect(&img).unwrap();
    c.bench_function("orb_compute_640x480", |b| {
        b.iter(|| orb.compute(black_box(&img), black_box(&kps)).unwrap())
    });

    c.bench_function("orb_build_pyramid_640x480", |b| {
        b.iter(|| build_orb_pyramid(black_box(&img), 8, 1.2).unwrap())
    });
}

criterion_group!(benches, bench_features2d, bench_orb_textured);
criterion_main!(benches);
