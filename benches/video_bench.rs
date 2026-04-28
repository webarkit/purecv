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

use criterion::{criterion_group, criterion_main, Criterion};
use purecv::core::types::{Point2f, Size2i, TermCriteria, TermType};
use purecv::core::Matrix;
use purecv::video::optical_flow::calc_optical_flow_pyramid_lk;
use std::hint::black_box;

fn bench_optical_flow(c: &mut Criterion) {
    let size = 512;
    let shift = 3;

    // Create a smooth gaussian blob image
    let mut prev_data = vec![0u8; size * size];
    let cx = size as f32 / 2.0;
    let cy = size as f32 / 2.0;
    let sigma = 40.0f32;
    for r in 0..size {
        for c_idx in 0..size {
            let dx = c_idx as f32 - cx;
            let dy = r as f32 - cy;
            let val = (200.0 * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()).round() as u8;
            prev_data[r * size + c_idx] = val;
        }
    }

    let mut next_data = vec![0u8; size * size];
    for r in 0..size {
        for c_idx in 0..size {
            let src_c = if c_idx >= shift { c_idx - shift } else { 0 };
            next_data[r * size + c_idx] = prev_data[r * size + src_c];
        }
    }

    let prev = Matrix::<u8>::from_vec(size, size, 1, prev_data);
    let next = Matrix::<u8>::from_vec(size, size, 1, next_data);

    // Track a grid of points
    let mut pts = Vec::new();
    let step = size / 8; // 7x7 grid = 49 points
    for r in (step..size - step).step_by(step) {
        for c_idx in (step..size - step).step_by(step) {
            pts.push(Point2f::new(c_idx as f32, r as f32));
        }
    }

    let criteria = TermCriteria::new(TermType::Both, 30, 0.01);
    let win_size = Size2i::new(21, 21);
    let max_level = 3;

    c.bench_function("calc_optical_flow_pyr_lk_512x512_pts_49", |b| {
        b.iter(|| {
            calc_optical_flow_pyramid_lk(
                black_box(&prev),
                black_box(&next),
                black_box(&pts),
                None,
                win_size,
                max_level,
                criteria,
                0,
                1e-4,
            )
            .unwrap()
        })
    });
}

criterion_group!(benches, bench_optical_flow);
criterion_main!(benches);
