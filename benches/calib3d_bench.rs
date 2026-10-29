/*
 *  calib3d_bench.rs
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

//! Calib3d module benchmarks: the Levenberg-Marquardt solver and the
//! iterative PnP that refines with it.
//!
//! ```sh
//! cargo bench --bench calib3d_bench
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use purecv::calib3d::{solve_pnp, LmSolver, SolvePnPMethod};
use purecv::core::types::{Point2f, Point3f};
use purecv::core::Matrix;
use std::hint::black_box;

/// Samples of `y = 2 exp(-0.7 x) + 0.5` with a deterministic ripple.
fn exp_samples(n: usize) -> (Vec<f64>, Vec<f64>) {
    let xs: Vec<f64> = (0..n).map(|i| i as f64 * 5.0 / n as f64).collect();
    let ys = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| 2.0 * (-0.7 * x).exp() + 0.5 + 1e-3 * (i as f64 * 1.7).sin())
        .collect();
    (xs, ys)
}

fn bench_lm_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("lm_solver_exp_fit");
    for &n in &[32usize, 256, 2048] {
        let (xs, ys) = exp_samples(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let solver = LmSolver::new(100);
            b.iter(|| {
                let mut p = [1.0, -0.1, 0.0];
                let mut cb = |p: &[f64], err: &mut Vec<f64>, jac: Option<&mut Vec<f64>>| {
                    err.clear();
                    err.extend(
                        xs.iter()
                            .zip(&ys)
                            .map(|(&x, &y)| p[0] * (p[1] * x).exp() + p[2] - y),
                    );
                    if let Some(j) = jac {
                        j.clear();
                        for &x in &xs {
                            let e = (p[1] * x).exp();
                            j.extend([e, p[0] * x * e, 1.0]);
                        }
                    }
                    true
                };
                black_box(solver.run(&mut p, &mut cb).unwrap());
                black_box(p)
            });
        });
    }
    group.finish();
}

/// Points on a 3D grid seen by a camera with `fx != fy`, with a small
/// deterministic pixel perturbation so the refinement has work to do.
fn pnp_fixture(n: usize) -> (Vec<Point3f>, Vec<Point2f>, Matrix<f64>) {
    let (fx, fy, cx, cy) = (800.0, 600.0, 320.0, 240.0);
    let k = Matrix::from_vec(3, 3, 1, vec![fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]);
    // A rotation of about 0.2 rad around (1, 1, 0) / sqrt(2), then t = (0.1, -0.05, 4).
    let (s, co) = (0.2f64.sin(), 0.2f64.cos());
    let u = std::f64::consts::FRAC_1_SQRT_2;
    let r = [
        co + u * u * (1.0 - co),
        u * u * (1.0 - co),
        u * s,
        u * u * (1.0 - co),
        co + u * u * (1.0 - co),
        -u * s,
        -u * s,
        u * s,
        co,
    ];
    let t = [0.1, -0.05, 4.0];
    let mut obj = Vec::with_capacity(n);
    let mut img = Vec::with_capacity(n);
    for i in 0..n {
        let (x, y, z) = (
            (i % 7) as f64 * 0.2 - 0.6,
            ((i / 7) % 7) as f64 * 0.2 - 0.6,
            (i % 5) as f64 * 0.15 - 0.3,
        );
        let pc = [
            r[0] * x + r[1] * y + r[2] * z + t[0],
            r[3] * x + r[4] * y + r[5] * z + t[1],
            r[6] * x + r[7] * y + r[8] * z + t[2],
        ];
        let du = 0.5 * (i as f64 * 0.9).sin();
        let dv = 0.5 * (i as f64 * 1.3).cos();
        obj.push(Point3f::new(x as f32, y as f32, z as f32));
        img.push(Point2f::new(
            (fx * pc[0] / pc[2] + cx + du) as f32,
            (fy * pc[1] / pc[2] + cy + dv) as f32,
        ));
    }
    (obj, img, k)
}

fn bench_solve_pnp(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_pnp_iterative");
    for &n in &[12usize, 100, 1000] {
        let (obj, img, k) = pnp_fixture(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut rvec = Matrix::<f64>::new(3, 1, 1);
                let mut tvec = Matrix::<f64>::new(3, 1, 1);
                let ok = solve_pnp(
                    black_box(&obj),
                    black_box(&img),
                    &k,
                    None,
                    &mut rvec,
                    &mut tvec,
                    false,
                    SolvePnPMethod::Iterative,
                )
                .unwrap();
                black_box((ok, rvec, tvec))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_lm_solver, bench_solve_pnp);
criterion_main!(benches);
