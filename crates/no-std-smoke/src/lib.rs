/*
 *  lib.rs
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

//! Build-only `no_std` smoke test for `purecv` (issue #83).
//!
//! This crate never runs; compiling it for a bare-metal target such as
//! `thumbv7em-none-eabihf` proves that the `purecv` core API is usable
//! from a `no_std` + `alloc` consumer:
//!
//! ```sh
//! cargo build --target thumbv7em-none-eabihf
//! ```

#![no_std]

extern crate alloc;

use alloc::vec;
use purecv::core::error::Result;
use purecv::core::types::{BorderTypes, Size2i};
use purecv::core::{add, determinant, mean, Matrix};
use purecv::imgproc::gaussian_blur;

/// Exercises matrix construction, arithmetic, statistics, and linear algebra.
pub fn smoke() -> Result<f64> {
    let a = Matrix::<f32>::from_vec(2, 2, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let b = Matrix::<f32>::from_vec(2, 2, 1, vec![5.0, 6.0, 7.0, 8.0]);

    let sum = add(&a, &b)?;
    let avg = mean(&sum);
    let det = determinant(&sum);

    Ok(avg.v[0] + det)
}

/// Exercises the Phase 2 imgproc path: a scalar `gaussian_blur` under no_std.
pub fn smoke_imgproc() -> Result<f32> {
    let src = Matrix::<f32>::from_vec(
        3,
        3,
        1,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    let blurred = gaussian_blur(
        &src,
        Size2i::new(3, 3),
        0.0,
        0.0,
        BorderTypes::Reflect101,
    )?;
    Ok(blurred.data[4])
}
