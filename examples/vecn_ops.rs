/*
 *  vecn_ops.rs
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

//! Demonstrates the `VecN<T, N>` generic vector type and its OpenCV-compatible
//! aliases (`Vec2i`, `Vec3f`, `Vec6d`, …).

use purecv::core::{Scalar, Vec2i, Vec3b, Vec3f, Vec4i, Vec6f, VecN};
use purecv::version;

fn main() {
    println!("--- purecv VecN Operations Example ---");
    println!("purecv v{}", version::get_version());

    // -----------------------------------------------------------------------
    // 1. Construction
    // -----------------------------------------------------------------------
    println!("\n--- 1. Construction ---");

    // Using the typed new() constructor (mirrors cv::Vec3f(x, y, z))
    let v3f = Vec3f::new(1.0_f32, 2.0, 3.0);
    println!("Vec3f::new(1, 2, 3)        → {:?}", v3f.val);

    // Using from_array
    let v2i = Vec2i::from_array([10, 20]);
    println!("Vec2i::from_array([10, 20]) → {:?}", v2i.val);

    // zeros() — additive-identity vector
    let zeros: Vec3f = VecN::zeros();
    println!("Vec3f::zeros()             → {:?}", zeros.val);

    // all(v) — broadcast a constant to every channel
    let ones: Vec4i = VecN::all(1);
    println!("Vec4i::all(1)              → {:?}", ones.val);

    // -----------------------------------------------------------------------
    // 2. Indexing
    // -----------------------------------------------------------------------
    println!("\n--- 2. Indexing ---");

    let mut color = Vec3b::new(255_u8, 128, 64);
    println!("Vec3b R={} G={} B={}", color[0], color[1], color[2]);

    // Mutable index
    color[1] = 200;
    println!("After color[1]=200 → {:?}", color.val);

    // -----------------------------------------------------------------------
    // 3. Element-wise arithmetic
    // -----------------------------------------------------------------------
    println!("\n--- 3. Element-wise arithmetic ---");

    let a = Vec3f::new(1.0_f32, 2.0, 3.0);
    let b = Vec3f::new(4.0_f32, 5.0, 6.0);

    println!("a = {:?}", a.val);
    println!("b = {:?}", b.val);
    println!("a + b = {:?}", (a.clone() + b.clone()).val);
    println!("a - b = {:?}", (a.clone() - b.clone()).val);
    println!("a * b = {:?}", (a.clone() * b.clone()).val); // element-wise

    // Scalar broadcast multiply / divide
    println!("a * 2.0 = {:?}", (a.clone() * 2.0_f32).val);
    println!("a / 2.0 = {:?}", (a.clone() / 2.0_f32).val);

    // -----------------------------------------------------------------------
    // 4. Dot product
    // -----------------------------------------------------------------------
    println!("\n--- 4. Dot product ---");

    let dot = a.dot(&b); // 1*4 + 2*5 + 3*6 = 32
    println!("dot(a, b) = {}", dot);
    assert!((dot - 32.0_f32).abs() < 1e-6);

    // -----------------------------------------------------------------------
    // 5. Scalar broadcast addition / subtraction (OpenCV parity)
    // -----------------------------------------------------------------------
    println!("\n--- 5. VecN +/- Scalar broadcast ---");

    let v = Vec3f::new(1.0_f32, 2.0, 3.0);
    let s = Scalar::new(10.0_f32, 20.0, 30.0, 0.0);
    println!("v        = {:?}", v.val);
    println!("scalar   = ({}, {}, {}, {})", s[0], s[1], s[2], s[3]);
    println!("v + s    = {:?}", (v.clone() + s).val);
    println!("v - s    = {:?}", (v.clone() - s).val);

    // Vec6f: channels 4–5 receive T::default() (0.0) from the Scalar
    let v6 = Vec6f::new(1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0);
    let s4 = Scalar::new(10.0_f32, 10.0, 10.0, 10.0);
    println!(
        "\nVec6f + Scalar (channels 4-5 unchanged) = {:?}",
        (v6 + s4).val
    );

    // -----------------------------------------------------------------------
    // 6. map() — element-wise type conversion
    // -----------------------------------------------------------------------
    println!("\n--- 6. map() ---");

    let bytes = Vec3b::new(100_u8, 150, 200);
    // Normalise to [0.0, 1.0]
    let normalised: Vec3f = bytes.clone().map(|x| x as f32 / 255.0);
    println!(
        "u8 {:?}  → normalised f32 [{:.3}, {:.3}, {:.3}]",
        bytes.val, normalised[0], normalised[1], normalised[2]
    );

    // -----------------------------------------------------------------------
    // 7. to_array / From<[T;N]>
    // -----------------------------------------------------------------------
    println!("\n--- 7. to_array / From trait ---");

    let arr = Vec3f::new(7.0_f32, 8.0, 9.0).to_array();
    println!("to_array: {:?}", arr);

    let from_arr: Vec3f = [1.0_f32, 2.0, 3.0].into();
    println!("From<[f32;3]>: {:?}", from_arr.val);

    println!("\nDone.");
}
