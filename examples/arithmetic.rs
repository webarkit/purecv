/*
 *  arithmetic.rs
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

use purecv::core::Matrix;
use purecv::core::arithm::{add, multiply};

fn main() {
    println!("--- purecv Basic Arithmetic Example ---");

    // 1. Create two 3x3 matrices with 1 channel (u8)
    let m1 = Matrix::<u8>::from_vec(3, 3, 1, vec![
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    ]);

    let m2 = Matrix::<u8>::from_vec(3, 3, 1, vec![
        10, 10, 10,
        10, 10, 10,
        10, 10, 10
    ]);

    println!("Matrix 1 (3x3):\n{:?}", m1.data);
    println!("Matrix 2 (3x3, constant 10):\n{:?}", m2.data);

    // 2. Perform addition
    let sum = add(&m1, &m2).expect("Addition failed");
    println!("\nSum (m1 + m2):\n{:?}", sum.data);

    // 3. Perform multiplication
    let prod = multiply(&m1, &m2).expect("Multiplication failed");
    println!("\nProduct (m1 * m2):\n{:?}", prod.data);

    // 4. Large matrix demonstration (Parallelism/SIMD check if enabled)
    println!("\nPerforming operation on a larger matrix (1024x1024)...");
    let large_m1 = Matrix::<f32>::new(1024, 1024, 1);
    let large_m2 = Matrix::<f32>::new(1024, 1024, 1);
    
    let start = std::time::Instant::now();
    let _large_sum = add(&large_m1, &large_m2).unwrap();
    let duration = start.elapsed();
    
    println!("Large matrix addition completed in: {:?}", duration);
    println!("Done.");
}
