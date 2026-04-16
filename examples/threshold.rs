/*
 *  threshold.rs
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

use purecv::core::Matrix;
use purecv::imgproc::{threshold, ThresholdTypes};
use purecv::version;

fn main() {
    println!("--- purecv Threshold Example ---\n");
    println!("purecv v{}", version::get_version());

    // 1. Create a 4x4 grayscale matrix with a gradient
    let data: Vec<u8> = vec![
        10, 50, 100, 130, 150, 170, 200, 220, 30, 60, 90, 127, 128, 180, 210, 255,
    ];
    let src = Matrix::<u8>::from_vec(4, 4, 1, data);
    println!("Source matrix (4x4, 1ch):");
    print_matrix_u8(&src);

    // 2. BINARY threshold at 127
    let (thresh_val, binary) =
        threshold(&src, 127.0, 255.0, ThresholdTypes::THRESH_BINARY).unwrap();
    println!("\nBINARY (thresh={thresh_val}):");
    print_matrix_u8(&binary);

    // 3. BINARY_INV threshold
    let (_, binary_inv) = threshold(&src, 127.0, 255.0, ThresholdTypes::THRESH_BINARY_INV).unwrap();
    println!("\nBINARY_INV:");
    print_matrix_u8(&binary_inv);

    // 4. TRUNC threshold
    let (_, trunc) = threshold(&src, 127.0, 255.0, ThresholdTypes::THRESH_TRUNC).unwrap();
    println!("\nTRUNC:");
    print_matrix_u8(&trunc);

    // 5. TOZERO threshold
    let (_, tozero) = threshold(&src, 127.0, 255.0, ThresholdTypes::THRESH_TOZERO).unwrap();
    println!("\nTOZERO:");
    print_matrix_u8(&tozero);

    // 6. TOZERO_INV threshold
    let (_, tozero_inv) = threshold(&src, 127.0, 255.0, ThresholdTypes::THRESH_TOZERO_INV).unwrap();
    println!("\nTOZERO_INV:");
    print_matrix_u8(&tozero_inv);

    println!("\nDone.");
}

/// Pretty-print a u8 matrix row by row.
fn print_matrix_u8(m: &Matrix<u8>) {
    for r in 0..m.rows {
        let start = r * m.cols * m.channels;
        let end = start + m.cols * m.channels;
        let row = &m.data[start..end];
        println!("  {:?}", row);
    }
}
