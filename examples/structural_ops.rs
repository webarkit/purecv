/*
 *  structural_ops.rs
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

use purecv::core::structural::{copy_make_border, flip, merge, repeat, rotate, split, transpose};
use purecv::core::{Matrix, Scalar};
use purecv::version;

fn main() {
    println!("--- purecv Structural Operations Example ---");
    println!("purecv v{}", version::get_version());

    // Start with a 2x3 matrix
    let m = Matrix::<u8>::from_vec(2, 3, 1, vec![1, 2, 3, 4, 5, 6]);
    println!("Original Matrix (2x3):\n{:?}", m.data);

    // 1. Flip
    let m_flip = flip(&m, 0).unwrap(); // Vertical flip
    println!("\nFlipped Vertically:\n{:?}", m_flip.data);

    // 2. Transpose
    let m_trans = transpose(&m).unwrap();
    println!("\nTransposed (3x2):\n{:?}", m_trans.data);

    // 3. Rotate
    let m_rot = rotate(&m, 0).unwrap(); // 90 deg clockwise
    println!("\nRotated 90 deg CW:\n{:?}", m_rot.data);

    // 4. Repeat
    let m_rep = repeat(&m, 2, 2).unwrap();
    println!("\nRepeated (4x6):\n{:?}", m_rep.data);

    // 5. Border
    let m_border = copy_make_border(&m, 1, 1, 1, 1, 0, Scalar::all(255)).unwrap();
    println!("\nConstant Border (white 255):\n{:?}", m_border.data);

    // 6. Split/Merge
    let m_rgb = Matrix::<u8>::from_vec(
        2,
        2,
        3,
        vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255],
    );
    println!("\nRGB Matrix (2x2, 3ch):\n{:?}", m_rgb.data);

    let planes = split(&m_rgb).unwrap();
    println!("Red plane:   {:?}", planes[0].data);
    println!("Green plane: {:?}", planes[1].data);
    println!("Blue plane:  {:?}", planes[2].data);

    let merged = merge(&planes).unwrap();
    assert_eq!(m_rgb.data, merged.data);
    println!("\nMerge successful! Data matches original RGB.");

    println!("\nDone.");
}
