/*
 *  color_conversion.rs
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
use purecv::imgproc::cvt_color_rgb_to_gray;

fn main() {
    println!("--- purecv Color Conversion Example ---");

    // 1. Create a 3x3 RGB matrix (3 channels)
    // Representing a simple gradient
    let rgb_data = vec![
        255, 0, 0,   0, 255, 0,   0, 0, 255,
        128, 128, 0, 0, 128, 128, 128, 0, 128,
        255, 255, 255, 128, 128, 128, 0, 0, 0
    ];
    let m_rgb = Matrix::<u8>::from_vec(3, 3, 3, rgb_data);
    println!("Original RGB Matrix (3x3, 3ch):\n{:?}", m_rgb.data);

    // 2. Convert to Grayscale
    let m_gray = cvt_color_rgb_to_gray(&m_rgb).expect("Color conversion failed");
    
    println!("\nGrayscale Matrix (3x3, 1ch):\n{:?}", m_gray.data);
    
    assert_eq!(m_gray.channels, 1);
    assert_eq!(m_gray.rows, 3);
    assert_eq!(m_gray.cols, 3);

    println!("\nConversion successful!");
    println!("Done.");
}
