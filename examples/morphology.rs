/*
 *  morphology.rs
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

//! Morphological operations example.
//!
//! Demonstrates: erode, dilate, open, close, gradient, tophat, blackhat.
//!
//! Run from the project root:
//! ```
//! cargo run --example morphology
//! ```

use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};
use purecv::core::{BorderTypes, Matrix, Point, Size};
use purecv::imgproc::{
    cvt_color_rgb_to_gray, dilate, erode, get_structuring_element, morphology_ex, MorphShapes,
    MorphTypes,
};
use purecv::version;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- purecv Morphology Example ---");
    println!("purecv v{}", version::get_version());

    // 1. Load the image
    let img_path = "examples/data/Morphology_1_Tutorial_Theory_Original_Image.webp";
    if !Path::new(img_path).exists() {
        eprintln!("Error: {} not found. Run from the project root.", img_path);
        return Ok(());
    }

    let img = image::open(img_path)?;
    let (width, height) = img.dimensions();
    println!("Loaded image: {} ({}x{})", img_path, width, height);

    // 2. Convert to purecv Matrix<u8> (grayscale)
    let rgb_img = img.to_rgb8();
    let mat_rgb = Matrix::from_vec(height as usize, width as usize, 3, rgb_img.into_raw());
    let mat_gray = cvt_color_rgb_to_gray(&mat_rgb)?;

    // 3. Create structuring elements
    let kernel_rect = get_structuring_element(
        MorphShapes::Rect,
        Size::new(5_usize, 5_usize),
        Point::new(-1_i32, -1_i32),
    )?;

    let kernel_ellipse = get_structuring_element(
        MorphShapes::Ellipse,
        Size::new(5_usize, 5_usize),
        Point::new(-1_i32, -1_i32),
    )?;

    let kernel_cross = get_structuring_element(
        MorphShapes::Cross,
        Size::new(5_usize, 5_usize),
        Point::new(-1_i32, -1_i32),
    )?;

    // Print structuring element shapes
    println!("\n--- Structuring Elements ---");
    print_kernel("Rect 5x5", &kernel_rect);
    print_kernel("Ellipse 5x5", &kernel_ellipse);
    print_kernel("Cross 5x5", &kernel_cross);

    let anchor = Point::new(-1_i32, -1_i32);
    let border = BorderTypes::Constant;

    // --- Apply Morphological Operations ---

    // Erode
    println!("\nApplying Erode...");
    let eroded = erode(&mat_gray, &kernel_rect, anchor, 1, border)?;
    save_matrix_gray(&eroded, "examples/data/out/output_erode.png")?;

    // Dilate
    println!("Applying Dilate...");
    let dilated = dilate(&mat_gray, &kernel_rect, anchor, 1, border)?;
    save_matrix_gray(&dilated, "examples/data/out/output_dilate.png")?;

    // Open (remove small bright noise)
    println!("Applying Open...");
    let opened = morphology_ex(&mat_gray, MorphTypes::Open, &kernel_rect, anchor, 1, border)?;
    save_matrix_gray(&opened, "examples/data/out/output_open.png")?;

    // Close (fill small dark holes)
    println!("Applying Close...");
    let closed = morphology_ex(
        &mat_gray,
        MorphTypes::Close,
        &kernel_rect,
        anchor,
        1,
        border,
    )?;
    save_matrix_gray(&closed, "examples/data/out/output_close.png")?;

    // Gradient (edge detection via morphology)
    println!("Applying Gradient...");
    let gradient = morphology_ex(
        &mat_gray,
        MorphTypes::Gradient,
        &kernel_rect,
        anchor,
        1,
        border,
    )?;
    save_matrix_gray(&gradient, "examples/data/out/output_gradient.png")?;

    // TopHat (isolates bright features smaller than kernel)
    println!("Applying TopHat...");
    let tophat = morphology_ex(
        &mat_gray,
        MorphTypes::TopHat,
        &kernel_ellipse,
        anchor,
        1,
        border,
    )?;
    save_matrix_gray(&tophat, "examples/data/out/output_tophat.png")?;

    // BlackHat (isolates dark features smaller than kernel)
    println!("Applying BlackHat...");
    let blackhat = morphology_ex(
        &mat_gray,
        MorphTypes::BlackHat,
        &kernel_ellipse,
        anchor,
        1,
        border,
    )?;
    save_matrix_gray(&blackhat, "examples/data/out/output_blackhat.png")?;

    // Erode with Cross kernel, 2 iterations
    println!("Applying Erode (cross, 2 iterations)...");
    let eroded_cross = erode(&mat_gray, &kernel_cross, anchor, 2, border)?;
    save_matrix_gray(
        &eroded_cross,
        "examples/data/out/output_erode_cross_2iter.png",
    )?;

    println!("\nAll morphological operations applied successfully!");
    println!("Check the examples/data/out/ directory for output images.");
    Ok(())
}

/// Pretty-prints a kernel matrix to the console.
fn print_kernel(name: &str, kernel: &Matrix<u8>) {
    println!("  {name}:");
    for r in 0..kernel.rows {
        print!("    ");
        for c in 0..kernel.cols {
            print!("{} ", kernel.get(r, c, 0).unwrap());
        }
        println!();
    }
}

fn save_matrix_gray(mat: &Matrix<u8>, filename: &str) -> image::ImageResult<()> {
    let img: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_raw(mat.cols as u32, mat.rows as u32, mat.data.clone())
            .expect("Failed to create image buffer");
    DynamicImage::ImageLuma8(img).save(filename)
}
