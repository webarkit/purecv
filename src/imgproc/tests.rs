/*
 *  tests.rs
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

#[cfg(test)]
mod imgproc_tests {
    use crate::core::*;
    use crate::imgproc::*;

    #[test]
    fn test_blur() {
        let m = Matrix::from_vec(3, 3, 1, vec![10u8, 10, 10, 10, 10, 10, 10, 10, 10]);
        let ksize = Size2i::new(3, 3);
        let res = blur(&m, ksize, Point2i::new(-1, -1), BorderTypes::Reflect101).unwrap();
        assert_eq!(res.data, vec![10u8; 9]);
    }

    #[test]
    fn test_box_filter() {
        let m = Matrix::from_vec(3, 3, 1, vec![1u8, 1, 1, 1, 1, 1, 1, 1, 1]);
        let ksize = Size2i::new(3, 3);
        // Sum should be 9 for each pixel because of border reflection
        let res = box_filter(
            &m,
            ksize,
            Point2i::new(-1, -1),
            false,
            BorderTypes::Reflect101,
        )
        .unwrap();
        for val in res.data {
            assert_eq!(val, 9u8);
        }
    }

    #[test]
    fn test_gaussian_blur() {
        let m = Matrix::from_vec(5, 5, 1, vec![100u8; 25]);
        let ksize = Size2i::new(3, 3);
        let res = gaussian_blur(&m, ksize, 1.0, 1.0, BorderTypes::Reflect101).unwrap();
        // Since all pixels are 100, the result should be 100 (normalized)
        for val in res.data {
            assert!((99..=101).contains(&val)); // Allow small deviation for rounding
        }
    }

    #[test]
    fn test_median_blur() {
        // Create a 3x3 matrix with an outlier
        let mut data = vec![10u8; 9];
        data[4] = 255; // Center is outlier
        let src = Matrix::from_vec(3, 3, 1, data);

        // Median of [10, 10, 10, 10, 255, 10, 10, 10, 10] is 10
        let blur = median_blur(&src, 3).unwrap();
        assert_eq!(*blur.at(1, 1, 0).unwrap(), 10);
    }

    #[test]
    fn test_bilateral_filter() {
        // Create a 5x5 matrix with an edge
        let mut data = vec![0u8; 25];
        for x in 3..5 {
            for y in 0..5 {
                data[y * 5 + x] = 255;
            }
        }
        let src = Matrix::from_vec(5, 5, 1, data);

        let res = bilateral_filter(&src, 5, 50.0, 50.0, BorderTypes::Reflect101).unwrap();

        // Edge should be preserved.
        // Row 2, Col 2 is x=2, y=2. Value is 0.
        // Row 2, Col 3 is x=3, y=2. Value is 255.
        let val_at_2_2 = *res.at(2, 2, 0).unwrap();
        let val_at_2_3 = *res.at(2, 3, 0).unwrap();

        assert!(val_at_2_2 < 50);
        assert!(val_at_2_3 > 200);
    }

    #[test]
    fn test_sobel() {
        // Vertical edge: left half 0, right half 255
        let mut data = vec![0u8; 100];
        for x in 5..10 {
            for y in 0..10 {
                data[y * 10 + x] = 255;
            }
        }
        let src = Matrix::from_vec(10, 10, 1, data);

        // Sobel dx=1, dy=0 should detect the vertical edge (x direction derivative)
        // With ksize=3, scale=1.0, delta=0.0
        let _res = sobel(&src, 1, 0, 3, 1.0, 0.0, BorderTypes::Reflect101).unwrap();

        // At the edge (x=4 to x=5), the derivative should be high.
        // Neighbors: x=4 (0), x=6 (255) -> 255 - 0 = 255.
        // Sobel dx=1: [-1, 0, 1] smoothed by [1, 2, 1] vertically.
        // Total weight is 4. Result should be around 255 * 4 = 1020,
        // but it's cast back to u8 if T is u8.
        // Let's use f32 to avoid overflow for testing.

        let src_f32: Matrix<f32> =
            Matrix::from_vec(10, 10, 1, src.data.iter().map(|&v| v as f32).collect());
        let res_f32 = sobel(&src_f32, 1, 0, 3, 1.0, 0.0, BorderTypes::Reflect101).unwrap();

        let edge_val = *res_f32.at(5, 5, 0).unwrap();
        assert!(edge_val > 500.0); // 255 * 4 = 1020 expected for Sobel ksize=3
    }

    #[test]
    fn test_scharr() {
        let mut data = [0u8; 100];
        for x in 5..10 {
            for y in 0..10 {
                data[y * 10 + x] = 255;
            }
        }
        let src = Matrix::<f32>::from_vec(10, 10, 1, data.iter().map(|&v| v as f32).collect());

        let res = scharr(&src, 1, 0, 1.0, 0.0, BorderTypes::Reflect101).unwrap();
        let edge_val = *res.at(5, 5, 0).unwrap();
        // Scharr weight for center row is 10, total 3+10+3 = 16.
        // Expected: 255 * 16 = 4080
        assert!(edge_val > 2000.0);
    }

    #[test]
    fn test_laplacian() {
        // Uniform image
        let src = Matrix::<f32>::from_vec(5, 5, 1, vec![100.0; 25]);
        let res = laplacian(&src, 1, 1.0, 0.0, BorderTypes::Reflect101).unwrap();

        // Laplacian of a uniform field should be 0
        for &val in &res.data {
            assert!(val.abs() < 1e-5);
        }

        // Image with a peak at (2, 2)
        let mut src_peak = Matrix::<f64>::new(5, 5, 1);
        src_peak.set(2, 2, 0, 255.0);
        let res_peak = laplacian(&src_peak, 1, 1.0, 0.0, BorderTypes::Reflect101).unwrap();
        // Laplacian [0, 1, 0; 1, -4, 1; 0, 1, 0] * 255 = [..., -4*255, ...] = -1020
        assert!(((*res_peak.at(2, 2, 0).unwrap() + 1020.0f64).abs()) < 1e-5);
    }

    #[test]
    fn test_canny() {
        let mut data = vec![0u8; 100];
        for x in 4..7 {
            for y in 0..10 {
                data[y * 10 + x] = 255;
            }
        }
        let src = Matrix::from_vec(10, 10, 1, data);

        // Edge should be at x=4 and x=7
        let edges = canny(&src, 50.0, 150.0, 3, false).unwrap();

        assert_eq!(*edges.at(5, 4, 0).unwrap(), 255);
        assert_eq!(*edges.at(5, 7, 0).unwrap(), 255);
        assert_eq!(*edges.at(5, 0, 0).unwrap(), 0);
    }

    #[test]
    fn test_cvt_color() {
        let mut data = vec![0u8; 12];
        // 2x2 RGB image. R,G,B
        data[0] = 255; // Red
        data[4] = 255; // Green
        data[8] = 255; // Blue
        let rgb_src = Matrix::from_vec(2, 2, 3, data);

        let gray = cvt_color(&rgb_src, ColorConversionCode::COLOR_RGB2GRAY).unwrap();

        // Allow ±1 tolerance: SIMD fixed-point (77/150/29) and scalar float
        // (0.299/0.587/0.114) differ by at most 1 LSB.
        assert!(
            (*gray.at(0, 0, 0).unwrap() as i16 - 76).abs() <= 1,
            "Red expected ~76, got {}",
            *gray.at(0, 0, 0).unwrap()
        );
        assert!(
            (*gray.at(0, 1, 0).unwrap() as i16 - 150).abs() <= 1,
            "Green expected ~150, got {}",
            *gray.at(0, 1, 0).unwrap()
        );
        assert!(
            (*gray.at(1, 0, 0).unwrap() as i16 - 29).abs() <= 1,
            "Blue expected ~29, got {}",
            *gray.at(1, 0, 0).unwrap()
        );
        assert_eq!(*gray.at(1, 1, 0).unwrap(), 0);

        let mut bgr_data = vec![0u8; 12];
        bgr_data[0] = 255; // Blue
        bgr_data[4] = 255; // Green
        bgr_data[8] = 255; // Red
        let bgr_src = Matrix::from_vec(2, 2, 3, bgr_data);

        let gray_bgr = cvt_color(&bgr_src, ColorConversionCode::COLOR_BGR2GRAY).unwrap();
        assert!(
            (*gray_bgr.at(0, 0, 0).unwrap() as i16 - 29).abs() <= 1,
            "Blue expected ~29, got {}",
            *gray_bgr.at(0, 0, 0).unwrap()
        );
        assert!(
            (*gray_bgr.at(0, 1, 0).unwrap() as i16 - 150).abs() <= 1,
            "Green expected ~150, got {}",
            *gray_bgr.at(0, 1, 0).unwrap()
        );
        assert!(
            (*gray_bgr.at(1, 0, 0).unwrap() as i16 - 76).abs() <= 1,
            "Red expected ~76, got {}",
            *gray_bgr.at(1, 0, 0).unwrap()
        );
        assert_eq!(*gray_bgr.at(1, 1, 0).unwrap(), 0);
    }

    #[test]
    fn test_threshold() {
        let data = vec![10u8, 50, 100, 150, 200, 250];
        let src = Matrix::from_vec(1, 6, 1, data);

        // Binary threshold at 127
        let (_, res) = threshold(&src, 127.0, 255.0, ThresholdTypes::THRESH_BINARY).unwrap();
        assert_eq!(res.data, vec![0, 0, 0, 255, 255, 255]);

        // Binary Inv threshold at 127
        let (_, res_inv) =
            threshold(&src, 127.0, 255.0, ThresholdTypes::THRESH_BINARY_INV).unwrap();
        assert_eq!(res_inv.data, vec![255, 255, 255, 0, 0, 0]);

        // Truncate at 127
        let (_, res_trunc) = threshold(&src, 127.0, 255.0, ThresholdTypes::THRESH_TRUNC).unwrap();
        assert_eq!(res_trunc.data, vec![10, 50, 100, 127, 127, 127]);

        // ToZero at 127
        let (_, res_zero) = threshold(&src, 127.0, 255.0, ThresholdTypes::THRESH_TOZERO).unwrap();
        assert_eq!(res_zero.data, vec![0, 0, 0, 150, 200, 250]);

        // ToZero Inv at 127
        let (_, res_zero_inv) =
            threshold(&src, 127.0, 255.0, ThresholdTypes::THRESH_TOZERO_INV).unwrap();
        assert_eq!(res_zero_inv.data, vec![10, 50, 100, 0, 0, 0]);
    }

    // -------------------------------------------------------------------
    //  Color conversion: larger-image equivalence tests
    // -------------------------------------------------------------------

    #[test]
    fn test_cvt_color_rgb_to_gray_large() {
        // 64×64 RGB image with gradient pattern
        let rows = 64;
        let cols = 64;
        let mut data = vec![0u8; rows * cols * 3];
        for i in 0..rows * cols {
            data[i * 3] = (i % 256) as u8; // R
            data[i * 3 + 1] = ((i * 3) % 256) as u8; // G
            data[i * 3 + 2] = ((i * 7) % 256) as u8; // B
        }
        let src = Matrix::from_vec(rows, cols, 3, data.clone());
        let gray = cvt_color(&src, ColorConversionCode::COLOR_RGB2GRAY).unwrap();

        assert_eq!(gray.rows, rows);
        assert_eq!(gray.cols, cols);
        assert_eq!(gray.channels, 1);

        // Verify every pixel is within ±1 of the float formula
        for i in 0..rows * cols {
            let r = data[i * 3] as f32;
            let g = data[i * 3 + 1] as f32;
            let b = data[i * 3 + 2] as f32;
            let expected = (0.299 * r + 0.587 * g + 0.114 * b).round() as i16;
            let actual = gray.data[i] as i16;
            assert!(
                (expected - actual).abs() <= 1,
                "Pixel {i}: expected ~{expected}, got {actual}"
            );
        }
    }

    #[test]
    fn test_cvt_color_bgr_to_gray_large() {
        let rows = 64;
        let cols = 64;
        let mut data = vec![0u8; rows * cols * 3];
        for i in 0..rows * cols {
            data[i * 3] = ((i * 7) % 256) as u8; // B
            data[i * 3 + 1] = ((i * 3) % 256) as u8; // G
            data[i * 3 + 2] = (i % 256) as u8; // R
        }
        let src = Matrix::from_vec(rows, cols, 3, data.clone());
        let gray = cvt_color(&src, ColorConversionCode::COLOR_BGR2GRAY).unwrap();

        for i in 0..rows * cols {
            let b = data[i * 3] as f32;
            let g = data[i * 3 + 1] as f32;
            let r = data[i * 3 + 2] as f32;
            let expected = (0.299 * r + 0.587 * g + 0.114 * b).round() as i16;
            let actual = gray.data[i] as i16;
            assert!(
                (expected - actual).abs() <= 1,
                "Pixel {i}: expected ~{expected}, got {actual}"
            );
        }
    }

    #[test]
    fn test_cvt_color_rgba_to_gray_large() {
        let rows = 32;
        let cols = 32;
        let mut data = vec![0u8; rows * cols * 4];
        for i in 0..rows * cols {
            data[i * 4] = (i % 256) as u8; // R
            data[i * 4 + 1] = ((i * 3) % 256) as u8; // G
            data[i * 4 + 2] = ((i * 7) % 256) as u8; // B
            data[i * 4 + 3] = 255; // A
        }
        let src = Matrix::from_vec(rows, cols, 4, data.clone());
        let gray = cvt_color(&src, ColorConversionCode::COLOR_RGBA2GRAY).unwrap();

        for i in 0..rows * cols {
            let r = data[i * 4] as f32;
            let g = data[i * 4 + 1] as f32;
            let b = data[i * 4 + 2] as f32;
            let expected = (0.299 * r + 0.587 * g + 0.114 * b).round() as i16;
            let actual = gray.data[i] as i16;
            assert!(
                (expected - actual).abs() <= 1,
                "Pixel {i}: expected ~{expected}, got {actual}"
            );
        }
    }

    // -------------------------------------------------------------------
    //  Threshold: all types equivalence tests
    // -------------------------------------------------------------------

    #[test]
    fn test_threshold_all_types_u8() {
        // Larger input to exercise SIMD paths
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let src = Matrix::from_vec(1, 256, 1, data);

        // BINARY
        let (_, res) = threshold(&src, 127.0, 200.0, ThresholdTypes::THRESH_BINARY).unwrap();
        for i in 0..256 {
            let expected: u8 = if (i as u8) > 127 { 200 } else { 0 };
            assert_eq!(res.data[i], expected, "BINARY mismatch at {i}");
        }

        // BINARY_INV
        let (_, res) = threshold(&src, 127.0, 200.0, ThresholdTypes::THRESH_BINARY_INV).unwrap();
        for i in 0..256 {
            let expected: u8 = if (i as u8) > 127 { 0 } else { 200 };
            assert_eq!(res.data[i], expected, "BINARY_INV mismatch at {i}");
        }

        // TRUNC
        let (_, res) = threshold(&src, 127.0, 200.0, ThresholdTypes::THRESH_TRUNC).unwrap();
        for i in 0..256 {
            let expected: u8 = if (i as u8) > 127 { 127 } else { i as u8 };
            assert_eq!(res.data[i], expected, "TRUNC mismatch at {i}");
        }

        // TOZERO
        let (_, res) = threshold(&src, 127.0, 200.0, ThresholdTypes::THRESH_TOZERO).unwrap();
        for i in 0..256 {
            let expected: u8 = if (i as u8) > 127 { i as u8 } else { 0 };
            assert_eq!(res.data[i], expected, "TOZERO mismatch at {i}");
        }

        // TOZERO_INV
        let (_, res) = threshold(&src, 127.0, 200.0, ThresholdTypes::THRESH_TOZERO_INV).unwrap();
        for i in 0..256 {
            let expected: u8 = if (i as u8) > 127 { 0 } else { i as u8 };
            assert_eq!(res.data[i], expected, "TOZERO_INV mismatch at {i}");
        }
    }

    #[test]
    fn test_threshold_f32() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let src = Matrix::from_vec(1, 100, 1, data);

        let (_, res) = threshold(&src, 0.5, 1.0, ThresholdTypes::THRESH_BINARY).unwrap();
        for i in 0..100 {
            let val = i as f32 * 0.01;
            let expected: f32 = if val > 0.5 { 1.0 } else { 0.0 };
            assert!(
                (res.data[i] - expected).abs() < 1e-5,
                "f32 BINARY mismatch at {i}: expected {expected}, got {}",
                res.data[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Gray-to-color conversions
    // -----------------------------------------------------------------------

    #[test]
    fn test_cvt_color_gray_to_rgb() {
        let m = Matrix::from_vec(1, 2, 1, vec![100u8, 200]);
        let r = cvt_color_gray_to_rgb(&m).unwrap();
        assert_eq!(r.channels, 3);
        assert_eq!(r.data, vec![100u8, 100, 100, 200, 200, 200]);
    }

    #[test]
    fn test_cvt_color_gray_to_bgr() {
        let m = Matrix::from_vec(1, 2, 1, vec![100u8, 200]);
        let r = cvt_color_gray_to_bgr(&m).unwrap();
        assert_eq!(r.channels, 3);
        assert_eq!(r.data, vec![100u8, 100, 100, 200, 200, 200]);
    }

    #[test]
    fn test_cvt_color_gray_to_rgba() {
        let m = Matrix::from_vec(1, 2, 1, vec![100u8, 200]);
        let r = cvt_color_gray_to_rgba(&m).unwrap();
        assert_eq!(r.channels, 4);
        assert_eq!(r.data, vec![100u8, 100, 100, 255, 200, 200, 200, 255]);
    }

    #[test]
    fn test_cvt_color_gray_to_bgra() {
        let m = Matrix::from_vec(1, 2, 1, vec![100u8, 200]);
        let r = cvt_color_gray_to_bgra(&m).unwrap();
        assert_eq!(r.channels, 4);
        assert_eq!(r.data, vec![100u8, 100, 100, 255, 200, 200, 200, 255]);
    }

    #[test]
    fn test_gray_to_color_error() {
        // Non-grayscale input (3 channels) should error
        let m = Matrix::from_vec(1, 1, 3, vec![1u8, 2, 3]);
        assert!(cvt_color_gray_to_rgb(&m).is_err());
        assert!(cvt_color_gray_to_bgr(&m).is_err());
        assert!(cvt_color_gray_to_rgba(&m).is_err());
        assert!(cvt_color_gray_to_bgra(&m).is_err());
    }

    // -----------------------------------------------------------------------
    // Kernel helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_gaussian_kernel() {
        // Single element
        let k = get_gaussian_kernel(1, 1.0);
        assert_eq!(k.len(), 1);
        assert!((k[0] - 1.0).abs() < 1e-9);

        // 3-element kernel: symmetric, sums to ~1.0, center is max
        let k = get_gaussian_kernel(3, 1.0);
        assert_eq!(k.len(), 3);
        let sum: f64 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        assert!((k[0] - k[2]).abs() < 1e-9); // symmetric
        assert!(k[1] > k[0]); // center is max
    }

    #[test]
    fn test_get_sobel_kernels() {
        // dx=1, dy=0: Sobel X derivative
        let (kx, ky) = get_sobel_kernels(3, 1, 0);
        assert_eq!(kx, vec![-1.0, 0.0, 1.0]);
        assert_eq!(ky, vec![1.0, 2.0, 1.0]);

        // dx=0, dy=1: Sobel Y derivative
        let (kx, ky) = get_sobel_kernels(3, 0, 1);
        assert_eq!(kx, vec![1.0, 2.0, 1.0]);
        assert_eq!(ky, vec![-1.0, 0.0, 1.0]);
    }

    // -----------------------------------------------------------------------
    // Feature / corner detection tests
    // -----------------------------------------------------------------------

    /// Build a 20×20 image with a bright square (corners visible at four locations).
    fn make_corner_image() -> Matrix<f32> {
        let rows = 20;
        let cols = 20;
        let mut data = vec![0.0f32; rows * cols];
        // Fill the interior rectangle [6..14, 6..14] with 255.
        for y in 6..14 {
            for x in 6..14 {
                data[y * cols + x] = 255.0;
            }
        }
        Matrix::from_vec(rows, cols, 1, data)
    }

    #[test]
    fn test_corner_harris_uniform_image() {
        // Uniform image → no corners → Harris response should be near zero everywhere.
        let src = Matrix::<f32>::from_vec(10, 10, 1, vec![50.0; 100]);
        let result = corner_harris(&src, 3, 3, 0.04, BorderTypes::Reflect101).unwrap();
        for &v in &result.data {
            assert!(
                v.abs() < 1e-3,
                "uniform image Harris response {v} not near 0"
            );
        }
    }

    #[test]
    fn test_corner_harris_detects_corners() {
        let src = make_corner_image();
        let response = corner_harris(&src, 3, 3, 0.04, BorderTypes::Reflect101).unwrap();

        // The four actual corners of the bright rectangle are at approximately
        // (6,6), (6,13), (13,6), (13,13).  Harris response should be positive there.
        for &(row, col) in &[(6usize, 6usize), (6, 13), (13, 6), (13, 13)] {
            let v = *response.at(row as i32, col as i32, 0).unwrap();
            assert!(
                v > 0.0,
                "expected positive Harris response at ({row},{col}), got {v}"
            );
        }
    }

    #[test]
    fn test_corner_min_eigen_val_uniform() {
        let src = Matrix::<f32>::from_vec(10, 10, 1, vec![128.0; 100]);
        let result = corner_min_eigen_val(&src, 3, 3, BorderTypes::Reflect101).unwrap();
        for &v in &result.data {
            assert!(v.abs() < 1e-3, "uniform image min-eigenval {v} not near 0");
        }
    }

    #[test]
    fn test_corner_min_eigen_val_detects_corners() {
        let src = make_corner_image();
        let response = corner_min_eigen_val(&src, 3, 3, BorderTypes::Reflect101).unwrap();

        // Min-eigenvalue (Shi-Tomasi) should be positive at actual corners.
        for &(row, col) in &[(6usize, 6usize), (6, 13), (13, 6), (13, 13)] {
            let v = *response.at(row as i32, col as i32, 0).unwrap();
            assert!(
                v > 0.0,
                "expected positive min-eigenval at ({row},{col}), got {v}"
            );
        }
    }

    #[test]
    fn test_corner_eigen_vals_and_vecs_shape() {
        let src = make_corner_image();
        let result = corner_eigen_vals_and_vecs(&src, 3, 3, BorderTypes::Reflect101).unwrap();

        // Output must be 6-channel, same spatial size as input.
        assert_eq!(result.rows, src.rows);
        assert_eq!(result.cols, src.cols);
        assert_eq!(result.channels, 6);
    }

    #[test]
    fn test_corner_eigen_vals_and_vecs_ordering() {
        let src = make_corner_image();
        let result = corner_eigen_vals_and_vecs(&src, 3, 3, BorderTypes::Reflect101).unwrap();

        // λ1 ≥ λ2 everywhere.
        for i in 0..src.rows * src.cols {
            let lambda1 = result.data[i * 6];
            let lambda2 = result.data[i * 6 + 1];
            assert!(
                lambda1 >= lambda2 - 1e-5,
                "pixel {i}: λ1={lambda1} < λ2={lambda2}"
            );
        }
    }

    #[test]
    fn test_good_features_to_track_count() {
        let src = make_corner_image();
        let corners = good_features_to_track(&src, 10, 0.01, 3.0, 3, false, 0.04).unwrap();

        // The bright square has 4 visible corners; we should detect at least some.
        assert!(!corners.is_empty(), "expected at least one corner detected");
        assert!(
            corners.len() <= 10,
            "more corners than max_corners: {}",
            corners.len()
        );
    }

    #[test]
    fn test_good_features_to_track_min_distance() {
        let src = make_corner_image();
        let corners = good_features_to_track(&src, 20, 0.01, 5.0, 3, false, 0.04).unwrap();

        // Check that no two corners are within min_distance of each other.
        for i in 0..corners.len() {
            for j in (i + 1)..corners.len() {
                let dx = corners[i].x - corners[j].x;
                let dy = corners[i].y - corners[j].y;
                let dist = ((dx * dx + dy * dy) as f64).sqrt();
                assert!(
                    dist >= 4.9,
                    "corners {i} and {j} are too close: dist={dist:.3}"
                );
            }
        }
    }

    #[test]
    fn test_good_features_to_track_invalid_quality() {
        let src = make_corner_image();
        assert!(good_features_to_track(&src, 10, 0.0, 3.0, 3, false, 0.04).is_err());
        assert!(good_features_to_track(&src, 10, 1.5, 3.0, 3, false, 0.04).is_err());
    }

    #[test]
    fn test_corner_sub_pix_empty_corners() {
        use crate::core::types::{TermCriteria, TermType};
        let src = make_corner_image();
        let mut corners: Vec<crate::core::Point2f> = Vec::new();
        let result = corner_sub_pix(
            &src,
            &mut corners,
            crate::core::Size2i::new(5, 5),
            crate::core::Size2i::new(-1, -1),
            TermCriteria::new(TermType::Both, 30, 0.01),
        );
        assert!(result.is_ok());
        assert!(corners.is_empty());
    }

    #[test]
    fn test_corner_sub_pix_refines_toward_corner() {
        use crate::core::types::{TermCriteria, TermType};

        let src = make_corner_image();

        // Detect corners first.
        let mut corners = good_features_to_track(&src, 4, 0.01, 3.0, 3, false, 0.04).unwrap();
        if corners.is_empty() {
            return; // no corners detected — skip refinement test
        }

        let initial_x = corners[0].x;
        let initial_y = corners[0].y;

        corner_sub_pix(
            &src,
            &mut corners,
            crate::core::Size2i::new(5, 5),
            crate::core::Size2i::new(-1, -1),
            TermCriteria::new(TermType::Both, 40, 0.001),
        )
        .unwrap();

        let refined_x = corners[0].x;
        let refined_y = corners[0].y;

        // Refined position must remain within the image bounds.
        assert!(
            refined_x >= 0.0 && refined_x < src.cols as f32,
            "refined x {refined_x} out of image width {}",
            src.cols
        );
        assert!(
            refined_y >= 0.0 && refined_y < src.rows as f32,
            "refined y {refined_y} out of image height {}",
            src.rows
        );

        // The refinement should not move the corner by more than win_size pixels.
        let dx = (refined_x - initial_x).abs();
        let dy = (refined_y - initial_y).abs();
        assert!(
            dx <= 6.0 && dy <= 6.0,
            "corner moved too far: Δx={dx:.2}, Δy={dy:.2}"
        );
    }

    #[test]
    fn test_pre_corner_detect_uniform_image() {
        // Uniform image → all derivatives are zero → pre_corner_detect result ≈ 0.
        let src = Matrix::<f32>::from_vec(10, 10, 1, vec![100.0; 100]);
        let result = pre_corner_detect(&src, 3, BorderTypes::Reflect101).unwrap();
        for &v in &result.data {
            assert!(
                v.abs() < 1e-2,
                "uniform image pre_corner_detect {v} not near 0"
            );
        }
    }

    #[test]
    fn test_pre_corner_detect_output_shape() {
        let src = make_corner_image();
        let result = pre_corner_detect(&src, 3, BorderTypes::Reflect101).unwrap();
        assert_eq!(result.rows, src.rows);
        assert_eq!(result.cols, src.cols);
        assert_eq!(result.channels, 1);
    }
}
