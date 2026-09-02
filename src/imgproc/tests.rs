/*
 *  tests.rs
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

    // -------------------------------------------------------------------
    //  Morphological operations
    // -------------------------------------------------------------------

    #[test]
    fn test_get_structuring_element_rect() {
        let kernel = get_structuring_element(
            MorphShapes::Rect,
            types::Size::new(3_usize, 3_usize),
            types::Point::new(-1_i32, -1_i32),
        )
        .unwrap();
        assert_eq!(kernel.rows, 3);
        assert_eq!(kernel.cols, 3);
        // A 3×3 rect kernel should be all 1s
        assert!(kernel.data.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_get_structuring_element_cross() {
        let kernel = get_structuring_element(
            MorphShapes::Cross,
            types::Size::new(5_usize, 5_usize),
            types::Point::new(-1_i32, -1_i32),
        )
        .unwrap();
        assert_eq!(kernel.rows, 5);
        assert_eq!(kernel.cols, 5);

        // Centre row (row 2) should be all 1s
        for c in 0..5 {
            assert_eq!(*kernel.get(2, c, 0).unwrap(), 1, "centre row col {c}");
        }
        // Centre column (col 2) in non-centre rows should be 1
        for r in 0..5 {
            assert_eq!(*kernel.get(r, 2, 0).unwrap(), 1, "centre col row {r}");
        }
        // Corners should be 0
        assert_eq!(*kernel.get(0, 0, 0).unwrap(), 0);
        assert_eq!(*kernel.get(0, 4, 0).unwrap(), 0);
        assert_eq!(*kernel.get(4, 0, 0).unwrap(), 0);
        assert_eq!(*kernel.get(4, 4, 0).unwrap(), 0);
    }

    #[test]
    fn test_get_structuring_element_ellipse() {
        let kernel = get_structuring_element(
            MorphShapes::Ellipse,
            types::Size::new(5_usize, 5_usize),
            types::Point::new(-1_i32, -1_i32),
        )
        .unwrap();
        assert_eq!(kernel.rows, 5);
        assert_eq!(kernel.cols, 5);

        // Centre pixel should always be 1
        assert_eq!(*kernel.get(2, 2, 0).unwrap(), 1);
        // Corners of a 5×5 ellipse should be 0
        assert_eq!(*kernel.get(0, 0, 0).unwrap(), 0);
        assert_eq!(*kernel.get(0, 4, 0).unwrap(), 0);
        assert_eq!(*kernel.get(4, 0, 0).unwrap(), 0);
        assert_eq!(*kernel.get(4, 4, 0).unwrap(), 0);
        // Centre row should be fully filled
        for c in 0..5 {
            assert_eq!(*kernel.get(2, c, 0).unwrap(), 1, "centre row col {c}");
        }
    }

    #[test]
    fn test_erode_basic() {
        // 5×5 image with a bright 3×3 block in the centre
        let mut data = vec![0u8; 25];
        for r in 1..4 {
            for c in 1..4 {
                data[r * 5 + c] = 255;
            }
        }
        let src = Matrix::from_vec(5, 5, 1, data);
        let kernel = get_structuring_element(
            MorphShapes::Rect,
            types::Size::new(3_usize, 3_usize),
            types::Point::new(-1_i32, -1_i32),
        )
        .unwrap();

        let eroded = erode(
            &src,
            &kernel,
            types::Point::new(-1, -1),
            1,
            BorderTypes::Constant,
        )
        .unwrap();

        // After erosion with 3×3 rect, only the very centre pixel survives
        assert_eq!(*eroded.get(2, 2, 0).unwrap(), 255);
        // Edge pixels of the block should be eroded to 0
        assert_eq!(*eroded.get(1, 1, 0).unwrap(), 0);
        assert_eq!(*eroded.get(1, 2, 0).unwrap(), 0);
    }

    #[test]
    fn test_dilate_basic() {
        // 5×5 image with a single bright pixel at the centre
        let mut data = vec![0u8; 25];
        data[2 * 5 + 2] = 255;
        let src = Matrix::from_vec(5, 5, 1, data);
        let kernel = get_structuring_element(
            MorphShapes::Rect,
            types::Size::new(3_usize, 3_usize),
            types::Point::new(-1_i32, -1_i32),
        )
        .unwrap();

        let dilated = dilate(
            &src,
            &kernel,
            types::Point::new(-1, -1),
            1,
            BorderTypes::Constant,
        )
        .unwrap();

        // After dilation with 3×3 rect, the centre pixel should spread to a 3×3 block
        for r in 1..4 {
            for c in 1..4 {
                assert_eq!(
                    *dilated.get(r, c, 0).unwrap(),
                    255,
                    "should be 255 at ({r},{c})"
                );
            }
        }
        // Corners should remain 0
        assert_eq!(*dilated.get(0, 0, 0).unwrap(), 0);
        assert_eq!(*dilated.get(4, 4, 0).unwrap(), 0);
    }

    #[test]
    fn test_erode_iterations() {
        // 7×7 image with a 5×5 bright block in the centre
        let mut data = vec![0u8; 49];
        for r in 1..6 {
            for c in 1..6 {
                data[r * 7 + c] = 255;
            }
        }
        let src = Matrix::from_vec(7, 7, 1, data);
        let kernel = get_structuring_element(
            MorphShapes::Rect,
            types::Size::new(3_usize, 3_usize),
            types::Point::new(-1_i32, -1_i32),
        )
        .unwrap();

        let eroded1 = erode(
            &src,
            &kernel,
            types::Point::new(-1, -1),
            1,
            BorderTypes::Constant,
        )
        .unwrap();
        let eroded2 = erode(
            &src,
            &kernel,
            types::Point::new(-1, -1),
            2,
            BorderTypes::Constant,
        )
        .unwrap();

        // 1 iteration: 5×5 → 3×3 bright block remains
        let sum1: u32 = eroded1.data.iter().map(|&v| v as u32).sum();
        // 2 iterations: 5×5 → 1×1 bright block remains
        let sum2: u32 = eroded2.data.iter().map(|&v| v as u32).sum();

        assert!(sum1 > sum2, "2 iterations should erode more than 1");
        assert_eq!(sum2, 255); // Only centre pixel
    }

    #[test]
    fn test_morphology_ex_open() {
        // Open = erode → dilate → removes isolated bright noise
        let mut data = vec![0u8; 49];
        // Single bright pixel (noise)
        data[3 * 7 + 3] = 255;
        let src = Matrix::from_vec(7, 7, 1, data);
        let kernel = get_structuring_element(
            MorphShapes::Rect,
            types::Size::new(3_usize, 3_usize),
            types::Point::new(-1_i32, -1_i32),
        )
        .unwrap();

        let opened = morphology_ex(
            &src,
            MorphTypes::Open,
            &kernel,
            types::Point::new(-1, -1),
            1,
            BorderTypes::Constant,
        )
        .unwrap();

        // A single pixel should be removed by opening
        assert!(opened.data.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_morphology_ex_close() {
        // Close = dilate → erode → fills isolated dark holes
        let mut data = vec![255u8; 49];
        // Single dark pixel (hole)
        data[3 * 7 + 3] = 0;
        let src = Matrix::from_vec(7, 7, 1, data);
        let kernel = get_structuring_element(
            MorphShapes::Rect,
            types::Size::new(3_usize, 3_usize),
            types::Point::new(-1_i32, -1_i32),
        )
        .unwrap();

        let closed = morphology_ex(
            &src,
            MorphTypes::Close,
            &kernel,
            types::Point::new(-1, -1),
            1,
            BorderTypes::Constant,
        )
        .unwrap();

        // The dark hole should be filled; all interior pixels should be 255
        // (edges may be affected by border)
        assert_eq!(*closed.get(3, 3, 0).unwrap(), 255);
    }

    #[test]
    fn test_morphology_ex_gradient() {
        // Gradient = dilate − erode → detects edges
        let mut data = vec![0u8; 49];
        for r in 2..5 {
            for c in 2..5 {
                data[r * 7 + c] = 255;
            }
        }
        let src = Matrix::from_vec(7, 7, 1, data);
        let kernel = get_structuring_element(
            MorphShapes::Rect,
            types::Size::new(3_usize, 3_usize),
            types::Point::new(-1_i32, -1_i32),
        )
        .unwrap();

        let gradient = morphology_ex(
            &src,
            MorphTypes::Gradient,
            &kernel,
            types::Point::new(-1, -1),
            1,
            BorderTypes::Constant,
        )
        .unwrap();

        // Interior of the block should be 0 (dilate == erode for interior)
        assert_eq!(*gradient.get(3, 3, 0).unwrap(), 0);
        // Edges should be non-zero
        let edge_sum: u32 = gradient.data.iter().map(|&v| v as u32).sum();
        assert!(edge_sum > 0);
    }

    // -------------------------------------------------------------------
    //  Pyramid operations
    // -------------------------------------------------------------------

    #[test]
    fn test_pyr_down_size() {
        let src = Matrix::<u8>::new(8, 8, 1);
        let dst = pyr_down(&src, None, BorderTypes::Reflect101).unwrap();
        assert_eq!(dst.rows, 4);
        assert_eq!(dst.cols, 4);
    }

    #[test]
    fn test_pyr_up_size() {
        let src = Matrix::<u8>::new(4, 4, 1);
        let dst = pyr_up(&src, None, BorderTypes::Reflect101).unwrap();
        assert_eq!(dst.rows, 8);
        assert_eq!(dst.cols, 8);
    }

    #[test]
    fn test_pyr_down_uniform() {
        // A uniform image should remain uniform after pyr_down
        let src = Matrix::<u8>::from_vec(8, 8, 1, vec![100u8; 64]);
        let dst = pyr_down(&src, None, BorderTypes::Reflect101).unwrap();
        for &v in &dst.data {
            assert!(
                (v as i16 - 100).abs() <= 1,
                "Uniform pyr_down should preserve value, got {v}"
            );
        }
    }

    #[test]
    fn test_pyr_down_odd_size() {
        // 7×9 → (7+1)/2 × (9+1)/2 = 4×5
        let src = Matrix::<u8>::new(9, 7, 1);
        let dst = pyr_down(&src, None, BorderTypes::Reflect101).unwrap();
        assert_eq!(dst.rows, 5);
        assert_eq!(dst.cols, 4);
    }

    #[test]
    fn test_build_pyramid_levels() {
        let src = Matrix::<u8>::new(64, 64, 1);
        let pyramid = build_pyramid(&src, 3, BorderTypes::Reflect101).unwrap();
        assert_eq!(pyramid.len(), 4);
        assert_eq!(pyramid[0].rows, 64);
        assert_eq!(pyramid[1].rows, 32);
        assert_eq!(pyramid[2].rows, 16);
        assert_eq!(pyramid[3].rows, 8);
    }

    #[test]
    fn test_build_pyramid_multichannel() {
        let src = Matrix::<u8>::new(32, 32, 3);
        let pyramid = build_pyramid(&src, 2, BorderTypes::Reflect101).unwrap();
        assert_eq!(pyramid.len(), 3);
        assert_eq!(pyramid[0].channels, 3);
        assert_eq!(pyramid[1].channels, 3);
        assert_eq!(pyramid[2].channels, 3);
        assert_eq!(pyramid[1].rows, 16);
        assert_eq!(pyramid[2].rows, 8);
    }

    // -------------------------------------------------------------------
    //  Hough Transform operations
    // -------------------------------------------------------------------

    #[test]
    fn test_hough_lines() {
        let mut data = vec![0u8; 100];
        // Draw a diagonal line
        for i in 0..10 {
            data[i * 10 + i] = 255;
        }
        let src = Matrix::from_vec(10, 10, 1, data);

        let lines = hough_lines(&src, 1.0, CV_PI / 180.0, 5, 0.0, CV_PI).unwrap();

        assert!(
            !lines.is_empty(),
            "Hough lines failed to detect the main diagonal line"
        );

        let mut found_diag = false;
        for line in lines {
            let rho = line[0];
            let theta = line[1];
            // Diagonal line from 0,0 to 10,10 should have theta around pi/4 or 3pi/4.
            // Using standard representation, theta ≈ 2.356 (135 degrees) and rho ≈ 0
            if (theta - CV_PI as f32 * 0.75).abs() < 0.1 && rho.abs() < 2.0 {
                found_diag = true;
                break;
            }
        }
        assert!(found_diag, "Did not detect the correct diagonal line");
    }

    #[test]
    fn test_hough_lines_p() {
        let mut data = vec![0u8; 100];
        // Draw a horizontal line segment
        for i in 2..8 {
            data[5 * 10 + i] = 255;
        }
        let src = Matrix::from_vec(10, 10, 1, data);

        let lines = hough_lines_p(&src, 1.0, CV_PI / 180.0, 3, 5.0, 2.0).unwrap();

        assert!(
            !lines.is_empty(),
            "Hough Lines P failed to detect the line segment"
        );

        let mut found_segment = false;
        for line in lines {
            let (x1, y1, x2, y2) = (line[0], line[1], line[2], line[3]);
            if y1 == 5 && y2 == 5 && (x1 - x2).abs() >= 5 {
                found_segment = true;
                break;
            }
        }
        assert!(
            found_segment,
            "Did not detect the horizontal line segment correctly"
        );
    }

    #[test]
    fn test_hough_circles() {
        let mut data = vec![0u8; 400];
        // Draw a basic circle of radius 5 at center (10, 10) in a 20x20 image
        let cx = 10;
        let cy = 10;
        let r = 5;
        for y in 0..20 {
            for x in 0..20 {
                let dx = x as f32 - cx as f32;
                let dy = y as f32 - cy as f32;
                let dist = (dx * dx + dy * dy).sqrt();
                if (dist - r as f32).abs() < 0.5 {
                    data[y * 20 + x] = 255;
                }
            }
        }

        let src = Matrix::from_vec(20, 20, 1, data);

        let circles = hough_circles(&src, 1.0, 10.0, 10.0, 5.0, 3, 10).unwrap();

        assert!(
            !circles.is_empty(),
            "Hough circles failed to detect the circle"
        );

        let mut found_circle = false;
        for c in circles {
            let cx_f = c[0];
            let cy_f = c[1];
            let r_f = c[2];

            if (cx_f - 10.0).abs() < 2.0 && (cy_f - 10.0).abs() < 2.0 && (r_f - 5.0).abs() < 2.0 {
                found_circle = true;
                break;
            }
        }
        assert!(
            found_circle,
            "Did not detect the circle at the correct location"
        );
    }

    #[test]
    fn test_resize_size() {
        let src = Matrix::<u8>::new(4, 4, 1);
        let dst = resize(&src, Size::new(2, 2)).unwrap();
        assert_eq!(dst.rows, 2);
        assert_eq!(dst.cols, 2);
        assert_eq!(dst.channels, 1);

        let dst2 = resize(&src, Size::new(8, 8)).unwrap();
        assert_eq!(dst2.rows, 8);
        assert_eq!(dst2.cols, 8);
        assert_eq!(dst2.channels, 1);
    }

    #[test]
    fn test_resize_uniform() {
        let src = Matrix::<u8>::from_vec(4, 4, 1, vec![100; 16]);
        let dst = resize(&src, Size::new(8, 8)).unwrap();
        for &val in &dst.data {
            assert_eq!(val, 100);
        }
    }

    #[test]
    fn test_resize_bilinear_interpolation() {
        // Simple 2x2 image:
        // [ 10,  30 ]
        // [ 50,  70 ]
        let src = Matrix::<u8>::from_vec(2, 2, 1, vec![10, 30, 50, 70]);
        // Resize to 4x4 using bilinear interpolation
        let dst = resize(&src, Size::new(4, 4)).unwrap();
        assert_eq!(dst.rows, 4);
        assert_eq!(dst.cols, 4);

        // Verify the 4 corners retain original values due to clamping at boundaries
        assert_eq!(*dst.at(0, 0, 0).unwrap(), 10);
        assert_eq!(*dst.at(0, 3, 0).unwrap(), 30);
        assert_eq!(*dst.at(3, 0, 0).unwrap(), 50);
        assert_eq!(*dst.at(3, 3, 0).unwrap(), 70);

        // Center pixel should be around the average of the whole patch (around 40)
        let center_val = *dst.at(1, 1, 0).unwrap();
        assert!(center_val > 10 && center_val < 70);
    }

    #[test]
    fn test_resize_errors() {
        let src = Matrix::<u8>::new(4, 4, 1);
        assert!(resize(&src, Size::new(0, 4)).is_err());
        assert!(resize(&src, Size::new(4, 0)).is_err());
    }

    #[test]
    fn test_remap_nearest() {
        let src = Matrix::from_vec(3, 3, 1, vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90]);
        let mut map1 = Matrix::<f32>::new(3, 3, 1);
        let mut map2 = Matrix::<f32>::new(3, 3, 1);
        for y in 0..3 {
            for x in 0..3 {
                *map1.at_mut(y, x, 0).unwrap() = (x as f32) + 1.0;
                *map2.at_mut(y, x, 0).unwrap() = y as f32;
            }
        }

        let res = remap(
            &src,
            &map1,
            &map2,
            InterpolationFlags::Nearest,
            BorderTypes::Constant,
            Scalar::all(0u8),
        )
        .unwrap();

        assert_eq!(res.data, vec![20, 30, 0, 50, 60, 0, 80, 90, 0,]);
    }

    #[test]
    fn test_remap_bilinear() {
        let src = Matrix::from_vec(2, 2, 1, vec![10.0f32, 20.0, 30.0, 40.0]);
        let mut map1 = Matrix::<f32>::new(2, 2, 1);
        let mut map2 = Matrix::<f32>::new(2, 2, 1);
        for y in 0..2 {
            for x in 0..2 {
                *map1.at_mut(y, x, 0).unwrap() = 0.5;
                *map2.at_mut(y, x, 0).unwrap() = 0.5;
            }
        }

        let res = remap(
            &src,
            &map1,
            &map2,
            InterpolationFlags::Linear,
            BorderTypes::Constant,
            Scalar::all(0.0f32),
        )
        .unwrap();

        for val in res.data {
            assert!((val - 25.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_warp_perspective_identity() {
        let src = Matrix::from_vec(2, 2, 1, vec![10u8, 20, 30, 40]);
        let m = Matrix::from_vec(3, 3, 1, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let res = warp_perspective(
            &src,
            &m,
            Size2i::new(2, 2),
            InterpolationFlags::Nearest,
            BorderTypes::Constant,
            Scalar::all(0u8),
        )
        .unwrap();

        assert_eq!(res.data, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_warp_perspective_translation() {
        let src = Matrix::from_vec(2, 2, 1, vec![10f32, 20.0, 30.0, 40.0]);
        // Translate by dx=1.0, dy=0.0
        let m = Matrix::from_vec(3, 3, 1, vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let res = warp_perspective(
            &src,
            &m,
            Size2i::new(2, 2),
            InterpolationFlags::Nearest,
            BorderTypes::Constant,
            Scalar::all(0.0f32),
        )
        .unwrap();

        assert_eq!(res.data, vec![0.0, 10.0, 0.0, 30.0]);
    }

    #[test]
    fn test_compare_hist_simd_coverage() {
        let mut h1 = Matrix::<f32>::new(1, 10, 1);
        let mut h2 = Matrix::<f32>::new(1, 10, 1);
        for i in 0..10 {
            *h1.at_mut(0, i, 0).unwrap() = i as f32;
            *h2.at_mut(0, i, 0).unwrap() = (9 - i) as f32;
        }

        let c = compare_hist(&h1, &h2, HistCompMethods::Correl).unwrap();
        assert!(c > -1.1 && c < 1.1);

        let c = compare_hist(&h1, &h2, HistCompMethods::ChiSqr).unwrap();
        assert!(c >= 0.0);

        let c = compare_hist(&h1, &h2, HistCompMethods::Intersection).unwrap();
        assert!(c >= 0.0);

        let c = compare_hist(&h1, &h2, HistCompMethods::Bhattacharyya).unwrap();
        assert!(c >= 0.0);
    }

    #[test]
    fn test_calc_hist_uniform_1d() {
        let data: Vec<u8> = (0..16).collect();
        let img = Matrix::from_vec(4, 4, 1, data);
        let hist = calc_hist(
            &[&img],
            &[0],
            None,
            &[16],
            &[RangeSpec::Uniform(0.0, 16.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist.data.len(), 16);
        for &v in hist.data.iter() {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn test_calc_hist_uniform_1d_fewer_bins() {
        let data: Vec<u8> = (0..16).collect();
        let img = Matrix::from_vec(4, 4, 1, data);
        let hist = calc_hist(
            &[&img],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 16.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist.data.len(), 4);
        for &v in hist.data.iter() {
            assert_eq!(v, 4.0);
        }
    }

    #[test]
    fn test_calc_hist_mask() {
        let img = Matrix::from_vec(3, 4, 1, vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        // Partial mask: checkerboard
        let mask = Matrix::from_vec(3, 4, 1, vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
        let h = calc_hist(
            &[&img],
            &[0],
            Some(&mask),
            &[4],
            &[RangeSpec::Uniform(0.0, 12.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(h.data, vec![1.0, 2.0, 1.0, 2.0]);

        // All-zero mask: nothing counted
        let img2 = Matrix::from_vec(2, 2, 1, vec![0u8, 1, 2, 3]);
        let mask_zero = Matrix::from_vec(2, 2, 1, vec![0u8; 4]);
        let h = calc_hist(
            &[&img2],
            &[0],
            Some(&mask_zero),
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(h.data, vec![0.0, 0.0, 0.0, 0.0]);

        // All-one mask: all counted
        let mask_one = Matrix::from_vec(2, 2, 1, vec![1u8; 4]);
        let h = calc_hist(
            &[&img2],
            &[0],
            Some(&mask_one),
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(h.data, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_calc_hist_multichannel_mask_error() {
        // Regression test: a mask must be single-channel. Previously only
        // rows/cols were checked, so a multichannel mask was silently
        // accepted and only its channel 0 was ever read.
        let img = Matrix::from_vec(2, 2, 1, vec![0u8, 1, 2, 3]);
        let mask = Matrix::from_vec(2, 2, 3, vec![1u8; 12]);
        assert!(calc_hist(
            &[&img],
            &[0],
            Some(&mask),
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .is_err());
    }

    #[test]
    fn test_calc_hist_multi_image_channel_indexing() {
        // images[0] has 2 channels, images[1] has 1 channel
        let img0 = Matrix::from_vec(2, 1, 2, vec![10, 20, 30, 40]);
        let img1 = Matrix::from_vec(2, 1, 1, vec![100, 200]);

        let hist = calc_hist(
            &[&img0, &img1],
            &[0, 2],
            None,
            &[2, 2],
            &[
                RangeSpec::Uniform(0.0, 50.0),
                RangeSpec::Uniform(0.0, 250.0),
            ],
            false,
            None,
        )
        .unwrap();

        // pixel(0,0): ch0=10->bin0, ch2=100->bin0 -> idx=0
        // pixel(1,0): ch0=30->bin1, ch2=200->bin1 -> idx=3
        assert_eq!(hist.data[0], 1.0);
        assert_eq!(hist.data[1], 0.0);
        assert_eq!(hist.data[2], 0.0);
        assert_eq!(hist.data[3], 1.0);
    }

    #[test]
    fn test_calc_hist_nonuniform() {
        let data: Vec<u8> = (0..20).collect();
        let img = Matrix::from_vec(4, 5, 1, data);
        // Non-uniform: boundaries [0, 5, 20] -> 2 bins: [0,5) and [5,20)
        let hist = calc_hist(
            &[&img],
            &[0],
            None,
            &[2],
            &[RangeSpec::NonUniform(vec![0.0, 5.0, 20.0])],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist.data[0], 5.0); // values 0..4
        assert_eq!(hist.data[1], 15.0); // values 5..19
    }

    #[test]
    fn test_calc_back_project() {
        let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let img = Matrix::from_vec(2, 4, 1, data);
        let hist = Matrix::from_vec(4, 1, 1, vec![10.0, 20.0, 30.0, 40.0]);
        let bp = calc_back_project(
            &[&img],
            &[0],
            &[4],
            &hist,
            &[RangeSpec::Uniform(0.0, 8.0)],
            1.0,
        )
        .unwrap();
        assert_eq!(
            bp.data,
            vec![10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0]
        );
    }

    #[test]
    fn test_calc_back_project_2d_shape() {
        // Regression test: without an explicit hist_size, a flat 8-bin
        // histogram used to have its shape *guessed* from its length alone
        // (infer_hist_size), which silently produced the wrong shape for any
        // non-perfect-power dims (e.g. [2, 4] was inferred as [1, 8]) and
        // corrupted the bin math. hist_size is now required explicitly.
        let img = Matrix::from_vec(1, 1, 2, vec![1u8, 7]);
        let hist = Matrix::from_vec(
            8,
            1,
            1,
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        );
        let bp = calc_back_project(
            &[&img],
            &[0, 1],
            &[2, 4],
            &hist,
            &[RangeSpec::Uniform(0.0, 2.0), RangeSpec::Uniform(0.0, 8.0)],
            1.0,
        )
        .unwrap();
        // channel 0 (value 1, range [0,2), 2 bins) -> bin 1
        // channel 1 (value 7, range [0,8), 4 bins)  -> bin 3
        // flat index with strides [4, 1] -> 1*4 + 3 = 7 -> hist.data[7] = 80.0
        assert_eq!(bp.data, vec![80.0]);
    }

    #[test]
    fn test_calc_back_project_hist_size_mismatch_error() {
        let img = Matrix::from_vec(1, 4, 1, vec![0u8, 1, 2, 3]);
        let hist = Matrix::from_vec(4, 1, 1, vec![10.0, 20.0, 30.0, 40.0]);
        assert!(calc_back_project(
            &[&img],
            &[0],
            &[3], // doesn't match hist's 4 bins
            &hist,
            &[RangeSpec::Uniform(0.0, 4.0)],
            1.0,
        )
        .is_err());
    }

    #[test]
    fn test_calc_back_project_invalid_uniform_range_error() {
        // Regression test: calc_back_project used to skip the range
        // validation calc_hist already had (lo < hi, NaN rejection,
        // non-uniform boundary shape/ordering), silently producing a
        // plausible-but-wrong projection instead of an error.
        let img = Matrix::from_vec(1, 4, 1, vec![0u8, 1, 2, 3]);
        let hist = Matrix::from_vec(4, 1, 1, vec![10.0, 20.0, 30.0, 40.0]);

        // NaN bounds: lo.partial_cmp(hi) is None, must be rejected.
        assert!(calc_back_project(
            &[&img],
            &[0],
            &[4],
            &hist,
            &[RangeSpec::Uniform(f32::NAN, f32::NAN)],
            1.0,
        )
        .is_err());

        // lo >= hi.
        assert!(calc_back_project(
            &[&img],
            &[0],
            &[4],
            &hist,
            &[RangeSpec::Uniform(4.0, 0.0)],
            1.0,
        )
        .is_err());
    }

    #[test]
    fn test_calc_back_project_invalid_nonuniform_boundaries_error() {
        let img = Matrix::from_vec(1, 4, 1, vec![0u8, 1, 2, 3]);
        let hist = Matrix::from_vec(4, 1, 1, vec![10.0, 20.0, 30.0, 40.0]);

        // Wrong length: needs hist_size[0] + 1 = 5 boundaries, only 3 given.
        assert!(calc_back_project(
            &[&img],
            &[0],
            &[4],
            &hist,
            &[RangeSpec::NonUniform(vec![0.0, 2.0, 4.0])],
            1.0,
        )
        .is_err());

        // Not strictly increasing.
        assert!(calc_back_project(
            &[&img],
            &[0],
            &[4],
            &hist,
            &[RangeSpec::NonUniform(vec![0.0, 2.0, 1.0, 3.0, 4.0])],
            1.0,
        )
        .is_err());
    }

    #[test]
    fn test_compare_hist_correl_identical() {
        let h1 = Matrix::from_vec(4, 1, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let h2 = Matrix::from_vec(4, 1, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let corr = compare_hist(&h1, &h2, HistCompMethods::Correl).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compare_hist_correl_opposite() {
        let h1 = Matrix::from_vec(4, 1, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let h2 = Matrix::from_vec(4, 1, 1, vec![4.0, 3.0, 2.0, 1.0]);
        let corr = compare_hist(&h1, &h2, HistCompMethods::Correl).unwrap();
        assert!((corr - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_compare_hist_intersect() {
        let h1 = Matrix::from_vec(4, 1, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let h2 = Matrix::from_vec(4, 1, 1, vec![4.0, 3.0, 2.0, 1.0]);
        let inter = compare_hist(&h1, &h2, HistCompMethods::Intersection).unwrap();
        assert!((inter - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_compare_hist_chi_sqr() {
        let h1 = Matrix::from_vec(3, 1, 1, vec![1.0, 2.0, 3.0]);
        let h2 = Matrix::from_vec(3, 1, 1, vec![1.0, 2.0, 3.0]);
        let chi = compare_hist(&h1, &h2, HistCompMethods::ChiSqr).unwrap();
        assert!((chi - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compare_hist_bhattacharyya_identical() {
        let h1 = Matrix::from_vec(4, 1, 1, vec![0.25, 0.25, 0.25, 0.25]);
        let h2 = Matrix::from_vec(4, 1, 1, vec![0.25, 0.25, 0.25, 0.25]);
        let bc = compare_hist(&h1, &h2, HistCompMethods::Bhattacharyya).unwrap();
        assert!(bc.abs() < 1e-10);
    }

    #[test]
    fn test_compare_hist_kl_divergence() {
        let h1 = Matrix::from_vec(3, 1, 1, vec![0.5, 0.3, 0.2]);
        let h2 = Matrix::from_vec(3, 1, 1, vec![0.5, 0.3, 0.2]);
        let kl = compare_hist(&h1, &h2, HistCompMethods::KullbackLeibler).unwrap();
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_equalize_hist_uniform() {
        let img = Matrix::from_vec(4, 4, 1, vec![128u8; 16]);
        let dst = equalize_hist(&img).unwrap();
        for &v in dst.data.iter() {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_equalize_hist_gradient() {
        let data: Vec<u8> = (0..=255).collect();
        let img = Matrix::from_vec(16, 16, 1, data);
        let dst = equalize_hist(&img).unwrap();
        assert_eq!(dst.rows, 16);
        assert_eq!(dst.cols, 16);
        assert_eq!(dst.channels, 1);
        let min_val = *dst.data.iter().min().unwrap();
        let max_val = *dst.data.iter().max().unwrap();
        assert_eq!(min_val, 0);
        assert_eq!(max_val, 255);
    }

    #[test]
    fn test_clahe_basic() {
        let data: Vec<u8> = (0..=255).collect();
        let img = Matrix::from_vec(16, 16, 1, data);
        let clahe = create_clahe(40.0, Size2i::new(4, 4));
        let dst = clahe.apply_u8(&img).unwrap();
        assert_eq!(dst.rows, 16);
        assert_eq!(dst.cols, 16);
        assert_eq!(dst.channels, 1);
        assert_eq!(dst.data.len(), 256);
    }

    #[test]
    fn test_clahe_u16() {
        let data: Vec<u16> = (0..=1023).collect();
        let img = Matrix::from_vec(32, 32, 1, data);
        let clahe = create_clahe(40.0, Size2i::new(4, 4));
        let dst = clahe.apply_u16(&img).unwrap();
        assert_eq!(dst.rows, 32);
        assert_eq!(dst.cols, 32);
        assert_eq!(dst.channels, 1);
    }

    #[test]
    fn test_calc_hist_2d() {
        let img = Matrix::from_vec(3, 3, 1, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        let hist = calc_hist(
            &[&img, &img],
            &[0, 0],
            None,
            &[3, 3],
            &[RangeSpec::Uniform(0.0, 9.0), RangeSpec::Uniform(0.0, 9.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist.data.len(), 9);
        assert_eq!(hist.data[0], 3.0);
        assert_eq!(hist.data[1], 0.0);
        assert_eq!(hist.data[2], 0.0);
        assert_eq!(hist.data[3], 0.0);
        assert_eq!(hist.data[4], 3.0);
        assert_eq!(hist.data[5], 0.0);
        assert_eq!(hist.data[6], 0.0);
        assert_eq!(hist.data[7], 0.0);
        assert_eq!(hist.data[8], 3.0);
    }

    #[test]
    fn test_calc_hist_empty_images_error() {
        assert!(calc_hist::<u8>(
            &[],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .is_err());
    }

    #[test]
    fn test_calc_hist_mismatched_channels_error() {
        let img = Matrix::from_vec(2, 2, 1, vec![0u8; 4]);
        assert!(calc_hist(
            &[&img],
            &[0, 1],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .is_err());
    }

    #[test]
    fn test_calc_back_project_empty_channels_error() {
        let img = Matrix::from_vec(2, 2, 1, vec![0u8; 4]);
        let hist = Matrix::from_vec(4, 1, 1, vec![0.0; 4]);
        assert!(calc_back_project::<u8>(
            &[&img],
            &[],
            &[],
            &hist,
            &[RangeSpec::Uniform(0.0, 4.0)],
            1.0,
        )
        .is_err());
    }

    #[test]
    fn test_calc_back_project_zero_width() {
        // Regression test: chunks_mut/par_chunks_mut panic on a zero chunk
        // size regardless of slice length, so a zero-width image must not
        // reach the row-chunking dispatch.
        let img = Matrix::<u8>::from_vec(3, 0, 1, vec![]);
        let hist = Matrix::from_vec(4, 1, 1, vec![0.0; 4]);
        let dst = calc_back_project(
            &[&img],
            &[0],
            &[4],
            &hist,
            &[RangeSpec::Uniform(0.0, 4.0)],
            1.0,
        )
        .unwrap();
        assert_eq!(dst.rows, 3);
        assert_eq!(dst.cols, 0);
        assert_eq!(dst.data.len(), 0);
    }

    #[test]
    fn test_calc_hist_u16_input() {
        let data: Vec<u16> = vec![0, 100, 200, 300, 400, 500];
        let img = Matrix::from_vec(2, 3, 1, data);
        let hist = calc_hist(
            &[&img],
            &[0],
            None,
            &[3],
            &[RangeSpec::Uniform(0.0, 600.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist.data, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_calc_hist_f32_input() {
        let data: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5];
        let img = Matrix::from_vec(2, 2, 1, data);
        let hist = calc_hist(
            &[&img],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist.data, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_calc_hist_boundary_exclusion() {
        let img = Matrix::from_vec(1, 4, 1, vec![0u8, 4, 8, 12]);
        let hist = calc_hist(
            &[&img],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 16.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist.data, vec![1.0, 1.0, 1.0, 1.0]);

        // Value at exact hi boundary should be excluded
        let img2 = Matrix::from_vec(1, 2, 1, vec![4u8, 4]);
        let hist2 = calc_hist(
            &[&img2],
            &[0],
            None,
            &[2],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist2.data, vec![0.0, 0.0]);

        // Value just below hi should be included
        let img3 = Matrix::from_vec(1, 1, 1, vec![3u8]);
        let hist3 = calc_hist(
            &[&img3],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist3.data[3], 1.0);
    }

    #[test]
    fn test_calc_hist_accumulate() {
        let img = Matrix::from_vec(1, 4, 1, vec![0u8, 1, 2, 3]);
        let h1 = calc_hist(
            &[&img],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(h1.data, vec![1.0, 1.0, 1.0, 1.0]);

        let h2 = calc_hist(
            &[&img],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            true,
            Some(&h1),
        )
        .unwrap();
        assert_eq!(h2.data, vec![2.0, 2.0, 2.0, 2.0]);

        let h3 = calc_hist(
            &[&img],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            true,
            Some(&h2),
        )
        .unwrap();
        assert_eq!(h3.data, vec![3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_compare_hist_edge_cases() {
        let z = Matrix::from_vec(3, 1, 1, vec![0.0, 0.0, 0.0]);
        assert!((compare_hist(&z, &z, HistCompMethods::Correl).unwrap() - 1.0).abs() < 1e-10);
        assert!((compare_hist(&z, &z, HistCompMethods::Intersection).unwrap()).abs() < 1e-10);
        assert!(
            (compare_hist(&z, &z, HistCompMethods::Bhattacharyya).unwrap() - 1.0).abs() < 1e-10
        );

        let p1 = Matrix::from_vec(4, 1, 1, vec![0.5, 0.25, 0.125, 0.125]);
        let p2 = Matrix::from_vec(4, 1, 1, vec![0.5, 0.25, 0.125, 0.125]);
        assert!((compare_hist(&p1, &p2, HistCompMethods::Correl).unwrap() - 1.0).abs() < 1e-10);
        assert!((compare_hist(&p1, &p2, HistCompMethods::ChiSqr).unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_calc_back_project_scale() {
        let img = Matrix::from_vec(1, 4, 1, vec![0u8, 1, 2, 3]);
        let hist = Matrix::from_vec(4, 1, 1, vec![10.0, 20.0, 30.0, 40.0]);
        let bp = calc_back_project(
            &[&img],
            &[0],
            &[4],
            &hist,
            &[RangeSpec::Uniform(0.0, 4.0)],
            2.0,
        )
        .unwrap();
        assert_eq!(bp.data, vec![20.0, 40.0, 60.0, 80.0]);
    }

    #[test]
    fn test_calc_hist_errors() {
        let img = Matrix::from_vec(2, 2, 1, vec![0u8; 4]);
        assert!(calc_hist::<u8>(
            &[],
            &[0],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .is_err());
        assert!(calc_hist(
            &[&img],
            &[0, 1],
            None,
            &[4],
            &[RangeSpec::Uniform(0.0, 4.0)],
            false,
            None,
        )
        .is_err());
        let img2 = Matrix::from_vec(3, 3, 1, vec![0u8; 9]);
        assert!(calc_hist(
            &[&img, &img2],
            &[0, 0],
            None,
            &[2, 2],
            &[RangeSpec::Uniform(0.0, 2.0), RangeSpec::Uniform(0.0, 2.0)],
            false,
            None,
        )
        .is_err());
        let hist = Matrix::from_vec(4, 1, 1, vec![0.0; 4]);
        assert!(calc_back_project::<u8>(
            &[&img],
            &[],
            &[],
            &hist,
            &[RangeSpec::Uniform(0.0, 4.0)],
            1.0,
        )
        .is_err());
    }

    #[test]
    fn test_calc_hist_multichannel_select() {
        let img = Matrix::from_vec(
            1,
            3,
            3,
            vec![
                10, 100, 200, // pixel 0
                20, 150, 250, // pixel 1
                30, 50, 100, // pixel 2
            ],
        );
        let hist = calc_hist(
            &[&img],
            &[1],
            None,
            &[3],
            &[RangeSpec::Uniform(0.0, 200.0)],
            false,
            None,
        )
        .unwrap();
        assert_eq!(hist.data, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_clahe_no_clip() {
        let data: Vec<u8> = (0..=255).collect();
        let img = Matrix::from_vec(16, 16, 1, data);
        let clahe = create_clahe(0.0, Size2i::new(4, 4));
        let dst = clahe.apply_u8(&img).unwrap();
        assert_eq!(dst.data.len(), 256);
    }

    #[test]
    fn test_clahe_tile_interp_weights_are_bounded() {
        // Regression test: the interpolation fraction used to be derived
        // *after* clamping the tile index, which produced weights outside
        // [0, 1] (extrapolation) at the top/left edges - e.g. at y=0 with
        // tile_rows=4, tyf=-0.5 used to yield ya=-0.5, ya1=1.5 instead of a
        // convex blend. Check the invariant holds across the coordinate
        // range that actually occurs (coord_f = pixel*inv_tile_extent - 0.5,
        // for pixel in 0..src_extent), for several tile counts.
        use crate::imgproc::histogram::tile_interp_weights;

        for num_tiles in [1usize, 2, 3, 5] {
            let tile_extent = 4usize; // pixels per tile
            let inv = 1.0 / tile_extent as f64;
            let src_extent = num_tiles * tile_extent;

            for pixel in 0..src_extent {
                let coord_f = pixel as f64 * inv - 0.5;
                let (idx1, idx2, w1, w2) = tile_interp_weights(coord_f, num_tiles);

                assert!(
                    (0.0..=1.0).contains(&w1) && (0.0..=1.0).contains(&w2),
                    "weights out of [0,1] at pixel={pixel}, num_tiles={num_tiles}: w1={w1}, w2={w2}"
                );
                assert!(
                    (w1 + w2 - 1.0).abs() < 1e-12,
                    "weights don't sum to 1 at pixel={pixel}: w1={w1}, w2={w2}"
                );
                assert!(idx1 < num_tiles && idx2 < num_tiles);
            }
        }

        // The exact case from the bug report: y=0, tile_rows=4 -> tyf=-0.5.
        let (idx1, idx2, w1, w2) = tile_interp_weights(-0.5, 2);
        assert_eq!((idx1, idx2), (0, 0)); // both clamp to the first tile
        assert!((w1 - 0.5).abs() < 1e-12 && (w2 - 0.5).abs() < 1e-12);
    }
}
