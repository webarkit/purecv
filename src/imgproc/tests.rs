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
mod tests {
    use crate::core::*;
    use crate::imgproc::*;

    #[test]
    fn test_blur() {
        let m = Matrix::from_vec(3, 3, 1, vec![
            10u8, 10, 10,
            10, 10, 10,
            10, 10, 10,
        ]);
        let ksize = Size2i::new(3, 3);
        let res = blur(&m, ksize, Point2i::new(-1, -1), BorderTypes::REFLECT_101).unwrap();
        assert_eq!(res.data, vec![10u8; 9]);
    }

    #[test]
    fn test_box_filter() {
        let m = Matrix::from_vec(3, 3, 1, vec![
            1u8, 1, 1,
            1, 1, 1,
            1, 1, 1,
        ]);
        let ksize = Size2i::new(3, 3);
        // Sum should be 9 for each pixel because of border reflection
        let res = box_filter(&m, ksize, Point2i::new(-1, -1), false, BorderTypes::REFLECT_101).unwrap();
        for val in res.data {
            assert_eq!(val, 9u8);
        }
    }

    #[test]
    fn test_gaussian_blur() {
        let m = Matrix::from_vec(5, 5, 1, vec![100u8; 25]);
        let ksize = Size2i::new(3, 3);
        let res = gaussian_blur(&m, ksize, 1.0, 1.0, BorderTypes::REFLECT_101).unwrap();
        // Since all pixels are 100, the result should be 100 (normalized)
        for val in res.data {
            assert!(val >= 99 && val <= 101); // Allow small deviation for rounding
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
        
        let res = bilateral_filter(&src, 5, 50.0, 50.0, BorderTypes::REFLECT_101).unwrap();
        
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
        let _res = sobel(&src, 1, 0, 3, 1.0, 0.0, BorderTypes::REFLECT_101).unwrap();
        
        // At the edge (x=4 to x=5), the derivative should be high.
        // Neighbors: x=4 (0), x=6 (255) -> 255 - 0 = 255.
        // Sobel dx=1: [-1, 0, 1] smoothed by [1, 2, 1] vertically.
        // Total weight is 4. Result should be around 255 * 4 = 1020, 
        // but it's cast back to u8 if T is u8.
        // Let's use f32 to avoid overflow for testing.
        
        let src_f32: Matrix<f32> = Matrix::from_vec(10, 10, 1, src.data.iter().map(|&v| v as f32).collect());
        let res_f32 = sobel(&src_f32, 1, 0, 3, 1.0, 0.0, BorderTypes::REFLECT_101).unwrap();
        
        let edge_val = *res_f32.at(5, 5, 0).unwrap();
        assert!(edge_val > 500.0); // 255 * 4 = 1020 expected for Sobel ksize=3
    }

    #[test]
    fn test_scharr() {
        let mut data = vec![0u8; 100];
        for x in 5..10 {
            for y in 0..10 {
                data[y * 10 + x] = 255;
            }
        }
        let src = Matrix::<f32>::from_vec(10, 10, 1, data.iter().map(|&v| v as f32).collect());
        
        let res = scharr(&src, 1, 0, 1.0, 0.0, BorderTypes::REFLECT_101).unwrap();
        let edge_val = *res.at(5, 5, 0).unwrap();
        // Scharr weight for center row is 10, total 3+10+3 = 16.
        // Expected: 255 * 16 = 4080
        assert!(edge_val > 2000.0);
    }

    #[test]
    fn test_laplacian() {
        // Uniform image
        let src = Matrix::<f32>::from_vec(5, 5, 1, vec![100.0; 25]);
        let res = laplacian(&src, 1, 1.0, 0.0, BorderTypes::REFLECT_101).unwrap();
        
        // Laplacian of a uniform field should be 0
        for &val in &res.data {
            assert!(val.abs() < 1e-5);
        }
        
        // Image with a peak at (2, 2)
        let mut src_peak = Matrix::<f64>::new(5, 5, 1);
        src_peak.set(2, 2, 0, 255.0);
        let res_peak = laplacian(&src_peak, 1, 1.0, 0.0, BorderTypes::REFLECT_101).unwrap();
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
}
