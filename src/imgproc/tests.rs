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
}
