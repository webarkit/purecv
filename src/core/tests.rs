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
mod core_tests {
    use crate::core::arithm::*;
    use crate::core::types::*;
    use crate::core::utils::*;
    use crate::core::*;

    #[test]
    fn test_point_add() {
        let p1 = Point2i::new(10, 20);
        let p2 = Point2i::new(5, 5);
        let p3 = p1 + p2;
        assert_eq!(p3.x, 15);
        assert_eq!(p3.y, 25);
    }

    #[test]
    fn test_size_area() {
        let sz = Size2i::new(100, 50);
        assert_eq!(sz.area(), 5000);
    }

    #[test]
    fn test_matrix_from_size() {
        let sz = Size2i::new(100, 200);
        let mat: Matrix<u8> =
            Matrix::from_size(Size::new(sz.width as usize, sz.height as usize), 3);

        assert_eq!(mat.cols, 100);
        assert_eq!(mat.rows, 200);
        assert_eq!(mat.channels, 3);
        assert_eq!(mat.data.len(), 100 * 200 * 3);
    }

    #[test]
    fn test_rect_tl_br() {
        let r = Rect2i::new(10, 10, 100, 50);
        assert_eq!(r.tl(), Point2i::new(10, 10));
        assert_eq!(r.br(), Point2i::new(110, 60));
    }

    #[test]
    fn test_range() {
        let r = Range::new(10, 20);
        assert_eq!(r.size(), 10);
        assert!(!r.empty());

        let r_all = Range::all();
        assert_eq!(r_all.start, i32::MIN);
    }

    #[test]
    fn test_scalar() {
        let s = Scalar::<u8>::all(255);
        assert_eq!(s.v, [255, 255, 255, 255]);
    }

    #[test]
    fn test_arithmetic() {
        let m1 = Matrix::from_vec(2, 2, 1, vec![10, 20, 30, 40]);
        let m2 = Matrix::from_vec(2, 2, 1, vec![5, 5, 5, 5]);

        let sum = add(&m1, &m2).unwrap();
        assert_eq!(sum.data, vec![15, 25, 35, 45]);

        let diff = subtract(&m1, &m2).unwrap();
        assert_eq!(diff.data, vec![5, 15, 25, 35]);

        let prod = multiply(&m1, &m2).unwrap();
        assert_eq!(prod.data, vec![50, 100, 150, 200]);

        let quot = divide(&m1, &m2).unwrap();
        assert_eq!(quot.data, vec![2, 4, 6, 8]);

        let abs_diff = absdiff(&m2, &m1).unwrap();
        assert_eq!(abs_diff.data, vec![5, 15, 25, 35]);

        // has_no_zero
        let m_zeros = Matrix::from_vec(2, 2, 1, vec![0, 20, 30, 40]);
        assert!(has_no_zero(&m1).unwrap());
        assert!(!has_no_zero(&m_zeros).unwrap());
    }

    #[test]
    fn test_bitwise() {
        let m1 = Matrix::from_vec(1, 4, 1, vec![0b1010, 0b1100, 0b1111, 0b0000]);
        let m2 = Matrix::from_vec(1, 4, 1, vec![0b0101, 0b0110, 0b0000, 0b1111]);

        let and = bitwise_and(&m1, &m2).unwrap();
        assert_eq!(and.data, vec![0b0000, 0b0100, 0b0000, 0b0000]);

        let or = bitwise_or(&m1, &m2).unwrap();
        assert_eq!(or.data, vec![0b1111, 0b1110, 0b1111, 0b1111]);

        let xor = bitwise_xor(&m1, &m2).unwrap();
        assert_eq!(xor.data, vec![0b1111, 0b1010, 0b1111, 0b1111]);

        let m3 = Matrix::from_vec(1, 1, 1, vec![0u8]);
        let not = bitwise_not(&m3).unwrap();
        assert_eq!(not.data, vec![255u8]);
    }

    #[test]
    fn test_weighted() {
        let m1 = Matrix::from_vec(1, 2, 1, vec![100u8, 200u8]);
        let m2 = Matrix::from_vec(1, 2, 1, vec![50u8, 10u8]);

        // dst = m1*0.5 + m2*0.1 + 10.0
        let res = add_weighted(&m1, 0.5, &m2, 0.1, 10.0).unwrap();
        assert_eq!(res.data, vec![65, 111]);
    }

    #[test]
    fn test_scalar_term() {
        use crate::core::types::{Scalar, TermCriteria, TermType};
        let s = Scalar::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(s.v[0], 1.0);
        assert_eq!(s.v[3], 4.0);

        let term = TermCriteria::new(TermType::Both, 100, 0.001);
        assert_eq!(term.max_count, 100);
        assert_eq!(term.epsilon, 0.001);
    }

    #[test]
    fn test_convert_scale_abs() {
        let m = Matrix::<f32>::from_vec(1, 3, 1, vec![-10.0, 0.0, 10.0]);
        // |-10*1 + 0| = 10
        // |0*1 + 0| = 0
        // |10*1 + 0| = 10
        let res = convert_scale_abs(&m, 1.0, 0.0).unwrap();
        assert_eq!(res.data[0], 10);
        assert_eq!(res.data[1], 0);
        assert_eq!(res.data[2], 10);

        // Saturation test: |100*2 + 100| = 300 -> 255
        let m2 = Matrix::<f32>::from_vec(1, 1, 1, vec![100.0]);
        let res2 = convert_scale_abs(&m2, 2.0, 100.0).unwrap();
        assert_eq!(res2.data[0], 255);
    }

    #[test]
    fn test_structural() {
        use crate::core::structural::*;

        // Flip test
        let m = Matrix::<u8>::from_vec(2, 2, 1, vec![1, 2, 3, 4]);
        let f_v = flip(&m, 0).unwrap(); // vertical
        assert_eq!(f_v.data, vec![3, 4, 1, 2]);
        let f_h = flip(&m, 1).unwrap(); // horizontal
        assert_eq!(f_h.data, vec![2, 1, 4, 3]);

        // flip_nd test
        let f_nd_v = flip_nd(&m, 0).unwrap(); // vertical (axis 0)
        assert_eq!(f_nd_v.data, vec![3, 4, 1, 2]);
        let f_nd_h = flip_nd(&m, 1).unwrap(); // horizontal (axis 1)
        assert_eq!(f_nd_h.data, vec![2, 1, 4, 3]);
        let f_nd_both = flip_nd(&m, -1).unwrap(); // both
        assert_eq!(f_nd_both.data, vec![4, 3, 2, 1]);

        // Transpose test
        let m_rect = Matrix::<u8>::from_vec(2, 3, 1, vec![1, 2, 3, 4, 5, 6]);
        let t = transpose(&m_rect).unwrap();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.data, vec![1, 4, 2, 5, 3, 6]);

        // Split/Merge test
        let m_rgb = Matrix::<u8>::from_vec(1, 1, 3, vec![10, 20, 30]);
        let channels = split(&m_rgb).unwrap();
        assert_eq!(channels.len(), 3);
        assert_eq!(channels[0].data[0], 10);
        assert_eq!(channels[1].data[0], 20);
        assert_eq!(channels[2].data[0], 30);

        let merged = merge(&channels).unwrap();
        assert_eq!(merged.data, vec![10, 20, 30]);

        // Rotate test
        let m_rot = Matrix::<u8>::from_vec(2, 2, 1, vec![1, 2, 3, 4]);
        let rot90 = rotate(&m_rot, 0).unwrap();
        assert_eq!(rot90.data, vec![3, 1, 4, 2]);

        // Repeat test
        let m_rep = Matrix::<u8>::from_vec(1, 2, 1, vec![1, 2]);
        let rep = repeat(&m_rep, 2, 2).unwrap();
        assert_eq!(rep.rows, 2);
        assert_eq!(rep.cols, 4);
        assert_eq!(rep.data, vec![1, 2, 1, 2, 1, 2, 1, 2]);

        // mixChannels test: swap R and B in an RGB matrix
        let m_rgb = Matrix::<u8>::from_vec(1, 1, 3, vec![1, 2, 3]);
        let m_bgr = Matrix::<u8>::new(1, 1, 3);
        let mut dst_vec = vec![m_bgr];
        // 0 -> 2 (R to B), 1 -> 1 (G to G), 2 -> 0 (B to R)
        mix_channels(&[m_rgb], &mut dst_vec, &[(0, 2), (1, 1), (2, 0)]).unwrap();
        assert_eq!(dst_vec[0].data, vec![3, 2, 1]);

        // copyMakeBorder test
        let m_pad = Matrix::<u8>::from_vec(1, 1, 1, vec![100]);
        let padded =
            copy_make_border(&m_pad, 1, 1, 1, 1, 0, crate::core::types::Scalar::all(0)).unwrap();
        assert_eq!(padded.rows, 3);
        assert_eq!(padded.cols, 3);
        assert_eq!(padded.data, vec![0, 0, 0, 0, 100, 0, 0, 0, 0]);

        // reshape test
        let m_reshape = Matrix::<u8>::from_vec(1, 4, 1, vec![1, 2, 3, 4]);
        let reshaped = reshape(&m_reshape, 1, 2).unwrap();
        assert_eq!(reshaped.rows, 2);
        assert_eq!(reshaped.cols, 2);
        assert_eq!(reshaped.data, vec![1, 2, 3, 4]);

        // hconcat test
        let m1 = Matrix::<u8>::from_vec(2, 1, 1, vec![1, 2]);
        let m2 = Matrix::<u8>::from_vec(2, 1, 1, vec![3, 4]);
        let h_concat = hconcat(&[m1.clone(), m2.clone()]).unwrap();
        assert_eq!(h_concat.rows, 2);
        assert_eq!(h_concat.cols, 2);
        assert_eq!(h_concat.data, vec![1, 3, 2, 4]);

        // vconcat test
        let v_concat = vconcat(&[m1, m2]).unwrap();
        assert_eq!(v_concat.rows, 4);
        assert_eq!(v_concat.cols, 1);
        assert_eq!(v_concat.data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_math() {
        let m1 = Matrix::from_vec(1, 4, 1, vec![1.0f32, 4.0, 9.0, 16.0]);
        let s = sqrt(&m1).unwrap();
        assert_eq!(s.data, vec![1.0, 2.0, 3.0, 4.0]);

        let m2 = Matrix::from_vec(1, 2, 1, vec![0.0f32, 1.0]);
        let e = exp(&m2).unwrap();
        assert!((e.data[0] - 1.0).abs() < 1e-6);
        assert!((e.data[1] - std::f32::consts::E).abs() < 1e-6);

        let m3 = Matrix::from_vec(1, 2, 1, vec![1.0f32, std::f32::consts::E]);
        let l = crate::core::arithm::log(&m3).unwrap();
        assert!((l.data[0] - 0.0).abs() < 1e-6);
        assert!((l.data[1] - 1.0).abs() < 1e-6);

        let m4 = Matrix::from_vec(1, 2, 1, vec![2.0f32, 3.0]);
        let p = pow(&m4, 2.0).unwrap();
        assert_eq!(p.data, vec![4.0, 9.0]);
    }

    #[test]
    fn test_convert_to() {
        let m = Matrix::from_vec(2, 2, 1, vec![1u8, 2u8, 3u8, 4u8]);
        let m_f32 = m.convert_to::<f32>().unwrap();
        assert_eq!(m_f32.data, vec![1.0f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_norm_normalize() {
        let m = Matrix::from_vec(1, 3, 1, vec![1.0f64, 2.0, 3.0]);

        // Norms
        assert_eq!(norm(&m, NormTypes::Inf, None).unwrap(), 3.0);
        assert_eq!(norm(&m, NormTypes::L1, None).unwrap(), 6.0);
        assert_eq!(
            norm(&m, NormTypes::L2, None).unwrap(),
            (1.0f64 + 4.0 + 9.0).sqrt()
        );

        // Normalize MINMAX to [0, 1]
        let mut m_minmax = Matrix::<f64>::new(1, 3, 1);
        normalize(&m, &mut m_minmax, 0.0, 1.0, NormTypes::MinMax, -1, None).unwrap();
        assert_eq!(m_minmax.data[0], 0.0);
        assert_eq!(m_minmax.data[2], 1.0);
        assert!((m_minmax.data[1] - 0.5).abs() < 1e-6);

        // Normalize L2 to norm 1
        let mut m_l2 = Matrix::<f64>::new(1, 3, 1);
        normalize(&m, &mut m_l2, 1.0, 0.0, NormTypes::L2, -1, None).unwrap();
        let n_l2 = norm(&m_l2, NormTypes::L2, None).unwrap();
        assert!((n_l2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_stats() {
        let m = Matrix::from_vec(2, 2, 1, vec![10.0f64, 20.0, 30.0, 40.0]);

        // Sum
        let s = sum(&m);
        assert_eq!(s.v[0], 100.0);

        // Mean
        let mn = mean(&m);
        assert_eq!(mn.v[0], 25.0);

        // MinMaxLoc
        let (min_val, max_val, min_loc, max_loc) = min_max_loc(&m);
        assert_eq!(min_val, 10.0);
        assert_eq!(max_val, 40.0);
        assert_eq!(min_loc.0, 0);
        assert_eq!(min_loc.1, 0);
        assert_eq!(max_loc.0, 1);
        assert_eq!(max_loc.1, 1);

        // MeanStdDev
        let (mn2, sd) = mean_std_dev(&m);
        assert_eq!(mn2.v[0], 25.0);
        // Variance = ((10-25)^2 + (20-25)^2 + (30-25)^2 + (40-25)^2) / 4
        // Variance = (225 + 25 + 25 + 225) / 4 = 500 / 4 = 125
        // StdDev = sqrt(125)
        assert!((sd.v[0] - 125.0f64.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_factories() {
        // zeros
        let m_zeros = Matrix::<u8>::zeros(2, 2, 1);
        assert_eq!(m_zeros.data, vec![0, 0, 0, 0]);

        // ones
        let m_ones = Matrix::<f32>::ones(1, 4, 1);
        assert_eq!(m_ones.data, vec![1.0, 1.0, 1.0, 1.0]);

        // eye
        let m_eye = Matrix::<i32>::eye(3, 3, 1);
        let expected_eye = vec![1, 0, 0, 0, 1, 0, 0, 0, 1];
        assert_eq!(m_eye.data, expected_eye);

        // diag
        let diag_vals = vec![1.0, 2.0, 3.0];
        let m_diag = Matrix::diag(&diag_vals);
        assert_eq!(m_diag.rows, 3);
        assert_eq!(m_diag.cols, 3);
        let expected_diag = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        assert_eq!(m_diag.data, expected_diag);
    }

    #[test]
    fn test_mat_type_integration() {
        // Test DataType trait depth mapping
        assert_eq!(u8::depth(), Depth::CV_8U);
        assert_eq!(f32::depth(), Depth::CV_32F);

        // Test Matrix::new_with_type
        let mat = Matrix::<u8>::new_with_type(10, 20, CV_8UC3).unwrap();
        assert_eq!(mat.rows, 10);
        assert_eq!(mat.cols, 20);
        assert_eq!(mat.channels, 3);
        assert_eq!(mat.mat_type(), CV_8UC3);

        // Test Matrix::zeros_with_type
        let z = Matrix::<f32>::zeros_with_type(5, 5, CV_32FC1).unwrap();
        assert_eq!(z.data.len(), 25);
        assert!(z.data.iter().all(|&v| v == 0.0));
        assert_eq!(z.mat_type(), CV_32FC1);

        // Test Matrix::ones_with_type
        let o = Matrix::<i16>::ones_with_type(2, 2, CV_16SC2).unwrap();
        assert_eq!(o.data.len(), 8);
        assert!(o.data.iter().all(|&v| v == 1));
        assert_eq!(o.mat_type(), CV_16SC2);

        // Test error when depth mismatch on new_with_type
        let err_res = Matrix::<u8>::new_with_type(10, 20, CV_32FC1);
        assert!(matches!(
            err_res,
            Err(crate::core::error::PureCvError::InvalidInput(_))
        ));

        // Test error when depth mismatch on create_with_type
        let mut mat = Matrix::<u8>::zeros(1, 1, 1);
        let err_create = mat.create_with_type(10, 20, CV_32FC1);
        assert!(matches!(
            err_create,
            Err(crate::core::error::PureCvError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_dot_cross_trace() {
        let m1 = Matrix::from_vec(1, 3, 1, vec![1.0, 2.0, 3.0]);
        let m2 = Matrix::from_vec(1, 3, 1, vec![4.0, 5.0, 6.0]);

        // Dot product: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(dot(&m1, &m2).unwrap(), 32.0);

        // Cross product
        let c = cross(&m1, &m2).unwrap();
        // [2*6-3*5, 3*4-1*6, 1*5-2*4] = [12-15, 12-6, 5-8] = [-3, 6, -3]
        assert_eq!(c.data, vec![-3.0, 6.0, -3.0]);

        // Trace of eye(3) should be 3
        let m_eye = Matrix::<f64>::eye(3, 3, 1);
        assert_eq!(trace(&m_eye).v[0], 3.0);
    }

    #[test]
    fn test_set_identity() {
        let mut m = Matrix::<f64>::zeros(3, 3, 1);
        set_identity(&mut m, Scalar::all(5.0));
        assert_eq!(m.data, vec![5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn test_check_range() {
        let m = Matrix::from_vec(1, 3, 1, vec![1.0, 2.0, 3.0]);
        assert!(check_range(&m, 0.0, 4.0));
        assert!(!check_range(&m, 0.0, 2.5));

        let m_nan = Matrix::from_vec(1, 1, 1, vec![f64::NAN]);
        assert!(!check_range(&m_nan, 0.0, 10.0));
    }

    #[test]
    fn test_gemm() {
        let a = Matrix::from_vec(2, 2, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, 1, vec![5.0, 6.0, 7.0, 8.0]);
        let c = Matrix::from_vec(2, 2, 1, vec![1.0, 1.0, 1.0, 1.0]);

        // res = 1.0 * A * B + 1.0 * C
        let res = gemm(&a, &b, 1.0, &c, 1.0, 0).unwrap();
        assert_eq!(res.data, vec![20.0, 23.0, 44.0, 51.0]);

        // Test with transpose A
        let empty = Matrix::<f64>::new(0, 0, 1);
        let res_t = gemm(&a, &b, 1.0, &empty, 0.0, GEMM_1_T).unwrap();
        assert_eq!(res_t.data, vec![26.0, 30.0, 38.0, 44.0]);
    }

    // ---- RNG tests ----

    #[test]
    fn test_randu_basic() {
        use crate::core::rng::{randu, set_rng_seed};

        set_rng_seed(1234);
        let mut mat = Matrix::<f64>::new(10, 10, 1);
        randu(&mut mat, Scalar::all(0.0), Scalar::all(1.0)).unwrap();

        for &v in &mat.data {
            assert!((0.0..1.0).contains(&v), "value {} out of [0, 1)", v);
        }
    }

    #[test]
    fn test_randn_basic() {
        use crate::core::rng::{randn, set_rng_seed};

        set_rng_seed(5678);
        let mut mat = Matrix::<f64>::new(10, 10, 1);
        randn(&mut mat, Scalar::all(0.0), Scalar::all(1.0)).unwrap();

        // Just verify the matrix was filled (not all zeros).
        let any_nonzero = mat.data.iter().any(|&v| v != 0.0);
        assert!(any_nonzero, "randn produced all zeros");
    }

    #[test]
    fn test_set_rng_seed_reproducible() {
        use crate::core::rng::{randu, set_rng_seed};

        set_rng_seed(999);
        let mut a = Matrix::<f32>::new(5, 5, 3);
        randu(&mut a, Scalar::all(0.0), Scalar::all(255.0)).unwrap();

        set_rng_seed(999);
        let mut b = Matrix::<f32>::new(5, 5, 3);
        randu(&mut b, Scalar::all(0.0), Scalar::all(255.0)).unwrap();

        assert_eq!(a.data, b.data);
    }

    // ---- transform / perspective_transform tests ----

    #[test]
    fn test_transform_identity() {
        // 3-channel input, 3×3 identity matrix → output equals input.
        let src = Matrix::from_vec(1, 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m = Matrix::from_vec(3, 3, 1, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let mut dst = Matrix::<f64>::new(0, 0, 0);
        transform(&src, &mut dst, &m).unwrap();
        assert_eq!(dst.data, src.data);
    }

    #[test]
    fn test_transform_swap_channels() {
        // Swap R and B: matrix [[0,0,1],[0,1,0],[1,0,0]]
        let src = Matrix::from_vec(1, 1, 3, vec![10.0, 20.0, 30.0]);
        let m = Matrix::from_vec(3, 3, 1, vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        let mut dst = Matrix::<f64>::new(0, 0, 0);
        transform(&src, &mut dst, &m).unwrap();
        assert_eq!(dst.data, vec![30.0, 20.0, 10.0]);
    }

    #[test]
    fn test_transform_affine() {
        // 2-channel input, 2×3 affine matrix (identity + translation (5, 10))
        let src = Matrix::from_vec(1, 2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m = Matrix::from_vec(2, 3, 1, vec![1.0, 0.0, 5.0, 0.0, 1.0, 10.0]);
        let mut dst = Matrix::<f64>::new(0, 0, 0);
        transform(&src, &mut dst, &m).unwrap();
        assert_eq!(dst.data, vec![6.0, 12.0, 8.0, 14.0]);
    }

    #[test]
    fn test_transform_reduce_channels() {
        // Convert 3-channel to 1-channel grayscale using a 1×3 matrix.
        let src = Matrix::from_vec(1, 1, 3, vec![100.0, 150.0, 200.0]);
        let m = Matrix::from_vec(1, 3, 1, vec![0.299, 0.587, 0.114]);
        let mut dst = Matrix::<f64>::new(0, 0, 0);
        transform(&src, &mut dst, &m).unwrap();
        assert_eq!(dst.channels, 1);
        let expected = 0.299 * 100.0 + 0.587 * 150.0 + 0.114 * 200.0;
        assert!((dst.data[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_perspective_transform_identity() {
        // 2-channel input, 3×3 identity → output equals input.
        let src = Matrix::from_vec(1, 2, 2, vec![10.0, 20.0, 30.0, 40.0]);
        let m = Matrix::from_vec(3, 3, 1, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let mut dst = Matrix::<f64>::new(0, 0, 0);
        perspective_transform(&src, &mut dst, &m).unwrap();
        assert_eq!(dst.channels, 2);
        assert!((dst.data[0] - 10.0).abs() < 1e-10);
        assert!((dst.data[1] - 20.0).abs() < 1e-10);
        assert!((dst.data[2] - 30.0).abs() < 1e-10);
        assert!((dst.data[3] - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_perspective_transform_translation() {
        // Translation by (5, 10) via perspective matrix.
        #[rustfmt::skip]
        let m = Matrix::from_vec(3, 3, 1, vec![
            1.0, 0.0, 5.0,
            0.0, 1.0, 10.0,
            0.0, 0.0, 1.0,
        ]);
        let src = Matrix::from_vec(1, 1, 2, vec![3.0, 7.0]);
        let mut dst = Matrix::<f64>::new(0, 0, 0);
        perspective_transform(&src, &mut dst, &m).unwrap();
        assert!((dst.data[0] - 8.0).abs() < 1e-10);
        assert!((dst.data[1] - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_perspective_transform_scaling() {
        // A projective matrix that scales by 0.5 via the w component.
        #[rustfmt::skip]
        let m = Matrix::from_vec(3, 3, 1, vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 2.0,
        ]);
        let src = Matrix::from_vec(1, 1, 2, vec![10.0, 20.0]);
        let mut dst = Matrix::<f64>::new(0, 0, 0);
        perspective_transform(&src, &mut dst, &m).unwrap();
        // w = 2, so result = (10/2, 20/2) = (5, 10)
        assert!((dst.data[0] - 5.0).abs() < 1e-10);
        assert!((dst.data[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_perspective_transform_3d() {
        // 3-channel (3D) input with 4×4 identity.
        #[rustfmt::skip]
        let m = Matrix::from_vec(4, 4, 1, vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]);
        let src = Matrix::from_vec(1, 1, 3, vec![1.0, 2.0, 3.0]);
        let mut dst = Matrix::<f64>::new(0, 0, 0);
        perspective_transform(&src, &mut dst, &m).unwrap();
        assert_eq!(dst.channels, 3);
        assert!((dst.data[0] - 1.0).abs() < 1e-10);
        assert!((dst.data[1] - 2.0).abs() < 1e-10);
        assert!((dst.data[2] - 3.0).abs() < 1e-10);
    }

    // ---- solvePoly tests ----

    #[test]
    fn test_solve_poly_cubic() {
        use crate::core::arithm::solve_poly;

        // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        // coeffs = [-6, 11, -6, 1]
        let coeffs = Matrix::from_vec(1, 4, 1, vec![-6.0, 11.0, -6.0, 1.0]);
        let mut roots = Matrix::<f64>::new(0, 0, 0);
        let residual = solve_poly(&coeffs, &mut roots, 0).unwrap();
        assert!(residual < 1e-6);
        assert_eq!(roots.rows, 3);

        let mut reals: Vec<f64> = (0..3).map(|k| roots.data[k * 2]).collect();
        reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((reals[0] - 1.0).abs() < 1e-6);
        assert!((reals[1] - 2.0).abs() < 1e-6);
        assert!((reals[2] - 3.0).abs() < 1e-6);
    }

    // ---- sort / sortIdx tests ----

    #[test]
    fn test_sort_integration() {
        use crate::core::arithm::{sort, sort_idx};

        let src = Matrix::from_vec(2, 3, 1, vec![9i32, 3, 6, 1, 7, 4]);
        let mut dst = Matrix::<i32>::new(0, 0, 0);
        sort(&src, &mut dst, 0).unwrap(); // rows ascending
        assert_eq!(dst.data, vec![3, 6, 9, 1, 4, 7]);

        // sortIdx
        let mut idx = Matrix::<i32>::new(0, 0, 0);
        sort_idx(&src, &mut idx, 0).unwrap();
        // Row 0: [9,3,6] → sorted indices [1,2,0]
        assert_eq!(&idx.data[0..3], &[1, 2, 0]);
    }

    // ---- kmeans tests ----

    #[test]
    fn test_kmeans_multidim() {
        use crate::core::arithm::kmeans;
        use crate::core::types::{TermCriteria, TermType, KMEANS_PP_CENTERS};

        // 6 points in 2D: cluster A near (0,0), cluster B near (10,10)
        let mut data = Matrix::<f32>::new(6, 2, 1);
        data.data = vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 10.0, 10.0, 10.1, 10.1, 10.2, 10.2,
        ];
        let mut labels = Matrix::<i32>::new(0, 0, 0);
        let criteria = TermCriteria::new(TermType::Both, 100, 1e-6);
        let mut centers = Some(Matrix::<f32>::new(0, 0, 0));

        let comp = kmeans(
            &data,
            2,
            &mut labels,
            criteria,
            3,
            KMEANS_PP_CENTERS,
            &mut centers,
        )
        .unwrap();

        assert!(comp < 1.0);
        // First 3 should share a label, last 3 another
        let la = labels.data[0];
        let lb = labels.data[3];
        assert_ne!(la, lb);
        for i in 0..3 {
            assert_eq!(labels.data[i], la);
        }
        for i in 3..6 {
            assert_eq!(labels.data[i], lb);
        }
    }

    // ---- Matrix accessor / mutator tests ----

    #[test]
    fn test_get_and_get_mut() {
        let mut mat = Matrix::<u8>::from_vec(
            2,
            3,
            2,
            vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        );

        // get: valid indices
        assert_eq!(mat.get(0, 0, 0), Some(&10u8));
        assert_eq!(mat.get(0, 2, 1), Some(&60u8));
        assert_eq!(mat.get(1, 1, 0), Some(&90u8));

        // get: out-of-bounds returns None
        assert_eq!(mat.get(2, 0, 0), None);
        assert_eq!(mat.get(0, 3, 0), None);
        assert_eq!(mat.get(0, 0, 2), None);

        // get_mut: modify a value and verify
        if let Some(v) = mat.get_mut(1, 2, 1) {
            *v = 255;
        }
        assert_eq!(mat.get(1, 2, 1), Some(&255u8));

        // get_mut: out-of-bounds returns None
        assert_eq!(mat.get_mut(5, 0, 0), None);
    }

    #[test]
    fn test_at_and_at_mut() {
        let mut mat = Matrix::<u8>::from_vec(2, 2, 1, vec![1, 2, 3, 4]);

        // at: positive valid indices
        assert_eq!(mat.at(0, 0, 0), Some(&1u8));
        assert_eq!(mat.at(1, 1, 0), Some(&4u8));

        // at: negative indices must return None
        assert_eq!(mat.at(-1, 0, 0), None);
        assert_eq!(mat.at(0, -1, 0), None);
        assert_eq!(mat.at(-1, -1, 0), None);

        // at: out-of-bounds positive
        assert_eq!(mat.at(2, 0, 0), None);

        // at_mut: negative indices must return None
        assert_eq!(mat.at_mut(-1, 0, 0), None);
        assert_eq!(mat.at_mut(0, -5, 0), None);

        // at_mut: modify a value
        if let Some(v) = mat.at_mut(0, 1, 0) {
            *v = 99;
        }
        assert_eq!(mat.at(0, 1, 0), Some(&99u8));
    }

    #[test]
    fn test_set() {
        let mut mat = Matrix::<f32>::zeros(3, 3, 1);

        // set a value and verify with get
        mat.set(1, 2, 0, crate::core::constants::CV_PI as f32);
        assert_eq!(
            mat.get(1, 2, 0),
            Some(&(crate::core::constants::CV_PI as f32))
        );

        // set at out-of-bounds: must be a silent no-op, not a panic
        mat.set(99, 99, 0, 1.0);
        // if we get here without panic the test passes
    }

    #[test]
    fn test_as_slice_and_as_mut_slice() {
        let mut mat = Matrix::<u8>::from_vec(2, 2, 1, vec![1, 2, 3, 4]);

        // as_slice: correct length and values
        let s = mat.as_slice();
        assert_eq!(s.len(), 4);
        assert_eq!(s, &[1u8, 2, 3, 4]);

        // as_mut_slice: modify through slice, verify via get
        let ms = mat.as_mut_slice();
        ms[0] = 42;
        assert_eq!(mat.get(0, 0, 0), Some(&42u8));
    }

    // ---- Constructor variants ----

    #[test]
    fn test_zeros_from_size_and_ones_from_size() {
        use crate::core::Size;

        let sz = Size::new(4usize, 3usize); // width=4, height=3
        let z = Matrix::<f32>::zeros_from_size(sz, 1);
        assert_eq!(z.rows, 3);
        assert_eq!(z.cols, 4);
        assert_eq!(z.channels, 1);
        assert_eq!(z.data.len(), 12);
        assert!(z.data.iter().all(|&v| v == 0.0));

        let sz2 = Size::new(2usize, 2usize);
        let o = Matrix::<u8>::ones_from_size(sz2, 3);
        assert_eq!(o.rows, 2);
        assert_eq!(o.cols, 2);
        assert_eq!(o.channels, 3);
        assert_eq!(o.data.len(), 12);
        assert!(o.data.iter().all(|&v| v == 1));
    }

    // ---- with_type error cases ----

    #[test]
    fn test_zeros_with_type_error() {
        // depth mismatch: u8 matrix with f32 MatType
        let err = Matrix::<u8>::zeros_with_type(4, 4, CV_32FC1);
        assert!(matches!(
            err,
            Err(crate::core::error::PureCvError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_ones_with_type_error() {
        // depth mismatch: f32 matrix with u8 MatType
        let err = Matrix::<f32>::ones_with_type(4, 4, CV_8UC1);
        assert!(matches!(
            err,
            Err(crate::core::error::PureCvError::InvalidInput(_))
        ));
    }

    // ---- dims_match ----

    #[test]
    fn test_dims_match() {
        let a = Matrix::<u8>::new(4, 4, 3);
        let b = Matrix::<u8>::new(4, 4, 3);
        let c = Matrix::<u8>::new(4, 4, 1);
        let d = Matrix::<f32>::new(2, 4, 3); // different type, same dims as a

        assert!(a.dims_match(&b));

        // different channels
        assert!(!a.dims_match(&c));

        // different rows — Matrix<f32> vs Matrix<u8>, dims_match is generic over U
        assert!(!a.dims_match(&d));

        // single-element matrix matches itself
        let e = Matrix::<i32>::new(1, 1, 1);
        assert!(e.dims_match(&e.clone()));
    }

    // ---- create no-op branch ----

    #[test]
    fn test_create_noop() {
        let mut mat = Matrix::<u8>::from_vec(2, 3, 1, vec![1, 2, 3, 4, 5, 6]);

        // Calling create with the same dims must not reallocate (data preserved)
        mat.create(2, 3, 1);
        assert_eq!(mat.rows, 2);
        assert_eq!(mat.cols, 3);
        assert_eq!(mat.channels, 1);
        assert_eq!(mat.data, vec![1u8, 2, 3, 4, 5, 6], "data must be unchanged");

        // Calling create with different dims resets to default
        mat.create(10, 20, 3);
        assert_eq!(mat.rows, 10);
        assert_eq!(mat.cols, 20);
        assert_eq!(mat.channels, 3);
        assert_eq!(mat.data.len(), 10 * 20 * 3);

        let mut a = Matrix::<u8>::zeros(2, 2, 1);
        let mut b = Matrix::<u8>::ones(2, 2, 1);
        a.swap(&mut b);
        assert_eq!(a.data, vec![1, 1, 1, 1]);
        assert_eq!(b.data, vec![0, 0, 0, 0]);
    }

    // ---- Matrix scalar constructor tests ----

    #[test]
    fn test_matrix_new_with_scalar_1ch() {
        let s = Scalar::new(42u8, 0u8, 0u8, 0u8);
        let m = Matrix::new_with_scalar(2, 3, 1, s);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.channels, 1);
        assert!(m.data.iter().all(|&v| v == 42));
    }

    #[test]
    fn test_matrix_new_with_scalar_3ch() {
        let s = Scalar::new(10u8, 20u8, 30u8, 0u8);
        let m = Matrix::new_with_scalar(1, 2, 3, s);
        // pixel (0,0): [10,20,30]  pixel (0,1): [10,20,30]
        assert_eq!(m.data, vec![10, 20, 30, 10, 20, 30]);
    }

    #[test]
    fn test_matrix_new_with_scalar_channels_beyond_4() {
        let s = Scalar::new(1u8, 2u8, 3u8, 4u8);
        let m = Matrix::new_with_scalar(1, 1, 6, s);
        // channels 4 and 5 default to 0
        assert_eq!(m.data, vec![1, 2, 3, 4, 0, 0]);
    }

    #[test]
    fn test_matrix_new_with_scalar_from_size() {
        use crate::core::Size;
        let s = Scalar::new(7u8, 8u8, 9u8, 0u8);
        let m = Matrix::new_with_scalar_from_size(Size::new(3usize, 2usize), 3, s);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.channels, 3);
        assert_eq!(m.data.len(), 2 * 3 * 3);
    }

    #[test]
    fn test_matrix_new_with_scalar_typed_from_size_ok() {
        use crate::core::matrix::{CV_32FC1, CV_8UC3};
        use crate::core::Size;

        let s = Scalar::new(128u8, 64u8, 32u8, 0u8);
        let m =
            Matrix::<u8>::new_with_scalar_typed_from_size(Size::new(4usize, 4usize), CV_8UC3, s)
                .unwrap();
        assert_eq!(m.rows, 4);
        assert_eq!(m.cols, 4);
        assert_eq!(m.channels, 3);
        // spot-check first pixel
        assert_eq!(m.data[0], 128);
        assert_eq!(m.data[1], 64);
        assert_eq!(m.data[2], 32);

        let sf = Scalar::new(1.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mf =
            Matrix::<f32>::new_with_scalar_typed_from_size(Size::new(2usize, 2usize), CV_32FC1, sf)
                .unwrap();
        assert_eq!(mf.channels, 1);
        assert!(mf.data.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_matrix_new_with_scalar_typed_from_size_depth_mismatch() {
        use crate::core::matrix::CV_32FC1;
        use crate::core::Size;

        // u8 matrix but CV_32FC1 (f32 depth) → error
        let s = Scalar::new(0u8, 0u8, 0u8, 0u8);
        let result =
            Matrix::<u8>::new_with_scalar_typed_from_size(Size::new(2usize, 2usize), CV_32FC1, s);
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_set_to() {
        let mut m = Matrix::<u8>::zeros(2, 2, 3);
        m.set_to(Scalar::new(10u8, 20u8, 30u8, 0u8));
        // every pixel should be [10, 20, 30]
        for i in (0..m.data.len()).step_by(3) {
            assert_eq!(m.data[i], 10);
            assert_eq!(m.data[i + 1], 20);
            assert_eq!(m.data[i + 2], 30);
        }
    }

    #[test]
    fn test_matrix_set_to_masked_ok() {
        let mut m = Matrix::<u8>::zeros(2, 2, 1);
        // mask: only (0,0) and (1,1) are non-zero
        let mask = Matrix::from_vec(2, 2, 1, vec![1u8, 0u8, 0u8, 1u8]);
        m.set_to_masked(Scalar::new(255u8, 0u8, 0u8, 0u8), &mask)
            .unwrap();
        assert_eq!(m.data, vec![255, 0, 0, 255]);
    }

    #[test]
    fn test_matrix_set_to_masked_size_mismatch() {
        let mut m = Matrix::<u8>::zeros(3, 3, 1);
        let mask = Matrix::from_vec(2, 2, 1, vec![1u8, 0u8, 0u8, 1u8]);
        assert!(m
            .set_to_masked(Scalar::new(255u8, 0u8, 0u8, 0u8), &mask)
            .is_err());
    }

    #[test]
    fn test_matrix_set_to_f32() {
        let mut m = Matrix::<f32>::zeros(1, 2, 2);
        m.set_to(Scalar::new(0.5f32, 1.5f32, 0.0f32, 0.0f32));
        assert_eq!(m.data, vec![0.5, 1.5, 0.5, 1.5]);
    }

    // -----------------------------------------------------------------------
    // Scalar arithmetic variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_scalar_arithmetic() {
        let m = Matrix::from_vec(2, 2, 1, vec![10u8, 20, 30, 40]);
        let s = Scalar::<f64>::all(5.0);

        let r = add_scalar(&m, s).unwrap();
        assert_eq!(r.data, vec![15u8, 25, 35, 45]);

        let r = subtract_scalar(&m, s).unwrap();
        assert_eq!(r.data, vec![5u8, 15, 25, 35]);

        let r = multiply_scalar(&m, s).unwrap();
        assert_eq!(r.data, vec![50u8, 100, 150, 200]);

        let r = divide_scalar(&m, s).unwrap();
        assert_eq!(r.data, vec![2u8, 4, 6, 8]);
    }

    #[test]
    fn test_abs_diff_scalar() {
        let m = Matrix::from_vec(1, 4, 1, vec![10u8, 20, 30, 40]);
        let s = Scalar::<u8>::all(25);
        let r = abs_diff_scalar(&m, s).unwrap();
        assert_eq!(r.data, vec![15u8, 5, 5, 15]);
    }

    #[test]
    fn test_bitwise_scalar() {
        let m = Matrix::from_vec(1, 4, 1, vec![0b1010u8, 0b1100, 0b1111, 0b0000]);
        let s = Scalar::<u8>::all(0b0101);

        let r = bitwise_and_scalar(&m, s).unwrap();
        assert_eq!(r.data, vec![0b0000u8, 0b0100, 0b0101, 0b0000]);

        let r = bitwise_or_scalar(&m, s).unwrap();
        assert_eq!(r.data, vec![0b1111u8, 0b1101, 0b1111, 0b0101]);

        let r = bitwise_xor_scalar(&m, s).unwrap();
        assert_eq!(r.data, vec![0b1111u8, 0b1001, 0b1010, 0b0101]);
    }

    // -----------------------------------------------------------------------
    // Min / Max
    // -----------------------------------------------------------------------

    #[test]
    fn test_min_max() {
        let m1 = Matrix::from_vec(2, 2, 1, vec![1i32, 5, 3, 7]);
        let m2 = Matrix::from_vec(2, 2, 1, vec![4i32, 2, 6, 0]);

        let r = min(&m1, &m2).unwrap();
        assert_eq!(r.data, vec![1i32, 2, 3, 0]);

        let r = max(&m1, &m2).unwrap();
        assert_eq!(r.data, vec![4i32, 5, 6, 7]);
    }

    #[test]
    fn test_min_max_scalar() {
        let m = Matrix::from_vec(1, 4, 1, vec![1i32, 5, 3, 7]);
        let s = Scalar::<i32>::all(4);

        let r = min_scalar(&m, s).unwrap();
        assert_eq!(r.data, vec![1i32, 4, 3, 4]);

        let r = max_scalar(&m, s).unwrap();
        assert_eq!(r.data, vec![4i32, 5, 4, 7]);
    }

    // -----------------------------------------------------------------------
    // Comparison operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_compare() {
        let m1 = Matrix::from_vec(1, 4, 1, vec![1i32, 2, 3, 4]);
        let m2 = Matrix::from_vec(1, 4, 1, vec![2i32, 2, 2, 2]);

        let r = compare(&m1, &m2, CmpTypes::Eq).unwrap();
        assert_eq!(r.data, vec![0u8, 255, 0, 0]);

        let r = compare(&m1, &m2, CmpTypes::Gt).unwrap();
        assert_eq!(r.data, vec![0u8, 0, 255, 255]);

        let r = compare(&m1, &m2, CmpTypes::Ge).unwrap();
        assert_eq!(r.data, vec![0u8, 255, 255, 255]);

        let r = compare(&m1, &m2, CmpTypes::Lt).unwrap();
        assert_eq!(r.data, vec![255u8, 0, 0, 0]);

        let r = compare(&m1, &m2, CmpTypes::Le).unwrap();
        assert_eq!(r.data, vec![255u8, 255, 0, 0]);

        let r = compare(&m1, &m2, CmpTypes::Ne).unwrap();
        assert_eq!(r.data, vec![255u8, 0, 255, 255]);
    }

    #[test]
    fn test_compare_scalar() {
        let m = Matrix::from_vec(1, 4, 1, vec![1i32, 2, 3, 4]);
        let s = Scalar::<i32>::all(2);

        let r = compare_scalar(&m, s, CmpTypes::Eq).unwrap();
        assert_eq!(r.data, vec![0u8, 255, 0, 0]);

        let r = compare_scalar(&m, s, CmpTypes::Gt).unwrap();
        assert_eq!(r.data, vec![0u8, 0, 255, 255]);
    }

    #[test]
    fn test_in_range() {
        let src = Matrix::from_vec(1, 4, 1, vec![10u8, 50, 100, 200]);
        let lower = Matrix::from_vec(1, 4, 1, vec![20u8, 20, 20, 20]);
        let upper = Matrix::from_vec(1, 4, 1, vec![150u8, 150, 150, 150]);
        let mut dst = Matrix::<u8>::zeros(1, 4, 1);

        in_range(&src, &lower, &upper, &mut dst).unwrap();
        assert_eq!(dst.data, vec![0u8, 255, 255, 0]);
    }

    #[test]
    fn test_in_range_scalar() {
        let src = Matrix::from_vec(1, 4, 1, vec![10u8, 50, 100, 200]);
        let mut dst = Matrix::<u8>::zeros(1, 4, 1);

        in_range_scalar(&src, &[20u8], &[150u8], &mut dst).unwrap();
        assert_eq!(dst.data, vec![0u8, 255, 255, 0]);
    }

    // -----------------------------------------------------------------------
    // Reduction operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduce_sum_avg() {
        let m = Matrix::from_vec(2, 3, 1, vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Reduce along rows (dim=0) → 1×3
        let r = reduce(&m, 0, ReduceTypes::Sum).unwrap();
        assert_eq!(r.rows, 1);
        assert_eq!(r.cols, 3);
        assert_eq!(r.data, vec![5.0, 7.0, 9.0]);

        // Reduce along cols (dim=1) → 2×1
        let r = reduce(&m, 1, ReduceTypes::Sum).unwrap();
        assert_eq!(r.rows, 2);
        assert_eq!(r.cols, 1);
        assert_eq!(r.data, vec![6.0, 15.0]);

        // Average along rows
        let r = reduce(&m, 0, ReduceTypes::Avg).unwrap();
        assert_eq!(r.data, vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_reduce_min_max() {
        let m = Matrix::from_vec(2, 3, 1, vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let r = reduce(&m, 0, ReduceTypes::Max).unwrap();
        assert_eq!(r.data, vec![4.0, 5.0, 6.0]);

        let r = reduce(&m, 0, ReduceTypes::Min).unwrap();
        assert_eq!(r.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_count_non_zero() {
        let m = Matrix::from_vec(1, 6, 1, vec![0i32, 1, 0, 3, 0, 5]);
        assert_eq!(count_non_zero(&m).unwrap(), 3);

        // All zeros
        let m_zeros = Matrix::from_vec(1, 4, 1, vec![0i32, 0, 0, 0]);
        assert_eq!(count_non_zero(&m_zeros).unwrap(), 0);

        // All non-zero
        let m_all = Matrix::from_vec(1, 3, 1, vec![1i32, 2, 3]);
        assert_eq!(count_non_zero(&m_all).unwrap(), 3);

        // Multi-channel should error
        let m_multi = Matrix::from_vec(1, 3, 2, vec![0i32, 1, 0, 3, 0, 5]);
        assert!(count_non_zero(&m_multi).is_err());
    }

    // -----------------------------------------------------------------------
    // LUT (Look-Up Table)
    // -----------------------------------------------------------------------

    #[test]
    fn test_lut_identity() {
        // Identity LUT: output should equal input
        let src = Matrix::from_vec(1, 5, 1, vec![0u8, 50, 100, 200, 255]);
        let table_data: Vec<u8> = (0..=255).collect();
        let table = Matrix::from_vec(1, 256, 1, table_data);
        let dst = lut(&src, &table).unwrap();
        assert_eq!(dst.data, vec![0u8, 50, 100, 200, 255]);
    }

    #[test]
    fn test_lut_invert() {
        // Invert LUT: 255 - x
        let src = Matrix::from_vec(1, 3, 1, vec![0u8, 127, 255]);
        let table_data: Vec<u8> = (0..=255).rev().collect();
        let table = Matrix::from_vec(1, 256, 1, table_data);
        let dst = lut(&src, &table).unwrap();
        assert_eq!(dst.data, vec![255u8, 128, 0]);
    }

    #[test]
    fn test_lut_type_conversion() {
        // u8 src -> f32 output via LUT
        let src = Matrix::from_vec(1, 3, 1, vec![0u8, 1, 2]);
        let table_data: Vec<f32> = (0..256).map(|x| x as f32 * 0.5).collect();
        let table = Matrix::from_vec(1, 256, 1, table_data);
        let dst = lut(&src, &table).unwrap();
        assert_eq!(dst.data, vec![0.0f32, 0.5, 1.0]);
    }

    #[test]
    fn test_lut_multichannel_broadcast() {
        // 3-channel src with 1-channel LUT (broadcast)
        let src = Matrix::from_vec(1, 2, 3, vec![0u8, 1, 2, 10, 20, 30]);
        let table_data: Vec<u8> = (0..=255).map(|x: u8| x.wrapping_mul(2)).collect();
        let table = Matrix::from_vec(1, 256, 1, table_data);
        let dst = lut(&src, &table).unwrap();
        assert_eq!(dst.data, vec![0u8, 2, 4, 20, 40, 60]);
    }

    #[test]
    fn test_lut_wrong_size_error() {
        let src = Matrix::from_vec(1, 3, 1, vec![0u8, 1, 2]);
        let table = Matrix::from_vec(1, 128, 1, vec![0u8; 128]);
        assert!(lut(&src, &table).is_err());
    }

    #[test]
    fn test_lut_channel_mismatch_error() {
        let src = Matrix::from_vec(1, 2, 3, vec![0u8; 6]);
        let table = Matrix::from_vec(1, 256, 2, vec![0u8; 512]);
        assert!(lut(&src, &table).is_err());
    }

    // -----------------------------------------------------------------------
    // Polar / Cartesian conversions
    // -----------------------------------------------------------------------

    #[test]
    fn test_magnitude() {
        let x = Matrix::from_vec(1, 3, 1, vec![3.0f64, 0.0, 1.0]);
        let y = Matrix::from_vec(1, 3, 1, vec![4.0f64, 5.0, 0.0]);
        let mut dst = Matrix::<f64>::new(1, 3, 1);

        magnitude(&x, &y, &mut dst).unwrap();
        assert!((dst.data[0] - 5.0).abs() < 1e-9);
        assert!((dst.data[1] - 5.0).abs() < 1e-9);
        assert!((dst.data[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_phase() {
        let x = Matrix::from_vec(1, 2, 1, vec![1.0f64, 0.0]);
        let y = Matrix::from_vec(1, 2, 1, vec![0.0f64, 1.0]);
        let mut angle = Matrix::<f64>::new(1, 2, 1);

        // Degrees
        phase(&x, &y, &mut angle, true).unwrap();
        assert!((angle.data[0] - 0.0).abs() < 1e-9);
        assert!((angle.data[1] - 90.0).abs() < 1e-9);

        // Radians
        phase(&x, &y, &mut angle, false).unwrap();
        assert!((angle.data[0] - 0.0).abs() < 1e-9);
        assert!((angle.data[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    }

    #[test]
    fn test_cart_to_polar() {
        let x = Matrix::from_vec(1, 2, 1, vec![3.0f64, 0.0]);
        let y = Matrix::from_vec(1, 2, 1, vec![4.0f64, 5.0]);
        let mut mag = Matrix::<f64>::new(1, 2, 1);
        let mut ang = Matrix::<f64>::new(1, 2, 1);

        cart_to_polar(&x, &y, &mut mag, &mut ang, true).unwrap();
        assert!((mag.data[0] - 5.0).abs() < 1e-9);
        assert!((mag.data[1] - 5.0).abs() < 1e-9);
        // atan2(4,3) ≈ 53.13°
        assert!((ang.data[0] - 53.13010235415598).abs() < 1e-6);
        assert!((ang.data[1] - 90.0).abs() < 1e-9);
    }

    #[test]
    fn test_polar_to_cart() {
        let mag = Matrix::from_vec(1, 2, 1, vec![5.0f64, 1.0]);
        let ang = Matrix::from_vec(1, 2, 1, vec![0.0f64, 90.0]);
        let mut x_out = Matrix::<f64>::new(1, 2, 1);
        let mut y_out = Matrix::<f64>::new(1, 2, 1);

        polar_to_cart(&mag, &ang, &mut x_out, &mut y_out, true).unwrap();
        assert!((x_out.data[0] - 5.0).abs() < 1e-9);
        assert!((y_out.data[0] - 0.0).abs() < 1e-9);
        assert!((x_out.data[1] - 0.0).abs() < 1e-9);
        assert!((y_out.data[1] - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Linear algebra
    // -----------------------------------------------------------------------

    #[test]
    fn test_determinant() {
        // 2×2: det([1,2;3,4]) = 1*4 - 2*3 = -2
        let m2 = Matrix::from_vec(2, 2, 1, vec![1.0f64, 2.0, 3.0, 4.0]);
        assert!((determinant(&m2) - (-2.0)).abs() < 1e-9);

        // 3×3: det([1,2,3;0,1,4;5,6,0]) = 1
        let m3 = Matrix::from_vec(
            3,
            3,
            1,
            vec![1.0f64, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0],
        );
        assert!((determinant(&m3) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_invert() {
        // inv([4,7;2,6]) = [0.6,-0.7;-0.2,0.4], det = 10
        let m = Matrix::from_vec(2, 2, 1, vec![4.0f64, 7.0, 2.0, 6.0]);
        let mut dst = Matrix::<f64>::new(2, 2, 1);

        let det = invert(&m, &mut dst, DecompTypes::DECOMP_LU).unwrap();
        assert!((det - 10.0).abs() < 1e-9);
        assert!((dst.data[0] - 0.6).abs() < 1e-9);
        assert!((dst.data[1] - (-0.7)).abs() < 1e-9);
        assert!((dst.data[2] - (-0.2)).abs() < 1e-9);
        assert!((dst.data[3] - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_solve() {
        // x + y = 2, x - y = 0 => x=1, y=1
        let a = Matrix::from_vec(2, 2, 1, vec![1.0f64, 1.0, 1.0, -1.0]);
        let b = Matrix::from_vec(2, 1, 1, vec![2.0f64, 0.0]);
        let mut x = Matrix::<f64>::new(2, 1, 1);

        assert!(solve(&a, &b, &mut x, DecompTypes::DECOMP_LU).unwrap());
        assert!((x.data[0] - 1.0).abs() < 1e-9);
        assert!((x.data[1] - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Channel operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_insert_channel() {
        use crate::core::structural::{extract_channel, insert_channel};

        // 1×2, 3 channels: pixel0=[10,20,30], pixel1=[40,50,60]
        let m = Matrix::from_vec(1, 2, 3, vec![10u8, 20, 30, 40, 50, 60]);

        let ch0 = extract_channel(&m, 0).unwrap();
        assert_eq!(ch0.channels, 1);
        assert_eq!(ch0.data, vec![10u8, 40]);

        let ch1 = extract_channel(&m, 1).unwrap();
        assert_eq!(ch1.data, vec![20u8, 50]);

        let ch2 = extract_channel(&m, 2).unwrap();
        assert_eq!(ch2.data, vec![30u8, 60]);

        // Out-of-bounds channel should error
        assert!(extract_channel(&m, 3).is_err());

        // Insert channel: place [99, 88] at channel 1 of a zero matrix
        let src_ch = Matrix::from_vec(1, 2, 1, vec![99u8, 88]);
        let mut dst = Matrix::<u8>::zeros(1, 2, 3);
        insert_channel(&src_ch, &mut dst, 1).unwrap();
        assert_eq!(dst.data, vec![0u8, 99, 0, 0, 88, 0]);
    }

    // -----------------------------------------------------------------------
    // DynamicMatrix
    // -----------------------------------------------------------------------

    #[test]
    fn test_dynamic_matrix() {
        // Constructor via MatType
        let dm = DynamicMatrix::new(2, 3, CV_8UC3).unwrap();
        assert_eq!(dm.rows(), 2);
        assert_eq!(dm.cols(), 3);
        assert_eq!(dm.channels(), 3);
        assert_eq!(dm.depth_name(), "u8");
        assert_eq!(dm.total(), 18);

        // zeros
        let dm = DynamicMatrix::zeros(2, 2, CV_8UC1).unwrap();
        assert_eq!(dm.data_u8(), Some(&[0u8, 0, 0, 0][..]));

        // ones
        let dm = DynamicMatrix::ones(2, 2, CV_32FC1).unwrap();
        assert_eq!(dm.data_f32(), Some(&[1.0f32, 1.0, 1.0, 1.0][..]));

        // new_u8 + at_f64
        let dm = DynamicMatrix::new_u8(1, 2, 1, vec![10, 20]).unwrap();
        assert_eq!(dm.at_f64(0, 0, 0), Some(10.0));
        assert_eq!(dm.at_f64(0, 1, 0), Some(20.0));

        // as_matrix_u8 / as_matrix_f32 type checks
        assert!(dm.as_matrix_u8().is_some());
        assert!(dm.as_matrix_f32().is_none());

        // new_f32 + as_matrix_f32
        let dm = DynamicMatrix::new_f32(1, 2, 1, vec![1.5, 2.5]).unwrap();
        let mat = dm.as_matrix_f32().unwrap();
        assert_eq!(mat.data, vec![1.5f32, 2.5]);

        // mat_type
        let dm = DynamicMatrix::new(1, 1, CV_8UC3).unwrap();
        let mt = dm.mat_type();
        assert_eq!(mt.channels(), 3);

        // convert_to
        let dm = DynamicMatrix::new_u8(1, 2, 1, vec![10, 20]).unwrap();
        let converted = dm.convert_to("f32").unwrap();
        assert_eq!(converted.depth_name(), "f32");
        assert_eq!(converted.data_f32(), Some(&[10.0f32, 20.0][..]));

        // Error: length mismatch
        assert!(DynamicMatrix::new_u8(1, 2, 1, vec![10]).is_err());
    }

    // ---------------------------------------------------------------
    // Mathematical constants (constants.rs)
    // ---------------------------------------------------------------

    #[test]
    fn test_cv_pi_value() {
        assert_eq!(CV_PI, std::f64::consts::PI);
    }

    #[test]
    fn test_cv_pi_2_value() {
        assert_eq!(CV_PI_2, std::f64::consts::FRAC_PI_2);
    }

    #[test]
    fn test_cv_2pi_value() {
        assert_eq!(CV_2PI, 2.0 * std::f64::consts::PI);
    }

    #[test]
    fn test_cv_pi_4_value() {
        assert_eq!(CV_PI_4, std::f64::consts::FRAC_PI_4);
    }

    #[test]
    fn test_cv_log2_value() {
        assert_eq!(CV_LOG2, std::f64::consts::LOG2_E);
    }

    #[test]
    fn test_cv_ln2_value() {
        assert_eq!(CV_LN2, std::f64::consts::LN_2);
    }

    #[test]
    fn test_constants_are_f64() {
        let _: f64 = CV_PI;
        let _: f64 = CV_PI_2;
        let _: f64 = CV_2PI;
        let _: f64 = CV_PI_4;
        let _: f64 = CV_LOG2;
        let _: f64 = CV_LN2;
    }

    #[test]
    fn test_constants_mathematical_relationships() {
        assert!((CV_PI / 2.0 - CV_PI_2).abs() < 1e-15);
        assert!((CV_2PI - 2.0 * CV_PI).abs() < 1e-15);
        assert!((CV_PI / 4.0 - CV_PI_4).abs() < 1e-15);
        assert!((CV_LOG2 * CV_LN2 - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_metrics() {
        use crate::core::metrics::*;

        let mut m1 = Matrix::<u8>::zeros(2, 2, 1);
        let mut m2 = Matrix::<u8>::zeros(2, 2, 1);

        // Identical matrices, PSNR should be infinite, but we cap log(0) basically, wait our PSNR returns 0.0 or inf?
        // Let's test with a difference.
        m1.data = vec![10, 20, 30, 40];
        m2.data = vec![10, 20, 30, 40]; // same

        // Let's make them differ by 10 per element
        m1.data = vec![10, 20, 30, 40];
        m2.data = vec![20, 30, 40, 50]; // mse = 100

        let p = psnr(&m1, &m2).unwrap();
        // 10 * log10(255^2 / 100) = 10 * log10(650.25) ≈ 10 * 2.81308 = 28.1308
        assert!((p - 28.1308).abs() < 1e-3);

        let v1 = Matrix::<f64>::from_vec(2, 1, 1, vec![1.0, 2.0]);
        let v2 = Matrix::<f64>::from_vec(2, 1, 1, vec![2.0, 3.0]);
        let icov = Matrix::<f64>::eye(2, 2, 1);
        let mahal = mahalanobis(&v1, &v2, &icov).unwrap();
        // Difference is [-1, -1]
        // dot product with I is [-1, -1], then dot with diff is 1 + 1 = 2
        // sqrt(2) ≈ 1.4142
        assert!((mahal - crate::core::constants::CV_SQRT2).abs() < 1e-3);
    }

    #[test]
    fn test_math_constants() {
        use crate::core::constants::*;
        assert!((CV_PI - std::f64::consts::PI).abs() < f64::EPSILON);
        assert!((CV_2PI - 2.0 * std::f64::consts::PI).abs() < f64::EPSILON);
        assert!((CV_PI_2 - std::f64::consts::FRAC_PI_2).abs() < f64::EPSILON);
        assert!((CV_PI_4 - std::f64::consts::FRAC_PI_4).abs() < f64::EPSILON);
        assert!((CV_LOG2 - std::f64::consts::LOG2_E).abs() < f64::EPSILON);
        assert!((CV_LN2 - std::f64::consts::LN_2).abs() < f64::EPSILON);
        assert!((CV_SQRT2 - std::f64::consts::SQRT_2).abs() < f64::EPSILON);
        assert!((CV_E - std::f64::consts::E).abs() < f64::EPSILON);
        assert!((CV_LN10 - std::f64::consts::LN_10).abs() < f64::EPSILON);
    }
}
