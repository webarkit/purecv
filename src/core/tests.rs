/*
 *  tests.rs (Internal test module)
 */

#[cfg(test)]
mod tests {
    use crate::core::types::*;
    use crate::core::utils::*;
    use crate::core::Matrix;
    use crate::core::arithm::*;

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
        let padded = copy_make_border(&m_pad, 1, 1, 1, 1, 0, Scalar::all(0)).unwrap();
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
        let l = log(&m3).unwrap();
        assert!((l.data[0] - 0.0).abs() < 1e-6);
        assert!((l.data[1] - 1.0).abs() < 1e-6);

        let m4 = Matrix::from_vec(1, 2, 1, vec![2.0f32, 3.0]);
        let p = pow(&m4, 2.0).unwrap();
        assert_eq!(p.data, vec![4.0, 9.0]);
    }
}
