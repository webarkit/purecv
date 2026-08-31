#![cfg(target_arch = "wasm32")]

use purecv_wasm::{
    calc_hist_uniform, compare_hist, equalize_hist, find_homography_wasm, hist_cmp_correl,
    rodrigues_wasm, solve_pnp_wasm, Clahe, Mat, MatVector, Point2fVector, Point3fVector,
};
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn test_rodrigues() {
    let src_data = vec![0.0, 0.0, 0.0];
    let mut src = Mat::from_f64_data(3, 1, 1, &src_data).unwrap();

    let dst_data = vec![0.0; 9];
    let mut dst = Mat::from_f64_data(3, 3, 1, &dst_data).unwrap();

    rodrigues_wasm(&src, &mut dst).unwrap();

    let dst_array = dst.data_f64().unwrap();
    assert_eq!(dst_array[0], 1.0);
    assert_eq!(dst_array[4], 1.0);
    assert_eq!(dst_array[8], 1.0);
}

fn rvec_to_rmat(rx: f64, ry: f64, rz: f64) -> [f64; 9] {
    let theta = (rx * rx + ry * ry + rz * rz).sqrt();
    if theta < 1e-8 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let c = theta.cos();
    let s = theta.sin();
    let c1 = 1.0 - c;
    let itheta = 1.0 / theta;
    let x = rx * itheta;
    let y = ry * itheta;
    let z = rz * itheta;
    [
        c + x * x * c1,
        x * y * c1 - z * s,
        x * z * c1 + y * s,
        y * x * c1 + z * s,
        c + y * y * c1,
        y * z * c1 - x * s,
        z * x * c1 - y * s,
        z * y * c1 + x * s,
        c + z * z * c1,
    ]
}

#[wasm_bindgen_test]
fn test_solve_pnp() {
    let rvec_true = [0.1, 0.05, 0.02];
    let tvec_true = [0.0, 0.0, 5.0];
    let k = [800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0];
    let r = rvec_to_rmat(rvec_true[0], rvec_true[1], rvec_true[2]);

    let raw_obj = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5],
        [-0.5, 0.3, 0.2],
    ];

    let mut obj_points = Point3fVector::new();
    let mut img_points = Point2fVector::new();

    for p in &raw_obj {
        obj_points.push(p[0] as f32, p[1] as f32, p[2] as f32);
        let cx = r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + tvec_true[0];
        let cy = r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + tvec_true[1];
        let cz = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + tvec_true[2];
        let u = k[0] * (cx / cz) + k[2];
        let v = k[4] * (cy / cz) + k[5];
        img_points.push(u as f32, v as f32);
    }

    let camera_matrix = Mat::from_f64_data(3, 3, 1, &k).unwrap();

    let mut rvec = Mat::from_f64_data(3, 1, 1, &vec![0.0; 3]).unwrap();
    let mut tvec = Mat::from_f64_data(3, 1, 1, &vec![0.0; 3]).unwrap();

    solve_pnp_wasm(
        &obj_points,
        &img_points,
        &camera_matrix,
        None,
        &mut rvec,
        &mut tvec,
        false,
        0, // Iterative
    )
    .expect("solve_pnp_wasm failed");

    let tvec_data = tvec.data_f64().unwrap();
    assert!((tvec_data[2] - 5.0).abs() < 1e-2);
}

#[wasm_bindgen_test]
fn test_find_homography() {
    let mut src_points = Point2fVector::new();
    src_points.push(0.0, 0.0);
    src_points.push(1.0, 0.0);
    src_points.push(0.0, 1.0);
    src_points.push(1.0, 1.0);

    let mut dst_points = Point2fVector::new();
    dst_points.push(10.0, 10.0);
    dst_points.push(20.0, 10.0);
    dst_points.push(10.0, 20.0);
    dst_points.push(20.0, 20.0);

    let mut h = Mat::from_f64_data(3, 3, 1, &vec![0.0; 9]).unwrap();

    find_homography_wasm(
        &src_points,
        &dst_points,
        8, // Ransac
        3.0,
    )
    .unwrap();
}

#[wasm_bindgen_test]
fn test_equalize_hist() {
    let data = vec![0, 50, 100, 150, 200, 255];
    let src = Mat::from_u8_data(2, 3, 1, &data).unwrap();
    let eq = equalize_hist(&src).unwrap();
    assert_eq!(eq.rows(), 2);
    assert_eq!(eq.cols(), 3);
}

#[wasm_bindgen_test]
fn test_clahe() {
    let data = vec![128u8; 64];
    let src = Mat::from_u8_data(8, 8, 1, &data).unwrap();
    let mut clahe = Clahe::new(40.0, 8, 8);
    assert_eq!(clahe.clip_limit(), 40.0);
    clahe.set_clip_limit(20.0);
    assert_eq!(clahe.clip_limit(), 20.0);
    let dst = clahe.apply(&src).unwrap();
    assert_eq!(dst.rows(), 8);
    assert_eq!(dst.cols(), 8);
}

#[wasm_bindgen_test]
fn test_calc_hist_and_compare() {
    let mut mv = MatVector::new();
    let data = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
    let img = Mat::from_u8_data(2, 4, 1, &data).unwrap();
    mv.push(&img);
    assert_eq!(mv.length(), 1);

    let hist_size = vec![4];
    let ranges = vec![0.0, 8.0];
    let channels = vec![0];
    let h1 = calc_hist_uniform(&mv, &channels, None, &hist_size, &ranges, false).unwrap();
    assert_eq!(h1.rows(), 4);
    assert_eq!(h1.cols(), 1);

    let score = compare_hist(&h1, &h1, hist_cmp_correl()).unwrap();
    assert!((score - 1.0).abs() < 1e-4);
}
