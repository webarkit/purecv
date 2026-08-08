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

use crate::core::types::Point2f;
use crate::features2d::KeyPoint;

#[test]
fn test_keypoint_default() {
    let kp = KeyPoint::default();
    assert_eq!(kp.pt.x, 0.0);
    assert_eq!(kp.pt.y, 0.0);
    assert_eq!(kp.size, 0.0);
    assert_eq!(kp.angle, -1.0);
    assert_eq!(kp.response, 0.0);
    assert_eq!(kp.octave, 0);
    assert_eq!(kp.class_id, -1);
}

#[test]
fn test_keypoint_new() {
    let pt = Point2f::new(10.5, 20.7);
    let kp = KeyPoint::new(pt, 12.0, 45.0, 1.5, 2, 7);
    assert_eq!(kp.pt.x, 10.5);
    assert_eq!(kp.pt.y, 20.7);
    assert_eq!(kp.size, 12.0);
    assert_eq!(kp.angle, 45.0);
    assert_eq!(kp.response, 1.5);
    assert_eq!(kp.octave, 2);
    assert_eq!(kp.class_id, 7);
}

#[test]
fn test_keypoint_convert_to_points() {
    let kps = vec![
        KeyPoint::new(Point2f::new(1.0, 2.0), 10.0, 0.0, 1.0, 0, -1),
        KeyPoint::new(Point2f::new(3.0, 4.0), 10.0, 0.0, 2.0, 0, -1),
    ];
    let pts = KeyPoint::convert_to_points(&kps);
    assert_eq!(pts.len(), 2);
    assert_eq!(pts[0].x, 1.0);
    assert_eq!(pts[0].y, 2.0);
    assert_eq!(pts[1].x, 3.0);
    assert_eq!(pts[1].y, 4.0);
}

#[test]
fn test_keypoint_convert_to_points_masked() {
    let kps = vec![
        KeyPoint::new(Point2f::new(1.0, 2.0), 10.0, 0.0, 1.0, 0, -1),
        KeyPoint::new(Point2f::new(3.0, 4.0), 10.0, 0.0, 2.0, 0, -1),
        KeyPoint::new(Point2f::new(5.0, 6.0), 10.0, 0.0, 3.0, 0, -1),
    ];

    // Valid indices
    let pts = KeyPoint::convert_to_points_masked(&kps, &[2, 0]).unwrap();
    assert_eq!(pts.len(), 2);
    assert_eq!(pts[0].x, 5.0);
    assert_eq!(pts[0].y, 6.0);
    assert_eq!(pts[1].x, 1.0);
    assert_eq!(pts[1].y, 2.0);

    // Negative index error
    let err_neg = KeyPoint::convert_to_points_masked(&kps, &[-1]);
    assert!(err_neg.is_err());

    // Out of bounds error
    let err_out = KeyPoint::convert_to_points_masked(&kps, &[3]);
    assert!(err_out.is_err());
}

#[test]
fn test_keypoint_convert_from_points() {
    let pts = vec![Point2f::new(1.0, 2.0), Point2f::new(3.0, 4.0)];
    let kps = KeyPoint::convert_from_points(&pts, 8.0, 0.5, 1, 3);
    assert_eq!(kps.len(), 2);
    for (i, kp) in kps.iter().enumerate() {
        assert_eq!(kp.pt, pts[i]);
        assert_eq!(kp.size, 8.0);
        assert_eq!(kp.angle, -1.0); // OpenCV default conversion angle is always -1.0
        assert_eq!(kp.response, 0.5);
        assert_eq!(kp.octave, 1);
        assert_eq!(kp.class_id, 3);
    }
}

#[test]
fn test_keypoint_sort_by_response() {
    let mut kps = vec![
        KeyPoint::new(Point2f::default(), 10.0, 0.0, 1.5, 0, -1),
        KeyPoint::new(Point2f::default(), 10.0, 0.0, 0.5, 0, -1),
        KeyPoint::new(Point2f::default(), 10.0, 0.0, 3.0, 0, -1),
    ];

    // Descending sort (default for selecting best)
    KeyPoint::sort_by_response(&mut kps, true);
    assert_eq!(kps[0].response, 3.0);
    assert_eq!(kps[1].response, 1.5);
    assert_eq!(kps[2].response, 0.5);

    // Ascending sort
    KeyPoint::sort_by_response(&mut kps, false);
    assert_eq!(kps[0].response, 0.5);
    assert_eq!(kps[1].response, 1.5);
    assert_eq!(kps[2].response, 3.0);
}

#[test]
fn test_keypoint_retain_best() {
    let mut kps = vec![
        KeyPoint::new(Point2f::default(), 10.0, 0.0, 1.5, 0, -1),
        KeyPoint::new(Point2f::default(), 10.0, 0.0, 0.5, 0, -1),
        KeyPoint::new(Point2f::default(), 10.0, 0.0, 3.0, 0, -1),
    ];

    KeyPoint::retain_best(&mut kps, 2);
    assert_eq!(kps.len(), 2);
    assert_eq!(kps[0].response, 3.0);
    assert_eq!(kps[1].response, 1.5);
}

#[test]
fn test_keypoint_overlap() {
    // 1. Circles too far apart
    let kp1 = KeyPoint::new(Point2f::new(0.0, 0.0), 10.0, 0.0, 0.0, 0, -1);
    let kp2 = KeyPoint::new(Point2f::new(20.0, 0.0), 10.0, 0.0, 0.0, 0, -1);
    assert_eq!(KeyPoint::overlap(&kp1, &kp2), 0.0);

    // 2. One circle completely enclosed inside another
    let kp3 = KeyPoint::new(Point2f::new(0.0, 0.0), 10.0, 0.0, 0.0, 0, -1); // size 10 -> radius 5
    let kp4 = KeyPoint::new(Point2f::new(0.0, 0.0), 20.0, 0.0, 0.0, 0, -1); // size 20 -> radius 10
                                                                            // area_1 = pi * 25, area_2 = pi * 100
                                                                            // enclosed overlap = 25 / 100 = 0.25
    let enclosed_overlap = KeyPoint::overlap(&kp3, &kp4);
    assert!((enclosed_overlap - 0.25).abs() < 1e-5);

    // 3. Partial overlap (distance 4.0, radii 5.0 and 5.0)
    let kp5 = KeyPoint::new(Point2f::new(0.0, 0.0), 10.0, 0.0, 0.0, 0, -1);
    let kp6 = KeyPoint::new(Point2f::new(4.0, 0.0), 10.0, 0.0, 0.0, 0, -1);
    let overlap_val = KeyPoint::overlap(&kp5, &kp6);
    // Calculated theoretical overlap ratio is approx 0.337463
    assert!((overlap_val - 0.337463).abs() < 1e-4);
}

#[test]
fn test_fast_grayscale_validation() {
    use crate::core::Matrix;
    use crate::features2d::FastFeatureDetector;

    // Create a 3-channel matrix (RGB)
    let img = Matrix::<u8>::new(10, 10, 3);
    let detector = FastFeatureDetector::default();
    let res = detector.detect(&img);
    assert!(res.is_err());
    if let Err(e) = res {
        assert!(e.to_string().contains("grayscale"));
    }
}

#[test]
fn test_fast_uniform_image() {
    use crate::core::Matrix;
    use crate::features2d::{FastFeatureDetector, FastType};

    let img = Matrix::<u8>::from_vec(20, 20, 1, vec![128; 400]);

    for fast_type in &[FastType::Type5_8, FastType::Type7_12, FastType::Type9_16] {
        let detector = FastFeatureDetector::new(10, true, *fast_type);
        let kps = detector.detect(&img).unwrap();
        assert_eq!(kps.len(), 0);
    }
}

#[test]
fn test_fast_synthetic_corner() {
    use crate::core::Matrix;
    use crate::features2d::{FastFeatureDetector, FastType};

    // Construct an 11x11 grayscale image with a dark region in the top-left and bright region in the bottom-right
    let rows = 11;
    let cols = 11;
    let mut data = vec![100u8; rows * cols];
    for y in 5..rows {
        for x in 5..cols {
            data[y * cols + x] = 200;
        }
    }
    let img = Matrix::<u8>::from_vec(rows, cols, 1, data.clone());

    // Check FAST-9 (Type9_16)
    let detector_9 = FastFeatureDetector::new(10, false, FastType::Type9_16);
    let kps_9 = detector_9.detect(&img).unwrap();

    println!("Detected Keypoints count: {}", kps_9.len());
    for kp in &kps_9 {
        println!(
            "Keypoint at: ({}, {}), response: {}",
            kp.pt.x, kp.pt.y, kp.response
        );
    }

    assert!(
        !kps_9.is_empty(),
        "FAST-9 should detect at least one keypoint at the corner boundary"
    );

    // Check if the corner is found around the boundary region (5, 5)
    let has_near_boundary = kps_9
        .iter()
        .any(|kp| (kp.pt.x - 5.0).abs() <= 1.0 && (kp.pt.y - 5.0).abs() <= 1.0);
    assert!(has_near_boundary, "FAST-9 keypoint should be near (5, 5)");

    // Check FAST-7 (Type7_12)
    let detector_7 = FastFeatureDetector::new(10, false, FastType::Type7_12);
    let kps_7 = detector_7.detect(&img).unwrap();
    assert!(
        !kps_7.is_empty(),
        "FAST-7 should detect at least one keypoint"
    );

    // Check FAST-5 (Type5_8)
    let detector_5 = FastFeatureDetector::new(10, false, FastType::Type5_8);
    let kps_5 = detector_5.detect(&img).unwrap();
    assert!(
        !kps_5.is_empty(),
        "FAST-5 should detect at least one keypoint"
    );
}

#[test]
fn test_fast_nonmax_suppression() {
    use crate::core::Matrix;
    use crate::features2d::{FastFeatureDetector, FastType};

    let rows = 11;
    let cols = 11;
    let mut data = vec![100u8; rows * cols];
    for y in 5..rows {
        for x in 5..cols {
            data[y * cols + x] = 200;
        }
    }
    let img = Matrix::<u8>::from_vec(rows, cols, 1, data);

    // With NMS
    let detector_nms = FastFeatureDetector::new(10, true, FastType::Type9_16);
    let kps_nms = detector_nms.detect(&img).unwrap();

    // Without NMS
    let detector_no_nms = FastFeatureDetector::new(10, false, FastType::Type9_16);
    let kps_no_nms = detector_no_nms.detect(&img).unwrap();

    assert!(
        kps_no_nms.len() >= kps_nms.len(),
        "Without NMS, we should detect at least as many (and usually more) keypoints than with NMS"
    );
}

#[test]
fn test_orb_pyramid_grayscale_validation() {
    use crate::core::Matrix;
    use crate::features2d::build_orb_pyramid;

    // Create a 3-channel matrix (RGB)
    let img = Matrix::<u8>::new(10, 10, 3);
    let res = build_orb_pyramid(&img, 3, 1.2);
    assert!(res.is_err());
    if let Err(e) = res {
        assert!(e.to_string().contains("grayscale"));
    }
}

// miri: ORB scale-pyramid construction takes ~30s under interpretation.
// No `unsafe` on this path. See .agents/MIRI_PLAN.md §4.
#[cfg_attr(miri, ignore)]
#[test]
fn test_orb_pyramid_dimensions() {
    use crate::core::Matrix;
    use crate::features2d::build_orb_pyramid;

    // Create a 120x120 single-channel matrix
    let img = Matrix::<u8>::new(120, 120, 1);
    let pyr = build_orb_pyramid(&img, 4, 1.2).unwrap();

    assert_eq!(pyr.len(), 4);

    // Level 0: 120x120
    assert_eq!(pyr[0].cols, 120);
    assert_eq!(pyr[0].rows, 120);

    // Level 1: round(120 / 1.2) = 100x100
    assert_eq!(pyr[1].cols, 100);
    assert_eq!(pyr[1].rows, 100);

    // Level 2: round(120 / 1.44) = 83x83
    assert_eq!(pyr[2].cols, 83);
    assert_eq!(pyr[2].rows, 83);

    // Level 3: round(120 / 1.728) = 69x69
    assert_eq!(pyr[3].cols, 69);
    assert_eq!(pyr[3].rows, 69);
}

#[test]
fn test_orb_pyramid_errors() {
    use crate::core::Matrix;
    use crate::features2d::build_orb_pyramid;

    let img = Matrix::<u8>::new(10, 10, 1);

    // Invalid nlevels = 0
    assert!(build_orb_pyramid(&img, 0, 1.2).is_err());

    // Invalid scale_factor <= 1.0
    assert!(build_orb_pyramid(&img, 3, 0.9).is_err());
    assert!(build_orb_pyramid(&img, 3, 1.0).is_err());

    // Image dimension shunk to 0
    assert!(build_orb_pyramid(&img, 20, 1.2).is_err());
}

#[test]
fn test_orb_orientation() {
    use crate::core::Matrix;
    use crate::features2d::{compute_orientation, precompute_umax};

    // Precompute umax for patch size 31 (half size = 15)
    let u_max = precompute_umax(15);
    assert_eq!(u_max.len(), 17);
    assert_eq!(u_max[0], 15); // Center maximum width is 15

    // Create a 40x40 grayscale image filled with 0 (dark)
    let mut img = Matrix::<u8>::new(40, 40, 1);

    // Set a vertical edge: left half (columns < 20) is 0, right half is 255 (bright)
    for y in 0..40 {
        for x in 20..40 {
            img.set(y, x, 0, 255);
        }
    }

    // Keypoint at (20, 20) should have a centroid pointing straight east (0 degrees)
    let angle_east = compute_orientation(&img, 20, 20, 31, &u_max).unwrap();
    // Angle should be very close to 0.0 degrees
    assert!(angle_east.abs() < 1e-3 || (angle_east - 360.0).abs() < 1e-3);

    // Create another image for a horizontal edge: bottom half (rows >= 20) is 255 (bright)
    let mut img2 = Matrix::<u8>::new(40, 40, 1);
    for y in 20..40 {
        for x in 0..40 {
            img2.set(y, x, 0, 255);
        }
    }

    // Keypoint at (20, 20). Since screen Y increases downwards:
    // Bottom half is bright -> centroid is on positive Y axis (downwards in screen coordinates)
    // Positive Y axis corresponds to 90 degrees.
    let angle_south = compute_orientation(&img2, 20, 20, 31, &u_max).unwrap();
    assert!((angle_south - 90.0).abs() < 1.0);
}

#[test]
fn test_orb_descriptors() {
    use crate::core::types::Point2f;
    use crate::core::Matrix;
    use crate::features2d::{compute_orb_descriptor, KeyPoint, BIT_PATTERN_31};

    // Create a simple grayscale image with a gradient
    let mut img = Matrix::<u8>::new(40, 40, 1);
    for y in 0..40 {
        for x in 0..40 {
            img.set(y, x, 0, (x * 5) as u8);
        }
    }

    // Keypoint at (20, 20) with angle = 0.0 degrees
    let mut kp = KeyPoint::new(Point2f::new(20.0, 20.0), 31.0, 0.0, 0.0, 0, -1);
    let desc0 = compute_orb_descriptor(&img, &kp, 31, &BIT_PATTERN_31).unwrap();
    assert_eq!(desc0.len(), 32);

    // Keypoint at same location with angle = 90.0 degrees
    kp.angle = 90.0;
    let desc90 = compute_orb_descriptor(&img, &kp, 31, &BIT_PATTERN_31).unwrap();

    // Rotating the keypoint should steer the pattern, resulting in different descriptor values
    assert_ne!(desc0, desc90);
}

// miri: full ORB detect+describe pipeline takes ~806s under interpretation.
// No `unsafe` on this path. See .agents/MIRI_PLAN.md §4.
#[cfg_attr(miri, ignore)]
#[test]
fn test_orb_full_pipeline() {
    use crate::core::Matrix;
    use crate::features2d::Orb;

    // Create a 100x100 synthetic image with a clear corner to detect
    let mut img = Matrix::<u8>::new(100, 100, 1);
    // Draw a bright square corner
    for y in 40..60 {
        for x in 40..60 {
            img.set(y, x, 0, 255);
        }
    }

    let orb = Orb::default();
    let (kpts, descriptors) = orb.detect_and_compute(&img).unwrap();

    // The pipeline should run and find at least the corners of our square
    assert!(!kpts.is_empty());
    assert_eq!(descriptors.rows, kpts.len());
    assert_eq!(descriptors.cols, 32);

    // All keypoints should have a valid octave and angle
    for kp in kpts.iter() {
        assert!(kp.angle >= 0.0 && kp.angle < 360.0);
        assert!(kp.octave >= 0 && kp.octave < 8);
    }
}

#[test]
fn test_bfmatcher_validation() {
    use crate::core::Matrix;
    use crate::features2d::{BFMatcher, DescriptorMatcher, NormType};

    let matcher = BFMatcher::<f32>::new(NormType::NormHamming, false).unwrap();
    let query = Matrix::<f32>::new(1, 4, 1);
    let train = Matrix::<f32>::new(1, 4, 1);

    // Hamming is invalid for f32, should return InvalidInput error
    let res = matcher.match_descriptors(&query, &train);
    assert!(res.is_err());
}

#[test]
fn test_bfmatcher_hamming_match() {
    use crate::core::Matrix;
    use crate::features2d::{BFMatcher, DescriptorMatcher, NormType};

    let matcher = BFMatcher::<u8>::new(NormType::NormHamming, false).unwrap();

    // Query: 2 descriptors of size 4
    // [0b0000_1111, 0, 0, 0] -> hamming popcount differences
    let query_data = vec![15, 0, 0, 0, 240, 0, 0, 0];
    let query = Matrix::from_vec(2, 4, 1, query_data);

    // Train: 3 descriptors
    let train_data = vec![
        0, 0, 0, 0, // Index 0: dist to Q0=4, to Q1=4
        15, 0, 0, 0, // Index 1: dist to Q0=0, to Q1=8
        240, 0, 0, 0, // Index 2: dist to Q0=8, to Q1=0
    ];
    let train = Matrix::from_vec(3, 4, 1, train_data);

    let matches = matcher.match_descriptors(&query, &train).unwrap();
    assert_eq!(matches.len(), 2);

    assert_eq!(matches[0].query_idx, 0);
    assert_eq!(matches[0].train_idx, 1);
    assert_eq!(matches[0].distance, 0.0);

    assert_eq!(matches[1].query_idx, 1);
    assert_eq!(matches[1].train_idx, 2);
    assert_eq!(matches[1].distance, 0.0);
}

#[test]
fn test_bfmatcher_l2_match() {
    use crate::core::Matrix;
    use crate::features2d::{BFMatcher, DescriptorMatcher, NormType};

    let matcher = BFMatcher::<f32>::new(NormType::NormL2, false).unwrap();

    let query = Matrix::from_vec(2, 3, 1, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    let train = Matrix::from_vec(2, 3, 1, vec![1.0, 2.0, 2.5, 0.1, 0.1, 0.0]);

    let matches = matcher.match_descriptors(&query, &train).unwrap();
    assert_eq!(matches.len(), 2);

    // Q0 closest to T1 (dist = sqrt(0.1^2 + 0.1^2) = 0.1414)
    assert_eq!(matches[0].query_idx, 0);
    assert_eq!(matches[0].train_idx, 1);

    // Q1 closest to T0 (dist = sqrt(0^2 + 0^2 + 0.5^2) = 0.5)
    assert_eq!(matches[1].query_idx, 1);
    assert_eq!(matches[1].train_idx, 0);
    assert_eq!(matches[1].distance, 0.5);
}

#[test]
fn test_bfmatcher_knn() {
    use crate::core::Matrix;
    use crate::features2d::{BFMatcher, DescriptorMatcher, NormType};

    let matcher = BFMatcher::<u8>::new(NormType::NormHamming, false).unwrap();
    let query = Matrix::from_vec(1, 2, 1, vec![0, 0]);
    let train = Matrix::from_vec(
        3,
        2,
        1,
        vec![
            1, 0, // dist = 1
            1, 1, // dist = 2
            0, 0, // dist = 0
        ],
    );

    let knn = matcher.knn_match(&query, &train, 2).unwrap();
    assert_eq!(knn.len(), 1);
    let matches = &knn[0];
    assert_eq!(matches.len(), 2);

    // Nearest should be T2
    assert_eq!(matches[0].train_idx, 2);
    assert_eq!(matches[0].distance, 0.0);

    // Second nearest should be T0
    assert_eq!(matches[1].train_idx, 0);
    assert_eq!(matches[1].distance, 1.0);
}

#[test]
fn test_bfmatcher_cross_check() {
    use crate::core::Matrix;
    use crate::features2d::{BFMatcher, DescriptorMatcher, NormType};

    let matcher = BFMatcher::<u8>::new(NormType::NormHamming, true).unwrap();
    let query = Matrix::from_vec(2, 2, 1, vec![0, 0, 1, 1]);
    let train = Matrix::from_vec(
        2,
        2,
        1,
        vec![
            0, 0, // matches Q0
            0, 0, // also matches Q0
        ],
    );

    let matches = matcher.match_descriptors(&query, &train).unwrap();
    // Q0 matches T0. T0 matches Q0. (Mutual!)
    // Q1 matches T0. T0 matches Q0. (Not Mutual!)
    // Only 1 match should survive
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].query_idx, 0);
    assert_eq!(matches[0].train_idx, 0);
}

#[test]
fn test_filter_matches() {
    use crate::features2d::{filter_matches, DMatch};

    let knn = vec![
        vec![DMatch::new(0, 0, 0, 1.0), DMatch::new(0, 1, 0, 2.5)], // ratio 0.4 < 0.6 => should pass
        vec![DMatch::new(1, 0, 0, 2.0), DMatch::new(1, 1, 0, 2.2)], // ratio ~0.9 => should fail
    ];

    let filtered = filter_matches(&knn, 0.6);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].query_idx, 0);
}

#[test]
fn test_drawing_primitives() {
    use crate::core::types::Point2f;
    use crate::core::types::Scalar;
    use crate::core::Matrix;
    use crate::features2d::{draw_keypoints, draw_matches, DMatch, KeyPoint};

    let img = Matrix::<u8>::zeros(10, 10, 1);
    let kps = vec![KeyPoint::new(Point2f::new(5.0, 5.0), 4.0, 90.0, 1.0, 0, -1)];

    let drawn_kps = draw_keypoints(&img, &kps, Scalar::all(255)).unwrap();
    assert_eq!(drawn_kps.rows, 10);
    assert_eq!(drawn_kps.cols, 10);
    assert_eq!(drawn_kps.channels, 1);

    // Verify some pixels were set to 255
    let mut num_white_pixels = 0;
    for &val in drawn_kps.as_slice() {
        if val == 255 {
            num_white_pixels += 1;
        }
    }
    assert!(num_white_pixels > 0);

    let img2 = Matrix::<u8>::zeros(10, 10, 1);
    let kps2 = vec![KeyPoint::new(Point2f::new(3.0, 3.0), 4.0, -1.0, 1.0, 0, -1)];
    let matches = vec![DMatch::new(0, 0, 0, 0.0)];

    let drawn_matches = draw_matches(
        &img,
        &kps,
        &img2,
        &kps2,
        &matches,
        Some(Scalar::all(255)),
        Some(Scalar::all(128)),
    )
    .unwrap();
    assert_eq!(drawn_matches.rows, 10);
    assert_eq!(drawn_matches.cols, 20); // 10 + 10
}
