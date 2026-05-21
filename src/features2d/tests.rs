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
