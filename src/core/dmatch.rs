/*
 *  dmatch.rs
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

//! `DMatch`: one match between a query and a train descriptor, mirroring `cv::DMatch`.
//!
//! OpenCV defines it in the `core` module
//! ([`types.hpp`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L841)),
//! so purecv does too; `purecv::features2d::DMatch` remains available as a re-export.

/// Structure for storing match information.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DMatch {
    /// Index of the descriptor in the query set.
    pub query_idx: i32,
    /// Index of the descriptor in the train set.
    pub train_idx: i32,
    /// Index of the train image (relevant for multi-image matching).
    pub train_img_idx: i32,
    /// Distance between the two descriptors.
    pub distance: f32,
}

impl DMatch {
    pub fn new(query_idx: i32, train_idx: i32, train_img_idx: i32, distance: f32) -> Self {
        Self {
            query_idx,
            train_idx,
            train_img_idx,
            distance,
        }
    }
}
