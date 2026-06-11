/*
 *  matcher.rs
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

use crate::core::error::{PureCvError, Result};
use crate::core::Matrix;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Enum for matching distance metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    NormHamming,
    NormL2,
}

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

/// A modular trait for all descriptor matching algorithms.
pub trait DescriptorMatcher<T> {
    /// Find the single best match for each query descriptor.
    fn match_descriptors(
        &self,
        query_descriptors: &Matrix<T>,
        train_descriptors: &Matrix<T>,
    ) -> Result<Vec<DMatch>>;

    /// Find the k-best matches for each query descriptor.
    fn knn_match(
        &self,
        query_descriptors: &Matrix<T>,
        train_descriptors: &Matrix<T>,
        k: usize,
    ) -> Result<Vec<Vec<DMatch>>>;
}

/// Helper trait to calculate distance between two descriptor vectors.
pub trait DescriptorDistance: Sized {
    fn compute_distance(a: &[Self], b: &[Self], norm_type: NormType) -> Result<f32>;
}

impl DescriptorDistance for u8 {
    #[inline]
    fn compute_distance(a: &[Self], b: &[Self], norm_type: NormType) -> Result<f32> {
        if a.len() != b.len() {
            return Err(PureCvError::InvalidInput(format!(
                "Descriptor length mismatch: {} vs {}",
                a.len(),
                b.len()
            )));
        }
        match norm_type {
            NormType::NormHamming => {
                let dist = a
                    .iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| (x ^ y).count_ones())
                    .sum::<u32>();
                Ok(dist as f32)
            }
            NormType::NormL2 => {
                let sum: f32 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| {
                        let diff = (x as f32) - (y as f32);
                        diff * diff
                    })
                    .sum();
                Ok(sum.sqrt())
            }
        }
    }
}

impl DescriptorDistance for f32 {
    #[inline]
    fn compute_distance(a: &[Self], b: &[Self], norm_type: NormType) -> Result<f32> {
        if a.len() != b.len() {
            return Err(PureCvError::InvalidInput(format!(
                "Descriptor length mismatch: {} vs {}",
                a.len(),
                b.len()
            )));
        }
        match norm_type {
            NormType::NormHamming => Err(PureCvError::InvalidInput(
                "Hamming distance is not supported for floating point descriptors".to_string(),
            )),
            NormType::NormL2 => {
                let sum: f32 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| {
                        let diff = x - y;
                        diff * diff
                    })
                    .sum();
                Ok(sum.sqrt())
            }
        }
    }
}

/// Brute-force matcher for feature descriptors.
pub struct BFMatcher<T> {
    pub norm_type: NormType,
    pub cross_check: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DescriptorDistance + Send + Sync> BFMatcher<T> {
    /// Create a new brute-force matcher.
    pub fn new(norm_type: NormType, cross_check: bool) -> Result<Self> {
        Ok(Self {
            norm_type,
            cross_check,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Internal non-cross-checked match logic to avoid circular calling.
    fn match_internal(
        &self,
        query_descriptors: &Matrix<T>,
        train_descriptors: &Matrix<T>,
    ) -> Result<Vec<DMatch>> {
        if query_descriptors.rows == 0 || train_descriptors.rows == 0 {
            return Ok(Vec::new());
        }

        let q_cols = query_descriptors.cols * query_descriptors.channels;
        let t_cols = train_descriptors.cols * train_descriptors.channels;
        if q_cols != t_cols {
            return Err(PureCvError::InvalidInput(format!(
                "Query descriptor size ({}) does not match train descriptor size ({})",
                q_cols, t_cols
            )));
        }

        let norm_type = self.norm_type;

        let find_best_match = |i: usize| -> Result<DMatch> {
            let q_start = i * q_cols;
            let q_slice = &query_descriptors.data[q_start..(q_start + q_cols)];

            let mut min_dist = f32::MAX;
            let mut best_idx = -1;

            for j in 0..train_descriptors.rows {
                let t_start = j * t_cols;
                let t_slice = &train_descriptors.data[t_start..(t_start + t_cols)];

                let dist = T::compute_distance(q_slice, t_slice, norm_type)?;
                if dist < min_dist {
                    min_dist = dist;
                    best_idx = j as i32;
                }
            }

            Ok(DMatch::new(i as i32, best_idx, 0, min_dist))
        };

        #[cfg(feature = "parallel")]
        {
            (0..query_descriptors.rows)
                .into_par_iter()
                .map(find_best_match)
                .collect::<Result<Vec<DMatch>>>()
        }

        #[cfg(not(feature = "parallel"))]
        {
            (0..query_descriptors.rows)
                .map(find_best_match)
                .collect::<Result<Vec<DMatch>>>()
        }
    }
}

impl<T: DescriptorDistance + Send + Sync> DescriptorMatcher<T> for BFMatcher<T> {
    fn match_descriptors(
        &self,
        query_descriptors: &Matrix<T>,
        train_descriptors: &Matrix<T>,
    ) -> Result<Vec<DMatch>> {
        if !self.cross_check {
            return self.match_internal(query_descriptors, train_descriptors);
        }

        // Cross check matching
        let matches_q2t = self.match_internal(query_descriptors, train_descriptors)?;
        let matches_t2q = self.match_internal(train_descriptors, query_descriptors)?;

        let mut mutual_matches = Vec::new();
        for m_q2t in matches_q2t {
            if m_q2t.train_idx >= 0 && (m_q2t.train_idx as usize) < matches_t2q.len() {
                let m_t2q = matches_t2q[m_q2t.train_idx as usize];
                if m_t2q.train_idx == m_q2t.query_idx {
                    mutual_matches.push(m_q2t);
                }
            }
        }

        Ok(mutual_matches)
    }

    fn knn_match(
        &self,
        query_descriptors: &Matrix<T>,
        train_descriptors: &Matrix<T>,
        k: usize,
    ) -> Result<Vec<Vec<DMatch>>> {
        if self.cross_check && k > 1 {
            return Err(PureCvError::InvalidInput(
                "Cross-check is only supported for 1-to-1 matching (k=1)".to_string(),
            ));
        }

        if query_descriptors.rows == 0 || train_descriptors.rows == 0 {
            return Ok(vec![Vec::new(); query_descriptors.rows]);
        }

        let q_cols = query_descriptors.cols * query_descriptors.channels;
        let t_cols = train_descriptors.cols * train_descriptors.channels;
        if q_cols != t_cols {
            return Err(PureCvError::InvalidInput(format!(
                "Query descriptor size ({}) does not match train descriptor size ({})",
                q_cols, t_cols
            )));
        }

        let norm_type = self.norm_type;

        let find_k_best_matches = |i: usize| -> Result<Vec<DMatch>> {
            let q_start = i * q_cols;
            let q_slice = &query_descriptors.data[q_start..(q_start + q_cols)];

            let mut matches = Vec::with_capacity(train_descriptors.rows);

            for j in 0..train_descriptors.rows {
                let t_start = j * t_cols;
                let t_slice = &train_descriptors.data[t_start..(t_start + t_cols)];

                let dist = T::compute_distance(q_slice, t_slice, norm_type)?;
                matches.push(DMatch::new(i as i32, j as i32, 0, dist));
            }

            // Sort by distance ascending
            matches.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

            // Truncate to k
            matches.truncate(k);

            Ok(matches)
        };

        #[cfg(feature = "parallel")]
        let raw_knn = {
            (0..query_descriptors.rows)
                .into_par_iter()
                .map(find_k_best_matches)
                .collect::<Result<Vec<Vec<DMatch>>>>()?
        };
        #[cfg(not(feature = "parallel"))]
        let raw_knn = {
            (0..query_descriptors.rows)
                .map(find_k_best_matches)
                .collect::<Result<Vec<Vec<DMatch>>>>()?
        };

        if !self.cross_check {
            return Ok(raw_knn);
        }

        // Apply cross check for k=1 (where raw_knn elements have length <= 1)
        let matches_t2q = self.match_internal(train_descriptors, query_descriptors)?;

        let mut filtered_knn = vec![Vec::new(); query_descriptors.rows];
        for (i, m_list) in raw_knn.into_iter().enumerate() {
            if let Some(m_q2t) = m_list.first() {
                if m_q2t.train_idx >= 0 && (m_q2t.train_idx as usize) < matches_t2q.len() {
                    let m_t2q = matches_t2q[m_q2t.train_idx as usize];
                    if m_t2q.train_idx == i as i32 {
                        filtered_knn[i] = vec![*m_q2t];
                    }
                }
            }
        }

        Ok(filtered_knn)
    }
}

/// Standalone helper implementing Lowe's ratio test to filter matches.
pub fn filter_matches(matches: &[Vec<DMatch>], ratio: f32) -> Vec<DMatch> {
    matches
        .iter()
        .filter(|m| m.len() >= 2 && m[0].distance < ratio * m[1].distance)
        .map(|m| m[0])
        .collect()
}
