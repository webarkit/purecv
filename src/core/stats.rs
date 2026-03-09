/*
 *  stats.rs
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

use crate::core::{Matrix, Point2i};
use num_traits::ToPrimitive;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Calculates the sum of all elements in the matrix per channel.
pub fn sum<T>(src: &Matrix<T>) -> Vec<f64>
where
    T: ToPrimitive + Copy + Sync + Send + Default + Clone,
{
    let channels = src.channels;
    let mut channel_sums = vec![0.0; channels];

    #[cfg(feature = "parallel")]
    {
        channel_sums = src.data.as_slice()
            .par_chunks_exact(channels)
            .fold(|| vec![0.0; channels], |mut acc, chunk| {
                for (i, &val) in chunk.iter().enumerate() {
                    acc[i] += val.to_f64().unwrap_or(0.0);
                }
                acc
            })
            .reduce(|| vec![0.0; channels], |mut acc1, acc2| {
                for (i, v) in acc2.iter().enumerate() {
                    acc1[i] += v;
                }
                acc1
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for chunk in src.data.chunks_exact(channels) {
            for (i, &val) in chunk.iter().enumerate() {
                channel_sums[i] += val.to_f64().unwrap_or(0.0);
            }
        }
    }

    channel_sums
}

/// Calculates the mean of all elements in the matrix per channel.
pub fn mean<T>(src: &Matrix<T>) -> Vec<f64>
where
    T: ToPrimitive + Copy + Sync + Send + Default + Clone,
{
    let sums = sum(src);
    let count = (src.rows * src.cols) as f64;
    if count == 0.0 {
        return sums;
    }
    sums.into_iter().map(|s| s / count).collect()
}

/// Finds the global minimum and maximum values and their locations.
pub fn min_max_loc<T>(src: &Matrix<T>) -> (Vec<f64>, Vec<f64>, Vec<Point2i>, Vec<Point2i>)
where
    T: ToPrimitive + Copy + PartialOrd + Sync + Send + Default + Clone,
{
    let channels = src.channels;
    let mut min_vals = vec![f64::MAX; channels];
    let mut max_vals = vec![f64::MIN; channels];
    let mut min_locs = vec![Point2i::new(0, 0); channels];
    let mut max_locs = vec![Point2i::new(0, 0); channels];

    for row in 0..src.rows {
        for col in 0..src.cols {
            for ch in 0..channels {
                let idx = src.flat_index(row, col, ch);
                let val_raw = src.data[idx];
                let val = val_raw.to_f64().unwrap_or(0.0);

                if val < min_vals[ch] {
                    min_vals[ch] = val;
                    min_locs[ch] = Point2i::new(col as i32, row as i32);
                }
                if val > max_vals[ch] {
                    max_vals[ch] = val;
                    max_locs[ch] = Point2i::new(col as i32, row as i32);
                }
            }
        }
    }

    (min_vals, max_vals, min_locs, max_locs)
}

/// Calculates the mean and standard deviation of matrix elements.
pub fn mean_std_dev<T>(src: &Matrix<T>) -> (Vec<f64>, Vec<f64>)
where
    T: ToPrimitive + Copy + Sync + Send + Default + Clone,
{
    let means = mean(src);
    let channels = src.channels;
    let count = (src.rows * src.cols) as f64;
    let mut std_devs = vec![0.0; channels];

    if count > 0.0 {
        let mut sq_diff_sums = vec![0.0; channels];

        #[cfg(feature = "parallel")]
        {
            sq_diff_sums = src.data.as_slice()
                .par_chunks_exact(channels)
                .fold(|| vec![0.0; channels], |mut acc, chunk| {
                    for (ch, &val) in chunk.iter().enumerate() {
                        let v = val.to_f64().unwrap_or(0.0);
                        let diff = v - means[ch];
                        acc[ch] += diff * diff;
                    }
                    acc
                })
                .reduce(|| vec![0.0; channels], |mut acc1, acc2| {
                    for (i, v) in acc2.iter().enumerate() {
                        acc1[i] += v;
                    }
                    acc1
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for chunk in src.data.chunks_exact(channels) {
                for (ch, &val) in chunk.iter().enumerate() {
                    let v = val.to_f64().unwrap_or(0.0);
                    let diff = v - means[ch];
                    sq_diff_sums[ch] += diff * diff;
                }
            }
        }

        for ch in 0..channels {
            std_devs[ch] = (sq_diff_sums[ch] / count).sqrt();
        }
    }

    (means, std_devs)
}
