/*
 *  norm.rs
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

use crate::core::Matrix;
use crate::core::error::Result;
use crate::core::stats::min_max_loc;
use num_traits::{ToPrimitive, FromPrimitive};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(i32)]
pub enum NormTypes {
    INF = 1,
    L1 = 2,
    L2 = 4,
    MINMAX = 32,
}

/// Calculates the absolute norm of a matrix.
/// Currently supports INF, L1, and L2 norms.
pub fn norm<T>(src: &Matrix<T>, norm_type: NormTypes) -> f64
where
    T: ToPrimitive + Copy + Sync + Send + Default + Clone + PartialOrd,
{
    match norm_type {
        NormTypes::INF => {
            #[cfg(feature = "parallel")]
            {
                src.data.as_slice().par_iter()
                    .map(|&x| x.to_f64().unwrap_or(0.0).abs())
                    .reduce(|| 0.0, f64::max)
            }
            #[cfg(not(feature = "parallel"))]
            {
                src.data.iter()
                    .map(|&x| x.to_f64().unwrap_or(0.0).abs())
                    .fold(0.0, f64::max)
            }
        }
        NormTypes::L1 => {
            #[cfg(feature = "parallel")]
            {
                src.data.as_slice().par_iter()
                    .map(|&x| x.to_f64().unwrap_or(0.0).abs())
                    .sum()
            }
            #[cfg(not(feature = "parallel"))]
            {
                src.data.iter()
                    .map(|&x| x.to_f64().unwrap_or(0.0).abs())
                    .sum()
            }
        }
        NormTypes::L2 => {
            let sum_sq: f64;
            #[cfg(feature = "parallel")]
            {
                sum_sq = src.data.as_slice().par_iter()
                    .map(|&x| {
                        let v = x.to_f64().unwrap_or(0.0);
                        v * v
                    })
                    .sum();
            }
            #[cfg(not(feature = "parallel"))]
            {
                sum_sq = src.data.iter()
                    .map(|&x| {
                        let v = x.to_f64().unwrap_or(0.0);
                        v * v
                    })
                    .sum();
            }
            sum_sq.sqrt()
        }
        NormTypes::MINMAX => {
            // NormTypes::MINMAX is not valid for norm() in OpenCV, usually returns 0 or errors.
            0.0
        }
    }
}

/// Normalizes the matrix to a specified range or norm.
pub fn normalize<T>(
    src: &Matrix<T>,
    alpha: f64,
    beta: f64,
    norm_type: NormTypes,
) -> Result<Matrix<T>>
where
    T: ToPrimitive + FromPrimitive + Copy + Sync + Send + Default + Clone + PartialOrd,
{
    let mut dst = Matrix::new(src.rows, src.cols, src.channels);
    let channels = src.channels;

    match norm_type {
        NormTypes::MINMAX => {
            let (min_vals, max_vals, _, _) = min_max_loc(src);
            let global_min = min_vals.into_iter().fold(f64::MAX, f64::min);
            let global_max = max_vals.into_iter().fold(f64::MIN, f64::max);

            if global_max <= global_min {
                #[cfg(feature = "parallel")]
                dst.data.as_mut_slice().par_iter_mut().for_each(|x| *x = T::from_f64(alpha).unwrap_or(T::default()));
                #[cfg(not(feature = "parallel"))]
                for x in &mut dst.data { *x = T::from_f64(alpha).unwrap_or(T::default()); }
            } else {
                let scale = (beta - alpha) / (global_max - global_min);
                #[cfg(feature = "parallel")]
                {
                    dst.data.as_mut_slice().par_chunks_exact_mut(channels)
                        .zip(src.data.as_slice().par_chunks_exact(channels))
                        .for_each(|(dout, din)| {
                            for i in 0..channels {
                                let v = din[i].to_f64().unwrap_or(0.0);
                                let normalized = (v - global_min) * scale + alpha;
                                dout[i] = T::from_f64(normalized).unwrap_or(T::default());
                            }
                        });
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for (dout, din) in dst.data.chunks_exact_mut(channels).zip(src.data.chunks_exact(channels)) {
                        for i in 0..channels {
                            let v = din[i].to_f64().unwrap_or(0.0);
                            let normalized = (v - global_min) * scale + alpha;
                            dout[i] = T::from_f64(normalized).unwrap_or(T::default());
                        }
                    }
                }
            }
        }
        NormTypes::L1 | NormTypes::L2 | NormTypes::INF => {
            let n = norm(src, norm_type);
            if n.abs() < f64::EPSILON {
                #[cfg(feature = "parallel")]
                dst.data.as_mut_slice().par_iter_mut().for_each(|x| *x = T::default());
                #[cfg(not(feature = "parallel"))]
                for x in &mut dst.data { *x = T::default(); }
            } else {
                let scale = alpha / n;
                #[cfg(feature = "parallel")]
                {
                    dst.data.as_mut_slice().par_chunks_exact_mut(channels)
                        .zip(src.data.as_slice().par_chunks_exact(channels))
                        .for_each(|(dout, din)| {
                            for i in 0..channels {
                                let v = din[i].to_f64().unwrap_or(0.0);
                                dout[i] = T::from_f64(v * scale).unwrap_or(T::default());
                            }
                        });
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for (dout, din) in dst.data.chunks_exact_mut(channels).zip(src.data.chunks_exact(channels)) {
                        for i in 0..channels {
                            let v = din[i].to_f64().unwrap_or(0.0);
                            dout[i] = T::from_f64(v * scale).unwrap_or(T::default());
                        }
                    }
                }
            }
        }
    }

    Ok(dst)
}
