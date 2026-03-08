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
 *  Copyright 2026 WebARKit.
 *
 *  Author(s): Walter Perdan @kalwalt https://github.com/kalwalt
 *
 */

use crate::core::Matrix;
use crate::core::error::{Result, PureCvError};
use num_traits::{ToPrimitive, FromPrimitive, NumCast};

#[derive(Debug, Clone, Copy, PartialEq)]
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
    T: ToPrimitive + Copy,
{
    match norm_type {
        NormTypes::INF => {
            src.data.iter()
                .map(|&x| x.to_f64().unwrap_or(0.0).abs())
                .fold(0.0, f64::max)
        }
        NormTypes::L1 => {
            src.data.iter()
                .map(|&x| x.to_f64().unwrap_or(0.0).abs())
                .sum()
        }
        NormTypes::L2 => {
            let sum_sq: f64 = src.data.iter()
                .map(|&x| {
                    let v = x.to_f64().unwrap_or(0.0);
                    v * v
                })
                .sum();
            sum_sq.sqrt()
        }
        NormTypes::MINMAX => 0.0, // Should probably return an error or handle differently
    }
}

/// Normalizes the matrix to a specified range or norm.
/// 
/// * `src` - Input matrix.
/// * `alpha` - Lower bound or norm value.
/// * `beta` - Upper bound (used in MINMAX).
/// * `norm_type` - Type of normalization.
pub fn normalize<T>(
    src: &Matrix<T>,
    alpha: f64,
    beta: f64,
    norm_type: NormTypes,
) -> Result<Matrix<T>>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy,
{
    let mut dst = Matrix::new(src.rows, src.cols, src.channels);
    let data_len = src.data.len();

    match norm_type {
        NormTypes::MINMAX => {
            let mut min_val = f64::MAX;
            let mut max_val = f64::MIN;

            for &x in &src.data {
                let v = x.to_f64().ok_or(PureCvError::InvalidInput("Invalid data".into()))?;
                if v < min_val { min_val = v; }
                if v > max_val { max_val = v; }
            }

            let range = max_val - min_val;
            let target_range = beta - alpha;

            if range.abs() < f64::EPSILON {
                for i in 0..data_len {
                    dst.data[i] = T::from(alpha).unwrap_or_default();
                }
            } else {
                let scale = target_range / range;
                for i in 0..data_len {
                    let v = src.data[i].to_f64().unwrap();
                    let normalized = (v - min_val) * scale + alpha;
                    dst.data[i] = T::from(normalized).unwrap_or_default();
                }
            }
        }
        NormTypes::L1 | NormTypes::L2 | NormTypes::INF => {
            let n = norm(src, norm_type);
            if n.abs() < f64::EPSILON {
                for i in 0..data_len {
                    dst.data[i] = T::default();
                }
            } else {
                let scale = alpha / n;
                for i in 0..data_len {
                    let v = src.data[i].to_f64().unwrap();
                    dst.data[i] = T::from(v * scale).unwrap_or_default();
                }
            }
        }
    }

    Ok(dst)
}
