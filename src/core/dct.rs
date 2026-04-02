/*
 *  dct.rs
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

use crate::core::error::{PureCvError, Result};
use crate::core::matrix::Matrix;
use std::f64::consts::PI;

/// Discrete Cosine Transform.
/// Currently implemented using a straightforward algorithm.
pub fn dct<T>(src: &Matrix<T>) -> Result<Matrix<f64>>
where
    T: Copy + Into<f64>,
{
    if src.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "DCT only supports 1-channel images".into(),
        ));
    }

    let rows = src.rows;
    let cols = src.cols;

    if rows == 0 || cols == 0 {
        return Err(PureCvError::InvalidDimensions(
            "Input must not be empty".into(),
        ));
    }

    let n = rows * cols;
    let mut dst = vec![0.0; n];

    // Basic 1D or 2D unoptimized naive transform over flattened entries for now,
    // or standard 2D. We will implement 1D on flattened array if cols == 1 or rows == 1,
    // otherwise 2D by doing 1D on rows then 1D on cols.

    // Simplification for standard OpenCV 1D DCT behavior for vectors
    if rows == 1 || cols == 1 {
        let length = if rows == 1 { cols } else { rows };
        for k in 0..length {
            let mut sum_val = 0.0;
            for n_idx in 0..length {
                sum_val += src.data[n_idx].into()
                    * (PI / (length as f64) * (n_idx as f64 + 0.5) * k as f64).cos();
            }
            let alpha = if k == 0 {
                1.0 / (length as f64).sqrt()
            } else {
                (2.0 / length as f64).sqrt()
            };
            dst[k] = alpha * sum_val;
        }
        return Ok(Matrix {
            rows,
            cols,
            channels: 1,
            data: dst,
        });
    }

    // 2D naive implementation
    for u in 0..rows {
        for v in 0..cols {
            let mut sum_val = 0.0;
            for x in 0..rows {
                for y in 0..cols {
                    sum_val += src.data[x * cols + y].into()
                        * (PI / (rows as f64) * (x as f64 + 0.5) * u as f64).cos()
                        * (PI / (cols as f64) * (y as f64 + 0.5) * v as f64).cos();
                }
            }
            let au = if u == 0 {
                1.0 / (rows as f64).sqrt()
            } else {
                (2.0 / rows as f64).sqrt()
            };
            let av = if v == 0 {
                1.0 / (cols as f64).sqrt()
            } else {
                (2.0 / cols as f64).sqrt()
            };

            dst[u * cols + v] = au * av * sum_val;
        }
    }

    Ok(Matrix {
        rows,
        cols,
        channels: 1,
        data: dst,
    })
}

/// Inverse Discrete Cosine Transform.
/// Currently implemented using a straightforward algorithm.
pub fn idct<T>(src: &Matrix<T>) -> Result<Matrix<f64>>
where
    T: Copy + Into<f64>,
{
    if src.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "IDCT only supports 1-channel images".into(),
        ));
    }

    let rows = src.rows;
    let cols = src.cols;

    if rows == 0 || cols == 0 {
        return Err(PureCvError::InvalidDimensions(
            "Input must not be empty".into(),
        ));
    }

    let n = rows * cols;
    let mut dst = vec![0.0; n];

    if rows == 1 || cols == 1 {
        let length = if rows == 1 { cols } else { rows };
        for n_idx in 0..length {
            let mut sum_val = 0.0;
            for k in 0..length {
                let alpha = if k == 0 {
                    1.0 / (length as f64).sqrt()
                } else {
                    (2.0 / length as f64).sqrt()
                };
                sum_val += alpha
                    * src.data[k].into()
                    * (PI / (length as f64) * (n_idx as f64 + 0.5) * k as f64).cos();
            }
            dst[n_idx] = sum_val;
        }
        return Ok(Matrix {
            rows,
            cols,
            channels: 1,
            data: dst,
        });
    }

    // 2D naive implementation
    for x in 0..rows {
        for y in 0..cols {
            let mut sum_val = 0.0;
            for u in 0..rows {
                for v in 0..cols {
                    let au = if u == 0 {
                        1.0 / (rows as f64).sqrt()
                    } else {
                        (2.0 / rows as f64).sqrt()
                    };
                    let av = if v == 0 {
                        1.0 / (cols as f64).sqrt()
                    } else {
                        (2.0 / cols as f64).sqrt()
                    };

                    sum_val += au
                        * av
                        * src.data[u * cols + v].into()
                        * (PI / (rows as f64) * (x as f64 + 0.5) * u as f64).cos()
                        * (PI / (cols as f64) * (y as f64 + 0.5) * v as f64).cos();
                }
            }
            dst[x * cols + y] = sum_val;
        }
    }

    Ok(Matrix {
        rows,
        cols,
        channels: 1,
        data: dst,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_1d() {
        let src = Matrix::from_vec(1, 4, 1, vec![10.0, 20.0, 30.0, 40.0]);
        let d = dct(&src).unwrap();
        let inv = idct(&d).unwrap();
        for i in 0..src.data.len() {
            assert!((inv.data[i] - src.data[i]).abs() < 1e-5);
        }
    }
}
