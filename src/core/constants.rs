/*
 *  constants.rs
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

//! Mathematical constants mirroring OpenCV's C++ `CV_PI`, `CV_2PI`, etc.
//!
//! All values are `pub const f64` backed by [`std::f64::consts`] for
//! maximum precision and cross-platform reproducibility.

/// Pi (same as `std::f64::consts::PI`).
pub const CV_PI: f64 = std::f64::consts::PI;

/// Pi divided by 2 (same as `std::f64::consts::FRAC_PI_2`).
pub const CV_PI_2: f64 = std::f64::consts::FRAC_PI_2;

/// 2 * Pi — full circle in radians.
pub const CV_2PI: f64 = 2.0 * std::f64::consts::PI;

/// Pi divided by 4 (same as `std::f64::consts::FRAC_PI_4`).
pub const CV_PI_4: f64 = std::f64::consts::FRAC_PI_4;

/// Log base 2 of e (same as `std::f64::consts::LOG2_E`).
pub const CV_LOG2: f64 = std::f64::consts::LOG2_E;

/// Natural logarithm of 2 (same as `std::f64::consts::LN_2`).
pub const CV_LN2: f64 = std::f64::consts::LN_2;

/// Square root of 2 (same as `std::f64::consts::SQRT_2`).
pub const CV_SQRT2: f64 = std::f64::consts::SQRT_2;
