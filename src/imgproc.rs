/*
 *  imgproc.rs
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

pub mod color;
pub mod derivatives;
pub mod edge;
pub mod filter;
pub mod threshold;

#[cfg(test)]
mod tests;

// Re-export common conversions: purecv::imgproc::cvt_color_...
pub use color::{
    cvt_color, cvt_color_bgr_to_gray, cvt_color_bgra_to_gray, cvt_color_gray_to_bgr,
    cvt_color_gray_to_bgra, cvt_color_gray_to_rgb, cvt_color_gray_to_rgba, cvt_color_rgb_to_gray,
    cvt_color_rgba_to_gray, ColorConversionCode,
};
pub use derivatives::*;
pub use edge::*;
pub use filter::*;
pub use threshold::*;
