/*
 *  utils.rs
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

/// Template class specifying a continuous subsequence (slice) of a sequence.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Range {
    pub start: i32,
    pub end: i32,
}

impl Range {
    pub fn new(start: i32, end: i32) -> Self {
        Self { start, end }
    }

    pub fn size(&self) -> i32 {
        self.end - self.start
    }

    pub fn empty(&self) -> bool {
        self.start == self.end
    }

    pub fn all() -> Self {
        Self::new(i32::MIN, i32::MAX)
    }
}

/// Sets the global log level.
/// This is a wrapper around the `log` crate's level filter.
pub fn set_log_level(level: log::LevelFilter) {
    log::set_max_level(level);
}

/// Returns the current global log level.
pub fn get_log_level() -> log::LevelFilter {
    log::max_level()
}

/// Border interpolation helper.
/// Returns the index of a pixel that would be at the given position `p`
/// in a sequence of length `len` if a specified border logic is applied.
pub fn border_interpolate(p: i32, len: i32, border_type: crate::core::types::BorderTypes) -> i32 {
    if p >= 0 && p < len {
        return p;
    }

    use crate::core::types::BorderTypes;
    match border_type {
        BorderTypes::REPLICATE => {
            if p < 0 {
                0
            } else {
                len - 1
            }
        }
        BorderTypes::REFLECT => {
            let mut p = p;
            if len == 1 {
                return 0;
            }
            loop {
                if p < 0 {
                    p = -p - 1;
                } else if p >= len {
                    p = 2 * len - p - 1;
                } else {
                    break;
                }
            }
            p
        }
        BorderTypes::REFLECT_101 => {
            let mut p = p;
            if len == 1 {
                return 0;
            }
            loop {
                if p < 0 {
                    p = -p;
                } else if p >= len {
                    p = 2 * len - p - 2;
                } else {
                    break;
                }
            }
            p
        }
        BorderTypes::WRAP => {
            if p < 0 {
                (p % len + len) % len
            } else {
                p % len
            }
        }
        BorderTypes::CONSTANT => -1,
        _ => -1,
    }
}

/// Fallback trait for seamless compilation when the "parallel" feature is disabled.
/// This allows `.par_iter()`, `.par_iter_mut()`, and `.par_chunks_mut()`
/// to resolve to their sequential standard library equivalents.
#[cfg(not(feature = "parallel"))]
pub trait ParIterFallback<'a, T: 'a> {
    type Iter: Iterator<Item = &'a T>;
    type IterMut: Iterator<Item = &'a mut T>;

    fn par_iter(&'a self) -> Self::Iter;
    fn par_iter_mut(&'a mut self) -> Self::IterMut;
    fn par_chunks_mut(&'a mut self, size: usize) -> std::slice::ChunksMut<'a, T>;
}

#[cfg(not(feature = "parallel"))]
impl<'a, T: 'a> ParIterFallback<'a, T> for [T] {
    type Iter = std::slice::Iter<'a, T>;
    type IterMut = std::slice::IterMut<'a, T>;

    fn par_iter(&'a self) -> Self::Iter {
        self.iter()
    }
    fn par_iter_mut(&'a mut self) -> Self::IterMut {
        self.iter_mut()
    }
    fn par_chunks_mut(&'a mut self, size: usize) -> std::slice::ChunksMut<'a, T> {
        self.chunks_mut(size)
    }
}
