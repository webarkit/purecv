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
 *  WebARKitLib-rs is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with WebARKitLib-rs.  If not, see <http://www.gnu.org/licenses/>.
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

/// Template class for a 4-element vector.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Scalar<T> {
    pub values: [T; 4],
}

impl<T: Default + Copy> Scalar<T> {
    pub fn new(v0: T, v1: T, v2: T, v3: T) -> Self {
        Self { values: [v0, v1, v2, v3] }
    }

    pub fn all(v: T) -> Self {
        Self { values: [v, v, v, v] }
    }
}

/// Termination criteria for iterative algorithms.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TermCriteria {
    pub criteria_type: i32,
    pub max_count: i32,
    pub epsilon: f64,
}

impl TermCriteria {
    pub const COUNT: i32 = 1;
    pub const EPS: i32 = 2;

    pub fn new(criteria_type: i32, max_count: i32, epsilon: f64) -> Self {
        Self {
            criteria_type,
            max_count,
            epsilon,
        }
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
