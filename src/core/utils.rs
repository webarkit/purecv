/*
 *  utils.rs
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

#[cfg(feature = "std")]
use std::sync::OnceLock;
#[cfg(feature = "std")]
use std::time::Instant;

#[cfg(feature = "std")]
static START_TIME: OnceLock<Instant> = OnceLock::new();

/// Returns the number of ticks.
/// In this implementation, it returns the number of nanoseconds since the first call.
///
/// Requires the `std` feature: bare-metal targets have no portable clock,
/// so embedded users should rely on their platform timer instead.
#[cfg(feature = "std")]
pub fn get_tick_count() -> i64 {
    let start = *START_TIME.get_or_init(Instant::now);
    start.elapsed().as_nanos() as i64
}

/// Returns the number of ticks per second.
///
/// Requires the `std` feature (see [`get_tick_count`]).
#[cfg(feature = "std")]
pub fn get_tick_frequency() -> f64 {
    1_000_000_000.0
}

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

// Log level functions have moved to `core::logging`.
// Re-exported here for backwards-compatible import paths.
pub use crate::core::logging::{get_log_level, set_log_level};

/// Border interpolation helper.
/// Returns the index of a pixel that would be at the given position `p`
/// in a sequence of length `len` if a specified border logic is applied.
///
/// `Reflect` and `Reflect101` map in constant time for any `p`, however far outside the
/// sequence. `Reflect`, `Reflect101` and `Wrap` return -1 for an empty sequence (`len <= 0`).
pub fn border_interpolate(p: i32, len: i32, border_type: crate::core::types::BorderTypes) -> i32 {
    if p >= 0 && p < len {
        return p;
    }

    use crate::core::types::BorderTypes;
    match border_type {
        BorderTypes::Replicate => {
            if p < 0 {
                0
            } else {
                len - 1
            }
        }
        // Both reflections repeat with a period, so the folded index takes one remainder
        // instead of one step per period. The arithmetic is i64: `2 * len` and `-p` overflow
        // an i32 for the extreme values.
        BorderTypes::Reflect | BorderTypes::Reflect101 if len <= 0 => -1,
        BorderTypes::Reflect | BorderTypes::Reflect101 if len == 1 => 0,
        BorderTypes::Reflect => {
            // fedcba|abcdefgh|hgfedcb: the period is `2 * len`.
            let len = i64::from(len);
            let m = i64::from(p).rem_euclid(2 * len);
            (if m < len { m } else { 2 * len - 1 - m }) as i32
        }
        BorderTypes::Reflect101 => {
            // gfedcb|abcdefgh|gfedcba: the period is `2 * len - 2`.
            let len = i64::from(len);
            let m = i64::from(p).rem_euclid(2 * len - 2);
            (if m < len { m } else { 2 * len - 2 - m }) as i32
        }
        BorderTypes::Wrap if len <= 0 => -1,
        BorderTypes::Wrap => {
            if p < 0 {
                (p % len + len) % len
            } else {
                p % len
            }
        }
        BorderTypes::Constant => -1,
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
    fn par_chunks_mut(&'a mut self, size: usize) -> core::slice::ChunksMut<'a, T>;
}

#[cfg(not(feature = "parallel"))]
impl<'a, T: 'a> ParIterFallback<'a, T> for [T] {
    type Iter = core::slice::Iter<'a, T>;
    type IterMut = core::slice::IterMut<'a, T>;

    fn par_iter(&'a self) -> Self::Iter {
        self.iter()
    }
    fn par_iter_mut(&'a mut self) -> Self::IterMut {
        self.iter_mut()
    }
    fn par_chunks_mut(&'a mut self, size: usize) -> core::slice::ChunksMut<'a, T> {
        self.chunks_mut(size)
    }
}

#[cfg(test)]
mod tests {
    use super::border_interpolate;
    use crate::core::types::BorderTypes;

    const ALL_BORDERS: [BorderTypes; 7] = [
        BorderTypes::Constant,
        BorderTypes::Replicate,
        BorderTypes::Reflect,
        BorderTypes::Wrap,
        BorderTypes::Reflect101,
        BorderTypes::Transparent,
        BorderTypes::Isolated,
    ];

    /// The step by step folding `border_interpolate` used before it mapped indices in constant
    /// time, kept verbatim (i32 arithmetic) as the reference.
    fn border_interpolate_loop(p: i32, len: i32, border_type: BorderTypes) -> i32 {
        if p >= 0 && p < len {
            return p;
        }
        match border_type {
            BorderTypes::Replicate => {
                if p < 0 {
                    0
                } else {
                    len - 1
                }
            }
            BorderTypes::Reflect => {
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
            BorderTypes::Reflect101 => {
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
            BorderTypes::Wrap => {
                if p < 0 {
                    (p % len + len) % len
                } else {
                    p % len
                }
            }
            _ => -1,
        }
    }

    /// The same folding in i64, so it cannot overflow for any i32 `p` and `len`.
    fn fold_i64(p: i32, len: i32, border_type: BorderTypes) -> i32 {
        let (len, mut p) = (i64::from(len), i64::from(p));
        let (neg, over) = match border_type {
            BorderTypes::Reflect => (1, 1),
            _ => (0, 2),
        };
        while !(0..len).contains(&p) {
            p = if p < 0 { -p - neg } else { 2 * len - p - over };
        }
        i32::try_from(p).unwrap()
    }

    #[test]
    fn border_interpolate_matches_the_stepwise_fold() {
        for border in ALL_BORDERS {
            for len in 1..=24 {
                for p in -500..=500 {
                    assert_eq!(
                        border_interpolate(p, len, border),
                        border_interpolate_loop(p, len, border),
                        "{border:?} p={p} len={len}"
                    );
                }
            }
        }
    }

    #[test]
    fn border_interpolate_zero_length_does_not_panic_or_hang() {
        // The stepwise fold never returned for Reflect and Reflect101 on an empty axis.
        for border in [BorderTypes::Reflect, BorderTypes::Reflect101] {
            for p in [i32::MIN, -7, -1, 0, 1, 7, i32::MAX] {
                assert_eq!(border_interpolate(p, 0, border), -1, "{border:?} p={p}");
            }
        }
        // Wrap used to divide by zero.
        for border in [BorderTypes::Wrap] {
            for p in [i32::MIN, -1, 0, 1, i32::MAX] {
                assert_eq!(border_interpolate(p, 0, border), -1, "{border:?} p={p}");
            }
        }
        // The other border types keep their behaviour.
        for border in [
            BorderTypes::Constant,
            BorderTypes::Replicate,
            BorderTypes::Transparent,
            BorderTypes::Isolated,
        ] {
            for p in [i32::MIN, -1, 0, 1, i32::MAX] {
                assert_eq!(
                    border_interpolate(p, 0, border),
                    border_interpolate_loop(p, 0, border)
                );
            }
        }
    }

    #[test]
    fn border_interpolate_extreme_indices_do_not_overflow() {
        let extremes = (0..40)
            .flat_map(|d| [i32::MIN + d, i32::MAX - d, -d, d])
            .collect::<Vec<_>>();
        for border in [BorderTypes::Reflect, BorderTypes::Reflect101] {
            for len in [1000, 65_536, 1 << 20, i32::MAX / 2 + 1, i32::MAX] {
                for &p in &extremes {
                    let got = border_interpolate(p, len, border);
                    assert!((0..len).contains(&got), "{border:?} p={p} len={len}: {got}");
                    assert_eq!(got, fold_i64(p, len, border), "{border:?} p={p} len={len}");
                }
            }
        }
    }
}
