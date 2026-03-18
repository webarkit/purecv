/*
 *  version.rs
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

//! # Version
//!
//! Provides the library version string sourced directly from `Cargo.toml` at
//! compile time, along with helper functions to retrieve and display it.

/// The current version of `purecv`, sourced from `Cargo.toml` at compile time.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the current version string of the library.
///
/// # Example
///
/// ```rust
/// use webarkitlib_rs::version::get_version;
/// let v = get_version();
/// assert!(!v.is_empty());
/// ```
pub fn get_version() -> &'static str {
    VERSION
}

/// Logs the library name and version using the [`log`] crate at `info` level.
///
/// Call this early in your application (e.g. during initialization) so the
/// version appears in the console / log output.
pub fn print_version() {
    log::info!("purecv v{}", VERSION);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_version_is_not_empty() {
        assert!(!get_version().is_empty());
    }

    #[test]
    fn test_version_constant_matches_getter() {
        assert_eq!(VERSION, get_version());
    }

    #[test]
    fn test_version_format() {
        // Version should be a valid semver-like string, e.g. "0.1.4" or "0.1.4-alpha.1"
        let v = get_version();

        // Only validate the numeric core (major.minor[.patch]) before any pre-release/build suffix.
        let core = v
            .split(|c| c == '-' || c == '+')
            .next()
            .unwrap_or(v);

        let parts: Vec<&str> = core.split('.').collect();
        assert!(
            parts.len() >= 2,
            "Version should have at least major.minor parts in the numeric core"
        );
        for part in &parts {
            assert!(
                part.parse::<u64>().is_ok(),
                "Each numeric core version part should be purely numeric"
            );
        }
    }
}
