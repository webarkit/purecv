/*
 *  logging.rs
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

//! OpenCV-compatible structured logging API.
//!
//! This module mirrors [`cv::utils::logging`](https://docs.opencv.org/4.x/d5/d14/namespacecv_1_1utils_1_1logging.html)
//! and is built on top of the [`log`](https://crates.io/crates/log) facade so
//! the output backend stays the application's choice (e.g. `env_logger`, `fern`,
//! `tracing`, `console_log` on WASM).
//!
//! # Level Mapping
//!
//! | [`LogLevel`]   | [`log::Level`]     | Notes                          |
//! |----------------|--------------------|--------------------------------|
//! | `Silent`       | *(off)*            | Disables all logging           |
//! | `Fatal`        | `Error`            | Collapsed onto `Error`         |
//! | `Error`        | `Error`            |                                |
//! | `Warning`      | `Warn`             |                                |
//! | `Info`         | `Info`             |                                |
//! | `Debug`        | `Debug`            |                                |
//! | `Verbose`      | `Trace`            | Collapsed onto `Trace`         |
//!
//! # Compile-time stripping
//!
//! The `log` crate provides cargo features `max_level_*` and
//! `release_max_level_*` that strip log calls at compile time, equivalent to
//! OpenCV's `CV_LOG_STRIP_LEVEL`.
//!
//! # Example
//!
//! ```rust
//! use purecv::core::logging::{self, tags, LogLevel};
//!
//! let prev = logging::set_log_level(LogLevel::Info);
//! purecv::cv_log_info!(tags::IMGPROC, "gaussian blur, ksize = {}", 5);
//! ```

use core::fmt;

// ── LogLevel enum ──────────────────────────────────────────────────

/// Log severity levels mirroring OpenCV's `cv::utils::logging::LogLevel`.
///
/// The ordering matches OpenCV: `Silent` is the most restrictive (no output)
/// and `Verbose` is the most permissive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogLevel {
    /// Suppress all log output.
    Silent = 0,
    /// Fatal error — maps to [`log::Level::Error`].
    Fatal = 1,
    /// Error — maps to [`log::Level::Error`].
    Error = 2,
    /// Warning — maps to [`log::Level::Warn`].
    Warning = 3,
    /// Informational — maps to [`log::Level::Info`].
    Info = 4,
    /// Debug — maps to [`log::Level::Debug`].
    Debug = 5,
    /// Verbose / trace — maps to [`log::Level::Trace`].
    Verbose = 6,
}

impl PartialOrd for LogLevel {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LogLevel {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogLevel::Silent => write!(f, "SILENT"),
            LogLevel::Fatal => write!(f, "FATAL"),
            LogLevel::Error => write!(f, "ERROR"),
            LogLevel::Warning => write!(f, "WARNING"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Verbose => write!(f, "VERBOSE"),
        }
    }
}

impl From<LogLevel> for log::LevelFilter {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Silent => log::LevelFilter::Off,
            LogLevel::Fatal | LogLevel::Error => log::LevelFilter::Error,
            LogLevel::Warning => log::LevelFilter::Warn,
            LogLevel::Info => log::LevelFilter::Info,
            LogLevel::Debug => log::LevelFilter::Debug,
            LogLevel::Verbose => log::LevelFilter::Trace,
        }
    }
}

impl From<log::LevelFilter> for LogLevel {
    fn from(filter: log::LevelFilter) -> Self {
        match filter {
            log::LevelFilter::Off => LogLevel::Silent,
            log::LevelFilter::Error => LogLevel::Error,
            log::LevelFilter::Warn => LogLevel::Warning,
            log::LevelFilter::Info => LogLevel::Info,
            log::LevelFilter::Debug => LogLevel::Debug,
            log::LevelFilter::Trace => LogLevel::Verbose,
        }
    }
}

impl From<log::Level> for LogLevel {
    fn from(level: log::Level) -> Self {
        match level {
            log::Level::Error => LogLevel::Error,
            log::Level::Warn => LogLevel::Warning,
            log::Level::Info => LogLevel::Info,
            log::Level::Debug => LogLevel::Debug,
            log::Level::Trace => LogLevel::Verbose,
        }
    }
}

// ── set / get ──────────────────────────────────────────────────────

/// Sets the global log level and returns the previous level.
///
/// This is the Rust equivalent of OpenCV's `cv::utils::logging::setLogLevel`.
///
/// # Example
///
/// ```rust
/// use purecv::core::logging::{set_log_level, get_log_level, LogLevel};
///
/// let prev = set_log_level(LogLevel::Debug);
/// assert_eq!(get_log_level(), LogLevel::Debug);
/// set_log_level(prev); // restore
/// ```
pub fn set_log_level(level: LogLevel) -> LogLevel {
    let previous = get_log_level();
    log::set_max_level(log::LevelFilter::from(level));
    previous
}

/// Returns the current global log level.
///
/// This is the Rust equivalent of OpenCV's `cv::utils::logging::getLogLevel`.
pub fn get_log_level() -> LogLevel {
    LogLevel::from(log::max_level())
}

struct SimpleLogger;

impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            println!(
                "[{}] {} - {}",
                record.level(),
                record.target(),
                record.args()
            );
        }
    }

    fn flush(&self) {}
}

static LOGGER: SimpleLogger = SimpleLogger;

/// Initializes a simple stdout logger for purecv log messages.
///
/// Returns an error if a logger has already been set.
///
/// This is helpful for CLI tools or examples to quickly view logs without
/// pulling in external dependencies like `env_logger`.
pub fn init_basic_logger() -> Result<(), crate::core::PureCvError> {
    log::set_logger(&LOGGER)
        .map_err(|e| crate::core::PureCvError::InvalidInput(format!("Logger already set: {}", e)))?;
    if log::max_level() == log::LevelFilter::Off {
        log::set_max_level(log::LevelFilter::Info);
    }
    Ok(())
}

// ── Subsystem tags ─────────────────────────────────────────────────

/// Per-subsystem tag constants for use with `cv_log_*!` macros.
///
/// These map onto the `log` crate's `target:` field, enabling per-module
/// filtering via standard tooling, e.g.:
///
/// ```text
/// RUST_LOG=purecv::imgproc=debug,purecv::core=warn
/// ```
pub mod tags {
    /// Root tag for the purecv crate.
    pub const PURECV: &str = "purecv";
    /// Tag for `purecv::core` subsystem.
    pub const CORE: &str = "purecv::core";
    /// Tag for `purecv::imgproc` subsystem.
    pub const IMGPROC: &str = "purecv::imgproc";
    /// Tag for `purecv::features2d` subsystem.
    pub const FEATURES2D: &str = "purecv::features2d";
    /// Tag for `purecv::calib3d` subsystem.
    pub const CALIB3D: &str = "purecv::calib3d";
    /// Tag for `purecv::video` subsystem.
    pub const VIDEO: &str = "purecv::video";
}

// ── Level macros ───────────────────────────────────────────────────

/// Log at **fatal** level (mapped to `log::error!`).
///
/// OpenCV equivalent: `CV_LOG_FATAL(tag, ...)`.
///
/// # Example
///
/// ```rust
/// use purecv::core::logging::tags;
/// purecv::cv_log_fatal!(tags::CORE, "unrecoverable: {}", "out of memory");
/// ```
#[macro_export]
macro_rules! cv_log_fatal {
    ($tag:expr, $($arg:tt)+) => {
        log::error!(target: $tag, $($arg)+)
    };
}

/// Log at **error** level.
///
/// OpenCV equivalent: `CV_LOG_ERROR(tag, ...)`.
#[macro_export]
macro_rules! cv_log_error {
    ($tag:expr, $($arg:tt)+) => {
        log::error!(target: $tag, $($arg)+)
    };
}

/// Log at **warning** level.
///
/// OpenCV equivalent: `CV_LOG_WARNING(tag, ...)`.
#[macro_export]
macro_rules! cv_log_warning {
    ($tag:expr, $($arg:tt)+) => {
        log::warn!(target: $tag, $($arg)+)
    };
}

/// Log at **info** level.
///
/// OpenCV equivalent: `CV_LOG_INFO(tag, ...)`.
#[macro_export]
macro_rules! cv_log_info {
    ($tag:expr, $($arg:tt)+) => {
        log::info!(target: $tag, $($arg)+)
    };
}

/// Log at **debug** level.
///
/// OpenCV equivalent: `CV_LOG_DEBUG(tag, ...)`.
#[macro_export]
macro_rules! cv_log_debug {
    ($tag:expr, $($arg:tt)+) => {
        log::debug!(target: $tag, $($arg)+)
    };
}

/// Log at **verbose** (trace) level.
///
/// OpenCV equivalent: `CV_LOG_VERBOSE(tag, v, ...)`.
///
/// The `v` parameter is the verbosity sub-level. Since the `log` crate does
/// not support sub-levels, `v` is accepted for API parity but not enforced.
/// All verbose messages map to [`log::trace!`].
#[macro_export]
macro_rules! cv_log_verbose {
    ($tag:expr, $v:expr, $($arg:tt)+) => {
        log::trace!(target: $tag, $($arg)+)
    };
}

// ── Once-per-call-site macros ──────────────────────────────────────

/// Log at **error** level, but only on the first invocation at each call site.
///
/// Uses a `core::sync::atomic::AtomicBool` per call site to track whether the
/// message has already been emitted. This is `no_std`-friendly.
#[macro_export]
macro_rules! cv_log_once_error {
    ($tag:expr, $($arg:tt)+) => {{
        static LOGGED: core::sync::atomic::AtomicBool =
            core::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, core::sync::atomic::Ordering::Relaxed) {
            log::error!(target: $tag, $($arg)+);
        }
    }};
}

/// Log at **warning** level, but only on the first invocation at each call site.
#[macro_export]
macro_rules! cv_log_once_warning {
    ($tag:expr, $($arg:tt)+) => {{
        static LOGGED: core::sync::atomic::AtomicBool =
            core::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, core::sync::atomic::Ordering::Relaxed) {
            log::warn!(target: $tag, $($arg)+);
        }
    }};
}

/// Log at **info** level, but only on the first invocation at each call site.
#[macro_export]
macro_rules! cv_log_once_info {
    ($tag:expr, $($arg:tt)+) => {{
        static LOGGED: core::sync::atomic::AtomicBool =
            core::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, core::sync::atomic::Ordering::Relaxed) {
            log::info!(target: $tag, $($arg)+);
        }
    }};
}

/// Log at **debug** level, but only on the first invocation at each call site.
#[macro_export]
macro_rules! cv_log_once_debug {
    ($tag:expr, $($arg:tt)+) => {{
        static LOGGED: core::sync::atomic::AtomicBool =
            core::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, core::sync::atomic::Ordering::Relaxed) {
            log::debug!(target: $tag, $($arg)+);
        }
    }};
}

// ── Conditional macros ─────────────────────────────────────────────

/// Log at **error** level if `condition` is true.
///
/// OpenCV equivalent: `CV_LOG_IF_ERROR(tag, condition, ...)`.
///
/// # Example
///
/// ```rust
/// use purecv::core::logging::tags;
/// let ksize = 4;
/// purecv::cv_log_if_error!(tags::CORE, ksize % 2 == 0, "even kernel size: {}", ksize);
/// ```
#[macro_export]
macro_rules! cv_log_if_error {
    ($tag:expr, $cond:expr, $($arg:tt)+) => {
        if $cond {
            log::error!(target: $tag, $($arg)+);
        }
    };
}

/// Log at **warning** level if `condition` is true.
///
/// OpenCV equivalent: `CV_LOG_IF_WARNING(tag, condition, ...)`.
#[macro_export]
macro_rules! cv_log_if_warning {
    ($tag:expr, $cond:expr, $($arg:tt)+) => {
        if $cond {
            log::warn!(target: $tag, $($arg)+);
        }
    };
}

/// Log at **info** level if `condition` is true.
///
/// OpenCV equivalent: `CV_LOG_IF_INFO(tag, condition, ...)`.
#[macro_export]
macro_rules! cv_log_if_info {
    ($tag:expr, $cond:expr, $($arg:tt)+) => {
        if $cond {
            log::info!(target: $tag, $($arg)+);
        }
    };
}

/// Log at **debug** level if `condition` is true.
///
/// OpenCV equivalent: `CV_LOG_IF_DEBUG(tag, condition, ...)`.
#[macro_export]
macro_rules! cv_log_if_debug {
    ($tag:expr, $cond:expr, $($arg:tt)+) => {
        if $cond {
            log::debug!(target: $tag, $($arg)+);
        }
    };
}
