/*
 *  opencv_reference_links.rs
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

use std::fs;
use std::path::{Path, PathBuf};

/// The OpenCV release every reference in the source tree and the docs must name.
const REFERENCE: &str = "4.10.0";

/// The commit the `REFERENCE` tag of opencv/opencv peels to.
const REFERENCE_COMMIT: &str = "71d3237a093b60a27601c20e9ee6c3e52154e8b1";

/// The commit the `REFERENCE` tag of opencv/opencv_contrib peels to.
const REFERENCE_CONTRIB_COMMIT: &str = "1ed3dd2c53888e3289afdb22ec4e9ebbff3dba87";

/// Directories to scan, relative to `CARGO_MANIFEST_DIR`, with the file extension to read.
const SCANNED: &[(&str, &str)] = &[
    ("src", "rs"),
    ("examples", "rs"),
    ("benches", "rs"),
    ("crates/wasm/src", "rs"),
    ("docs/parity", "md"),
];

/// URL prefixes after which the next path segment names an OpenCV version.
const VERSIONED_URLS: &[&str] = &[
    "docs.opencv.org/",
    "github.com/opencv/opencv/blob/",
    "github.com/opencv/opencv/tree/",
    "github.com/opencv/opencv_contrib/blob/",
    "github.com/opencv/opencv_contrib/tree/",
];

/// Source-file suffixes that, followed by a digit, form a bare `file.cpp:NN` citation.
const LINE_CITATIONS: &[&str] = &[".cpp:", ".hpp:", ".h:"];

/// URLs a markdown link must target for its text to be allowed to read `file.cpp:NN`.
const CITATION_TARGETS: &[&str] = &[
    "https://github.com/opencv/opencv/blob/",
    "https://github.com/opencv/opencv_contrib/blob/",
];

fn is_segment_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-')
}

/// True when the citation whose digits start at `rest` is the text of a markdown link to a
/// GitHub blob: only line numbers, `,`, `-` and backticks come before the next `](`, which
/// is followed by a blob URL. The URL itself is still judged by the version-segment rule.
fn is_link_text(rest: &str) -> bool {
    let Some(close) = rest.find("](") else {
        return false;
    };
    rest[..close]
        .bytes()
        .all(|b| b.is_ascii_digit() || matches!(b, b',' | b'-' | b'`'))
        && CITATION_TARGETS
            .iter()
            .any(|target| rest[close + 2..].starts_with(target))
}

/// One message per offending OpenCV reference on `line`.
fn line_violations(line: &str) -> Vec<String> {
    let bytes = line.as_bytes();
    let mut out = Vec::new();

    for prefix in VERSIONED_URLS {
        for (at, _) in line.match_indices(prefix) {
            let start = at + prefix.len();
            let len = bytes[start..]
                .iter()
                .take_while(|&&b| is_segment_byte(b))
                .count();
            let segment = &line[start..start + len];
            if segment != REFERENCE {
                out.push(format!(
                    "`{prefix}{segment}` names `{segment}`, expected `{REFERENCE}`"
                ));
            }
        }
    }

    for suffix in LINE_CITATIONS {
        for (at, _) in line.match_indices(suffix) {
            let start = at + suffix.len();
            if !bytes.get(start).is_some_and(u8::is_ascii_digit) {
                continue;
            }
            if is_link_text(&line[start..]) {
                continue;
            }
            // Widen the match to the whole `path/file.cpp:12-34` token for the message.
            let from = bytes[..at]
                .iter()
                .rposition(|&b| !(is_segment_byte(b) || b == b'/'))
                .map_or(0, |i| i + 1);
            let to = start
                + bytes[start..]
                    .iter()
                    .take_while(|&&b| b.is_ascii_digit() || matches!(b, b',' | b'-'))
                    .count();
            out.push(format!(
                "bare line citation `{}`; link it as https://github.com/opencv/opencv/blob/{REFERENCE}/<path>#L<n>",
                line[from..to].trim_end_matches([',', '-'])
            ));
        }
    }

    out
}

/// `(1-based line number, message)` for every offending reference in `text`.
fn content_violations(text: &str) -> Vec<(usize, String)> {
    text.lines()
        .enumerate()
        .flat_map(|(i, line)| {
            line_violations(line)
                .into_iter()
                .map(move |msg| (i + 1, msg))
        })
        .collect()
}

fn collect_files(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let entries =
        fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    for entry in entries {
        let entry =
            entry.unwrap_or_else(|e| panic!("cannot read an entry of {}: {e}", dir.display()));
        let path = entry.path();
        let kind = entry
            .file_type()
            .unwrap_or_else(|e| panic!("cannot stat {}: {e}", path.display()));
        if kind.is_dir() {
            collect_files(&path, ext, out);
        } else if kind.is_file() && path.extension().is_some_and(|x| x == ext) {
            out.push(path);
        }
    }
}

/// Scans the `(directory, extension)` pairs under `root`. Returns the number of files read
/// and the sorted `"<relpath>:<line>: <message>"` entries. Directories that do not exist are
/// skipped.
fn scan(root: &Path, dirs: &[(&str, &str)]) -> (usize, Vec<String>) {
    let mut files = Vec::new();
    for (dir, ext) in dirs {
        let dir = root.join(dir);
        if dir.is_dir() {
            collect_files(&dir, ext, &mut files);
        }
    }

    let mut violations = Vec::new();
    for file in &files {
        let text = fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", file.display()));
        let rel = file
            .strip_prefix(root)
            .unwrap_or(file)
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        for (line, msg) in content_violations(&text) {
            violations.push(format!("{rel}:{line}: {msg}"));
        }
    }
    violations.sort();
    (files.len(), violations)
}

#[test]
fn line_violations_detects_drift() {
    let cases: &[(&str, usize)] = &[
        ("https://docs.opencv.org/4.10.0/db/d95/classcv_1_1ORB.html", 0),
        ("<https://docs.opencv.org/4.10.0>", 0),
        ("https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html", 1),
        ("https://docs.opencv.org/4.13.0/db/d70/tutorial_akaze_matching.html", 1),
        ("https://docs.opencv.org/4.10/d4/dee/tutorial_optical_flow.html", 1),
        ("https://docs.opencv.org/4.10.0-kleidicv/x", 1),
        ("https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L862", 0),
        ("https://github.com/opencv/opencv/blob/master/modules/video/src/lkpyramid.cpp", 1),
        ("https://github.com/opencv/opencv_contrib/tree/4.x/modules/xfeatures2d", 1),
        ("(`modules/video/src/lkpyramid.cpp:115`)", 1),
        ("`core/include/opencv2/core/saturate.hpp:106-107`", 1),
        ("[`lkpyramid.cpp:726-807`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L726-L807)", 0),
        ("[`lkpyramid.cpp:726-807`](https://github.com/opencv/opencv/blob/4.x/modules/video/src/lkpyramid.cpp#L726)", 1),
        ("see lkpyramid.cpp for details", 0),
        ("num-traits `src/cast.rs:288`", 0),
        ("docs.opencv.org/4.x/a and docs.opencv.org/master/b", 2),
    ];
    for (line, expected) in cases {
        assert_eq!(line_violations(line).len(), *expected, "line: {line}");
    }
}

#[test]
fn content_violations_reports_crlf_line_numbers() {
    let v = content_violations("ok\r\nhttps://docs.opencv.org/4.x/x\r\n");
    assert_eq!(v.len(), 1);
    assert_eq!(v[0].0, 2);
}

#[test]
fn scan_skips_missing_roots() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert_eq!(scan(root, &[("does/not/exist", "rs")]), (0, vec![]));
}

#[test]
fn scan_counts_files_under_src() {
    let (files, _) = scan(Path::new(env!("CARGO_MANIFEST_DIR")), &[("src", "rs")]);
    assert!(files > 0, "guard scanned no files: wrong root?");
}

#[test]
fn bootstrap_pins_reference_commits() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("cpp_ref")
        .join("libraries.json");
    let json = match fs::read_to_string(&path) {
        Ok(json) => json,
        // The packaged crate excludes cpp_ref/; there is nothing to check there.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(e) => panic!("cannot read {}: {e}", path.display()),
    };
    for pin in [
        format!("\"branch\": \"{REFERENCE}\""),
        format!("\"commit\": \"{REFERENCE_COMMIT}\""),
        format!("\"commit\": \"{REFERENCE_CONTRIB_COMMIT}\""),
    ] {
        assert!(
            json.contains(&pin),
            "{} must contain `{pin}` (see cpp_ref/README.md)",
            path.display()
        );
    }
}

#[test]
fn opencv_references_are_pinned_to_reference_version() {
    let (_, v) = scan(Path::new(env!("CARGO_MANIFEST_DIR")), SCANNED);
    assert!(
        v.is_empty(),
        "OpenCV references must be pinned to {REFERENCE} (see CONTRIBUTING.md):\n{}",
        v.join("\n")
    );
}
