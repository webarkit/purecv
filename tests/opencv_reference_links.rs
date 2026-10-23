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

/// The OpenCV release every reference in the source tree and the docs must name, except
/// the frozen parity series under `docs/parity/<version>/` (see `required_version`).
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
    ("crates/wasm/www", "js"),
    ("docs/parity", "md"),
];

/// URL prefixes after which the next path segment names an OpenCV version. Lowercase: a line
/// is matched case-insensitively, since GitHub and the docs site ignore case in these parts.
/// `commit/<sha>` links are not listed: a commit is immutable, so it cannot drift.
const VERSIONED_URLS: &[&str] = &[
    "docs.opencv.org/",
    "github.com/opencv/opencv/blob/",
    "github.com/opencv/opencv/tree/",
    "github.com/opencv/opencv/raw/",
    "github.com/opencv/opencv/blame/",
    "github.com/opencv/opencv_contrib/blob/",
    "github.com/opencv/opencv_contrib/tree/",
    "github.com/opencv/opencv_contrib/raw/",
    "github.com/opencv/opencv_contrib/blame/",
    "github.com/opencv/opencv_extra/blob/",
    "github.com/opencv/opencv_extra/tree/",
    "github.com/opencv/opencv_extra/raw/",
    "github.com/opencv/opencv_extra/blame/",
    "raw.githubusercontent.com/opencv/opencv/",
    "raw.githubusercontent.com/opencv/opencv_contrib/",
    "raw.githubusercontent.com/opencv/opencv_extra/",
];

/// Ref namespaces GitHub accepts in front of a version: `refs/tags/4.10.0` names the same
/// tree as `4.10.0`. The version is read after them; `refs/heads/4.x` still names `4.x`.
const REF_PREFIXES: &[&str] = &["refs/tags/", "refs/heads/"];

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

/// One message per offending OpenCV reference on `line`, against the current reference.
fn line_violations(line: &str) -> Vec<String> {
    line_violations_for(line, REFERENCE)
}

/// One message per offending OpenCV reference on `line`, which must name `reference`.
fn line_violations_for(line: &str, reference: &str) -> Vec<String> {
    let bytes = line.as_bytes();
    // ASCII lowercasing keeps every byte offset, so offsets in `lower` index `line` too.
    let lower = line.to_ascii_lowercase();
    let mut out = Vec::new();

    for prefix in VERSIONED_URLS {
        for (at, _) in lower.match_indices(prefix) {
            let mut start = at + prefix.len();
            if let Some(ns) = REF_PREFIXES
                .iter()
                .find(|ns| lower[start..].starts_with(*ns))
            {
                start += ns.len();
            }
            let len = bytes[start..]
                .iter()
                .take_while(|&&b| is_segment_byte(b))
                .count();
            let segment = &line[start..start + len];
            if segment != reference {
                out.push(format!(
                    "`{}` names `{segment}`, expected `{reference}`",
                    &line[at..start + len]
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
                "bare line citation `{}`; link it as https://github.com/opencv/opencv/blob/{reference}/<path>#L<n>",
                line[from..to].trim_end_matches([',', '-'])
            ));
        }
    }

    out
}

/// `(1-based line number, message)` for every offending reference in `text`, against the
/// current reference.
fn content_violations(text: &str) -> Vec<(usize, String)> {
    content_violations_for(text, REFERENCE)
}

/// `(1-based line number, message)` for every reference in `text` that does not name
/// `reference`.
fn content_violations_for(text: &str, reference: &str) -> Vec<(usize, String)> {
    text.lines()
        .enumerate()
        .flat_map(|(i, line)| {
            line_violations_for(line, reference)
                .into_iter()
                .map(move |msg| (i + 1, msg))
        })
        .collect()
}

/// Frozen parity series live in `docs/parity/<version>/` (ADR 0001). A file inside one must
/// keep naming that folder's version after the reference is bumped; every other file must
/// name the current `REFERENCE`. `rel` is a `/`-separated path relative to the crate root.
fn required_version(rel: &str) -> &str {
    let mut parts = rel.split('/');
    if parts.next() == Some("docs") && parts.next() == Some("parity") {
        if let Some(folder) = parts.next() {
            // the folder must be followed by a file (or a subfolder), and look like `X.Y.Z`
            if parts.next().is_some() && is_version(folder) {
                return folder;
            }
        }
    }
    REFERENCE
}

/// `X.Y.Z` with decimal components, as the OpenCV release tags are named.
fn is_version(s: &str) -> bool {
    let parts: Vec<&str> = s.split('.').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
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
        for (line, msg) in content_violations_for(&text, required_version(&rel)) {
            violations.push(format!("{rel}:{line}: {msg}"));
        }
    }
    violations.sort();
    (files.len(), violations)
}

/// A parsed JSON value. Objects keep every member in order, repeated keys included.
#[derive(Debug)]
enum Json {
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
    /// A number, `true`, `false` or `null`; `pin_violations` never needs its text.
    Other,
}

/// A small recursive-descent JSON parser, so the guard needs no crate dependency. It accepts
/// what Python's `json.loads` accepts (strict mode, the default) and parses it to the same
/// values: RFC 8259 numbers and literals plus Python's `NaN`, `Infinity` and `-Infinity`; no
/// raw control characters in strings; `\u` with exactly four hex digits, surrogate pairs
/// decoded. One difference remains: a lone surrogate escape is an error here, since a Rust
/// `String` cannot hold it; `bootstrap.py` would load such a file, but no library name, branch
/// or commit can contain one.
struct Parser<'a> {
    text: &'a str,
    at: usize,
}

impl Parser<'_> {
    fn skip_ws(&mut self) {
        let rest = &self.text[self.at..];
        self.at += rest.len() - rest.trim_start_matches([' ', '\t', '\n', '\r']).len();
    }

    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.at).copied()
    }

    fn expect(&mut self, b: u8) -> Result<(), String> {
        self.skip_ws();
        if self.peek() != Some(b) {
            return Err(format!("expected `{}` at byte {}", char::from(b), self.at));
        }
        self.at += 1;
        Ok(())
    }

    fn value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'"') => self.string().map(Json::Str),
            Some(b'[') => {
                self.at += 1;
                self.items(b']', Self::value).map(Json::Arr)
            }
            Some(b'{') => {
                self.at += 1;
                self.items(b'}', |p| {
                    p.skip_ws();
                    let key = p.string()?;
                    p.expect(b':')?;
                    Ok((key, p.value()?))
                })
                .map(Json::Obj)
            }
            Some(b) if b == b'-' || b.is_ascii_alphanumeric() => {
                self.scalar().map(|()| Json::Other)
            }
            _ => Err(format!("expected a value at byte {}", self.at)),
        }
    }

    /// A number, `true`, `false`, `null`, or Python's `NaN`, `Infinity`, `-Infinity`.
    fn scalar(&mut self) -> Result<(), String> {
        let rest = &self.text[self.at..];
        for literal in ["true", "false", "null", "NaN", "Infinity", "-Infinity"] {
            if rest.starts_with(literal) {
                self.at += literal.len();
                return Ok(());
            }
        }
        // -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
        let b = rest.as_bytes();
        let digits = |from: usize| b[from..].iter().take_while(|c| c.is_ascii_digit()).count();
        let mut i = usize::from(b.first() == Some(&b'-'));
        match b.get(i) {
            Some(b'0') => i += 1,
            Some(c) if c.is_ascii_digit() => i += digits(i),
            _ => return Err(format!("invalid number at byte {}", self.at)),
        }
        if b.get(i) == Some(&b'.') {
            let n = digits(i + 1);
            if n == 0 {
                return Err(format!("invalid number at byte {}", self.at));
            }
            i += 1 + n;
        }
        if matches!(b.get(i), Some(b'e' | b'E')) {
            i += 1;
            if matches!(b.get(i), Some(b'+' | b'-')) {
                i += 1;
            }
            let n = digits(i);
            if n == 0 {
                return Err(format!("invalid number at byte {}", self.at));
            }
            i += n;
        }
        self.at += i;
        Ok(())
    }

    /// The four hex digits of a `\u` escape, from `chars`.
    fn hex4(chars: &mut std::str::CharIndices<'_>) -> Result<u32, String> {
        let hex: String = chars.by_ref().take(4).map(|(_, h)| h).collect();
        if hex.len() != 4 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("invalid escape `\\u{hex}`"));
        }
        u32::from_str_radix(&hex, 16).map_err(|e| e.to_string())
    }

    /// The comma-separated items after an opening bracket, through the closing `close`.
    fn items<T>(
        &mut self,
        close: u8,
        mut item: impl FnMut(&mut Self) -> Result<T, String>,
    ) -> Result<Vec<T>, String> {
        let mut out = Vec::new();
        self.skip_ws();
        if self.peek() == Some(close) {
            self.at += 1;
            return Ok(out);
        }
        loop {
            out.push(item(self)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b) if b == close => {
                    self.at += 1;
                    return Ok(out);
                }
                _ => {
                    return Err(format!(
                        "expected `,` or `{}` at byte {}",
                        char::from(close),
                        self.at
                    ))
                }
            }
        }
    }

    /// The string at the current `"`, with its escapes decoded.
    fn string(&mut self) -> Result<String, String> {
        if self.peek() != Some(b'"') {
            return Err(format!("expected a string at byte {}", self.at));
        }
        self.at += 1;
        let mut out = String::new();
        let mut chars = self.text[self.at..].char_indices();
        while let Some((i, c)) = chars.next() {
            match c {
                '"' => {
                    self.at += i + 1;
                    return Ok(out);
                }
                '\\' => {
                    let (_, e) = chars.next().ok_or("unterminated escape")?;
                    out.push(match e {
                        '"' | '\\' | '/' => e,
                        'b' => '\u{8}',
                        'f' => '\u{c}',
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        'u' => {
                            let mut code = Self::hex4(&mut chars)?;
                            if (0xD800..0xDC00).contains(&code) {
                                // a high surrogate: decode the pair, as Python does
                                let low = match (chars.next(), chars.next()) {
                                    (Some((_, '\\')), Some((_, 'u'))) => Self::hex4(&mut chars)?,
                                    _ => return Err(format!("lone surrogate `\\u{code:04x}`")),
                                };
                                if !(0xDC00..0xE000).contains(&low) {
                                    return Err(format!("lone surrogate `\\u{code:04x}`"));
                                }
                                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                            }
                            char::from_u32(code)
                                .ok_or_else(|| format!("lone surrogate `\\u{code:04x}`"))?
                        }
                        _ => return Err(format!("invalid escape `\\{e}`")),
                    });
                }
                c if c < ' ' => {
                    return Err(format!("raw control character {c:?} in a string"));
                }
                _ => out.push(c),
            }
        }
        Err("unterminated string".to_owned())
    }
}

fn parse_json(text: &str) -> Result<Json, String> {
    let mut parser = Parser { text, at: 0 };
    let value = parser.value()?;
    parser.skip_ws();
    if parser.at != text.len() {
        return Err(format!("unexpected data at byte {}", parser.at));
    }
    Ok(value)
}

/// The value of the one member of `object` named `key`, or how many members have that name.
fn only<'a>(object: &'a [(String, Json)], key: &str) -> Result<&'a Json, usize> {
    let found: Vec<&Json> = object
        .iter()
        .filter(|(k, _)| k == key)
        .map(|(_, v)| v)
        .collect();
    match found.as_slice() {
        [value] => Ok(value),
        _ => Err(found.len()),
    }
}

/// One message per pin that `json`, the text of `cpp_ref/libraries.json`, does not hold. Each
/// library needs exactly one entry, with exactly one `name` and one `source` object, and in
/// that object exactly one `branch` (`REFERENCE`) and one `commit` (the library's constant).
/// Those are the fields the bootstrap reads; the same strings anywhere else do not count.
/// Exactly one, because Python's `json` keeps the last of repeated keys.
fn pin_violations(json: &str) -> Vec<String> {
    let entries = match parse_json(json) {
        Ok(Json::Arr(entries)) => entries,
        Ok(_) => return vec!["expected a top-level array of library entries".to_owned()],
        Err(e) => return vec![format!("not valid JSON: {e}")],
    };
    let mut out = Vec::new();
    for (name, commit) in [
        ("opencv", REFERENCE_COMMIT),
        ("opencv_contrib", REFERENCE_CONTRIB_COMMIT),
    ] {
        let named: Vec<&[(String, Json)]> = entries
            .iter()
            .filter_map(|entry| match entry {
                Json::Obj(members)
                    if members
                        .iter()
                        .any(|(k, v)| k == "name" && matches!(v, Json::Str(s) if s == name)) =>
                {
                    Some(members.as_slice())
                }
                _ => None,
            })
            .collect();
        let &[entry] = named.as_slice() else {
            out.push(format!(
                "expected one `\"name\": \"{name}\"` entry, found {}",
                named.len()
            ));
            continue;
        };
        if let Err(n) = only(entry, "name") {
            out.push(format!(
                "the `{name}` entry has {n} `name` keys, expected 1"
            ));
            continue;
        }
        let source = match only(entry, "source") {
            Ok(Json::Obj(source)) => source,
            Ok(other) => {
                out.push(format!(
                    "the `{name}` entry's `source` must be an object, found {other:?}"
                ));
                continue;
            }
            Err(n) => {
                out.push(format!(
                    "the `{name}` entry has {n} `source` keys, expected 1"
                ));
                continue;
            }
        };
        for (key, pin) in [("branch", REFERENCE), ("commit", commit)] {
            match only(source, key) {
                Ok(Json::Str(value)) if value == pin => {}
                Ok(other) => out.push(format!(
                    "the `{name}` entry's `source.{key}` is {other:?}, expected \"{pin}\""
                )),
                Err(n) => out.push(format!(
                    "the `{name}` entry's `source` has {n} `{key}` keys, expected 1"
                )),
            }
        }
    }
    out
}

#[test]
fn line_violations_detects_drift() {
    let cases: &[(&str, usize)] = &[
        ("https://docs.opencv.org/4.10.0/db/d95/classcv_1_1ORB.html", 0),
        ("<https://docs.opencv.org/4.10.0>", 0),
        ("(https://docs.opencv.org/4.10.0)", 0),
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
        ("https://github.com/opencv/opencv/raw/4.10.0/modules/core/src/matrix.cpp", 0),
        ("https://github.com/opencv/opencv/raw/master/modules/core/src/matrix.cpp", 1),
        ("https://github.com/opencv/opencv/blame/4.10.0/modules/video/src/lkpyramid.cpp#L115", 0),
        ("https://github.com/opencv/opencv/blame/4.x/modules/video/src/lkpyramid.cpp#L115", 1),
        ("https://github.com/opencv/opencv_contrib/raw/4.10.0/modules/xfeatures2d/src/sift.cpp", 0),
        ("https://github.com/opencv/opencv_contrib/raw/master/modules/xfeatures2d/src/sift.cpp", 1),
        ("https://github.com/opencv/opencv_contrib/blame/4.10.0/modules/xfeatures2d/src/sift.cpp", 0),
        ("https://github.com/opencv/opencv_contrib/blame/4.x/modules/xfeatures2d/src/sift.cpp", 1),
        ("https://raw.githubusercontent.com/opencv/opencv/4.10.0/modules/imgproc/src/color.cpp", 0),
        ("https://raw.githubusercontent.com/opencv/opencv/4.x/modules/imgproc/src/color.cpp", 1),
        ("https://raw.githubusercontent.com/opencv/opencv_contrib/4.10.0/modules/xfeatures2d/src/sift.cpp", 0),
        ("https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/xfeatures2d/src/sift.cpp", 1),
        // #171, item 6: the refs/tags form names the same tree; refs/heads/<branch> drifts.
        ("https://raw.githubusercontent.com/opencv/opencv/refs/tags/4.10.0/samples/data/butterfly.jpg", 0),
        ("https://github.com/opencv/opencv/raw/refs/tags/4.10.0/samples/data/butterfly.jpg", 0),
        ("https://github.com/opencv/opencv/blob/refs/tags/4.13.0/modules/core/src/arithm.cpp", 1),
        ("https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/samples/data/butterfly.jpg", 1),
        // #171, item 7: host and owner match whatever their case; opencv_extra is covered.
        ("https://GitHub.com/OpenCV/opencv/blob/master/modules/core/src/arithm.cpp", 1),
        ("https://Docs.OpenCV.org/4.x/d4/dee/tutorial_optical_flow.html", 1),
        ("https://GitHub.com/OpenCV/opencv/blob/4.10.0/modules/core/src/arithm.cpp", 0),
        ("https://github.com/opencv/opencv_extra/blob/4.10.0/testdata/cv/shared/lena.png", 0),
        ("https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/cv/shared/lena.png", 1),
        // commit links are immutable, so they are not version-checked
        ("https://github.com/opencv/opencv/commit/71d3237a093b60a27601c20e9ee6c3e52154e8b1", 0),
    ];
    for (line, expected) in cases {
        assert_eq!(line_violations(line).len(), *expected, "line: {line}");
    }
}

#[test]
fn parse_json_matches_python_strict_mode() {
    // #171, item 9: what Python's json.loads accepts, and with the same values.
    let accepted: &[(&str, &str)] = &[
        (
            r#"[0, -0, 12, 1.5, -2e3, 1E+2, 3.25e-1, true, false, null, NaN, Infinity, -Infinity]"#,
            "",
        ),
        (r#""a\"b\\c\/d\b\f\n\r\t""#, "a\"b\\c/d\u{8}\u{c}\n\r\t"),
        (r#""\u0041\u00e9""#, "A\u{e9}"),
        (r#""\ud83d\ude00""#, "\u{1F600}"), // a surrogate pair, as Python decodes it
    ];
    for (text, decoded) in accepted {
        match parse_json(text) {
            Ok(Json::Str(s)) => assert_eq!(s, *decoded, "{text}"),
            Ok(_) => assert!(decoded.is_empty(), "{text}"),
            Err(e) => panic!("{text}: {e}"),
        }
    }
    let rejected: &[&str] = &[
        "01",
        "[01]",
        "1.",
        ".5",
        "1e",
        "+1",
        "tru",
        "nan",
        "[1,]",
        "\"a\nb\"",      // raw control character
        "\"tab\there\"", // raw tab
        r#""\u+041""#,   // not four hex digits
        r#""\u004""#,
        r#""\ud83d""#,       // lone high surrogate
        r#""\ude00""#,       // lone low surrogate
        r#""\ud83d\u0041""#, // a high surrogate followed by a non-surrogate
    ];
    for text in rejected {
        assert!(parse_json(text).is_err(), "accepted: {text}");
    }
}

#[test]
fn required_version_follows_the_parity_folder() {
    let cases: &[(&str, &str)] = &[
        ("docs/parity/4.10.0/core.md", "4.10.0"),
        ("docs/parity/4.13.0/README.md", "4.13.0"),
        ("docs/parity/4.10.0/notes/old.md", "4.10.0"),
        ("docs/parity/README.md", REFERENCE),
        ("docs/parity/AGENTS.md", REFERENCE),
        ("docs/parity/4.10/core.md", REFERENCE),
        ("docs/parity/4.10.0-rc/core.md", REFERENCE),
        ("docs/parity/4.10.0", REFERENCE),
        ("src/core/matrix.rs", REFERENCE),
        ("docs/other/4.13.0/x.md", REFERENCE),
    ];
    for (rel, expected) in cases {
        assert_eq!(required_version(rel), *expected, "path: {rel}");
    }
}

#[test]
fn frozen_folders_are_checked_against_their_own_version() {
    let link_4_13 = "[`orb.cpp`](https://github.com/opencv/opencv/blob/4.13.0/modules/features2d/src/orb.cpp#L862)";
    let link_4_10 = "https://docs.opencv.org/4.10.0/db/d95/classcv_1_1ORB.html";
    // a frozen 4.13.0 folder linking its own version is valid
    let rel = "docs/parity/4.13.0/features2d.md";
    assert!(content_violations_for(link_4_13, required_version(rel)).is_empty());
    // the same folder with a link drifting to 4.10.0 (or to 4.x) is not
    assert_eq!(
        content_violations_for(link_4_10, required_version(rel)).len(),
        1
    );
    assert_eq!(
        content_violations_for("https://docs.opencv.org/4.x/x", required_version(rel)).len(),
        1
    );
    // a nested path inside the folder follows the folder
    let nested = "docs/parity/4.13.0/notes/features2d.md";
    assert!(content_violations_for(link_4_13, required_version(nested)).is_empty());
    // outside a version folder the current reference applies
    assert_eq!(
        content_violations_for(link_4_13, required_version("docs/parity/README.md")).len(),
        if REFERENCE == "4.13.0" { 0 } else { 1 }
    );
    // the bare-citation hint names the folder's version
    let v = content_violations_for("`orb.cpp:862`", "4.13.0");
    assert_eq!(v.len(), 1);
    assert!(v[0].1.contains("/blob/4.13.0/"), "{}", v[0].1);
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
fn pin_violations_checks_each_library_entry() {
    let entry = |name: &str, commit: &str| {
        format!(
            "  {{\n    \"name\": \"{name}\",\n    \"source\": {{\n      \"type\": \"git\",\n      \
             \"branch\": \"{REFERENCE}\",\n      \"commit\": \"{commit}\"\n    }}\n  }}"
        )
    };
    let file = |entries: &[String]| format!("[\n{}\n]\n", entries.join(",\n"));
    let opencv = entry("opencv", REFERENCE_COMMIT);
    let contrib = entry("opencv_contrib", REFERENCE_CONTRIB_COMMIT);

    assert_eq!(
        pin_violations(&file(&[opencv.clone(), contrib.clone()])),
        Vec::<String>::new()
    );
    // Both commits are in the file, but each in the other library's entry.
    let swapped = [
        entry("opencv", REFERENCE_CONTRIB_COMMIT),
        entry("opencv_contrib", REFERENCE_COMMIT),
    ];
    assert_eq!(pin_violations(&file(&swapped)).len(), 2);
    assert_eq!(
        pin_violations(&file(std::slice::from_ref(&opencv))).len(),
        1
    );
    assert_eq!(
        pin_violations(&file(&[opencv.clone(), opencv, contrib])).len(),
        1
    );
}

#[test]
fn pin_violations_reads_the_source_object() {
    // The layout of the committed cpp_ref/libraries.json, which
    // `bootstrap_pins_reference_commits` checks as it is.
    let source = |repo: &str, branch: &str, commit: &str| {
        format!(
            "{{\n      \"type\": \"git\",\n      \"url\": \"https://github.com/opencv/{repo}.git\",\n      \
             \"branch\": \"{branch}\",\n      \"commit\": \"{commit}\",\n      \"recursive\": false\n    }}"
        )
    };
    let contrib = format!(
        "  {{\n    \"name\": \"opencv_contrib\",\n    \"source\": {}\n  }}",
        source("opencv_contrib", REFERENCE, REFERENCE_CONTRIB_COMMIT)
    );
    // `members` follow the opencv entry's `name`.
    let file = |members: &str| {
        format!("[\n  {{\n    \"name\": \"opencv\",\n    {members}\n  }},\n{contrib}\n]\n")
    };
    let drift = "0000000000000000000000000000000000000000";

    let pinned = format!(
        "\"source\": {}",
        source("opencv", REFERENCE, REFERENCE_COMMIT)
    );
    assert_eq!(pin_violations(&file(&pinned)), Vec::<String>::new());

    // Drifted `source` pins, with the expected strings in members outside `source`.
    let decoys = format!(
        "\"branch\": \"{REFERENCE}\",\n    \"commit\": \"{REFERENCE_COMMIT}\",\n    \
         \"source\": {},\n    \
         \"expected\": {{ \"branch\": \"{REFERENCE}\", \"commit\": \"{REFERENCE_COMMIT}\" }}",
        source("opencv", "4.x", drift)
    );
    assert_eq!(pin_violations(&file(&decoys)).len(), 2, "{decoys}");

    let unversioned = format!("\"source\": {}", source("opencv", "4.x", REFERENCE_COMMIT));
    assert_eq!(pin_violations(&file(&unversioned)).len(), 1);

    // Python's json keeps the last of repeated keys, so each key must occur exactly once.
    // In the second case the key `bra\u006ech` decodes to a second `branch`.
    for repeated in [
        format!(
            "\"source\": {{ \"branch\": \"{REFERENCE}\", \"commit\": \"{REFERENCE_COMMIT}\", \
             \"branch\": \"4.x\" }}"
        ),
        format!(
            "\"source\": {{ \"branch\": \"{REFERENCE}\", \"commit\": \"{REFERENCE_COMMIT}\", \
             \"bra\\u006ech\": \"4.x\" }}"
        ),
        format!(
            "{pinned},\n    \"source\": {}",
            source("opencv", "4.x", drift)
        ),
    ] {
        assert_eq!(pin_violations(&file(&repeated)).len(), 1, "{repeated}");
    }

    let missing = format!("\"source\": {{ \"branch\": \"{REFERENCE}\" }}");
    assert_eq!(pin_violations(&file(&missing)).len(), 1);
    let not_an_object = format!("\"source\": \"{REFERENCE_COMMIT}\"");
    assert_eq!(pin_violations(&file(&not_an_object)).len(), 1);
    assert!(!pin_violations(&file(&pinned)[1..]).is_empty());
}

#[test]
fn bootstrap_pins_reference_commits() {
    let cpp_ref = Path::new(env!("CARGO_MANIFEST_DIR")).join("cpp_ref");
    // The packaged crate excludes cpp_ref/; there is nothing to check there. A checkout always
    // has cpp_ref/, so a missing libraries.json in it is a failure.
    if !cpp_ref.is_dir() {
        return;
    }
    let path = cpp_ref.join("libraries.json");
    let json =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let v = pin_violations(&json);
    assert!(
        v.is_empty(),
        "{} does not pin OpenCV {REFERENCE} (see cpp_ref/README.md):\n{}",
        path.display(),
        v.join("\n")
    );
}

#[test]
fn opencv_references_are_pinned_to_reference_version() {
    let (_, v) = scan(Path::new(env!("CARGO_MANIFEST_DIR")), SCANNED);
    assert!(
        v.is_empty(),
        "OpenCV references must be pinned to {REFERENCE}, or to the folder's version under \
         docs/parity/<version>/ (see CONTRIBUTING.md and docs/adr/0001-opencv-parity-audit.md):\n{}",
        v.join("\n")
    );
}

// ---------------------------------------------------------------------------------------------
// Parity suites: every `#[ignore]` cites the issue that fixes it, every current-behaviour pin
// names the issue it is deleted with (docs/parity/AGENTS.md, rule 11).

/// The shape every `#[ignore]` attribute in `tests/parity_*.rs` must have.
const IGNORE_SHAPE: &str = "#[ignore = \"diverges from OpenCV <version>, see #N\"]";

/// Byte offsets of each `#N` issue reference in `text` (a `#` followed by digits, not part
/// of a longer token such as `C#1` or `##1`).
fn issue_references(text: &str) -> Vec<usize> {
    let bytes = text.as_bytes();
    text.match_indices('#')
        .filter(|&(at, _)| {
            let after_digit = bytes.get(at + 1).is_some_and(u8::is_ascii_digit);
            let before = at.checked_sub(1).map(|i| bytes[i]);
            after_digit && !before.is_some_and(|b| b.is_ascii_alphanumeric() || b == b'#')
        })
        .map(|(at, _)| at)
        .collect()
}

/// An outer attribute of a source file: the 1-based line it starts on, the line it ends on,
/// and its text joined onto one line. rustfmt splits long `#[cfg_attr(...)]` attributes over
/// several lines, so an attribute is collected until its brackets balance; brackets inside
/// string literals do not count.
struct Attribute {
    start: usize,
    end: usize,
    text: String,
}

/// Net `[`/`]` nesting of `s`, ignoring brackets inside string literals.
fn bracket_balance(s: &str) -> i32 {
    let mut depth = 0;
    let mut in_string = false;
    let mut escaped = false;
    for b in s.bytes() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'[' => depth += 1,
            b']' => depth -= 1,
            _ => {}
        }
    }
    depth
}

/// Every `#[...]` attribute of `src`, in order.
fn attributes(src: &str) -> Vec<Attribute> {
    let mut out = Vec::new();
    let mut lines = src.lines().enumerate().peekable();
    while let Some((i, line)) = lines.next() {
        let first = line.trim();
        if !first.starts_with("#[") {
            continue;
        }
        let mut text = first.to_string();
        let mut depth = bracket_balance(first);
        let mut end = i;
        while depth > 0 {
            let Some((j, line)) = lines.next() else {
                break;
            };
            let t = line.trim();
            text.push(' ');
            text.push_str(t);
            depth += bracket_balance(t);
            end = j;
        }
        // `#[cfg_attr( pred, ignore = "..." )]` after the join reads like the one-line form.
        let text = text.replace("( ", "(").replace(" )", ")");
        out.push(Attribute {
            start: i + 1,
            end: end + 1,
            text,
        });
    }
    out
}

/// True for `#[cfg_attr(<predicate>, ignore ...)]`: an ignore that applies to one build only.
fn is_cfg_attr_ignore(attr: &str) -> bool {
    attr.starts_with("#[cfg_attr(")
        && (attr.contains(", ignore = \"") || attr.ends_with(", ignore)]"))
}

/// True for an attribute that marks a test ignored, in either form.
fn is_ignore(attr: &str) -> bool {
    attr.starts_with("#[ignore") || is_cfg_attr_ignore(attr)
}

/// True when `reason` reads `diverges from OpenCV <version>[ <qualifier>], see #N`: one
/// issue number, after a version that may be followed by a build qualifier such as
/// `with --features simd (panics)` that names no issue itself.
fn is_well_formed_reason(reason: &str) -> bool {
    reason
        .strip_prefix("diverges from OpenCV ")
        .and_then(|r| r.split_once(", see #"))
        .is_some_and(|(head, number)| {
            let version = head.split(' ').next().unwrap_or(head);
            is_version(version)
                && issue_references(head).is_empty()
                && !number.is_empty()
                && number.bytes().all(|b| b.is_ascii_digit())
        })
}

/// One `(line, message)` per ignore attribute in a parity suite that does not read
/// `#[ignore = "diverges from OpenCV <version>, see #N"]` with exactly one issue number.
/// The same reason inside `#[cfg_attr(<predicate>, ignore = "...")]`, on one line or split
/// by rustfmt, is accepted for a divergence of one build only, and the reason may carry a
/// qualifier after the version (`diverges from OpenCV 4.10.0 with --features simd, see #N`).
fn ignore_citation_violations(src: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for Attribute { start, text, .. } in attributes(src) {
        let attr = text.as_str();
        let reason = if attr.starts_with("#[ignore") {
            attr.strip_prefix("#[ignore = \"")
                .and_then(|r| r.strip_suffix("\"]"))
        } else if is_cfg_attr_ignore(attr) {
            attr.strip_suffix("\")]")
                .and_then(|r| r.split_once(", ignore = \""))
                .map(|(_, reason)| reason)
        } else {
            continue;
        };
        if !reason.is_some_and(is_well_formed_reason) {
            let what = match issue_references(attr).len() {
                0 => "cites no issue",
                1 => "is malformed",
                _ => "cites more than one issue",
            };
            out.push((start, format!("`{attr}` {what}: expected `{IGNORE_SHAPE}`")));
        }
    }
    out
}

/// One `(line, message)` per `current_behaviour` test whose doc comment does not name
/// exactly one issue, the one whose fix deletes the test.
fn pin_test_violations(src: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut doc = String::new();
    let mut attrs = attributes(src).into_iter().peekable();
    for (i, line) in src.lines().enumerate() {
        let line_no = i + 1;
        // Lines of an attribute, including the continuation lines of a split one, keep the
        // doc comment that precedes the attribute.
        if attrs
            .peek()
            .is_some_and(|a| (a.start..=a.end).contains(&line_no))
        {
            if attrs.peek().is_some_and(|a| a.end == line_no) {
                attrs.next();
            }
            continue;
        }
        let t = line.trim();
        if let Some(d) = t.strip_prefix("///") {
            doc.push_str(d);
            doc.push('\n');
            continue;
        }
        if let Some(rest) = t.strip_prefix("fn ") {
            let name = rest.split(['(', '<']).next().unwrap_or(rest);
            if name.contains("current_behaviour") {
                let what = match issue_references(&doc).len() {
                    1 => None,
                    0 => Some("names no issue"),
                    _ => Some("names more than one issue"),
                };
                if let Some(what) = what {
                    out.push((
                        i + 1,
                        format!(
                            "pin test `{name}` {what} in its doc comment: name the issue (#N) \
                             whose fix deletes it"
                        ),
                    ));
                }
            }
        }
        doc.clear();
    }
    out
}

/// The parity suites of the checkout, `tests/parity_*.rs`, sorted by name.
fn parity_suites() -> Vec<PathBuf> {
    let tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut out: Vec<PathBuf> = fs::read_dir(&tests)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", tests.display()))
        .map(|entry| entry.expect("readable tests/ entry").path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("parity_") && n.ends_with(".rs"))
        })
        .collect();
    out.sort();
    out
}

#[test]
fn issue_references_finds_bare_numbers_only() {
    assert_eq!(issue_references("see #12 and #3").len(), 2);
    assert_eq!(issue_references("C#1, ##2, #x, a#3, #").len(), 0);
    assert_eq!(issue_references("(#192)").len(), 1);
}

#[test]
fn ignore_citation_violations_require_exactly_one_issue() {
    let ok = "#[test]\n#[ignore = \"diverges from OpenCV 4.10.0, see #229\"]\nfn a() {}\n";
    assert_eq!(ignore_citation_violations(ok), Vec::new());
    let gated = "#[cfg_attr(not(feature = \"simd\"), ignore = \"diverges from OpenCV 4.10.0, \
                 see #208\")]\nfn a() {}\n";
    assert_eq!(ignore_citation_violations(gated), Vec::new());
    // Other cfg_attr attributes are not ignores.
    let cfg = "#[cfg_attr(feature = \"simd\", allow(dead_code))]\nfn a() {}\n";
    assert_eq!(ignore_citation_violations(cfg), Vec::new());
    // The forms rustfmt writes in the parity suites: split over lines, with or without a
    // build qualifier after the version.
    let split = "#[test]\n#[cfg_attr(\n    not(feature = \"simd\"),\n    \
                 ignore = \"diverges from OpenCV 4.10.0, see #208\"\n)]\nfn a() {}\n";
    assert_eq!(ignore_citation_violations(split), Vec::new());
    let qualified = "#[cfg_attr(\n    feature = \"simd\",\n    ignore = \"diverges from OpenCV \
                     4.10.0 with --features simd (panics), see #234\"\n)]\nfn a() {}\n";
    assert_eq!(ignore_citation_violations(qualified), Vec::new());
    let one_line_qualified =
        "#[ignore = \"diverges from OpenCV 4.10.0 with --features simd, see #235\"]\n";
    assert_eq!(ignore_citation_violations(one_line_qualified), Vec::new());

    // A split attribute is reported on the line it starts on.
    let bad_split = "fn before() {}\n\n#[test]\n#[cfg_attr(\n    not(feature = \"simd\"),\n    \
                     ignore = \"see #208\"\n)]\nfn t() {}\n";
    let v = ignore_citation_violations(bad_split);
    assert_eq!(v.len(), 1, "{v:?}");
    assert_eq!(v[0].0, 4);
    assert!(v[0].1.contains("is malformed"), "{}", v[0].1);
    // Module docs that mention the attribute are not attributes.
    let doc = "//! Tests marked `#[ignore]` document a `diverges` row.\n";
    assert_eq!(ignore_citation_violations(doc), Vec::new());

    let rows = [
        ("#[ignore]", "cites no issue"),
        (
            "#[ignore = \"diverges from OpenCV 4.10.0\"]",
            "cites no issue",
        ),
        (
            "#[ignore = \"diverges from OpenCV 4.10.0, see 229\"]",
            "cites no issue",
        ),
        (
            "#[ignore = \"diverges from OpenCV 4.10.0, see #\"]",
            "cites no issue",
        ),
        (
            "#[ignore = \"diverges from OpenCV 4.10.0, see #229 and #157\"]",
            "cites more than one issue",
        ),
        (
            "#[ignore = \"diverges from OpenCV 4.10.0, see #229, #157\"]",
            "cites more than one issue",
        ),
        ("#[ignore = \"see #229\"]", "is malformed"),
        (
            "#[ignore = \"diverges from OpenCV 4.x, see #229\"]",
            "is malformed",
        ),
        (
            "#[ignore = \"diverges from OpenCV 4.10.0, see #229 (ORB)\"]",
            "is malformed",
        ),
        (
            "#[ignore = \"diverges from OpenCV 4.10.0, see #229\"] // x",
            "is malformed",
        ),
        ("#[cfg_attr(not(feature = \"simd\"), ignore)]", "cites no issue"),
        (
            "#[ignore = \"diverges from OpenCV 4.10.0 (#157), see #235\"]",
            "cites more than one issue",
        ),
        (
            "#[ignore = \"diverges from OpenCV 4.x with --features simd, see #235\"]",
            "is malformed",
        ),
        (
            "#[cfg_attr(not(feature = \"simd\"), ignore = \"see #208\")]",
            "is malformed",
        ),
        (
            "#[cfg_attr(not(feature = \"simd\"), ignore = \"diverges from OpenCV 4.10.0, see #208 and #189\")]",
            "cites more than one issue",
        ),
    ];
    for (attr, what) in rows {
        let src = format!("fn before() {{}}\n\n#[test]\n{attr}\nfn t() {{}}\n");
        let v = ignore_citation_violations(&src);
        assert_eq!(v.len(), 1, "{attr}: {v:?}");
        assert_eq!(v[0].0, 4, "{attr}");
        assert!(v[0].1.contains(what), "{attr}: {}", v[0].1);
    }
}

#[test]
fn attributes_join_split_attributes_and_skip_brackets_in_strings() {
    let src = "#[test]\n#[cfg_attr(\n    not(feature = \"simd\"),\n    \
               ignore = \"diverges [sic], see #1\"\n)]\n#[doc = \"a ] b\"]\nfn a() {}\n";
    let attrs = attributes(src);
    let got: Vec<(usize, usize, &str)> = attrs
        .iter()
        .map(|a| (a.start, a.end, a.text.as_str()))
        .collect();
    assert_eq!(
        got,
        vec![
            (1, 1, "#[test]"),
            (
                2,
                5,
                "#[cfg_attr(not(feature = \"simd\"), ignore = \"diverges [sic], see #1\")]"
            ),
            (6, 6, "#[doc = \"a ] b\"]"),
        ]
    );
    assert_eq!(bracket_balance("#[a(\"]\")"), 1);
    assert_eq!(bracket_balance("\"\\\"]\""), 0);
    // An unterminated attribute ends with the file instead of looping.
    assert_eq!(attributes("#[cfg_attr(\n    x,\n").len(), 1);
}

#[test]
fn pin_test_violations_require_the_issue_in_the_doc_comment() {
    let ok = "/// Pins today's output (the fact recorded in #192) so that a fix is visible.\n\
              #[test]\nfn convert_to_current_behaviour_truncates() {\n}\n";
    assert_eq!(pin_test_violations(ok), Vec::new());
    // Tests without `current_behaviour` in their name are not pins.
    let other = "/// No issue here.\n#[test]\nfn convert_to_matches_opencv() {}\n";
    assert_eq!(pin_test_violations(other), Vec::new());
    // A split attribute between the doc comment and the test keeps the doc comment.
    let split = "/// Pinned for #208.\n#[test]\n#[cfg_attr(\n    feature = \"simd\",\n    \
                 ignore = \"x\"\n)]\nfn x_current_behaviour() {}\n";
    assert_eq!(pin_test_violations(split), Vec::new());

    let rows = [
        (
            "#[test]\nfn x_current_behaviour_is_zero() {}\n",
            2,
            "names no issue",
        ),
        (
            "/// The current behaviour.\n#[test]\nfn x_current_behaviour() {}\n",
            3,
            "names no issue",
        ),
        // A doc comment that belongs to the previous item does not carry over.
        (
            "/// See #209.\nfn helper() {}\n\n#[test]\nfn x_current_behaviour() {}\n",
            5,
            "names no issue",
        ),
        (
            "/// See #209 and #189.\n#[test]\nfn x_current_behaviour() {}\n",
            3,
            "more than one",
        ),
    ];
    for (src, line, what) in rows {
        let v = pin_test_violations(src);
        assert_eq!(v.len(), 1, "{src}: {v:?}");
        assert_eq!(v[0].0, line, "{src}");
        assert!(v[0].1.contains(what), "{src}: {}", v[0].1);
    }
}

#[test]
fn parity_ignores_cite_one_issue_and_pins_name_theirs() {
    let suites = parity_suites();
    assert!(
        !suites.is_empty(),
        "no tests/parity_*.rs found under {}",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut ignored = 0usize;
    let mut violations = Vec::new();
    for path in &suites {
        let src = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        ignored += attributes(&src)
            .iter()
            .filter(|a| is_ignore(&a.text))
            .count();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        for (line, msg) in ignore_citation_violations(&src)
            .into_iter()
            .chain(pin_test_violations(&src))
        {
            violations.push(format!("tests/{name}:{line}: {msg}"));
        }
    }
    assert!(ignored > 0, "no #[ignore] in the parity suites");
    assert!(
        violations.is_empty(),
        "every parity `#[ignore]` cites exactly one issue and every current-behaviour pin \
         names the issue it is deleted with (docs/parity/AGENTS.md, rule 11; \
         scripts/parity_xfail.py runs the ignored tests):\n{}",
        violations.join("\n")
    );
}
