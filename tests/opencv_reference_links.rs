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
    ("docs/parity", "md"),
];

/// URL prefixes after which the next path segment names an OpenCV version.
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
    "raw.githubusercontent.com/opencv/opencv/",
    "raw.githubusercontent.com/opencv/opencv_contrib/",
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

/// One message per offending OpenCV reference on `line`, against the current reference.
fn line_violations(line: &str) -> Vec<String> {
    line_violations_for(line, REFERENCE)
}

/// One message per offending OpenCV reference on `line`, which must name `reference`.
fn line_violations_for(line: &str, reference: &str) -> Vec<String> {
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
            if segment != reference {
                out.push(format!(
                    "`{prefix}{segment}` names `{segment}`, expected `{reference}`"
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

/// A small recursive-descent JSON parser, so the guard needs no crate dependency. Any JSON
/// that Python's `json` module accepts parses to the same values, except that a `\u` escape
/// naming a surrogate is an error. It does not reject every input that Python rejects.
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
                self.at += self.text[self.at..]
                    .bytes()
                    .take_while(|&b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'))
                    .count();
                Ok(Json::Other)
            }
            _ => Err(format!("expected a value at byte {}", self.at)),
        }
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
                            let hex: String = chars.by_ref().take(4).map(|(_, h)| h).collect();
                            u32::from_str_radix(&hex, 16)
                                .ok()
                                .and_then(char::from_u32)
                                .ok_or_else(|| format!("unsupported escape `\\u{hex}`"))?
                        }
                        _ => return Err(format!("invalid escape `\\{e}`")),
                    });
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
    ];
    for (line, expected) in cases {
        assert_eq!(line_violations(line).len(), *expected, "line: {line}");
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
