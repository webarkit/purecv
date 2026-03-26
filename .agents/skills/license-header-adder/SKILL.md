---
name: license-header-adder
description: Automatically adds a license header to source files with a filename placeholder.
---

# License Header Adder Skill

This skill allows you to add a consistent LGPLv3 license header to all source files in the project.

## Instructions

1.  **Read the Template**: Locate the license template in `resources/HEADER.txt`.
2.  **Placeholder Replacement**: Before applying the header to a file, replace the `{{FILENAME}}` placeholder with the basename of the target file.
3.  **Supported Files**: apply this header to:
    - `.rs` files
    - `.c` and `.h` files
    - `.js` and `.ts` files
4.  **Application Logic**:
    - If the file already has a license header, skip it.
    - **Exclusions**: Never apply this to files in `benchmarks/c_benchmark/src/WebARKitLib` or third-party vendor directories.
    - Insert the header at the very top of the file.
    - Ensure there is a blank line after the header before the original content starts.

## Example Usage

When applying to `core/src/lib.rs`:
```rust
/*
 * lib.rs
 * 
 * WebARKitLib.rs - High-performance AR tracking library
 ...
 */

use wasm_bindgen::prelude::*;
...
```
