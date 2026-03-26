---
description: Apply LGPLv3 license headers to src files (.rs, .js, .ts)
---

This workflow automates the application of the project's license header to source files using the `license-header-adder` skill.

1. **Identify Target Files**: Search for files with extensions `.rs`, `.js`, or `.ts` exclusively within the `src/`, `benches/` and `examples/` directory.
   - **Exclusions**: Do not process files in `benchmarks/c_benchmark/src/WebARKitLib` or any third-party vendor directories.

2. **Check for Existing Headers**: For each file, check if it already contains the license header (look for the string "purecv is free software"). If it exists, skip the file.

3. **Apply the Header**:
   - Invoke the `license-header-adder` skill.
   - Ensure the `{{FILENAME}}` placeholder is replaced with the actual basename of the file.
   - Insert the header at the extreme top of the file, followed by a blank line.

4. **Verification**: Confirm that the headers are correctly applied and that no files were corrupted or doubled-up on headers.