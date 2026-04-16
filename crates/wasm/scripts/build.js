#!/usr/bin/env node
/*
 *  build.js
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

// build.js
// Detects the OS and runs the appropriate dual build script.

const { execFileSync } = require('child_process');
const path = require('path');

const scriptsDir = __dirname;

try {
  if (process.platform === 'win32') {
    // Windows: run the PowerShell script
    const script = path.join(scriptsDir, 'build-dual.ps1');
    execFileSync('powershell.exe', ['-ExecutionPolicy', 'RemoteSigned', '-File', script], { stdio: 'inherit' });
  } else {
    // Linux / macOS: run the bash script
    const script = path.join(scriptsDir, 'build-dual.sh');
    execFileSync('bash', [script], { stdio: 'inherit' });
  }
} catch (err) {
  const scriptName = process.platform === 'win32' ? 'build-dual.ps1' : 'build-dual.sh';
  console.error(`Error: Failed to execute ${scriptName}.`);
  console.error(err.message);
  process.exit(1);
}
