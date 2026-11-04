import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { WASI } from 'node:wasi';

const wasmPath = process.argv[2];
const wasi = new WASI({
  version: 'preview1',
  args: [wasmPath, ...process.argv.slice(3), '--test-threads=1'],
  env: process.env,
  preopens: { '.': process.cwd() },
  returnOnExit: true,
});
const module = new WebAssembly.Module(readFileSync(wasmPath));
const instance = new WebAssembly.Instance(module, wasi.getImportObject());
assert.equal(wasi.start(instance), 0, 'WASI tests failed');
