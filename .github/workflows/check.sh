#!/bin/bash

set -eux -o pipefail

cargo fmt --check

cargo clippy -- -Dwarnings

install -d tests/data/
pushd tests/data/
archive_url="https://github.com/open-spaced-repetition/fsrs-optimizer-burn/files/12394182/collection.anki21.zip"
archive_path="collection.anki21.zip"
if command -v wget >/dev/null 2>&1; then
  wget "${archive_url}" -O "${archive_path}"
else
  curl -fsSL "${archive_url}" -o "${archive_path}"
fi
unzip -o "${archive_path}"

RUSTDOCFLAGS="-D warnings" cargo doc --release

SKIP_TRAINING=1 cargo test --release
