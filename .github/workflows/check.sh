#!/bin/bash

set -eux -o pipefail

cargo fmt --check

cargo clippy -- -Dwarnings

install -d tests/data/
pushd tests/data/
wget https://github.com/open-spaced-repetition/fsrs-optimizer-burn/files/12394182/collection.anki21.zip
unzip *.zip

RUSTDOCFLAGS="-D warnings" cargo doc --release

SKIP_TRAINING=1 cargo test --release
