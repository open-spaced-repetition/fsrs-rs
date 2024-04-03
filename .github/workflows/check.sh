#!/bin/bash

set -e

cargo fmt --check || (
    echo
    echo "Please run 'cargo fmt' to format the code."
    exit 1
)

cargo clippy -- -Dwarnings
