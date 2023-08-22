CRATE = /target/debug
WHEEL = .venv/lib/python3.11/site-packages/fsrs_optimizer_rust
RUST = ${wildcard src/*.rs}

.PHONEY: test

${CRATE}: ${RUST}
	cargo build

.venv:
	python -m venv .venv

${WHEEL}: .venv ${RUST}
	maturin develop

test: ${WHEEL}
	.venv/bin/python py/test.py
	cargo test