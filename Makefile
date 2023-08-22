CRATE = /target/debug
WHEEL = .venv/lib/python3.11/site-packages/fsrs_optimizer_rust

.PHONEY: test

${CRATE}:
	cargo build

.venv:
	python -m venv .venv

${WHEEL}: .venv
	maturin develop

test: ${WHEEL}
	.venv/bin/python py/test.py
	cargo test