CRATE = /target/debug
WHEEL = /target/wheels

.PHONEY: test

${CRATE}:
	cargo build

.venv:
	python -m venv .venv

${WHEEL}:
	maturin develop

test: ${WHEEL}
	python py/test.py
	cargo test