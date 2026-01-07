#!/bin/bash
set -e

trap 'echo "❌ Pre-commit failed at: $BASH_COMMAND"' ERR

clear

echo "🧪 Running Rust build, clippy and format checks..."

cargo fmt

cargo fmt --all --check

cargo clippy --all-targets --all-features -- -D warnings

cargo build --all-features --release

cargo llvm-cov --all-features --html -- --test-threads=1

echo "✅ All checks passed. Good to go!"
