#!/bin/bash
set -e

trap 'echo "❌ Pre-commit failed at: $BASH_COMMAND"' ERR

clear

echo "🧪 Running pre-commit Rust checks..."

cargo fmt

cargo fmt --all --check

cargo clippy --all-targets --all-features -- -D warnings

cargo check --all-features

echo "✅ All checks passed. Good to go!"
