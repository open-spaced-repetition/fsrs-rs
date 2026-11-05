#!/bin/bash
# Prints a Markdown table that compares Criterion's `base` baseline with the
# latest run (`new`), using the median of each benchmark.
#
# Usage: bench-report.sh [criterion dir] [threshold in %]

set -euo pipefail

dir=${1:-target/criterion}
threshold=${2:-10}

# Formats a duration in nanoseconds.
fmt() {
  awk -v ns="$1" 'BEGIN {
    if (ns >= 1e9) printf "%.3f s", ns / 1e9
    else if (ns >= 1e6) printf "%.2f ms", ns / 1e6
    else if (ns >= 1e3) printf "%.2f µs", ns / 1e3
    else printf "%.1f ns", ns
  }'
}

median() {
  jq '.median.point_estimate' "$1"
}

benches=
if [ -d "$dir" ]; then
  benches=$(find "$dir" -path '*/base/estimates.json' -o -path '*/new/estimates.json' |
    sed 's|/[^/]*/estimates.json$||' | sort -u)
fi
if [ -z "$benches" ]; then
  echo "No benchmark results found."
  exit 0
fi

echo "| Benchmark | Base | Head | Change |"
echo "| --- | ---: | ---: | ---: |"
while IFS= read -r bench; do
  base="$bench/base/estimates.json"
  new="$bench/new/estimates.json"
  if [ -f "$new" ]; then
    id=$(jq -r '.full_id' "$bench/new/benchmark.json")
  else
    id=$(jq -r '.full_id' "$bench/base/benchmark.json")
  fi
  if [ -f "$base" ] && [ -f "$new" ]; then
    b=$(median "$base")
    h=$(median "$new")
    change=$(awk -v b="$b" -v h="$h" -v t="$threshold" 'BEGIN {
      pct = (h - b) / b * 100
      mark = pct > t ? " :red_circle:" : (pct < -t ? " :green_circle:" : "")
      printf "%+.1f%%%s", pct, mark
    }')
    echo "| \`$id\` | $(fmt "$b") | $(fmt "$h") | $change |"
  elif [ -f "$new" ]; then
    echo "| \`$id\` | – | $(fmt "$(median "$new")") | new |"
  else
    echo "| \`$id\` | $(fmt "$(median "$base")") | – | removed |"
  fi
done <<<"$benches"

echo
echo "Medians from Criterion. :red_circle: slower or :green_circle: faster by" \
  "more than ${threshold}%; smaller changes are within the noise of shared" \
  "GitHub-hosted runners."
