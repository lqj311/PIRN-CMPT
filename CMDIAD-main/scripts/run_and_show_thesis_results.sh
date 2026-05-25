#!/usr/bin/env bash
set -euo pipefail

# Run final thesis experiments, run full-shot upper bound, then show one merged result table.
#
# Usage:
#   cd /root/autodl-tmp/PIRN-CMPT/CMDIAD-main
#   bash scripts/run_and_show_thesis_results.sh
#
# Skip already-finished parts:
#   SKIP_FINAL=1 bash scripts/run_and_show_thesis_results.sh
#   SKIP_FULLSHOT=1 bash scripts/run_and_show_thesis_results.sh

if [[ "${SKIP_FINAL:-0}" != "1" ]]; then
  bash scripts/run_thesis_final_15configs.sh
fi

if [[ "${SKIP_FULLSHOT:-0}" != "1" ]]; then
  bash scripts/run_fullshot_upper_bound.sh
fi

python scripts/show_thesis_results.py --filter final_
