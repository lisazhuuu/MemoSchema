#!/usr/bin/env bash
# memexp36 driver: downstream extraction with the AutoSchema Phase-II extractor (field names and descriptions), specificity
# judging and report.
set -u
cd "$(dirname "$0")/.."
L=results/memexp36; mkdir -p $L
for pass in 1 2; do python3 scripts/memexp36.py extract >> $L/driver.log 2>&1; done
for pass in 1 2; do python3 scripts/memexp36.py judge >> $L/driver.log 2>&1; done
python3 scripts/memexp36.py report >> $L/driver.log 2>&1
