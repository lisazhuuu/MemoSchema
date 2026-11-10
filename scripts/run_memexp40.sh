#!/usr/bin/env bash
# memexp40 driver: part 2 (SR_DRAFT) -> part 1 (IA_SR) -> judge (2 reps, fine + strict) -> report.
set -u
cd "$(dirname "$0")/.."
L=results/memexp40; mkdir -p $L
echo "[m40] part2 $(date)"; python3 scripts/memexp40.py part2 >> $L/part2.log 2>&1
echo "[m40] SR_DRAFT $(ls $L/runs/*/s*/SR_DRAFT/chem/*/final_schema.json | wc -l)/81; part1 $(date)"
python3 scripts/memexp40.py part1 >> $L/part1.log 2>&1
echo "[m40] IA_SR $(ls $L/runs/*/s*/IA_SR/chem/*/final_schema.json | wc -l)/81; judge $(date)"
# IA_GEN holds the initial schemas that IA_SR revises; only IA_SR and SR_DRAFT are judged.
for r in $(ls -d $L/runs/e2e*/s*); do ARMS="IA_SR SR_DRAFT" scripts/eval_fine.sh "$r" 2 8; ARMS="IA_SR SR_DRAFT" scripts/eval_replay.sh "$r" 2 8; done
python3 scripts/memexp40.py report > /dev/null
echo "[m40] DONE $(date)"
