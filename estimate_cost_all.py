#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute total input tokens for ArabicMMLU (All/dev.csv + All/test.csv) using tiktoken.
Optionally back-calculate total output tokens from a given spend (USD) and number of rows processed,
then project full-dataset cost using the implied average output tokens per row.

Usage examples:

# 1) Count input tokens for full dataset, and estimate outputs from your 2,153-row spend of 9.85 USD
python calc_tokens_arabicmmlu.py \
  --split all \
  --spent-usd 9.85 \
  --rows-processed 2153

# 2) Do the same but only over the first 2,153 rows when computing input tokens
python calc_tokens_arabicmmlu.py \
  --split all \
  --limit 2153 \
  --spent-usd 9.85 \
  --rows-processed 2153

# 3) If you used caching for part of the prompt
python calc_tokens_arabicmmlu.py \
  --split all \
  --spent-usd 9.85 \
  --rows-processed 2153 \
  --assume-cached-frac 0.2
"""

import json
import argparse
from pathlib import Path

from datasets import load_dataset
import tiktoken

# HF CSVs
DEV_URL = "https://huggingface.co/datasets/MBZUAI/ArabicMMLU/resolve/main/All/dev.csv"
TEST_URL = "https://huggingface.co/datasets/MBZUAI/ArabicMMLU/resolve/main/All/test.csv"

# GPT-5 Standard pricing (USD per 1M tokens)
DEFAULT_PRICE_INPUT = 1.25
DEFAULT_PRICE_OUTPUT = 10.0
DEFAULT_PRICE_INPUT_CACHED = 0.125

SYS_MSG = (
    "You are an expert educational assessor. "
    "Classify the cognitive skills required to answer a given question using Bloom's Taxonomy. "
    "Return ONLY a JSON object with exactly these keys and integer percentages that sum to 100: "
    "Remember, Understand, Apply, Analyze, Evaluate. "
    "Do not include any extra text."
)

def build_user_msg(q: str, options: list[str]) -> str:
    parts = [f"Question:\n{q.strip()}"]
    if options:
        joined = "\n".join([f"- {o}" for o in options if o and str(o).strip() and str(o).lower() != "null"])
        if joined.strip():
            parts.append(f"Options:\n{joined}")
    return "\n\n".join(parts) + "\n\nOutput JSON only with the five keys. Values must be integers in [0,100] and sum to 100."

def build_prompt_text(q: str, options: list[str]) -> str:
    return f"SYSTEM: {SYS_MSG}\nUSER: {build_user_msg(q, options)}"

def load_arabicmmlu(split: str):
    if split == "dev":
        return [("dev", load_dataset("csv", data_files={"dev": DEV_URL}, encoding="utf-8")["dev"])]
    if split == "test":
        return [("test", load_dataset("csv", data_files={"test": TEST_URL}, encoding="utf-8")["test"])]
    ds = load_dataset("csv", data_files={"dev": DEV_URL, "test": TEST_URL}, encoding="utf-8")
    return [("dev", ds["dev"]), ("test", ds["test"])]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process this many rows (across the chosen split order).")
    ap.add_argument("--spent-usd", type=float, default=None, help="Actual USD spent for rows-processed. Used to back-calc output tokens.")
    ap.add_argument("--rows-processed", type=int, default=None, help="Number of rows that produced the given spend.")
    ap.add_argument("--assume-cached-frac", type=float, default=0.0, help="Fraction [0..1] of input tokens considered cached.")
    ap.add_argument("--price-input", type=float, default=DEFAULT_PRICE_INPUT)
    ap.add_argument("--price-output", type=float, default=DEFAULT_PRICE_OUTPUT)
    ap.add_argument("--price-input-cached", type=float, default=DEFAULT_PRICE_INPUT_CACHED)
    args = ap.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")

    splits = load_arabicmmlu(args.split)

    # count input tokens
    total_rows = 0
    total_input_tokens = 0

    for split_name, dset in splits:
        for row in dset:
            q = str(row.get("Question", "") or "").strip()
            if not q:
                continue
            options = []
            for k in ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"]:
                v = row.get(k, "")
                if v is not None and str(v).strip() and str(v).lower() != "null":
                    options.append(str(v))
            prompt = build_prompt_text(q, options)
            in_tok = len(enc.encode(prompt))
            total_input_tokens += in_tok
            total_rows += 1
            if args.limit and total_rows >= args.limit:
                break
        if args.limit and total_rows >= args.limit:
            break

    # cost of input side for the measured set
    cached_frac = max(0.0, min(args.assume_cached_frac, 1.0))
    cached_in = int(total_input_tokens * cached_frac)
    noncached_in = total_input_tokens - cached_in

    input_cost_usd = (noncached_in * args.price_input / 1_000_000.0) + (cached_in * args.price_input_cached / 1_000_000.0)

    result = {
        "measured_rows": total_rows,
        "measured_input_tokens": total_input_tokens,
        "assumed_cached_input_tokens": cached_in,
        "avg_input_tokens_per_row": round(total_input_tokens / total_rows, 3) if total_rows else 0.0,
        "input_cost_usd_for_measured_set": round(input_cost_usd, 6),
        "prices": {
            "input_per_mtok": args.price_input,
            "input_cached_per_mtok": args.price_input_cached,
            "output_per_mtok": args.price_output
        }
    }

    # If user provided spend and rows-processed, back-calc outputs and project to full dataset
    if args.spent_usd is not None and args.rows_processed:
        # If the measured set equals the rows that produced the spend, use its input tokens.
        # If not, scale input tokens per row to the provided rows_processed.
        if total_rows == args.rows_processed:
            input_tokens_for_spend = total_input_tokens
            cached_for_spend = cached_in
        else:
            avg_in_per_row = total_input_tokens / max(total_rows, 1)
            input_tokens_for_spend = int(round(avg_in_per_row * args.rows_processed))
            cached_for_spend = int(round(input_tokens_for_spend * cached_frac))

        input_cost_for_spend = (
            (input_tokens_for_spend - cached_for_spend) * args.price_input / 1_000_000.0
            + cached_for_spend * args.price_input_cached / 1_000_000.0
        )
        remaining_usd = max(args.spent_usd - input_cost_for_spend, 0.0)
        # back-calc total output tokens that explain the remaining spend
        output_tokens_for_spend = int(round(remaining_usd / (args.price_output / 1_000_000.0)))
        avg_output_tokens_per_row = output_tokens_for_spend / max(args.rows_processed, 1)

        result.update({
            "given_spend_usd": round(args.spent_usd, 6),
            "rows_processed_for_spend": args.rows_processed,
            "implied_output_tokens_for_spend": output_tokens_for_spend,
            "implied_avg_output_tokens_per_row": round(avg_output_tokens_per_row, 3),
            "input_tokens_used_for_spend_calc": input_tokens_for_spend,
            "input_cost_usd_for_spend_calc": round(input_cost_for_spend, 6),
        })

        # If user measured a subset, project to full dataset under same avg output size
        # Load full count if we only measured a subset under --limit
        if args.split == "all":
            # Count total rows in full dataset quickly
            ds_all = load_arabicmmlu("all")
            full_rows = sum(len(ds) for _, ds in ds_all)
            avg_in_per_row = total_input_tokens / max(total_rows, 1)
            full_input_tokens = int(round(avg_in_per_row * full_rows))
            full_cached_in = int(round(full_input_tokens * cached_frac))
            full_input_cost = (
                (full_input_tokens - full_cached_in) * args.price_input / 1_000_000.0
                + full_cached_in * args.price_input_cached / 1_000_000.0
            )
            full_output_tokens = int(round(avg_output_tokens_per_row * full_rows))
            full_output_cost = full_output_tokens * args.price_output / 1_000_000.0
            full_total_cost = full_input_cost + full_output_cost

            result.update({
                "projection_full_dataset": {
                    "rows": full_rows,
                    "input_tokens": full_input_tokens,
                    "output_tokens": full_output_tokens,
                    "cost_input_usd": round(full_input_cost, 6),
                    "cost_output_usd": round(full_output_cost, 6),
                    "grand_total_usd": round(full_total_cost, 6),
                    "avg_cost_per_row_usd": round(full_total_cost / full_rows, 6) if full_rows else 0.0
                }
            })

    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
