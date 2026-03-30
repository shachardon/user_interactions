# preprocess_nof.py

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# --- Configuration ---
DATASET_NAME = "shachardon/ShareLM"
SPLIT = "train"
OUTPUT_FILENAME = "nof_interactions.jsonl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default=None,
                   help="Directory to write output JSONL. Defaults to <repo_root>/data/nof")
    p.add_argument("--extraction_df", type=str,
                   default="/cs/labs/oabend/shachar.don/repo/naturally_occurring_feedback/model_responses/"
                           "Qwen_Qwen3-30B-A3B-Instruct-2507_/2026-01-18_17:51:01.895434/"
                           "merged_feedback_gpt-4.1-mini-2025-04-14_English.csv")
    p.add_argument("--debug", action="store_true",
                   help="Debug mode: limit extraction_df to 10 rows")
    return p.parse_args()


def normalize_conversation(raw_conv):
    """Convert WildChat messages (role/content) to (from/value), dropping empty/unknown roles."""
    out = []
    for m in raw_conv:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            out.append({"from": "human", "value": content})
        elif role == "assistant":
            out.append({"from": "gpt", "value": content})
    return out


def truncate_history_starting_with_human(history, max_messages=5):
    if not history:
        return None

    truncated = history[-max_messages:]

    while truncated and truncated[0]["from"] != "human":
        truncated = truncated[1:]

    if not truncated:
        return None

    normalized = [truncated[0]]
    for m in truncated[1:]:
        if m["from"] != normalized[-1]["from"]:
            normalized.append(m)

    if not normalized or normalized[0]["from"] != "human":
        return None

    return normalized


if __name__ == "__main__":
    cli = parse_args()

    print("Loading extraction DataFrame...")
    extraction_df = pd.read_csv(cli.extraction_df)
    if cli.debug:
        extraction_df = extraction_df.head(10)
        print("  [DEBUG] Limiting to 10 rows")

    # Build lookup: conversation_id -> list of extraction rows
    conv_lookup = defaultdict(list)
    for _, exrow in extraction_df.iterrows():
        conv_lookup[exrow["conversation_id"]].append(exrow)

    print(f"  Unique conversation IDs in extraction_df: {len(conv_lookup)}")
    print(f"  Total extraction rows: {len(extraction_df)}")

    print(f"Loading dataset {DATASET_NAME} [{SPLIT}] ...")
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    if cli.out_dir is not None:
        data_dir = Path(cli.out_dir).resolve()
    else:
        data_dir = (Path(__file__).parent.parent / "data" / "nof").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / OUTPUT_FILENAME

    stats = {
        "total_conversations_scanned": 0,
        "matched_conversations": 0,
        "kept_interactions": 0,
        "skipped_turn_out_of_range": 0,
        "skipped_wrong_roles": 0,
        "skipped_no_valid_history": 0,
    }

    with output_path.open("w", encoding="utf-8") as f:
        for row in ds:
            stats["total_conversations_scanned"] += 1

            conv_id = row.get("conversation_id")
            if conv_id not in conv_lookup:
                continue

            stats["matched_conversations"] += 1

            raw_conv = row.get("conversation") or []
            norm_conv = normalize_conversation(raw_conv)

            for exrow in conv_lookup[conv_id]:
                feedback_turn = int(exrow["feedback_turn"])
                turn_id = feedback_turn - 1  # index of the completion (GPT message)

                # Validate indices
                if turn_id < 0 or turn_id + 1 >= len(norm_conv):
                    stats["skipped_turn_out_of_range"] += 1
                    continue

                completion = norm_conv[turn_id]
                user_response = norm_conv[turn_id + 1]

                # Validate roles
                if completion["from"] != "gpt" or user_response["from"] != "human":
                    stats["skipped_wrong_roles"] += 1
                    continue

                history_full = norm_conv[:turn_id]
                prompt_history = truncate_history_starting_with_human(history_full)

                if prompt_history is None or prompt_history[-1]["from"] != "human":
                    stats["skipped_no_valid_history"] += 1
                    continue

                entry = {
                    "id": f"{conv_id}_{turn_id}",
                    "conversation_id": conv_id,
                    "turn_id": turn_id,
                    "feedback_turn": feedback_turn,
                    "feedback_category": exrow.get("feedback_category"),
                    "feedback_text": exrow.get("feedback_text"),
                    "model": exrow.get("model"),
                    "language": exrow.get("language"),
                    "prompt": prompt_history,
                    "completion": completion,
                    "user_response": user_response,
                    "prompt_len": len(prompt_history),
                    "completion_len": len(completion["value"]),
                    "user_response_len": len(user_response["value"]),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                stats["kept_interactions"] += 1

    print("\n=== Processing Statistics ===")
    for k, v in stats.items():
        print(f"{k:35s} {v}")
    print("============================\n")
    print(f"Saved to: {output_path}")
