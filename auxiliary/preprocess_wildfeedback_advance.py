# preprocess_wildfeedback_advance.py
#
# For each conversation in WildFeedback, find the matching conversation in
# WildChat (allenai/WildChat) using content hashing on the first assistant
# response (robust to conversations that are prefixes of longer WildChat
# conversations).  Then use WildChat's last (gpt, human) pair as the
# `completion` and `user_response` fields, building the prompt from the
# WildChat messages that precede that last pair.

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import re
from datasets import load_dataset

# --- Configuration ---
MAX_TOTAL_CONV_LENGTH = 100_000    # Drop if whole WildChat conversation > this many chars
MAX_COMPLETION_LENGTH = 4_096      # Drop if the chosen completion > this many chars
MAX_HISTORY_MESSAGES = 5           # Max number of messages in prompt history
OUTPUT_FILENAME = "wildfeedback_advanced_interactions.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_wildchat(raw_conv: list) -> list:
    """Convert WildChat role/content messages → from/value format, dropping empties."""
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


def normalize_wildfeedback(raw_conv: list) -> list:
    """Strip empty messages from WildFeedback conversations (already from/value)."""
    return [
        m for m in raw_conv
        if (m.get("value") or "").strip() and m.get("from") in {"human", "gpt"}
    ]


# def _hash_text(text: str) -> str:
#     return hashlib.sha256(text.encode("utf-8")).hexdigest()
def _hash_text(text: str) -> str:
    # 1. Lowercase to ensure 'A' == 'a'
    # text = text.lower()
    # 2. Keep only alphanumeric characters (\W matches any non-word character)
    # clean_text = re.sub(r'[^a-zA-Z0-9]', '', text)
    clean_text = text
    # print(f"Hashing text (length {len(text)} → {len(clean_text)} after cleaning): {text[:50]}... → {clean_text[:50]}...")
    result = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
    # print(f"Hash result: {result}")
    return result

def first_gpt_hash(conv: list) -> str | None:
    """Hash of the first gpt/assistant message value — used for matching."""
    for m in conv:
        if m.get("from") == "gpt":
            return _hash_text(m["value"])
    return None


def full_conv_hash(conv: list) -> str:
    """Hash of all message values concatenated (secondary matching key)."""
    # print(f"len(conv) = {len(conv)}")
    text = "\n---\n".join(m["value"] for m in conv) # use only first two messages for full hash since WF conversations are often missing the last (gpt, human) pair from the WC conversations
    return _hash_text(text)


def truncate_history_starting_with_human(history: list, max_messages: int) -> list | None:
    """
    Truncate history to at most `max_messages` messages such that:
    - starts with a human message
    - roles alternate
    - empty messages are already removed
    Returns None if no valid history remains.
    """
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


def find_last_gpt_human_pair(conv: list) -> int | None:
    """
    Return the index i such that conv[i] is gpt and conv[i+1] is human,
    scanning from the end.  Returns None if no such pair exists.
    """
    for i in range(len(conv) - 2, -1, -1):
        if conv[i]["from"] == "gpt" and conv[i + 1]["from"] == "human":
            return i
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_dir", type=str, default=None,
        help="Directory to write output JSONL. Defaults to <repo_root>/data/wildfeedback",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Debug mode: stop after 10 matched entries and save as CSV instead of JSONL.",
    )
    return p.parse_args()


def main() -> None:
    cli = parse_args()

    # ------------------------------------------------------------------
    # 1. Build WildChat lookup indices
    # ------------------------------------------------------------------
    print("Loading WildChat (allenai/WildChat) …")
    wc_ds = load_dataset("allenai/WildChat", split="train")

    # Primary index: first-gpt-message hash → list of (norm_conv, row_meta)
    # (list because collisions are theoretically possible)
    first_gpt_index: dict[str, list] = {}
    # Secondary index: full-conv hash → (norm_conv, row_meta) for exact-match tie-breaking
    full_hash_index: dict[str, tuple] = {}

    error_counter_full_hash = 0
    print(f"Indexing {len(wc_ds):,} WildChat conversations …")
    for row in wc_ds:
        raw_conv = row.get("conversation") or []
        norm = normalize_wildchat(raw_conv)
        if len(norm) <= 2:
            continue

        fgh = first_gpt_hash(norm)
        if fgh is None:
            continue

        meta = {
            "conversation_id": row.get("conversation_id"),
            "model": row.get("model"),
            "timestamp": str(row.get("timestamp")) if row.get("timestamp") is not None else None,
            "language": row.get("language"),
        }
        entry = (norm, meta)

        first_gpt_index.setdefault(fgh, []).append(entry)
        # full hash should avoid the last two messages since they are missing from the wilfeedback convs
        fch = full_conv_hash(norm[:-2])
        if fch in full_hash_index:
            error_counter_full_hash += 1
            # print(f"Warning: full-conv hash collision for conversation_id {meta['conversation_id']} (hash {fch})")
            # print(norm[:-2])
        full_hash_index[fch] = entry

    print(f"Indexed {len(first_gpt_index):,} unique first-gpt hashes.")
    print(f"Indexed {len(full_hash_index):,} unique full-conv hashes (using conv[:-2] for hashing).")
    print(f"Full-conv hash collisions: {error_counter_full_hash}")

    # ------------------------------------------------------------------
    # 2. Process WildFeedback
    # ------------------------------------------------------------------
    print("Loading WildFeedback (microsoft/WildFeedback) …")
    wf_ds = load_dataset("microsoft/WildFeedback", "wildfeedback", split="train")

    stats = {
        "total_wf_conversations": 0,
        "skipped_too_short": 0,
        "skipped_no_match": 0,
        "skipped_no_last_pair": 0,
        "skipped_total_too_long": 0,
        "skipped_completion_too_long": 0,
        "skipped_no_valid_history": 0,
        "kept": 0,
    }

    processed_data = []

    for original_idx, row in enumerate(wf_ds):
        stats["total_wf_conversations"] += 1

        conversation = row.get("conversations") or row.get("conversation")
        if not conversation:  # we don't remove conv < 2 messages until after matching to WildChat, but we can skip empty convs early
            stats["skipped_too_short"] += 1
            continue

        norm_wf = normalize_wildfeedback(conversation)

        # ----------------------------------------------------------
        # 2a. Match to WildChat via first-gpt hash
        # ----------------------------------------------------------
        fgh = first_gpt_hash(norm_wf)
        if fgh is None:
            stats["skipped_no_match"] += 1
            continue

        candidates = first_gpt_index.get(fgh, [])
        if not candidates:
            stats["skipped_no_match"] += 1
            continue

        # If multiple candidates, prefer exact full-conv match
        wc_conv, wc_meta = None, None
        if len(candidates) == 1:
            wc_conv, wc_meta = candidates[0]
        else:
            print(f"Warning: {len(candidates)} candidates for WF idx {original_idx} (first-gpt hash {fgh[:8]})")
            fch = full_conv_hash(norm_wf)
            if fch in full_hash_index:
                wc_conv, wc_meta = full_hash_index[fch]
            else:
                # Fall back to first candidate (same first-gpt message)
                wc_conv, wc_meta = candidates[0]
                print(f"  No full-conv hash match for WF idx {original_idx}, using first candidate with WC conversation_id {wc_meta['conversation_id']}")

        # ----------------------------------------------------------
        # 2b. Extract last (gpt, human) pair from WildChat
        # ----------------------------------------------------------
        last_gpt_idx = find_last_gpt_human_pair(wc_conv)
        if last_gpt_idx is None:
            stats["skipped_no_last_pair"] += 1
            continue

        completion = wc_conv[last_gpt_idx]
        user_response = wc_conv[last_gpt_idx + 1]
        history_full = wc_conv[:last_gpt_idx]

        # ----------------------------------------------------------
        # 2c. Filtering
        # ----------------------------------------------------------
        full_text_length = sum(len(m["value"]) for m in wc_conv)
        if full_text_length > MAX_TOTAL_CONV_LENGTH:
            stats["skipped_total_too_long"] += 1
            continue

        if len(completion["value"]) > MAX_COMPLETION_LENGTH:
            stats["skipped_completion_too_long"] += 1
            continue

        prompt_history = truncate_history_starting_with_human(history_full, MAX_HISTORY_MESSAGES)
        if prompt_history is None:
            stats["skipped_no_valid_history"] += 1
            continue

        # ----------------------------------------------------------
        # 2d. Build output entry
        # ----------------------------------------------------------
        entry = {
            "id": str(original_idx),
            "wf_original_idx": original_idx,
            "wc_conversation_id": wc_meta["conversation_id"],
            "wc_model": wc_meta["model"],
            "wc_timestamp": wc_meta["timestamp"],
            "wc_language": wc_meta["language"],
            "prompt": prompt_history,
            "completion": completion,
            "user_response": user_response,
            "prompt_len": len(prompt_history),
            "completion_len": len(completion["value"]),
            "user_response_len": len(user_response["value"]),
            "total_conv_len": full_text_length,
        }
        processed_data.append(entry)
        stats["kept"] += 1

        if cli.debug and stats["kept"] >= 10:
            break

    # ------------------------------------------------------------------
    # 3. Save
    # ------------------------------------------------------------------
    if cli.out_dir is not None:
        data_dir = Path(cli.out_dir).resolve()
    else:
        data_dir = (Path(__file__).parent.parent / "data" / "wildfeedback").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Processing Statistics ===")
    for k, v in stats.items():
        print(f"  {k:35s} {v}")
    print("================================\n")

    if processed_data:
        df = pd.DataFrame(processed_data)
        if cli.debug:
            output_path = data_dir / "wildfeedback_advanced_debug.jsonl"
            df.to_json(output_path, orient="records", lines=True, force_ascii=False)
        else:
            output_path = data_dir / OUTPUT_FILENAME
            df.to_json(output_path, orient="records", lines=True, force_ascii=False)
        print(f"Saved {len(processed_data):,} entries to: {output_path}")
    else:
        print("Warning: No data remained after filtering!")


if __name__ == "__main__":
    main()
