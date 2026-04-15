from __future__ import annotations

import os
import json
import time
import glob
import argparse
from datetime import timedelta
from typing import Dict, List, Any

import torch
from datasets import load_dataset, Dataset

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser(description="Generate model outputs for pairwise eval (single model, no judge)")

    # Dataset
    p.add_argument("--eval_n", type=int, default=805,
                   help="Number of examples to use (alpaca_eval has 805 total)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_prompt_tokens_filter", type=int, default=512)

    # Model
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--tokenizer_name_or_path", type=str, default=None)

    # Generation
    p.add_argument("--max_input_tokens", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)

    # Output
    p.add_argument("--out_dir", type=str, default="model_outputs")
    p.add_argument("--run_name", type=str, required=True)

    p.add_argument("--system_prompt", type=str, default="")

    return p.parse_args()



def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_get_prompt(example: Dict[str, Any]) -> str:
    for k in ("instruction", "prompt", "text", "article", "document", "source"):
        if k in example and isinstance(example[k], str) and example[k].strip():
            return example[k]
    msgs = example.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        c = msgs[0].get("content", "")
        if isinstance(c, str):
            return c
    return json.dumps(example, ensure_ascii=False)[:5000]


def build_messages(system_prompt: str, user_text: str) -> List[Dict[str, str]]:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def format_messages(tokenizer, messages: List[Dict[str, str]]) -> str:
    if getattr(tokenizer, "apply_chat_template", None) is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return messages[-1]["content"] + "\n\nAssistant:"


def load_and_prepare_eval_ds(
    accelerator: Accelerator,
    eval_n: int,
    seed: int,
    max_prompt_tokens: int,
    filter_tokenizer_name_or_path: str,
    system_prompt: str,
) -> Dataset:
    """
    Returns the alpaca_eval dataset (subsampled + filtered) with:
      - global_idx
      - raw_prompt
      - messages
    """
    with accelerator.main_process_first():
        eval_ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]

        eval_ds = eval_ds.map(lambda x, idx: {"global_idx": idx}, with_indices=True)
        eval_ds = eval_ds.shuffle(seed=seed).select(range(min(eval_n, len(eval_ds))))

        def add_raw_prompt(ex):
            raw = safe_get_prompt(ex)
            return {"raw_prompt": raw}

        eval_ds = eval_ds.map(add_raw_prompt)

        def add_messages(ex):
            return {"messages": build_messages(system_prompt, ex["raw_prompt"])}

        eval_ds = eval_ds.map(add_messages)

        tok = AutoTokenizer.from_pretrained(filter_tokenizer_name_or_path, trust_remote_code=True)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        def add_len(ex):
            text = format_messages(tok, ex["messages"])
            ids = tok(text, add_special_tokens=False).input_ids
            return {"lengths": len(ids)}

        eval_ds = eval_ds.map(add_len)
        eval_ds = eval_ds.filter(lambda l: l <= max_prompt_tokens, input_columns="lengths").remove_columns("lengths")

    return eval_ds


@torch.no_grad()
def generate_for_dataset(
    accelerator: Accelerator,
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    local_ds: Dataset,
    max_input_tokens: int,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
) -> Dict[int, str]:
    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(accelerator.device)
    model.eval()

    outputs: Dict[int, str] = {}

    for start in range(0, len(local_ds), batch_size):
        batch = local_ds.select(range(start, min(len(local_ds), start + batch_size)))
        messages = [ex["messages"] for ex in batch]
        prompts = [format_messages(tok, m) for m in messages]

        if start == 0 and accelerator.is_main_process:
            print(f"\n\nPrompt Example:", prompts[:1], flush=True)

        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
            add_special_tokens=False,
        ).to(accelerator.device)

        gen_kwargs = dict(
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen_kwargs["do_sample"] = temperature > 0.0
            if temperature > 0.0:
                gen_kwargs["temperature"] = temperature
                if top_p is not None:
                    gen_kwargs["top_p"] = top_p

        gen = model.generate(**enc, **gen_kwargs)

        base_len = enc["input_ids"].shape[1]
        for i, ex in enumerate(batch):
            glb = int(ex["global_idx"])
            gen_ids = gen[i, base_len:]
            out = tok.decode(gen_ids, skip_special_tokens=True).strip()
            outputs[glb] = out

    del model
    torch.cuda.empty_cache()

    return outputs


def main():
    args = parse_args()

    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(kwargs_handlers=[pg_kwargs])

    rank = accelerator.process_index
    world = accelerator.num_processes
    local_rank = accelerator.local_process_index

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.run_name}.json")
    part_path = out_path.replace(".json", f".rank{rank}.jsonl")

    if accelerator.is_main_process:
        print(f"[{now_ts()}] world_size={world}", flush=True)
        print(f"[{now_ts()}] out_path={out_path}", flush=True)

    tok_path = args.tokenizer_name_or_path or args.model_name_or_path

    eval_ds = load_and_prepare_eval_ds(
        accelerator=accelerator,
        eval_n=args.eval_n,
        seed=args.seed,
        max_prompt_tokens=args.max_prompt_tokens_filter,
        filter_tokenizer_name_or_path=tok_path,
        system_prompt=args.system_prompt,
    )

    indices = list(range(rank, len(eval_ds), world))
    local_ds = eval_ds.select(indices)

    if accelerator.is_main_process:
        print(f"[{now_ts()}] eval_ds={len(eval_ds)} local_shard~{len(local_ds)}", flush=True)

    outputs = generate_for_dataset(
        accelerator=accelerator,
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=tok_path,
        local_ds=local_ds,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Write per-rank JSONL
    with open(part_path, "w") as f:
        for ex in local_ds:
            glb = int(ex["global_idx"])
            f.write(
                json.dumps(
                    {
                        "global_idx": glb,
                        "dataset": ex.get("dataset", ""),
                        "instruction": ex["raw_prompt"],
                        "raw_prompt": ex["raw_prompt"],
                        "output": outputs[glb],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    accelerator.wait_for_everyone()

    # Merge rank files on main process
    if accelerator.is_main_process:
        rows = []
        for pf in sorted(glob.glob(out_path.replace(".json", ".rank*.jsonl"))):
            with open(pf, "r") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))

        rows.sort(key=lambda r: r["global_idx"])

        examples = [
            {
                "dataset": r["dataset"],
                "instruction": r["instruction"],
                "raw_prompt": r["raw_prompt"],
                "output": r["output"],
                "generator": args.run_name,
            }
            for r in rows
        ]

        meta = {
            "run_name": args.run_name,
            "timestamp": now_ts(),
            "dataset": "tatsu-lab/alpaca_eval",
            "t": args.eval_n,
            "eval_n_actual": len(rows),
            "seed": args.seed,
            "max_prompt_tokens_filter": args.max_prompt_tokens_filter,
            "model": args.model_name_or_path,
            "gen": {
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
        }

        report = [{"meta": meta}] + examples

        with open(out_path, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"[{now_ts()}] DONE  saved {len(rows)} examples to {out_path}", flush=True)

        for pf in glob.glob(out_path.replace(".json", ".rank*.jsonl")):
            try:
                os.remove(pf)
            except OSError:
                pass

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
