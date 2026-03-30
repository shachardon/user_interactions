# Offline SDPO from User Interactions
import os
from xml.parsers.expat import model
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from offline_sdpo_trainer import OfflineSDPOCollator, OfflineSDPOTrainer
import argparse


PROBE_INSTRUCTION = "How did US states get their names?"


class ProbeGenerationCallback(TrainerCallback):
    """Generates a fixed probe response every `every_n_steps` and appends it to a file."""

    def __init__(self, tokenizer, output_file, every_n_steps=200):
        self.tokenizer = tokenizer
        self.output_file = output_file
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return

        messages = [{"role": "user", "content": PROBE_INSTRUCTION}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        device = torch.device(f"cuda:{args.local_rank}" if args.local_rank >= 0 else "cuda")
        inputs = self.tokenizer(input_text, return_tensors="pt").to(device)

        was_training = model.training
        model.eval()
        use_cache_before = model.config.use_cache
        model.config.use_cache = True  # required for generation

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        model.config.use_cache = use_cache_before
        if was_training:
            model.train()

        # Only the main process writes to disk
        if args.process_index == 0:
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            with open(self.output_file, "a") as f:
                f.write(f"=== Step {state.global_step} ===\n{response}\n\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--learning_rate", type=float, default=2e-6)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--train_jsonl", type=str, required=True,
                   help="Path to training JSONL (e.g. wildfeedback_interactions.jsonl)")
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to checkpoint dir, or 'true' to resume from latest in output_dir")
    p.add_argument("--probe_every_n_steps", type=int, default=200,
                   help="Generate a probe response every N training steps (0 to disable)")
    return p.parse_args()


def main():
    args = parse_args()

    model_name_or_path = args.base_model
    output_dir = os.environ.get("OUTPUT_DIR", "./local_checkpoints")

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    grad_accum = args.grad_accum
    max_completion_len = 2048
    num_epochs = args.num_epochs

    print("Config:")
    print(f"Model:      {model_name_or_path}")
    print(f"Output dir: {output_dir}")
    print(f"Train JSONL:{args.train_jsonl}")
    print(f"LR:         {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Grad accum: {grad_accum}")
    print(f"Epochs:     {num_epochs}")

    print(f"Loading data from {args.train_jsonl}...")
    dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    train_dataset = dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    # Not used for generation in offline training, but required by some model configs
    model.generation_config.do_sample = True
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0

    # Enable gradient checkpointing and disable caching for training to save memory
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # IMPORTANT

    # for Llama as they don't have a pad token by default
    if model_name_or_path == "meta-llama/Meta-Llama-3.1-8B-Instruct" and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    print("pad_token_id:", tokenizer.pad_token_id, "pad_token:", tokenizer.pad_token)
    print("eos_token_id:", tokenizer.eos_token_id, "eos_token:", tokenizer.eos_token)

    collator = OfflineSDPOCollator(
        tokenizer=tokenizer,
        max_completion_length=max_completion_len
    )

    # --- ADDED: DEEPSPEED DETECTION LOGIC ---
    # Check if DeepSpeed is actually being used in this run
    is_deepspeed_run = (
        os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true" or 
        "DEEPSPEED_CONFIG" in os.environ
    )
    
    # Select optimizer based on DeepSpeed presence
    if is_deepspeed_run:
        print("DeepSpeed detected: Allowing DeepSpeed config to manage optimizer.")
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=num_epochs,

            fp16=False,
            bf16=True,
            gradient_checkpointing=True,

            logging_steps=10,
            save_strategy="steps",
            save_steps=200,
            report_to=["wandb"],
            # save_total_limit=1,

            warmup_ratio=0.05,
            max_grad_norm=10.0,
            lr_scheduler_type="cosine",

            remove_unused_columns=False,
            dataloader_num_workers=4,
            seed=42,
        )
    else:
        print("DeepSpeed not detected: Using adamw_bnb_8bit for memory efficiency.")
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=num_epochs,

            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            optim="adamw_bnb_8bit",

            logging_steps=10,
            save_strategy="steps",
            save_steps=200,
            report_to=["wandb"],
            # save_total_limit=1,

            warmup_ratio=0.05,
            max_grad_norm=10.0,
            lr_scheduler_type="cosine",

            remove_unused_columns=False,
            dataloader_num_workers=4,
            seed=42,
        )


    callbacks = []
    if args.probe_every_n_steps > 0:
        probe_output_file = os.path.join(output_dir, "probe_responses.txt")
        callbacks.append(ProbeGenerationCallback(tokenizer, probe_output_file, args.probe_every_n_steps))
        print(f"Probe generation every {args.probe_every_n_steps} steps -> {probe_output_file}")

    # KL regularization is not used (kl_beta=0, ref_model=None).
    trainer = OfflineSDPOTrainer(
        ignore_first_k=2,
        model=model,
        ref_model=None,
        kl_beta=0,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )

    resume = args.resume_from_checkpoint
    if resume and resume.lower() == "true":
        resume = True
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    print("Training complete.")


if __name__ == "__main__":
    main()
