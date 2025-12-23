#!/usr/bin/env python3
# train_prott5_infilling.py

import os
import json
import argparse
import numpy as np

from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from peft import LoraConfig, get_peft_model, TaskType

BAD_AA = set("UZOB")

def prott5_prepare_text(seq: str) -> str:
    """
    Rostlab ProtT5 expects space-separated amino acids.
    Also replace rare AAs with X.
    """
    seq = (seq or "").replace(" ", "").replace("\n", "").strip().upper()
    seq = "".join(("X" if c in BAD_AA else c) for c in seq)
    return " ".join(list(seq))

def format_for_prott5(example):
    # Convert AA strings inside input/target to spaced AA strings
    # We only need to space the sequences, not the sentinel tokens.
    # We'll do a simple approach: split input_text by spaces and re-encode AA runs.

    def spaceify_mixed(text: str) -> str:
        parts = text.split()
        out = []
        for p in parts:
            if p in ("<extra_id_0>", "<extra_id_1>"):
                out.append(p)
            elif p.startswith("LT:") or p.startswith("RT:"):
                out.append(p)
            else:
                # assume it's a raw AA sequence chunk
                out.append(prott5_prepare_text(p))
        return " ".join(out)

    example["input_text_spaced"] = spaceify_mixed(example["input_text"])
    # target looks like: "<extra_id_0> LINKER <extra_id_1>"
    # LINKER should be spaced.
    tgt_parts = example["target_text"].split()
    # tgt_parts: [<extra_id_0>, LINKER, <extra_id_1>] (as built)
    if len(tgt_parts) >= 3:
        linker = tgt_parts[1]
        example["target_text_spaced"] = f"{tgt_parts[0]} {prott5_prepare_text(linker)} {tgt_parts[2]}"
    else:
        example["target_text_spaced"] = example["target_text"]
    return example

def tokenize_fn(tokenizer, max_source_len, max_target_len):
    def _tok(batch):
        model_inputs = tokenizer(
            batch["input_text_spaced"],
            max_length=max_source_len,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target_text_spaced"],
                max_length=max_target_len,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return _tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_dir", required=True)

    # Good default for T5 infilling on proteins:
    ap.add_argument("--model_name", default="Rostlab/prot_t5_xl_uniref50")
    ap.add_argument("--max_source_len", type=int, default=1024)
    ap.add_argument("--max_target_len", type=int, default=256)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)

    ap.add_argument("--seed", type=int, default=0)

    # LoRA params
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load dataset from JSONL
    ds = load_dataset("json", data_files={"all": args.jsonl})["all"]
    ds = ds.map(format_for_prott5)

    # Split by group to reduce leakage (group by bgc_id)
    groups = np.array(ds["bgc_id"])
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))

    train_ds = ds.select(train_idx.tolist())
    val_ds = ds.select(val_idx.tolist())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Apply LoRA (recommended for ~4k samples)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q", "v"],  # works well for T5 attention blocks
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = train_ds.map(
        tokenize_fn(tokenizer, args.max_source_len, args.max_target_len),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        tokenize_fn(tokenizer, args.max_source_len, args.max_target_len),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Training args
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    print(f"Saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
