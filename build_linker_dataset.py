#!/usr/bin/env python3
# build_linker_dataset.py

import os
import re
import json
import argparse
from typing import List, Dict, Tuple
from Bio import SeqIO
from tqdm import tqdm

DOMAIN_RE = re.compile(r"Domain:\s*(.+?)\s*\((\d+)\s*-\s*(\d+)\)")

BAD_AA = set("UZOB")  # replace with X for ProtT5

def clean_aa(seq: str) -> str:
    seq = (seq or "").replace("\n", "").replace(" ", "").strip().upper()
    if not seq:
        return ""
    return "".join(("X" if c in BAD_AA else c) for c in seq)

def extract_domains_from_NRPS_PKS(cds_feature) -> List[Dict]:
    """
    Parse CDS qualifiers /NRPS_PKS lines:
      /NRPS_PKS="Domain: PKS_AT (573-845) ..."
    Returns sorted list of domains with 0-based python slice coords.
    """
    q = cds_feature.qualifiers
    hits = []
    for raw in q.get("NRPS_PKS", []) or []:
        s = " ".join(str(raw).split())
        m = DOMAIN_RE.search(s)
        if not m:
            continue
        name = m.group(1).strip()
        start1 = int(m.group(2))  # 1-based inclusive
        end1 = int(m.group(3))    # 1-based inclusive

        start0 = max(0, start1 - 1)
        end0 = max(start0, end1)  # inclusive -> exclusive
        hits.append({"name": name, "start0": start0, "end0": end0})

    hits.sort(key=lambda x: (x["start0"], x["end0"], x["name"]))
    # de-dup identical spans
    out, seen = [], set()
    for h in hits:
        k = (h["name"], h["start0"], h["end0"])
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out

def crop_for_context(seq: str, max_len: int, side: str) -> str:
    if len(seq) <= max_len:
        return seq
    if side == "left":
        return seq[-max_len:]
    if side == "right":
        return seq[:max_len]
    raise ValueError("side must be left or right")

def make_t5_pair(left_seq: str, right_seq: str, linker_seq: str,
                 left_type: str, right_type: str) -> Tuple[str, str]:
    """
    T5-style single-span infilling:
      input:  "LT:... RT:... LEFT <extra_id_0> RIGHT"
      target: "<extra_id_0> LINKER <extra_id_1>"
    """
    inp = f"LT:{left_type} RT:{right_type} {left_seq} <extra_id_0> {right_seq}"
    tgt = f"<extra_id_0> {linker_seq} <extra_id_1>"
    return inp, tgt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbk_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--min_linker_len", type=int, default=3)
    ap.add_argument("--max_linker_len", type=int, default=300)
    ap.add_argument("--max_left", type=int, default=256)
    ap.add_argument("--max_right", type=int, default=256)
    ap.add_argument("--file_suffixes", nargs="+", default=[".gbk", ".gb"])
    args = ap.parse_args()

    gbk_files = [f for f in os.listdir(args.gbk_dir)
                 if any(f.endswith(suf) for suf in args.file_suffixes)]
    gbk_files.sort()

    n_written = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as out_f:
        for fn in tqdm(gbk_files, desc="Parsing GBKs"):
            bgc_id = os.path.splitext(fn)[0]
            gbk_path = os.path.join(args.gbk_dir, fn)

            try:
                for record in SeqIO.parse(gbk_path, "genbank"):
                    for feat in record.features:
                        if feat.type != "CDS":
                            continue
                        q = feat.qualifiers
                        if "translation" not in q:
                            continue

                        prot = clean_aa(q.get("translation", [""])[0])
                        if not prot:
                            continue

                        protein_id = (q.get("protein_id") or [""])[0]
                        locus_tag = (q.get("locus_tag") or [""])[0]
                        gene = (q.get("gene") or [""])[0]
                        gene_id = protein_id or locus_tag or gene or "Unknown"

                        doms = extract_domains_from_NRPS_PKS(feat)
                        if len(doms) < 2:
                            continue

                        # clamp domains
                        for d in doms:
                            d["start0"] = max(0, min(d["start0"], len(prot)))
                            d["end0"] = max(d["start0"], min(d["end0"], len(prot)))

                        # adjacent pairs -> one linker each
                        for i in range(len(doms) - 1):
                            dL = doms[i]
                            dR = doms[i + 1]

                            left_seq_full = prot[dL["start0"]:dL["end0"]]
                            right_seq_full = prot[dR["start0"]:dR["end0"]]

                            linker = prot[dL["end0"]:dR["start0"]]
                            if not (args.min_linker_len <= len(linker) <= args.max_linker_len):
                                continue

                            left_seq = crop_for_context(left_seq_full, args.max_left, "left")
                            right_seq = crop_for_context(right_seq_full, args.max_right, "right")

                            inp, tgt = make_t5_pair(
                                left_seq=left_seq,
                                right_seq=right_seq,
                                linker_seq=linker,
                                left_type=dL["name"],
                                right_type=dR["name"],
                            )

                            rec = {
                                "bgc_id": bgc_id,
                                "gene_id": gene_id,
                                "left_type": dL["name"],
                                "right_type": dR["name"],
                                "left_seq": left_seq,
                                "right_seq": right_seq,
                                "linker_seq": linker,
                                "input_text": inp,
                                "target_text": tgt,
                                "linker_len": len(linker),
                            }
                            out_f.write(json.dumps(rec) + "\n")
                            n_written += 1
            except Exception:
                # skip malformed gbk
                continue

    print(f"Done. Wrote {n_written} examples to {args.out_jsonl}")

if __name__ == "__main__":
    main()
