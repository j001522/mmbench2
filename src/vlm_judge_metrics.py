# vlm_judge_metrics.py
"""
Generative-VLM baseline for clef_metrics.py: the same images, task text,
guideline and questions, asked to a plain chat VLM (default Qwen3.8-27B, the
base model Clef is post-trained from), scored from the next-token logits
instead of Clef's decision head:
    noul   P(Yes) / (P(Yes) + P(No))
    score  expected level over the digit tokens 0..n-1
Thinking is disabled (enable_thinking=False), so the answer is the first token.
One forward pass per question (Clef answers all jointly), hence only --questions.

Output: columns <tag>_<layout>_<question> in the same
<root>/_hidden_metrics/clef_<set>/<id>_<task>.csv files (source "clef:" in
eval_annotations.py), e.g. qwen27_stack_halluc next to clef_stack_halluc.

Env /projects/0/prjs0951/conda_envs/clef. Run from src/ (GPU, ~56 GB bf16):
    python vlm_judge_metrics.py --layout stack
"""
import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import torch

from clef_metrics import GUIDELINE, LAYOUT, QUESTIONS, ROOT, TASKS, TELEPORT, make_images, merge_write

ALL_QUESTIONS = {**QUESTIONS, **TELEPORT}


def prompt_text(task, layout, q):
    head = f"Task: {TASKS.get(task, task)}\n{LAYOUT[layout]}{GUIDELINE}\n\nQuestion: {q['instructions']}\n"
    if q["type"] == "noul":
        return head + "Answer with only Yes or No."
    levels = "\n".join(f"{i}: {c}" for i, c in enumerate(q["criteria"]))
    return head + f"Answer with only one digit:\n{levels}"


def single_token(tok, s):
    ids = tok.encode(s, add_special_tokens=False)
    assert len(ids) == 1, (s, ids)
    return ids[0]


@torch.inference_mode()
def score_sequence(model, processor, real, wm, ctx, task, args, tok_ids):
    jobs = []                                              # (t, question id, chat text, image)
    for t in range(ctx, len(real)):
        img = make_images(real, wm, t, args.stride, args.upscale, args.gap, args.layout)
        assert len(img) == 1, "single-image layouts only (side, stack)"
        for qid in args.questions:
            msgs = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": prompt_text(task, args.layout, ALL_QUESTIONS[qid])}]}]
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
            jobs.append((t, qid, text, img[0]))
    out = {}
    for i in range(0, len(jobs), args.batch):
        chunk = jobs[i:i + args.batch]
        inp = processor(text=[j[2] for j in chunk], images=[j[3] for j in chunk], padding=True,
                        return_tensors="pt").to(model.device)
        logp = model(**inp, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
        for (t, qid, _, _), lp in zip(chunk, logp):
            ids = tok_ids[qid]
            p = lp[ids].softmax(-1)                        # renormalized over the allowed answers
            out.setdefault(t, {})[qid] = (p[0].item() if ALL_QUESTIONS[qid]["type"] == "noul"
                                          else (p * torch.arange(len(ids), device=p.device)).sum().item())
    ntok = int(inp["attention_mask"][0].sum())
    return [out[t] for t in sorted(out)], ntok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="+", default=["dmc_v1", "dmc_v2", "mw_v1", "mw_v2"])
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--tag", default="qwen27")
    p.add_argument("--root", default=ROOT)
    p.add_argument("--layout", choices=["side", "stack"], default="stack")
    p.add_argument("--questions", nargs="+", default=["halluc", "drift", "severity"],
                   choices=list(ALL_QUESTIONS))
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--upscale", type=int, default=2)
    p.add_argument("--gap", type=int, default=8)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--ids", nargs="*", default=None, help="only these sequence ids (smoke test)")
    args = p.parse_args()

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
                                                            device_map={"": "cuda"}).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"              # last position = next token for every row
    tok = processor.tokenizer
    tok_ids = {q: ([single_token(tok, "Yes"), single_token(tok, "No")] if ALL_QUESTIONS[q]["type"] == "noul"
                   else [single_token(tok, str(i)) for i in range(len(ALL_QUESTIONS[q]["criteria"]))])
               for q in args.questions}
    print(f"[{args.tag}] loaded {args.model} in {time.time() - t0:.0f}s, "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; answer tokens {tok_ids}", flush=True)

    name = f"{args.tag}_{args.layout}"
    for s in args.sets:
        out_dir = os.path.join(args.root, "_hidden_metrics", f"clef_{s}")
        os.makedirs(out_dir, exist_ok=True)
        key = {r["id"]: r for r in csv.DictReader(open(os.path.join(args.root, s, "key.csv")))}
        t0, n = time.time(), 0
        for f in sorted(glob.glob(os.path.join(args.root, s, "data", "*.npz"))):
            sid = os.path.basename(f)[:-4]
            if args.ids and sid not in args.ids:
                continue
            d = np.load(f)
            ctx, task = json.loads(str(d["meta"]))["ctx"], key[sid]["task"]
            rows, ntok = score_sequence(model, processor, d["real"], d["wm"], ctx, task, args, tok_ids)
            merge_write(os.path.join(out_dir, f"{sid}_{task}.csv"), list(range(ctx, len(d["real"]))), name, rows)
            n += 1
            if n == 1:
                print(f"[{args.tag}] {s} {sid}: {ntok} tokens/prompt; halluc per frame "
                      + " ".join(f"{r.get('halluc', float('nan')):.2f}" for r in rows), flush=True)
        print(f"[{args.tag}] {s}: {n} seqs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
