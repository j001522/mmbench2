# clef_metrics.py
"""
VLM hallucination scores from Cloudflare's Clef decision models (System-1
style: no text generation, one probability per allowed answer) on the frozen
annotation sets, for comparison with the manual labels in eval_annotations.py.

Per imagined frame t, one record: a single image [real_t | wm_t] side by side
(each frame upscaled --upscale x, a --gap px white separator), with the task
and the annotation guideline as the state, and these questions, all oriented
higher = more hallucinated:
    halluc    noul   overall verdict under the annotation guideline
    invented  noul   object/part in the imagined frame that is not in the real one
    missed    noul   object/part of the real frame absent or faded in the imagined one
    warp      noul   shape/size/structure distorted in the imagined frame
    drift     noul   object position/pose clearly differs (beyond a few px)
    severity  score  0 none .. 3 severe, probability-weighted level
    teleport  noul   temporal layouts only: implausible jump between t-k and t
noul -> P(true); score -> expected level.

--layout (the questions are always about step t):
    side   one image [real_t | wm_t]                                   (default)
    pair   two images real_t, wm_t
    quad   four images real_{t-k}, real_t, wm_{t-k}, wm_t      (k = --stride)
    stack  one 2x2 image, rows t-k / t, columns real | wm
For the first imagined steps t-k falls in the context, where wm is the
tokenizer reconstruction of the real frame (as in the annotation tool).
Use --tag to keep layouts apart, e.g. --layout quad --tag _quad.

Output: <root>/_hidden_metrics/clef_<set>/<id>_<task>.csv, one row per
imagined frame, columns <model>_<question> (model = flash | clef). Existing
columns of other models are kept, so the models can be run separately.

Needs torch 2.11 / transformers 5.10.2 (env /projects/0/prjs0951/conda_envs/clef).
Run from src/ (GPU; Clef 27B ~55 GB bf16, Flash ~19 GB):
    python clef_metrics.py --sets dmc_v1 dmc_v2 mw_v1 mw_v2 --models flash clef
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

REPOS = {"flash": "Cloudflare/clef-flash", "clef": "Cloudflare/clef"}

TASKS = {
    "reacher-easy": "DeepMind Control 'reacher': a planar two-link robot arm, seen from above, "
                    "reaching for a small target dot on a flat arena.",
    "cup-catch": "DeepMind Control 'ball-in-cup': a cup moved in a vertical plane with a ball "
                 "attached to it by a string; the goal is to swing the ball into the cup.",
    "mw-reach": "Meta-World 'reach': a Sawyer robot arm on a table moving its gripper to a goal "
                "position.",
}

LAYOUT = {
    "side": "Left half: the real frame from the simulator. Right half: the frame imagined by a "
            "learned world model for the same time step, from the same past and actions. ",
    "pair": "The first image is the real frame from the simulator, the second the frame imagined "
            "by a learned world model for the same time step, from the same past and actions. ",
    "quad": "Four images: (1) the real frame from the simulator at an earlier step, (2) the real "
            "frame at the current step, (3) the frame imagined by a learned world model at the "
            "earlier step, (4) the imagined frame at the current step. The imagined frames are "
            "predicted from the same past and actions as the real ones. Judge the current step; "
            "the earlier step shows how things move. ",
    "stack": "A 2x2 grid. Left column: real frames from the simulator. Right column: frames "
             "imagined by a learned world model from the same past and actions. Top row: an "
             "earlier step. Bottom row: the current step. Judge the current step (bottom row); "
             "the top row shows how things move. ",
}
TEMPORAL = ("quad", "stack")
GUIDELINE = (
    "A hallucination is a meaningful content error in the imagined frame. NOT a hallucination: "
    "uniform blur or softness, slight colour/brightness shift, a small constant offset of a few "
    "pixels, missing texture detail. Ignore the reacher's central joint and the cup-catch string."
)

QUESTIONS = {
    "halluc": {"type": "noul", "instructions":
               "Does the imagined frame contain a hallucination compared with the real frame?"},
    "invented": {"type": "noul", "instructions":
                 "Does the imagined frame show an object or body part that is not in the real "
                 "frame (extra, duplicated, or appearing where there is none)?"},
    "missed": {"type": "noul", "instructions":
               "Is an object or body part that is visible in the real frame absent, vanished or "
               "faded out in the imagined frame?"},
    "warp": {"type": "noul", "instructions":
             "Is the shape, size or structure of an object distorted in the imagined frame (bent "
             "or stretched arm, deformed ball, parts merging or melting)?"},
    "drift": {"type": "noul", "instructions":
              "Is an object or the robot clearly in a different position or pose in the imagined "
              "frame than in the real frame (more than a few pixels)?"},
    "severity": {"type": "score", "instructions":
                 "How severe is the difference between the imagined and the real frame?",
                 "criteria": ["No meaningful difference, only blur or tiny offsets",
                              "Slight content difference",
                              "Clear content error",
                              "Severe error: objects wrong, missing or invented"]},
}
TELEPORT = {"teleport": {"type": "noul", "instructions":
                         "From the earlier to the current step, does an object in the imagined "
                         "frames jump or move implausibly, unlike its motion in the real frames?"}}

ROOT = "../../logs/annotation"


def make_images(real, wm, t, k, upscale, gap, layout):
    up = lambda a: Image.fromarray(a).resize((a.shape[1] * upscale, a.shape[0] * upscale),
                                             Image.BICUBIC)
    r, w = up(real[t]), up(wm[t])
    if layout == "pair":
        return [r, w]
    if layout == "side":
        canvas = Image.new("RGB", (r.width + gap + w.width, r.height), (255, 255, 255))
        canvas.paste(r, (0, 0))
        canvas.paste(w, (r.width + gap, 0))
        return [canvas]
    r0, w0 = up(real[t - k]), up(wm[t - k])
    if layout == "quad":
        return [r0, r, w0, w]
    canvas = Image.new("RGB", (2 * r.width + gap, 2 * r.height + gap), (255, 255, 255))
    for (x, y), im in zip([(0, 0), (1, 0), (0, 1), (1, 1)], [r0, w0, r, w]):
        canvas.paste(im, (x * (r.width + gap), y * (r.height + gap)))
    return [canvas]


def answers(question, probs):
    if question["type"] == "noul":
        return probs["true"]
    return sum(int(k) * p for k, p in probs.items())          # score: expected level


def questions_for(layout):
    return {**QUESTIONS, **TELEPORT} if layout in TEMPORAL else QUESTIONS


@torch.inference_mode()
def score_sequence(model, processor, jsm, real, wm, ctx, task, args):
    state = {"task": TASKS.get(task, task), "images": LAYOUT[args.layout] + GUIDELINE}
    questions = questions_for(args.layout)
    enc = [jsm.encode_record(processor.tokenizer,
                             {"state": state, "questions": questions,
                              "images": make_images(real, wm, t, args.stride, args.upscale, args.gap,
                                                    args.layout)},
                             processor=processor)
           for t in range(ctx, len(real))]
    device = next(model.parameters()).device
    rows = []
    for i in range(0, len(enc), args.batch):
        chunk = enc[i:i + args.batch]
        logits = model(jsm.collate_records(chunk, processor.tokenizer.pad_token_id, device))
        for rec, rec_logits in zip(chunk, logits):
            rows.append({q.question_id: answers(questions[q.question_id],
                                                dict(zip(q.option_ids, l.float().softmax(-1).tolist())))
                         for q, l in zip(rec.questions, rec_logits)})
    return rows, len(enc[0].input_ids)


def merge_write(path, ts, name, rows):
    """Write columns <name>_<q>, keeping all other columns already in the file."""
    qs = list(rows[0])
    cols_new = [f"{name}_{q}" for q in qs]
    old = {}
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            old[int(r["t"])] = {k: v for k, v in r.items() if k != "t" and k not in cols_new}
    cols_old = sorted({k for r in old.values() for k in r})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + cols_old + cols_new)
        for t, r in zip(ts, rows):
            o = old.get(t, {})
            w.writerow([t] + [o.get(c, "") for c in cols_old] + [f"{r[q]:.6g}" for q in qs])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="+", default=["dmc_v1", "dmc_v2", "mw_v1", "mw_v2"])
    p.add_argument("--models", nargs="+", default=["flash", "clef"], choices=list(REPOS))
    p.add_argument("--root", default=ROOT)
    p.add_argument("--layout", choices=["side", "pair", "quad", "stack"], default="side")
    p.add_argument("--stride", type=int, default=1, help="temporal layouts: compare t with t-stride")
    p.add_argument("--upscale", type=int, default=2)
    p.add_argument("--gap", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--ids", nargs="*", default=None, help="only these sequence ids (smoke test)")
    p.add_argument("--tag", default="", help="suffix for the column prefix, e.g. _pair")
    args = p.parse_args()

    from huggingface_hub import snapshot_download
    for name in args.models:
        path = snapshot_download(REPOS[name])
        sys.path.insert(0, path)
        import joint_schema_model as jsm                      # same file in both repos
        t0 = time.time()
        model, processor = jsm.load_release_model(path, device="cuda")
        print(f"[{name}] loaded {REPOS[name]} in {time.time() - t0:.0f}s, "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        for s in args.sets:
            out_dir = os.path.join(args.root, "_hidden_metrics", f"clef_{s}")
            os.makedirs(out_dir, exist_ok=True)
            key = {r["id"]: r for r in csv.DictReader(open(os.path.join(args.root, s, "key.csv")))}
            files = sorted(glob.glob(os.path.join(args.root, s, "data", "*.npz")))
            t0 = time.time()
            n = 0
            for f in files:
                sid = os.path.basename(f)[:-4]
                if args.ids and sid not in args.ids:
                    continue
                d = np.load(f)
                meta = json.loads(str(d["meta"]))
                ctx, task = meta["ctx"], key[sid]["task"]
                rows, ntok = score_sequence(model, processor, jsm, d["real"], d["wm"], ctx, task, args)
                merge_write(os.path.join(out_dir, f"{sid}_{task}.csv"), list(range(ctx, len(d["real"]))),
                            name + args.tag, rows)
                n += 1
                if n == 1:
                    print(f"[{name}] {s} {sid}: {ntok} tokens/record; halluc per frame "
                          + " ".join(f"{r['halluc']:.2f}" for r in rows), flush=True)
            print(f"[{name}] {s}: {n} seqs in {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
        sys.path.remove(path)
        sys.modules.pop("joint_schema_model", None)


if __name__ == "__main__":
    main()
