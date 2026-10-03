import os
from argparse import ArgumentParser

from huggingface_hub import snapshot_download


# Download the MMBench2 dataset from the Hugging Face Hub.
# Authenticate via the HF_TOKEN env var or a cached `hf auth login` if required.
DEFAULT_REPO_ID = "nicklashansen/mmbench2"


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--local_dir", type=str, default="./data")
    p.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID,
                   help=f"HF dataset repo id (default: {DEFAULT_REPO_ID}).")
    p.add_argument("--subset", type=str, nargs="*", default=None,
                   help="optional subset(s) to download, e.g. 'val' or 'expert'")
    p.add_argument("--tasks", type=str, nargs="*", default=None,
                   help="optional task(s) to download, e.g. 'cup-catch walker-run'. "
                        "Combines with --subset; without it, the tasks are fetched "
                        "from every subset.")
    args = p.parse_args()

    allow_patterns = None
    if args.subset or args.tasks:
        subsets = [s.rstrip("/") for s in args.subset] if args.subset else ["*"]
        if args.tasks:
            # Each task contributes its frame shards (<task>-<k>.png) and its
            # action/reward file (<task>.pt). The [0-9] guard keeps prefixes
            # apart: plain 'walker-run-*' would also match 'walker-run-backward-*'.
            allow_patterns = [f"{s}/{t}-[0-9]*.png" for s in subsets for t in args.tasks]
            allow_patterns += [f"{s}/{t}.pt" for s in subsets for t in args.tasks]
        else:
            allow_patterns = [f"{s}/*" for s in subsets]

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.local_dir,
        token=os.environ.get("HF_TOKEN"),
        allow_patterns=allow_patterns,
    )
