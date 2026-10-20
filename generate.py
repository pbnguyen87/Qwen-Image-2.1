#!/usr/bin/env python3
"""Generate images with Qwen-Image-2.1 (text-to-image and image editing).

    # text to image
    python generate.py "một quán cà phê nhỏ trên phố cổ Hà Nội, buổi sáng sớm"
    python generate.py "a lighthouse in a storm" --size 1328x1328 --steps 40

    # the 8-step Turbo checkpoint, much faster
    python generate.py "con mèo đang ngủ trên ghế gỗ" --turbo

    # edit an existing image, local file or URL
    python generate.py "make it snow" --image room.jpg
    python generate.py "put the subject on a beach" --image a.jpg --image b.jpg

    # transparent output
    python generate.py "a red maple leaf, plain background" --rgba

    # a batch of prompts, one per line, same settings for all
    python generate.py --prompt-file prompts.txt --out-dir out

Weights come from Hugging Face on first run: about 33 GB for Qwen-Image-2.1,
cached under ~/.cache/huggingface afterwards.

Requirements, per the model's own README:
    torch>=2.4.0, transformers>=5.17, accelerate, pillow
    diffusers from git: pip install git+https://github.com/huggingface/diffusers
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_T2I = "Qwen/Qwen-Image-2.1"
REPO_TURBO = "Qwen/Qwen-Image-2.1-Turbo"
# Sizes the model card uses. Qwen-Image works in several aspect ratios at roughly
# constant pixel count; these are the documented ones.
# The card: "For best results, use the recommended prompt format" for transparency.
RGBA_PREFIX = "This is an RGBA image with transparency. "
RGBA_SUFFIX = " The image has alpha channel and the background is transparent."

PRESETS = {
    "square": (1328, 1328),
    "landscape": (1664, 928),      # 16:9
    "portrait": (928, 1664),       # 9:16
    "photo": (1472, 1140),         # 4:3
    "poster": (1696, 2528),        # the README's tall poster example
}


def parse_size(value: str) -> tuple[int, int]:
    if value in PRESETS:
        return PRESETS[value]
    if "x" in value.lower():
        w, _, h = value.lower().partition("x")
        try:
            return int(w), int(h)
        except ValueError:
            pass
    raise SystemExit(f"--size wants WxH or one of {', '.join(PRESETS)}; got {value!r}")


def load_pipeline(repo: str, device: str | None, offload: bool):
    """Build the pipeline. Returns (pipe, device, dtype_name)."""
    import torch
    try:
        from diffusers import QwenImage21Pipeline
    except ImportError as e:
        raise SystemExit(
            f"cannot import QwenImage21Pipeline ({e}).\n"
            "This pipeline landed in diffusers main, not a tagged release:\n"
            "   pip install git+https://github.com/huggingface/diffusers"
        ) from None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    # bf16 on CUDA, fp32 on CPU where bf16 is slow and unsupported in places
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    print(f"loading {repo} on {device} ({str(dtype).split('.')[-1]}) ...", flush=True)
    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(repo, torch_dtype=dtype)
    if offload:
        # Keeps only the active component on the GPU. Slower per image, but the
        # 7.1B transformer plus its Qwen3-VL text encoder then fit a smaller card.
        pipe.enable_model_cpu_offload()
        print("   model CPU offload enabled")
    else:
        pipe = pipe.to(device)
    print(f"   ready in {time.time() - t0:.0f}s", flush=True)
    return pipe, device, str(dtype).split(".")[-1]


def load_inputs(paths: list[str], rgba: bool):
    """Load edit inputs from local paths or URLs."""
    from diffusers.utils import load_image

    out = []
    for p in paths:
        img = load_image(p) if p.startswith(("http://", "https://")) else load_image(str(Path(p).expanduser()))
        out.append(img.convert("RGBA") if rgba else img.convert("RGB"))
        print(f"   input {p} -> {out[-1].size[0]}x{out[-1].size[1]} {out[-1].mode}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", help="what to generate, or the edit instruction")
    ap.add_argument("--prompt-file", help="run every non-empty line of this file as a prompt")
    ap.add_argument("--image", action="append", default=[],
                    help="input image to edit; repeat for multi-reference, up to 10")
    ap.add_argument("--out-dir", default="outputs", help="where to write the images")
    ap.add_argument("--name", help="output file stem (single prompt only)")
    ap.add_argument("--turbo", action="store_true",
                    help=f"use {REPO_TURBO}, 8 steps instead of 40")
    ap.add_argument("--model", help="override the checkpoint repo or local path")
    ap.add_argument("--size", default="square",
                    help=f"WxH, or a preset: {', '.join(PRESETS)} (ignored when editing)")
    ap.add_argument("--steps", type=int, help="denoising steps (default 8 turbo / 40 base)")
    ap.add_argument("--cfg", type=float, help="true_cfg_scale, higher follows the prompt harder")
    ap.add_argument("--negative", default=" ", help="negative prompt")
    ap.add_argument("--seed", type=int, default=42, help="-1 for a random seed each image")
    ap.add_argument("--n", type=int, default=1, help="images per prompt, seed increments")
    ap.add_argument("--rgba", action="store_true", help="transparent output")
    ap.add_argument("--raw-prompt", action="store_true",
                    help="with --rgba, do not wrap the prompt in the recommended "
                         "transparency template")
    ap.add_argument("--device", help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--offload", action="store_true",
                    help="CPU-offload components to fit a smaller GPU")
    a = ap.parse_args()

    prompts: list[str] = []
    if a.prompt_file:
        pf = Path(a.prompt_file).expanduser()
        if not pf.is_file():
            raise SystemExit(f"no such prompt file: {pf}")
        prompts = [l.strip() for l in pf.read_text(encoding="utf-8").splitlines()
                   if l.strip() and not l.strip().startswith("#")]
        if not prompts:
            raise SystemExit(f"no prompts found in {a.prompt_file}")
    elif a.prompt:
        prompts = [a.prompt]
    else:
        ap.error("give a prompt, or --prompt-file")

    if len(a.image) > 10:
        raise SystemExit(f"the model takes at most 10 reference images, got {len(a.image)}")
    # Validate everything cheap first: a typo should not surface after a 33 GB download.
    width, height = parse_size(a.size)
    for p in a.image:
        if not p.startswith(("http://", "https://")) and not Path(p).expanduser().is_file():
            raise SystemExit(f"no such input image: {p}")

    import torch

    repo = a.model or (REPO_TURBO if a.turbo else REPO_T2I)
    steps = a.steps if a.steps is not None else (8 if a.turbo else 40)
    if a.turbo and a.steps is not None:
        # The card: the 8-step schedule is saved in the checkpoint and
        # "setting num_inference_steps alone does not override it".
        print(f"note: --steps {a.steps} is ignored by the Turbo checkpoint, which "
              f"carries its own 8-step schedule", file=sys.stderr)
    pipe, device, dtype = load_pipeline(repo, a.device, a.offload)
    inputs = load_inputs(a.image, a.rgba) if a.image else None

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    made = 0

    for pi, prompt in enumerate(prompts):
        for k in range(a.n):
            seed = torch.seed() % (2**31) if a.seed < 0 else a.seed + k
            gen = torch.Generator(device="cpu" if device == "mps" else device).manual_seed(seed)

            text = prompt
            if a.rgba and not a.raw_prompt and "RGBA" not in prompt:
                text = RGBA_PREFIX + prompt.rstrip(". ") + "." + RGBA_SUFFIX
            kw = dict(prompt=text, num_inference_steps=steps, generator=gen)
            if a.negative.strip():
                kw["negative_prompt"] = a.negative
            if a.cfg is not None:
                kw["true_cfg_scale"] = a.cfg
            if inputs:
                # Editing keeps the input's own resolution; width/height do not apply.
                kw["image"] = inputs if len(inputs) > 1 else inputs[0]
            else:
                kw["width"], kw["height"] = width, height
            if a.rgba:
                kw["output_type"] = "pil"

            label = prompt if len(prompt) < 60 else prompt[:57] + "..."
            print(f"\n[{made + 1}/{len(prompts) * a.n}] seed {seed}  {steps} steps  {label}",
                  flush=True)
            t0 = time.time()
            try:
                image = pipe(**kw).images[0]
            except TypeError as e:
                # A pipeline argument this diffusers build does not accept.
                raise SystemExit(f"pipeline rejected an argument: {e}") from None
            except torch.cuda.OutOfMemoryError:
                raise SystemExit(
                    "CUDA out of memory. Try --offload, a smaller --size, or free the GPU."
                ) from None
            took = time.time() - t0

            if a.name and len(prompts) == 1 and a.n == 1:
                stem = a.name
            else:
                stem = f"{stamp}_{pi:02d}_{seed}"
            path = out_dir / f"{stem}.png"
            image.save(path)
            made += 1
            print(f"   {took:.1f}s -> {path}  ({image.size[0]}x{image.size[1]} {image.mode})")

            with manifest.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "file": path.name, "prompt": prompt, "prompt_sent": text,
                    "seed": seed, "steps": steps,
                    "model": repo, "device": device, "dtype": dtype,
                    "size": list(image.size), "mode": image.mode,
                    "edit_inputs": a.image or None, "rgba": a.rgba,
                    "cfg": a.cfg, "negative_prompt": a.negative.strip() or None,
                    "seconds": round(took, 2),
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                }, ensure_ascii=False) + "\n")

    print(f"\n{made} image(s) -> {out_dir}, logged in {manifest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
