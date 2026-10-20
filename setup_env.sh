#!/usr/bin/env bash
# Build a conda environment for Qwen-Image-2.1.
#
#   bash setup_env.sh                 # env name qwenimage, CUDA 12.8 wheels
#   ENV=myenv CUDA=cu126 bash setup_env.sh
#
# Why its own environment: the model needs transformers>=5.17 and diffusers from
# git. Existing envs on a shared box are usually pinned lower, and upgrading them
# in place breaks whatever else they were built for.
set -euo pipefail

ENV="${ENV:-qwenimage}"
CUDA="${CUDA:-cu128}"            # match the driver; nvidia-smi shows what it supports
PY="${PY:-3.12}"
CONDA="${CONDA:-$HOME/miniconda3}"

echo "== creating conda env '$ENV' (python $PY)"
"$CONDA/bin/conda" create -y -q -n "$ENV" "python=$PY"
P="$CONDA/envs/$ENV/bin/python"

echo "== torch for $CUDA"
"$P" -m pip install -q --upgrade pip
"$P" -m pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA"

echo "== transformers, accelerate, pillow"
"$P" -m pip install -q "transformers>=5.17" accelerate pillow safetensors sentencepiece

echo "== diffusers from git (QwenImage21Pipeline is not in a tagged release yet)"
"$P" -m pip install -q "git+https://github.com/huggingface/diffusers"

echo "== verifying"
"$P" - <<'PY'
import torch, transformers, diffusers
print(f"   torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"   gpu          {torch.cuda.get_device_name(0)}  {free/1e9:.1f} GB free of {total/1e9:.0f} GB")
print(f"   transformers {transformers.__version__}")
print(f"   diffusers    {diffusers.__version__}")
try:
    from diffusers import QwenImage21Pipeline          # noqa: F401
    print("   QwenImage21Pipeline importable")
except ImportError as e:
    raise SystemExit(f"   QwenImage21Pipeline NOT importable: {e}")
PY

cat <<MSG

Environment ready. To generate:

    $P generate.py "a lighthouse in a storm" --turbo

The first run downloads about 33 GB of weights into ~/.cache/huggingface.
MSG
