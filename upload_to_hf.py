"""
Upload TGIF-trained ViT-GPT2 to HuggingFace and convert to ONNX for Transformers.js.

Run in Google Colab:
  1. Mount Drive
  2. pip install optimum[exporters] huggingface_hub
  3. huggingface-cli login
  4. python upload_to_hf.py
"""

import subprocess
import sys
import os

# ── Config ────────────────────────────────────────────────────
MODEL_PATH = "/content/drive/MyDrive/Thesis/gif-caption-model/vitgpt2_tgif_6frames/final"
HF_REPO    = "Patricijia/vitgpt2-gif-descriptor"  # Change to your HF username
ONNX_DIR   = "/content/vitgpt2_onnx"

# ── Step 1: Upload PyTorch model ──────────────────────────────
print("=" * 60)
print("STEP 1: Uploading PyTorch model to HuggingFace...")
print("=" * 60)

from huggingface_hub import HfApi
api = HfApi()

# Create repo if it doesn't exist
try:
    api.create_repo(HF_REPO, exist_ok=True)
except Exception as e:
    print(f"Repo creation: {e}")

api.upload_folder(
    folder_path=MODEL_PATH,
    repo_id=HF_REPO,
    commit_message="Upload TGIF-trained ViT-GPT2 (6-frame grid)",
)
print(f"Uploaded to: https://huggingface.co/{HF_REPO}")

# ── Step 2: Export to ONNX for Transformers.js ────────────────
print("\n" + "=" * 60)
print("STEP 2: Converting to ONNX...")
print("=" * 60)

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "optimum[exporters]", "-q"
])

subprocess.check_call([
    sys.executable, "-m", "optimum.exporters.onnx",
    "--model", MODEL_PATH,
    "--task", "image-to-text-with-past",
    ONNX_DIR,
])

print(f"\nONNX model exported to: {ONNX_DIR}")
print("Files:")
for f in sorted(os.listdir(ONNX_DIR)):
    size = os.path.getsize(os.path.join(ONNX_DIR, f))
    print(f"  {f}: {size / 1024 / 1024:.1f} MB")

# ── Step 3: Upload ONNX to HuggingFace (onnx branch) ─────────
print("\n" + "=" * 60)
print("STEP 3: Uploading ONNX to HuggingFace (onnx branch)...")
print("=" * 60)

api.upload_folder(
    folder_path=ONNX_DIR,
    repo_id=HF_REPO,
    revision="onnx",
    commit_message="Add ONNX export for Transformers.js",
)

print(f"\nDone! ONNX model at: https://huggingface.co/{HF_REPO}/tree/onnx")
print(f"\nIn Transformers.js, load with:")
print(f'  pipeline("image-to-text", "{HF_REPO}", {{ revision: "onnx" }})')
