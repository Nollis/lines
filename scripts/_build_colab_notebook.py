"""Build notebooks/colab_train_box.ipynb. Run once; the resulting notebook
is committed."""

import json


def md(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src,
            "outputs": [], "execution_count": None}


cells = []

cells.append(md([
    "# Train the autoregressive box model on Colab\n",
    "\n",
    "One-click GPU run for the **A3** plan (autoregressive CAD-as-language model).\n",
    "\n",
    "**Setup:** *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*.\n",
    "\n",
    "What this does (~4-6 min on a T4):\n",
    "1. Clone the repo, install deps, verify GPU.\n",
    "2. Regenerate the training+held-out box data (deterministic seeds, ~2 min).\n",
    "3. Train for 50 epochs (~3 min on T4).\n",
    "4. Score the model under the honest F1 metric vs the reference table.\n",
    "5. Render before/after wireframes so you can eyeball the result.\n",
    "6. (Optional) Save the checkpoint to Google Drive.\n",
    "\n",
    "**Reference points** the run will print at the end:\n",
    "\n",
    "| Predictor on held-out boxes | F1 |\n",
    "|-----|-----|\n",
    "| classical baseline | 0.000 |\n",
    "| set predictor + structure post-process (the failing approach) | 0.282 |\n",
    "| **target after this notebook** | **>= 0.7** |\n",
]))

cells.append(md([
    "## 1. Clone the repo\n",
    "\n",
    "Edit `REPO_URL` below to point at your fork/push of the project.\n",
]))

cells.append(code([
    "REPO_URL = 'https://github.com/YOUR_USER/lines.git'   # <-- edit me\n",
    "BRANCH = 'main'\n",
    "REPO_DIR = '/content/lines'\n",
    "\n",
    "import os, subprocess\n",
    "if not os.path.isdir(REPO_DIR):\n",
    "    subprocess.check_call(['git', 'clone', '--branch', BRANCH, REPO_URL, REPO_DIR])\n",
    "else:\n",
    "    subprocess.check_call(['git', '-C', REPO_DIR, 'pull', '--ff-only'])\n",
    "os.chdir(REPO_DIR)\n",
    "print('repo at', os.getcwd())\n",
    "print(subprocess.check_output(['git', '-C', REPO_DIR, 'log', '-1', '--oneline']).decode().strip())\n",
]))

cells.append(md(["## 2. Install deps + verify GPU\n"]))

cells.append(code([
    "!pip install -q -e . matplotlib\n",
    "import torch\n",
    "print('torch', torch.__version__, '|  CUDA available:', torch.cuda.is_available())\n",
    "if torch.cuda.is_available():\n",
    "    print('device:', torch.cuda.get_device_name(0))\n",
    "else:\n",
    "    print('NOTE: no GPU. Runtime > Change runtime type > T4 GPU, then Run all again.')\n",
]))

cells.append(md([
    "## 3. Regenerate the data (deterministic seeds -> identical content every time)\n",
]))

cells.append(code([
    "from pathlib import Path\n",
    "import subprocess\n",
    "for split, n, seed, extra in [\n",
    "    ('data/train64_box', 4000, 0, []),\n",
    "    ('data/test64_box',  400, 900_000, ['--no-randomize']),\n",
    "]:\n",
    "    if (Path(split) / 'manifest.json').exists():\n",
    "        print(f'{split}: already on disk, skipping')\n",
    "        continue\n",
    "    print(f'generating {split} ({n} samples)...')\n",
    "    subprocess.check_call(['python', 'scripts/build_box_data.py',\n",
    "                            '--out', split, '--n', str(n),\n",
    "                            '--seed0', str(seed), '--canvas-side', '64'] + extra)\n",
]))

cells.append(md([
    "## 4. Train (50 epochs, ~3 min on T4)\n",
    "\n",
    "Calls the training function directly so the loss log streams into this cell.\n",
]))

cells.append(code([
    "from pathlib import Path\n",
    "from lines.train.train_autoregressive import ARTrainConfig, train_autoregressive\n",
    "\n",
    "cfg = ARTrainConfig(\n",
    "    canvas_side=64,\n",
    "    epochs=50,\n",
    "    batch_size=64,\n",
    "    d_model=192,\n",
    "    n_decoder_layers=3,\n",
    "    lr=3e-4,\n",
    "    device='cuda',\n",
    ")\n",
    "result = train_autoregressive(\n",
    "    cfg,\n",
    "    train_dir=Path('data/train64_box'),\n",
    "    test_dir=Path('data/test64_box'),\n",
    "    out_dir=Path('checkpoints/ar_box64_gpu'),\n",
    ")\n",
]))

cells.append(md(["## 5. Score under the honest F1 metric\n"]))

cells.append(code([
    "import torch\n",
    "from lines.datagen.dataset import Dataset\n",
    "from lines.datagen.sampler2d import Canvas\n",
    "from lines.eval.harness import run_predictor\n",
    "from lines.models.autoregressive import AutoregressiveModel\n",
    "from lines.train.predictor_ar import AutoregressivePredictor\n",
    "from lines.baselines.classical import ClassicalBaseline\n",
    "\n",
    "ck = torch.load('checkpoints/ar_box64_gpu/model.pt', map_location='cuda', weights_only=False)\n",
    "cfg = ck['cfg']\n",
    "model = AutoregressiveModel(canvas_side=cfg['canvas_side'], d_model=cfg['d_model'],\n",
    "                            n_heads=cfg['n_heads'], n_decoder_layers=cfg['n_decoder_layers'],\n",
    "                            max_seq_len=cfg['max_seq_len']).cuda()\n",
    "model.load_state_dict(ck['model'])\n",
    "\n",
    "C = Canvas(cfg['canvas_side'], cfg['canvas_side'])\n",
    "ds = Dataset('data/test64_box')\n",
    "ar = AutoregressivePredictor(model, C, max_tokens=cfg['max_seq_len'], device='cuda')\n",
    "r_ar = run_predictor(ar, ds, C)\n",
    "r_base = run_predictor(ClassicalBaseline(), ds, C)\n",
    "\n",
    "print('Predictor                       F1       Precision  Recall')\n",
    "print(f'classical baseline              {r_base[\"mean_f1\"]:.3f}   {r_base[\"mean_precision\"]:.3f}      {r_base[\"mean_recall\"]:.3f}')\n",
    "print(f'set predictor + post-process    0.282    (recorded; ar should beat)')\n",
    "print(f'AUTOREGRESSIVE (this run)       {r_ar[\"mean_f1\"]:.3f}   {r_ar[\"mean_precision\"]:.3f}      {r_ar[\"mean_recall\"]:.3f}')\n",
    "print()\n",
    "target = 0.7\n",
    "if r_ar['mean_f1'] >= target:\n",
    "    print(f'OK  F1 {r_ar[\"mean_f1\"]:.3f} >= {target}: A3 succeeded; Stage 2 (ellipses) is unblocked.')\n",
    "elif r_ar['mean_f1'] > 0.282:\n",
    "    print(f'PARTIAL  F1 {r_ar[\"mean_f1\"]:.3f} > 0.282 (beats set predictor) but < {target} target.')\n",
    "    print('   First try: more epochs (cfg.epochs *= 2) -- training may not have converged.')\n",
    "else:\n",
    "    print(f'UNEXPECTED  F1 {r_ar[\"mean_f1\"]:.3f} <= 0.282. See docs/a3-gpu-handoff.md rubric.')\n",
]))

cells.append(md([
    "## 6. Eyeball the wireframes\n",
    "\n",
    "Six held-out boxes: input | model prediction | ground truth.\n",
]))

cells.append(code([
    "import matplotlib.pyplot as plt\n",
    "from lines.datagen.render import render_primitives\n",
    "\n",
    "N = 6\n",
    "fig, axes = plt.subplots(N, 3, figsize=(7, 2.2 * N))\n",
    "for k in range(N):\n",
    "    img, gt = ds[k]\n",
    "    pred = ar(img)\n",
    "    pred_img = render_primitives(pred, C.width, C.height)\n",
    "    gt_img = render_primitives(gt, C.width, C.height)\n",
    "    for ax, im, title in zip(axes[k], [img, pred_img, gt_img],\n",
    "                              ['input', 'prediction', 'ground truth']):\n",
    "        ax.imshow(im, cmap='gray', vmin=0, vmax=255)\n",
    "        ax.set_title(title if k == 0 else ''); ax.axis('off')\n",
    "plt.tight_layout(); plt.show()\n",
]))

cells.append(md([
    "## 7. (Optional) Save the trained checkpoint to Google Drive\n",
    "\n",
    "Colab sessions get reaped. Uncomment to persist the checkpoint.\n",
]))

cells.append(code([
    "# from google.colab import drive\n",
    "# drive.mount('/content/drive')\n",
    "# !mkdir -p /content/drive/MyDrive/lines_checkpoints\n",
    "# !cp checkpoints/ar_box64_gpu/model.pt /content/drive/MyDrive/lines_checkpoints/ar_box64.pt\n",
    "# print('saved to Drive')\n",
]))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}

with open("notebooks/colab_train_box.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"wrote notebooks/colab_train_box.ipynb ({len(cells)} cells)")
