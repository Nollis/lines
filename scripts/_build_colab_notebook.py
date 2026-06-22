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
    "# Train the autoregressive model on Colab (boxes or cylinders)\n",
    "\n",
    "One-click GPU run for the **A3 / Stage 2** autoregressive CAD-as-language model.\n",
    "Same recipe (d=256, 5 layers, 100 epochs, ~10k samples) works for either\n",
    "content type; switch via `DATA_KIND` in cell 3.\n",
    "\n",
    "**Setup:** *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*.\n",
    "\n",
    "What this does (~15–25 min on a T4):\n",
    "1. Clone the repo, install deps, verify GPU.\n",
    "2. Generate 10k training + 400 held-out scenes for the chosen content (~4 min).\n",
    "3. Train for 100 epochs with d_model=256, 5 decoder layers (~10–18 min on T4).\n",
    "4. Score under F1 + exact-match; compare against the corresponding reference.\n",
    "5. Beam-3 ablation on the same checkpoint.\n",
    "6. Render before/after wireframes.\n",
    "7. (Optional) Save the checkpoint to Google Drive.\n",
    "\n",
    "**Reference points** (boxes is the Stage 1 ship config; cylinders is Unit E6):\n",
    "\n",
    "| Held-out content | F1 | Exact-match |\n",
    "|-----|-----|-----|\n",
    "| classical baseline (boxes) | 0.000 | 0.000 |\n",
    "| AR big preset on **boxes** (Stage 1 ship) | **0.980** | **0.932** |\n",
    "| AR big preset on **cylinders** (this run target) | — | **>= 0.80** |\n",
]))

cells.append(md([
    "## 1. Clone the repo\n",
    "\n",
    "Edit `REPO_URL` below to point at your fork/push of the project.\n",
]))

cells.append(code([
    "REPO_URL = 'https://github.com/Nollis/lines.git'\n",
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
    "## 3. Pick the content type\n",
    "\n",
    "Switch this once at the top of the notebook; data dirs, training output dir,\n",
    "and eval references all derive from it. The same recipe ships either kind.\n",
]))

cells.append(code([
    "DATA_KIND = 'cylinders'   # 'boxes' | 'cylinders'\n",
    "\n",
    "DATASETS = {\n",
    "    'boxes': {\n",
    "        'script':       'scripts/build_box_data.py',\n",
    "        'train_dir':    'data/train64_box_big',\n",
    "        'test_dir':     'data/test64_box',\n",
    "        'out_dir':      'checkpoints/ar_box64_gpu_big',\n",
    "        'reference':    'boxes -- Stage 1 ship config: F1=0.980, exact=0.932',\n",
    "    },\n",
    "    'cylinders': {\n",
    "        'script':       'scripts/build_cylinder_data.py',\n",
    "        'train_dir':    'data/train64_cyl',\n",
    "        'test_dir':     'data/test64_cyl',\n",
    "        'out_dir':      'checkpoints/ar_cyl64',\n",
    "        'reference':    'cylinders -- Stage 2 Unit E6 first run; target exact >= 0.80',\n",
    "    },\n",
    "}\n",
    "_cfg = DATASETS[DATA_KIND]\n",
    "BUILD_SCRIPT = _cfg['script']\n",
    "TRAIN_DIR    = _cfg['train_dir']\n",
    "TEST_DIR     = _cfg['test_dir']\n",
    "OUT_DIR      = _cfg['out_dir']\n",
    "print(f'DATA_KIND = {DATA_KIND!r}')\n",
    "print(f'  train:  {TRAIN_DIR}')\n",
    "print(f'  test:   {TEST_DIR}')\n",
    "print(f'  out:    {OUT_DIR}')\n",
    "print(f'  ref:    {_cfg[\"reference\"]}')\n",
]))

cells.append(md([
    "## 4. Generate data (deterministic seeds -> identical content every time)\n",
]))

cells.append(code([
    "# 10k train + 400 held-out, deterministic seeds. ~4 min on Colab CPU. Disjoint\n",
    "# dirs per DATA_KIND so a previous content type on disk doesn't shadow this.\n",
    "from pathlib import Path\n",
    "import subprocess\n",
    "for split, n, seed, extra in [\n",
    "    (TRAIN_DIR, 10000, 0, []),\n",
    "    (TEST_DIR,    400, 900_000, ['--no-randomize']),\n",
    "]:\n",
    "    if (Path(split) / 'manifest.json').exists():\n",
    "        print(f'{split}: already on disk, skipping')\n",
    "        continue\n",
    "    print(f'generating {split} ({n} samples) via {BUILD_SCRIPT}...')\n",
    "    subprocess.check_call(['python', BUILD_SCRIPT,\n",
    "                            '--out', split, '--n', str(n),\n",
    "                            '--seed0', str(seed), '--canvas-side', '64'] + extra)\n",
]))

cells.append(md([
    "## 5. Train (BIG preset: 100 epochs, ~10–18 min on T4)\n",
    "\n",
    "Calls the training function directly so the loss log streams into this cell.\n",
    "If the cell dies mid-run (Colab disconnect, etc.) the per-5-epoch checkpoint\n",
    "in `OUT_DIR` is reloadable with `--init-from`.\n",
]))

cells.append(code([
    "from pathlib import Path\n",
    "from lines.train.train_autoregressive import ARTrainConfig, train_autoregressive\n",
    "\n",
    "cfg = ARTrainConfig(\n",
    "    canvas_side=64,\n",
    "    epochs=100,\n",
    "    batch_size=64,\n",
    "    d_model=256,\n",
    "    n_decoder_layers=5,\n",
    "    lr=3e-4,\n",
    "    device='cuda',\n",
    "    checkpoint_every=5,\n",
    ")\n",
    "result = train_autoregressive(\n",
    "    cfg,\n",
    "    train_dir=Path(TRAIN_DIR),\n",
    "    test_dir=Path(TEST_DIR),\n",
    "    out_dir=Path(OUT_DIR),\n",
    ")\n",
]))

cells.append(md([
    "## 6. Score under the honest F1 metric + per-image distribution\n",
    "\n",
    "Mean F1 averages over edges and hides the **distribution** of per-image quality.\n",
    "A 60% perfect / 40% catastrophic mix can still average F1=0.9, but only ~60% of\n",
    "drawings would be *usable*. So we report two extras:\n",
    "\n",
    "* `exact_match_rate` = fraction of held-out drawings reconstructed completely (F1 >= 0.99)\n",
    "* `near_match_rate`  = fraction with at most ~1 wrong edge per 10 (F1 >= 0.9)\n",
]))

cells.append(code([
    "import torch\n",
    "from lines.datagen.dataset import Dataset\n",
    "from lines.datagen.sampler2d import Canvas\n",
    "from lines.eval.harness import run_predictor\n",
    "from lines.models.autoregressive import AutoregressiveModel\n",
    "from lines.train.predictor_ar import AutoregressivePredictor\n",
    "from lines.baselines.classical import ClassicalBaseline\n",
    "\n",
    "from pathlib import Path\n",
    "ck = torch.load(Path(OUT_DIR) / 'model.pt', map_location='cuda', weights_only=False)\n",
    "cfg = ck['cfg']\n",
    "model = AutoregressiveModel(canvas_side=cfg['canvas_side'], d_model=cfg['d_model'],\n",
    "                            n_heads=cfg['n_heads'], n_decoder_layers=cfg['n_decoder_layers'],\n",
    "                            max_seq_len=cfg['max_seq_len']).cuda()\n",
    "model.load_state_dict(ck['model'])\n",
    "\n",
    "C = Canvas(cfg['canvas_side'], cfg['canvas_side'])\n",
    "ds = Dataset(TEST_DIR)\n",
    "ar_greedy = AutoregressivePredictor(model, C, max_tokens=cfg['max_seq_len'],\n",
    "                                     device='cuda', beam_size=1)\n",
    "r_greedy = run_predictor(ar_greedy, ds, C)\n",
    "r_base = run_predictor(ClassicalBaseline(), ds, C)\n",
    "\n",
    "print('Predictor                       F1       Exact   Near')\n",
    "print(f'classical baseline              {r_base[\"mean_f1\"]:.3f}    {r_base[\"exact_match_rate\"]:.3f}   {r_base[\"near_match_rate\"]:.3f}')\n",
    "if DATA_KIND == 'boxes':\n",
    "    print(f'set predictor + post-process    0.282    (recorded)')\n",
    "    print(f'AR small preset (recorded)      0.899    0.642   0.650')\n",
    "    print(f'AR BIG ({DATA_KIND}, this run)        {r_greedy[\"mean_f1\"]:.3f}    {r_greedy[\"exact_match_rate\"]:.3f}   {r_greedy[\"near_match_rate\"]:.3f}')\n",
    "else:\n",
    "    print(f'AR boxes (recorded ship)        0.980    0.932   0.932')\n",
    "    print(f'AR BIG ({DATA_KIND}, this run)    {r_greedy[\"mean_f1\"]:.3f}    {r_greedy[\"exact_match_rate\"]:.3f}   {r_greedy[\"near_match_rate\"]:.3f}')\n",
    "print()\n",
    "print(f'  perfect drawings: {r_greedy[\"n_perfect\"]}/{r_greedy[\"n\"]}'\n",
    "      f'  ({r_greedy[\"exact_match_rate\"]*100:.1f}%)')\n",
    "print(f'  near-perfect:     {r_greedy[\"n_near\"]}/{r_greedy[\"n\"]}'\n",
    "      f'  ({r_greedy[\"near_match_rate\"]*100:.1f}%)')\n",
    "ar = ar_greedy   # used by the visualization cell below\n",
    "r_ar = r_greedy\n",
    "\n",
    "# exact-match thresholds: boxes is the proven ship config (>=0.85 to match\n",
    "# Stage 1 result); cylinders is Unit E6's first run, target the plan's >=0.80.\n",
    "exact_target = 0.85 if DATA_KIND == 'boxes' else 0.80\n",
    "if r_greedy['exact_match_rate'] >= exact_target:\n",
    "    print(f'WIN  Exact {r_greedy[\"exact_match_rate\"]:.3f} >= {exact_target}: {DATA_KIND} reconstruction ships.')\n",
    "elif r_greedy['exact_match_rate'] >= 0.60:\n",
    "    print(f'PARTIAL  Exact {r_greedy[\"exact_match_rate\"]:.3f}: still improving. Try +50 epochs (warm-start) or larger model.')\n",
    "else:\n",
    "    print(f'PLATEAU  Exact {r_greedy[\"exact_match_rate\"]:.3f}: capacity may not be the bottleneck. Inspect which views fail.')\n",
]))

cells.append(md([
    "## 6b. Beam search vs greedy on the *same* trained model\n",
    "\n",
    "Greedy is myopic -- once it picks a bad token, the rest cascades (the row-3\n",
    "catastrophe in the panel below). Beam-3 explores 3 paths and reduces that\n",
    "rate. No retraining; just a different decoder. Slower than greedy on CPU but\n",
    "comparable on GPU; expect ~+0.02 to ~+0.05 F1 and a bigger jump on\n",
    "`exact_match_rate`.\n",
]))

cells.append(code([
    "ar_beam = AutoregressivePredictor(model, C, max_tokens=cfg['max_seq_len'],\n",
    "                                   device='cuda', beam_size=3)\n",
    "r_beam = run_predictor(ar_beam, ds, C)\n",
    "print('Predictor                       F1       Exact   Near')\n",
    "print(f'AR greedy (this run)            {r_greedy[\"mean_f1\"]:.3f}    {r_greedy[\"exact_match_rate\"]:.3f}   {r_greedy[\"near_match_rate\"]:.3f}')\n",
    "print(f'AR beam-3 (this run)            {r_beam[\"mean_f1\"]:.3f}    {r_beam[\"exact_match_rate\"]:.3f}   {r_beam[\"near_match_rate\"]:.3f}')\n",
    "print()\n",
    "delta_f1 = r_beam['mean_f1'] - r_greedy['mean_f1']\n",
    "delta_exact = r_beam['exact_match_rate'] - r_greedy['exact_match_rate']\n",
    "print(f'beam vs greedy: dF1 = {delta_f1:+.3f}  dExact = {delta_exact:+.3f}')\n",
    "# use the better predictor for the visualization cell\n",
    "if r_beam['mean_f1'] > r_greedy['mean_f1']:\n",
    "    ar = ar_beam\n",
    "    print('-> using beam-3 for the visualization below')\n",
]))

cells.append(md([
    "## 7. Eyeball the wireframes\n",
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
    "## 8. (Optional) Save the trained checkpoint to Google Drive\n",
    "\n",
    "Colab sessions get reaped. Uncomment to persist the checkpoint.\n",
]))

cells.append(code([
    "# from google.colab import drive\n",
    "# drive.mount('/content/drive')\n",
    "# !mkdir -p /content/drive/MyDrive/lines_checkpoints\n",
    "# import shutil\n",
    "# dst = f'/content/drive/MyDrive/lines_checkpoints/ar_{DATA_KIND}_64.pt'\n",
    "# shutil.copy(f'{OUT_DIR}/model.pt', dst)\n",
    "# print(f'saved to Drive: {dst}')\n",
]))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}

with open("notebooks/colab_train_box.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"wrote notebooks/colab_train_box.ipynb ({len(cells)} cells)")
