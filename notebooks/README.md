# Colab notebooks

## `colab_train_box.ipynb` — A3 box training on a free Colab T4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nollis/lines/blob/main/notebooks/colab_train_box.ipynb)

### One-time setup (you only do this once)

The repo is currently local-only. Push it to GitHub:

```bash
# create a new private (or public) repo on github.com first, then:
git remote add origin https://github.com/YOUR_USER/lines.git
git push -u origin main
```

That's the only step that isn't "click a button."

### Per-run workflow (the actual one-click)

1. Click the **Open in Colab** badge above (or open the `.ipynb` from GitHub
   in Colab manually).
2. *Runtime → Change runtime type → T4 GPU*.
3. *Runtime → Run all.*
4. Wait ~5 minutes. The notebook clones, installs, generates data, trains, scores
   against the reference table, and renders a 6-row before/after wireframe panel.
5. (Optional) Uncomment the Drive-save cell at the end if you want the
   checkpoint to survive the session getting reaped.

### What it doesn't do (and how to extend it)

- It doesn't push results back to git. The checkpoint stays in Colab; copy to
  Drive (cell 7) if you want it. To pull it back to your local repo, download
  the `.pt` from Drive once you're back at your laptop.
- It targets boxes only. For the concentric-circle bake-off or the held-out
  probe, point `--train-dir` / `--test-dir` at the relevant manifest dir.

### If the F1 score is below target (0.7)

The notebook's last cell tells you which branch of the rubric you hit; see
`docs/a3-gpu-handoff.md` for the diagnostic ladder (more epochs ->
sampling tweaks -> tokenizer canonical-order audit).
