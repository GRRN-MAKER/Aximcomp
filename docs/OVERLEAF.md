# Compiling the AXIM paper on Overleaf

The paper source is `PAPER.tex` (you may have renamed it `thesis.tex` on Overleaf —
either works). It references two figures:

- `axim_throughput.png`
- `axim_latency.png`

## The "File not found" error

Overleaf compiles from an **uploaded** project. If you upload only the `.tex` and not the
images, you get:

```
Package pdftex.def Error: File `docs/assets/axim_throughput.png' not found
```

The paper is written to survive this: it uses a safe `\aximfig` helper that renders a
labelled placeholder box instead of failing. So **it will now compile even without the
images** — you just won't see the charts until you upload them.

## To show the real charts — upload the two PNGs

The `.tex` searches both `docs/assets/` and the project root (via `\graphicspath`), so
either of these works:

**Option A — upload flat (simplest):**
1. In Overleaf, click **Upload** in the file panel.
2. Upload `axim_throughput.png` and `axim_latency.png` to the **project root**
   (same folder as `thesis.tex`).
3. Recompile. Done.

**Option B — mirror the repo layout:**
1. Create a folder `docs/assets/` in the Overleaf project.
2. Upload the two PNGs into it.
3. Recompile.

## Where to get the PNGs

They are in this repo at:

```
docs/assets/axim_throughput.png
docs/assets/axim_latency.png
```

Download them from GitHub
(<https://github.com/GRRN-MAKER/Aximcomp/tree/main/docs/assets>) or regenerate locally:

```bash
pip install matplotlib pandas
python3 scripts/make_bench_charts.py     # rewrites both PNGs
```

## Tip: upload the whole repo

Overleaf can import a GitHub repo (New Project → Import from GitHub) or a zip. Importing
the whole `Aximcomp` repo brings `PAPER.tex` **and** `docs/assets/*.png` together, so the
figures resolve with no manual upload.
