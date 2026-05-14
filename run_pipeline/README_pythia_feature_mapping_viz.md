# Pythia SAE feature mapping visualization

`pythia_feature_mapping_viz.py` builds a **single self-contained HTML file** that shows two linked SAE feature spaces (two Pythia checkpoints by default). It:

- Loads the same text batch for both models (Hugging Face `datasets`).
- Runs both base LMs and **Eleuther `sparsify` SAEs** on the residual stream at chosen layers.
- Picks active features, estimates **cross-model feature correspondence** via a full activation correlation matrix, then **greedy**, **Hungarian** (default one-to-one max-sum correlation over the pool), or **Sinkhorn OT** (POT) alignment; runs **UMAP** on decoder directions for the selected features; writes **JSON + HTML** with **quantitative similarity metrics** (linear CKA, RSA Spearman on activation RDMs, orthogonal Procrustes relative RMSE, pool correlation stats).

## Prerequisites

- Python 3.8+ (project often uses 3.11; see root `README.md` for conda setup).
- Install root dependencies, then SAE stack (as in the main README):

  ```bash
  pip install -r requirements.txt
  pip install sae_lens "git+https://github.com/wlg1/sparsify.git"
  ```

- **Hugging Face**: models, datasets, and SAE weights download on first run. For gated models, run `huggingface-cli login`.
- **GPU**: Uses CUDA when available (`torch.cuda.is_available()`). CPU is possible for the default small Pythia pair but will be slow.
- **Disk / RAM**: Default runs use small models; larger pairs need proportionally more VRAM.

## How to run

From the **repository root** (`feature_space_mapping_UI/`, parent of `run_pipeline/`):

```bash
mkdir -p outputs
python run_pipeline/pythia_feature_mapping_viz.py
```

This writes:

- `outputs/pythia_sae_feature_map.json` — full payload (for re-rendering or tooling).
- `outputs/pythia_sae_feature_map.html` — open in a browser (double-click or `file://` path).

Override paths with `--data-json` and `--output-html`.

If you run from inside `run_pipeline/`, use `python pythia_feature_mapping_viz.py` (the script falls back to sibling imports when `run_pipeline.*` is not on the path).

### Quick examples

```bash
# Smaller feature set, explicit outputs
python run_pipeline/pythia_feature_mapping_viz.py \
  --features-per-side 800 \
  --dataset-split "train[:256]" \
  --output-html outputs/pythia_sae_feature_map.html \
  --data-json outputs/pythia_sae_feature_map.json

# Re-render HTML from a saved JSON only (no model forward pass)
python run_pipeline/pythia_feature_mapping_viz.py \
  --from-json outputs/pythia_sae_feature_map.json \
  --output-html outputs/pythia_sae_feature_map_rerender.html
```

## Command-line reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-a-name` | `EleutherAI/pythia-70m` | HF causal LM id for side A. |
| `--model-b-name` | `EleutherAI/pythia-160m` | HF causal LM id for side B. |
| `--sae-a-name` | `EleutherAI/sae-pythia-70m-32k` | SAE hub id for A (`sparsify` / eleuther path). |
| `--sae-b-name` | `EleutherAI/sae-pythia-160m-32k` | SAE hub id for B. |
| `--layer-a` | `2` | Residual-stream layer index for A’s SAE hook. |
| `--layer-b` | `3` | Residual-stream layer index for B’s SAE hook. |
| `--dataset-name` | `roneneldan/TinyStories` | HF dataset for prompts. |
| `--dataset-split` | `train[:256]` | Split / slice (datasets syntax). |
| `--dataset-text-key` | `text` | Column containing text. |
| `--num-samples` | `256` | Number of rows to draw from the split. |
| `--max-length` | `128` | Tokenizer max length per example. |
| `--batch-size` | `16` | LM + SAE batching inside `get_sae_actvs`. |
| `--corr-batch-size` | `128` | Batch size when building the full activation correlation matrix. |
| `--corr-pool-size` | `3000` | Top active features per side before alignment (equal counts; required for Hungarian). |
| `--mapping-method` | `hungarian` | `greedy`, `hungarian` (one-to-one max-sum correlation), or `ot` (Sinkhorn + column argmax; needs `POT`). |
| `--ot-reg` | `0.05` | Sinkhorn regularization for `--mapping-method ot`. |
| `--features-per-side` | `1200` | Target number of features to show on the B side (A side is derived from mappings). |
| `--label-top-k` | `5` | Top activating tokens per feature for hover labels. |
| `--seed` | `42` | RNG seed (UMAP). |
| `--data-json` | `outputs/pythia_sae_feature_map.json` | Where to save the JSON payload. |
| `--output-html` | `outputs/pythia_sae_feature_map.html` | Where to save the HTML. |
| `--title` | `Pythia SAE Feature Space Mapping` | Browser / page title. |
| `--from-json` | *(unset)* | If set, skip compute; load payload from this path and only rebuild semantic groups + HTML. |
| `--semantic-categories-json` | *(unset)* | Optional JSON file defining semantic buckets for coloring (see below). |

Implementation detail: the script always calls `get_sae_actvs` with `sae_lib="eleuther"` (sparsify), not the `sae_lens` branch in `get_actv_fns.py`.

## Semantic categories (optional)

By default, points are grouped using **built-in keyword lists** (and optional regex per group) inside `pythia_feature_mapping_viz.py` (`DEFAULT_SEMANTIC_CATEGORIES`).

To override, pass `--semantic-categories-json` pointing to a JSON **array** of objects:

```json
[
  {
    "id": "code",
    "name": "Code-like",
    "keywords": ["def", "return", "import", "class"],
    "pattern": "[{}();]"
  }
]
```

- `keywords`: substring matches (case-insensitive) against each feature’s comma-separated top-token label string.
- `pattern`: optional regex string; if present, a label can also match via regex.

The same file is applied when using `--from-json` so you can recolor without recomputing activations.

## Using the HTML

- Open the generated `.html` locally; the bundle loads Plotly from a CDN, so **an internet connection** is required for the first plot render unless you change the template.
- **Hover** a point on one UMAP to highlight the mapped feature on the other side (when a correlation mapping exists).
- The UI includes **drag modes** (e.g. box select / paint) documented in on-page controls; selected pairs can appear in a table.

## Troubleshooting

- **`RuntimeError: No mapped features were selected`**: Increase `--corr-pool-size`, `--num-samples`, or `--features-per-side` less aggressively; ensure layers / SAE names match the base models.
- **OOM / CUDA errors**: Lower `--batch-size`, `--max-length`, `--num-samples`, or `--features-per-side`.
- **Import errors**: Install `umap-learn` and ensure you are running from repo root or have `run_pipeline` on `PYTHONPATH` as described above.

## Related files

- `get_actv_fns.py`, `correlation_fns.py`, `interpret_fns.py` — shared helpers from the main `run_pipeline` stack.
- Root `README.md` — paper repo overview and environment setup.
