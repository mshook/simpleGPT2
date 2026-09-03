# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch GPT-2 inference implementation in pure NumPy, derived from
[picoGPT](https://github.com/jaymody/picoGPT) and the blog post
[GPT in 60 Lines of Numpy](https://jaykmody.com/blog/gpt-from-scratch/). It is a teaching
artifact (there is an accompanying [YouTube walkthrough](https://www.youtube.com/embed/8NCL03ZKNxY)),
not a library: no tests, no build step, no package metadata, no linter config.

Training is out of scope — this is forward-pass/inference only, using OpenAI's released weights.

## Setup and running

```bash
pip install -r requirements.txt

# Entry points (all three are the same model, different levels of abstraction):
python gpt2.py "not all heroes wear capes" --n_tokens_to_generate 40   # fire CLI, returns text
python gpt2_pico.py "not all heroes wear capes"                        # same, minified
python simpleGPT2.py                                                   # prompt hardcoded at bottom of file
```

`gpt2.py` / `gpt2_pico.py` use [fire](https://github.com/google/python-fire), so any `main()`
kwarg becomes a flag (`--model_size 355M`, `--models_dir models`).

On first run `utils.load_encoder_hparams_and_params` downloads the checkpoint from OpenAI's blob
storage into `models/<model_size>/` (~500MB for 124M; sizes are `124M`, `355M`, `774M`, `1558M`).
There is no `.gitignore`, so avoid `git add -A` after a download.

`utils.py` imports **TensorFlow** purely to read the OpenAI TF checkpoint — that is the only reason
TF is a dependency. The model math itself never touches it.

## Architecture

Weights live in a single nested dict, `params`, whose shape mirrors the GPT-2 checkpoint's variable
names. Every function takes its slice of that dict and is called with `**`-splat, which is why
argument names (`g`, `b`, `w`, `c_fc`, `c_proj`, `c_attn`, `ln_1`, `ln_2`, `mlp`, `attn`, `wte`,
`wpe`, `ln_f`, `blocks`) are load-bearing — renaming a parameter breaks the splat:

```
params = {wte, wpe, ln_f: {g,b}, blocks: [ {ln_1:{g,b}, ln_2:{g,b},
                                            attn: {c_attn:{w,b}, c_proj:{w,b}},
                                            mlp:  {c_fc:{w,b},  c_proj:{w,b}}} ]}
```

`hparams` comes from the checkpoint's `hparams.json` (`n_ctx`, `n_head`, `n_layer`, `n_embd`, `n_vocab`).

Linear layers use the checkpoint's `[in, out]` weight convention (`x @ w + b`), so no transposition
is needed anywhere. Output logits are produced by tying to the embedding matrix: `x @ wte.T`.
Sampling is greedy `argmax` only; there is no temperature/top-k.

### The three implementations

| File | Role |
|---|---|
| `gpt2.py` | Reference version: shape comments on every line, functions decomposed (`mha`, `attention`, `ffn`, `transformer_block`, `gpt2`, `generate`). Change this one first when altering model behavior. |
| `gpt2_pico.py` | Same code with comments and intermediate variables stripped. Keep it behaviorally identical to `gpt2.py`. |
| `simpleGPT2.py` | The video's version: the whole transformer stack is inlined into `main()` as explicit loops so each step is visible. Prints tokens as they are generated instead of returning a string. |

Note `simpleGPT2.py` splits QKV differently from `gpt2.py`: it does one
`np.split(qkv, 3 * n_head, axis=-1)` and indexes head `i`'s q/k/v as `[i]`, `[i + n_head]`,
`[i + 2*n_head]`, whereas `gpt2.py` splits into q/k/v first and then into heads. Both produce the
same slices; don't "fix" one to match the other.

`encoder.py` is OpenAI's BPE tokenizer copied verbatim — treat it as vendored, not as code to
refactor.

## simpleGPT2_HF.ipynb

A self-contained rewrite that inlines `encoder.py` and `utils.py` into the notebook and pulls
weights/tokenizer from **Hugging Face** (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`) via a
hand-rolled safetensors parser — no TensorFlow, no local `models/` checkout, but it needs network
access to `huggingface.co`. HF's Conv1D weights are already `[in, out]`, so the loader only renames
flat `transformer.h.{i}...` keys into the nested `params` layout above. The modeling cells are
copied from `simpleGPT2.py`; keep them in sync when changing that file.
