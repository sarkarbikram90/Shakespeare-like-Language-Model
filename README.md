# Shakespeare Language Model

> A tiny model that produces Shakespeare-like grammar and style — trained and run entirely on CPU.

---

## Purpose

Train a miniature LLM on Shakespeare to build **intuition**, not performance.

After training, you'll see the model produce Shakespeare-like grammar and style. This is essentially a miniature version of what models like GPT-2 do internally.

---

## Setup

### Step 1 — Download the dataset

```bash
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### Step 2 — Install dependencies

```bash
pip install torch tqdm
```

### Step 3 — Run training

```bash
python train_shakespeare_tiny_llm.py
```

---

## What the model learns

After training, the model has learned — from characters only:

- English grammar
- Punctuation
- Speaker formatting
- Shakespeare style

---

## The big insight

You just recreated — on your laptop — the **core learning mechanism behind modern LLMs**.

Scaling this up gives models like GPT-3, PaLM, and LLaMA.

Same idea. Just millions of times bigger.

Happy LLMing...