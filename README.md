# Conformer ASR Model (PyTorch)

This repository implements an end-to-end Automatic Speech Recognition (ASR) model using the **Conformer** architecture, written in PyTorch. The architecture integrates convolutional subsampling, multi-head self-attention (with optional RoPE), feed-forward blocks, and convolution modules following the design in [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100).

## Features

* **SpecAugment** for robust data augmentation
* **Conformer blocks** with:

  * Feed-Forward layers
  * Self-Attention with optional Rotary Positional Embedding (RoPE)
  * Convolutional modules (GLU + Depthwise Conv)
* **ConvSubsampling** for input compression
* CTC-based decoder with **greedy** and **beam search** (KenLM optional)
* Weight initialization options: **Xavier** and **Kaiming**
* Inference wrapper with token collapsing

## Architecture Overview

```
Input Spectrogram → SpecAugment → ConvSubsampling
→ N × [FFN → SelfAttention (RoPE+GQA optional) → Convolution → FFN]
→ Linear Decoder (CTC) → Logits
```

## Usage

### Model Initialization

```python
model = Conformer(config, vocab_size)
```

### Forward Pass

```python
logits, lengths = model(spectrograms, lengths)
```

### Inference

```python
decoded = model.infer(
    spectrogram, spectrogram_length,
    tokenizer=your_tokenizer,
    decoding_strategy="beam",  # or "greedy"
    beam_width=5,
    kenlm_path="path/to/kenlm.arpa"
)
```

### Weight Initialization

```python
init_weights(model, init_type='xavier')  # or 'kaiming'
```

## Configuration Example

```python
config = {
    "feat_in": 80,
    "d_model": 256,
    "ff_expansion_factor": 4,
    "dropout": 0.1,
    "dropout_att": 0.1,
    "dropout_pre_encoder": 0.1,
    "n_heads": 4,
    "conv_kernel_size": 31,
    "n_layers": 16,
}
```

## Dependencies

* PyTorch ≥ 1.10
* `pyctcdecode` for beam decoding with KenLM

Install with:

```bash
pip install torch pyctcdecode
```

## Notes

* SelfAttention supports **causal masking**, **GQA**, and **Rotary Positional Embedding**.
* Decoder supports **CTC greedy** and **beam search** decoding with optional **KenLM**.
* Subsampling reduces time resolution by factor of 4.
* Use `SpecAugment` during training only.


