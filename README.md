<p align="center">
  <h1 align="center">🧠 Exact Linear Attention (ELA)</h1>
  <p align="center">
    <em>Exact $O(L)$ Attention — No Approximation, No Compromise</em>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=flat-square" alt="Paper"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License"></a>
    <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/github/stars/yauntyour/Exact-Linear-Attention?style=flat-square&logo=github" alt="Stars">
    <img src="https://img.shields.io/github/last-commit/yauntyour/Exact-Linear-Attention?style=flat-square" alt="Last Commit">
    <br>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=yauntyour.Exact-Linear-Attention&style=flat-square" alt="Visitors">
    <a href="https://github.com/yauntyour/Exact-Linear-Attention/issues"><img src="https://img.shields.io/github/issues/yauntyour/Exact-Linear-Attention?style=flat-square" alt="Issues"></a>
    <a href="https://github.com/yauntyour/Exact-Linear-Attention/forks"><img src="https://img.shields.io/github/forks/yauntyour/Exact-Linear-Attention?style=flat-square" alt="Forks"></a>
  </p>
</p>

---

## 📋 Overview

Exact Linear Attention (ELA) is a novel attention mechanism that achieves **exact** linear-time attention computation ($O(L)$) by exploiting the exact decomposition property of kernel functions. Unlike prior linear attention methods that rely on approximation to the softmax attention, ELA performs mathematically identical computation to the full quadratic form proposed in the original Transformer [Vaswani et al., 2017] — just reorganized to avoid materializing the $L \times L$ attention matrix.

---

## 🎯 Key Contributions

### 1️⃣ Exact Linear Attention Formulation

Standard attention computes $\text{softmax}(QK^\top)V$ in $O(L^2)$. ELA instead decomposes a kernel function $k(A_i, B_j) = \langle \phi(A_i), \phi(B_j) \rangle$ and reorders computation:

$$Y_i = \frac{\phi(A_i) \sum_{j=1}^L \phi(B_j) V_j}{\phi(A_i) \sum_{j=1}^L \phi(B_j)}$$

This yields $O(L)$ complexity for bidirectional attention and $O(L)$ with a prefix sum for causal (autoregressive) attention — **without any approximation error**.

### 2️⃣ Suitable Kernel Functions

An ideal attention kernel should be:
- ✅ **Exactly decomposable** — admits an expansion that factors into separate query/key feature maps (per Mercer's theorem [Mercer, 1909])
- ✅ **Sufficiently discriminative** — smooth, broad output range to avoid gradient vanishing/explosion
- ✅ **Non-negative** — guarantees all attention weights are non-negative
- ✅ **Geometrically interpretable** — clear meaning in the embedding space

The paper identifies and analyzes three kernel families:

| Kernel | Expression | Behavior |
| --- | --- | --- |
| **Summation Squared Euclidean Distance** | $\|A_i + B_j\|^2$ | Emphasizes keys **aligned** with the query (supporting evidence) |
| **Subtraction Squared Euclidean Distance** | $\|A_i - B_j\|^2$ | Emphasizes keys **opposite** to the query (contrastive learning) |
| **Hadamard Exp Kernel** 🌟 | $\exp(A_i) \cdot \exp(B_j)$ | Amplifies feature **co-activation** patterns (multimodal, noise-robust) |

> 🌟 **The Hadamard Exp Kernel is recommended as the default choice** due to its strong nonlinearity, exact decomposability, smoothness, and non-negativity.

### 3️⃣ Custom Kernel Construction

You can design your own kernel function satisfying the four criteria above. The paper provides an example that recovers properties of standard attention:

$$k(A_i, B_j) = (A_i \cdot B_j + 1) \cdot (\|A_i\|^2 + 1) \cdot (\|B_j\|^2 + 1)$$

### 4️⃣ Hyper-Link: Residual Pathway Reconstruction

A replacement for traditional residual connections, building on Hyper-Connections [Zhu et al., 2024] and inspired by manifold-constrained variants [Xie et al., 2025]:
- Removes the attention residual branch, treating the entire Transformer layer as an integrated whole
- Leverages gated FFN outputs to adaptively modulate layer signals
- Achieves faster convergence and mitigates gradient degradation

### 5️⃣ Memory Module (Transformation Flow)

Inspired by biological neural memory systems — including goal-directed top-down control [Buschman & Miller, 2010], neural context representation [Polyn & Kahana, 2008], memory contextualization [Zhang et al., 2018], and hippocampal-prefrontal memory organization [de Sousa et al., 2026] — this module captures **qualitative memory** (behavioral judgment and rules) as distinct from **factual memory** (stored in e.g. lookup-based memory modules [Cheng et al., 2026]). It computes bidirectional linear attention over the layer-wise *transformation flow* $\Delta X_k = X_k - X_{k-1}$, allowing the model to implicitly learn and reuse processing experience. The QKV weight matrices are **pluggable** and can be embedded into any semantic-transformation-based model.

### 6️⃣ Engineering Contributions

- Analysis of the MoE "black-box" interpretability problem [Jain & Wallace, 2019] and a proposal to use expert label vectors with linear attention for routing
- Analysis of communication overhead reduction via token-block-level routing, drawing on efficient large-scale training principles [Jia et al., 2020]
- Practical optimization strategies for both bidirectional and causal linear attention, including efficient prefix-sum implementations compatible with linear attention frameworks [Katharopoulos et al., 2020; MiniMax Team, 2025; Kimi Team, 2025]

---

## 📊 Results

Experiments on a **5.8M-parameter GPT-style model** (4 layers, 4 attention heads, MoE with 4 experts), built on the MiniMind framework [Jingyao, 2026], show:

| Metric | Result |
|--------|--------|
| **Training Performance** | ELA variants match full attention GPT |
| **Anti-overfitting** | ELA shows slight advantage over full attention |
| **Convergence Speed** | ~10 epochs with Memory vs ~30 epochs without |
| **Memory Module Effect** | Abrupt loss drop at step ~750 (bidirectional only) |

---

## 🗂️ Project Structure

```
├── ela/                  # Core ELA implementation
│   ├── kernels/          # Kernel function implementations
│   ├── attention/        # Bidirectional & causal attention
│   └── layers/           # Transformer layer with Hyper-Link & Memory
├── examples/             # Usage examples
├── experiments/          # Training and evaluation scripts
├── ELA.pdf               # Paper
└── README.md
```

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@article{exactlinearattention2026,
  title={Exact Linear Attention: Achieving O(L) Complexity Without Approximation},
  author={Yuntian Yao},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📚 References

The paper builds upon prior work across machine learning, neuroscience, and kernel theory:

| Year | Work | Venue |
|------|------|-------|
| 1909 | Mercer, *"Functions of Positive and Negative Type, and Their Connection with the Theory of Integral Equations"* | Phil. Trans. R. Soc. Lond. A |
| 2008 | Polyn & Kahana, *"Memory Search and the Neural Representation of Context"* | Trends in Cognitive Sciences |
| 2010 | Buschman & Miller, *"Goal-Direction and Top-Down Control"* | Phil. Trans. R. Soc. B |
| 2017 | Vaswani et al., *"Attention Is All You Need"* | NeurIPS |
| 2018 | Zhang et al., *"Memory Contextualization: The Role of the Left Inferior Frontal Gyrus"* | J. Cognitive Neuroscience |
| 2019 | Jain & Wallace, *"Attention is not Explanation"* | NAACL-HLT |
| 2020 | Katharopoulos et al., *"Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"* | ICML |
| 2020 | Jia et al., *"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"* | SC'20 |
| 2021 | Schlag, Irie & Schmidhuber, *"Linear Transformers Are Secretly Fast Weight Programmers"* | ICML |
| 2022 | Qin et al., *"The Devil in Linear Transformer"* | EMNLP |
| 2024 | Zhu et al., *"Hyper-Connections"* | arXiv |
| 2025 | MiniMax Team, *"MiniMax-01: Scaling Foundation Models with Lightning Attention"* | arXiv |
| 2025 | Kimi Team, *"Kimi Linear: A Novel Hybrid Linear Attention Architecture"* | arXiv |
| 2025 | Xie et al., *"mHC: Manifold-Constrained Hyper-Connections"* | arXiv |
| 2026 | Cheng et al., *"Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"* | arXiv |
| 2026 | Jingyao, *"MiniMind: Train a Tiny LLM from Scratch"* | GitHub |
| 2026 | de Sousa et al., *"The Prefrontal Cortex Controls Memory Organization in the Hippocampus"* | Nature Neuroscience |

---

<p align="center">
  <a href="#-exact-linear-attention-ela">⬆ Back to Top</a>
</p>
