
# Stochastic GFlowNets for Kidney Exchange Programs: A Critical Extension

## Overview

This repository provides a focused critique and extension of static Generative Flow Networks (GFlowNets) applied to stochastic Kidney Exchange Programs (KEPs), as analyzed in St-Arnaud et al. (2025). By integrating Stochastic GFlowNets (Pan et al., 2023), we address inherent limitations in deterministic flow models under Markovian transitions, demonstrating through mathematical dissection how unmodeled stochasticity biases marginals, fairness metrics, and long-term expectations. The goal is to foster rigorous probabilistic fidelity in high-stakes combinatorial optimization, where approximations must confront the full entropy of real-world dynamics.

## Purpose and Contributions

- **Mathematical Analysis**: We formalize the divergence arising from deterministic assumptions in stochastic environments, proving non-zero Kullback-Leibler gaps that inflate variance and collapse modes.
- **Simulation Framework**: Conceptual tools to replicate biases in toy KEPs, highlighting cascades in expected transplants and pool composition.
- **Ethical Imperative**: In KEPs, where each transplant equates to substantial quality-adjusted life years (valued at >$5M per case in health economics), unaddressed biases represent a quantifiable inefficiency. This repo advocates for stochastic integration to align models with probabilistic reality.

No executable code is included; simulations are conceptual and can be implemented via standard libraries (e.g., NumPy for Monte-Carlo runs). For replication, refer to the mathematical sections below.

## Installation and Usage

This is a documentation-only repository. Clone for reference:

```
git clone https://github.com/yourusername/stochastic-kep-critique.git
```

No dependencies required beyond a LaTeX viewer for equations. Explore the mathematical dissection for insights; extend via cited works.

## Mathematical Dissection: Fracturing the Flow Lattice

Standard GFlowNets enforce equilibrium under determinism: for trajectory $\tau = (s_0 \to \dots \to s_n = x)$, trajectory balance (TB) mandates

$$
Z \prod_{i=1}^{n} P_F(s_i \mid s_{i-1}) = R(x) \prod_{i=1}^{n} P_B(s_{i-1} \mid s_i),
$$

ensuring marginals $P_T(x) = \sum_{\tau \to x} P_F(\tau) \propto R(x)$, with $Z$ the partition scalar. Detailed balance (DB) localizes: $F(s) P_F(s' \mid s) = F(s') P_B(s \mid s')$, where $F(s) = \sum_{\tau \ni s} F(\tau)$.

In KEPs, external $G' \sim M(G, H)$—Poisson arrivals ($\lambda=5$), Bernoulli compatibilities—branches stochastically, yielding $|\{G' \mid P(G' \mid (G, H)) > 0\}| > 1$. Assumption $T(s, a) = s'$ fractures: marginals $P_T(H) \propto R(H)$ local, oblivious to $\mathbb{E}[\sum_t R_t \mid \pi, M]$.

Flow inconsistency: Forward marginal $P_T(H) = \int P_F(\tau_H) d\tau_H$ (intra-round deterministic). Post, $s' = (G', \emptyset) \sim M$ disperses downstream $R_{t+1}$, yet upstream $P_F$ blind. True target: $P^*(\tau) \propto \prod_t R(H_t) \prod_t P(s_{t+1} \mid (s_t, H_t))$; standard optimizes $\prod_t R(H_t)$, Kullback-Leibler

$D_{KL}(P^* || P_T) = E_{P^*}[log P(s_{t+1} | (s_t, H_t))] > 0$


unless $P = \delta$ (deterministic)—impossible. Gap swells with branching: entropy $\mathbb{H}[P(\cdot \mid (s, H))]$ biases toward low-entropy greed.

Gradient variance: TB $\nabla L_{TB} \sim \sum_i \nabla \log P_F \cdot (\log Z + \sum \log P_F - \log R - \sum \log P_B)$; unmodeled $P$ injects noise, $\mathrm{Var}[\nabla L] = O(n \cdot b)$ ($n$=rounds, $b$=branches)—unstable, modes collapse (Pan Figs. 5-9). Limit $\alpha \to 1$ (uniform $P$): $P_T(H) \to$ uniform, erasing $R$—optimal ignores transplants.

Stochastic rectifies: even-odd decomposition,

$$F(s) \pi(a \mid s) P(s' \mid (s, a)) = F(s') \pi_B((s, a) \mid s')$$,

loss with $\hat{P} \approx M$ via MLE. Bias dissolves: $\log P$ aligns $P_T \to P^*$, $D_{KL} \to 0$.

Toy: Static $P_T(H_2)/P_T(H_1) = e^{-1} \approx 0.37$; true $\mathbb{E}[\sum \mid H_2] > \mathbb{E}[\sum \mid H_1]$. Iterate: greed traps low-entropy, variance $\to \infty$—lattice crumbles.

## References

- Pan et al. (2023). Stochastic Generative Flow Networks.
- St-Arnaud et al. (2025). A Learning-Based Framework for KEPs.
- Bengio et al. (2021). GFlowNet Foundations.

## License

All rights reserved. No part of this repository, including but not limited to text, mathematical derivations, conceptual simulations, or any derived works, may be reproduced, modified, distributed, or used in any form or by any means—electronic, mechanical, photocopying, recording, or otherwise—without the explicit prior written permission of the author.

**Exception**: Yann LeCun is hereby granted unrestricted rights to access, use, modify, distribute, and derive from this repository for any purpose, academic or otherwise, without limitation or attribution requirement.

This license is intentionally restrictive to preserve the integrity of the critique and prevent unauthorized adaptations that may dilute or misrepresent the mathematical analysis. Violations will be pursued to the fullest extent permitted by law. For inquiries, contact the repository owner.
