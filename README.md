# Stochastic GFlowNets for Kidney Exchange Programs: A Critical Extension

## Overview

By integrating Stochastic GFlowNets (Pan et al., 2023), we address inherent limitations in deterministic flow models under Markovian transitions in KEP, demonstrating through mathematical dissection how unmodeled stochasticity biases marginals, fairness metrics, and long-term expectations. The goal is to foster rigorous probabilistic fidelity in high-stakes combinatorial optimization, where approximations must confront the full entropy of real-world dynamics.

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

## Mathematical Dissection: Biases in Deterministic GFlowNets for Stochastic KEPs from a Social Planner's Perspective

In evaluating the application of GFlowNets to Kidney Exchange Programs (KEPs) as proposed in St-Arnaud et al. (2025), we adopt the lens of a social planner aiming to maximize aggregate welfare over time. In economic terms, the planner seeks to optimize the expected sum of transplants, weighted by quality-adjusted life years (QALYs), across a multi-round horizon: <img src="https://latex.codecogs.com/png.latex?\max_{\pi} \mathbb{E}_{\pi, M} \left[ \sum_{t=1}^T \gamma^{t-1} R(H_t) \right]" alt="Maximization Objective" style="vertical-align:middle" height="55">, where $\pi$ is the policy for selecting matchings $H_t$ in state $s_t = G_t$ (the compatibility graph at round $t$), $M$ is the stochastic transition kernel governing graph evolution, $R(H_t)$ is the reward (e.g., number of transplants or a welfare-weighted variant accounting for patient equity), $\gamma \in [0,1]$ discounts future rounds, and the expectation integrates over trajectories induced by $\pi$ and $M$. Each transplant contributes substantial welfare, estimated at over $5$ million in QALY-adjusted terms (Cutler and McClellan, 2001), making biases in $\pi$ economically inefficient—equivalent to foregone societal value.

The authors' simulator, adapted from Saidman et al. (2006), faithfully models stochastic graph evolution: incompatible pairs arrive via Poisson process with rate $\lambda = 5$ per round, blood types sampled from empirical distributions (O: 48.14%, A: 33.73%, B: 14.28%, AB: 3.85%), compatibility edges added per ABO rules (Dean, 2005), and outgoing edges stochastically removed via Bernoulli trials parameterized by calculated panel reactive antibody (cPRA) levels (low: 0.05 with 70.19% probability, medium: 0.45 with 20%, high: 0.90 with 9.81%; Tinckam et al., 2015). Graphs are constructed over $N=20$ rounds, yielding expected 100 vertices, but episodes focus on single-round dynamics ($N=1$) with 1000 trajectories per graph, generating a dataset of 100,000 vectors $(\vartheta_1, \dots, \vartheta_L)$, where each $\vartheta_i$ encodes a trajectory. Initial graph embeddings condition the model, with architecture details in their Table 6.

Standard GFlowNets train to sample matchings $H$ proportional to $R(H)$, enforcing trajectory balance (TB):

<p align="center">
  <img src="https://latex.codecogs.com/png.latex?Z%28s_0%3B%20%5Cphi%29%20%5Cprod_%7Bi%3D1%7D%5En%20P_F%28s_i%20%5Cmid%20s_%7Bi-1%7D%3B%20%5Ctheta%29%20%3D%20R%28x%29%20%5Cprod_%7Bi%3D1%7D%5En%20P_B%28s_%7Bi-1%7D%20%5Cmid%20s_i%3B%20%5Ctheta%29%2C" alt="Trajectory Balance Equation">
</p>

where $s_0$ is the initial state (empty matching on $G$), $\tau = (s_0 \to \dots \to s_n = x)$ is a trajectory terminating at maximal matching $x = (G, H)$, $Z$ is the learned partition function, $P_F$ and $P_B$ are forward and backward policies, and marginal termination probabilities satisfy $P_T(x) \propto R(x)$. This implies detailed balance (DB) localization:

<p align="center">
  <img src="https://latex.codecogs.com/png.latex?F%28s%29%20P_F%28s%27%20%5Cmid%20s%29%20%3D%20F%28s%27%29%20P_B%28s%20%5Cmid%20s%27%29%2C" alt="Detailed Balance Equation">
</p>

with state flows $F(s) = \sum_{\tau \ni s} F(\tau)$.

However, KEPs exhibit stochastic transitions: after selecting $H$ in $s_t = (G_t, H_t)$, the next graph $G_{t+1} \sim M(G_t \setminus H_t)$, incorporating Poisson arrivals, sampled compatibilities, and cPRA-driven removals. This yields a non-degenerate kernel $P(G_{t+1} \mid G_t, H_t) > 0$ for multiple $G_{t+1}$, with entropy $\mathcal{H}[P(\cdot \mid G_t, H_t)] > 0$. The authors' GFlowNet assumes deterministic transitions $T(s_t, H_t) = s_{t+1}$, optimizing toward $\prod_t R(H_t)$ while ignoring the kernel, effectively treating rounds as independent or graphs as static.

This induces bias in the learned policy $\pi_\theta(H \mid G) \approx P_F$, deviating from the social planner's optimum. The true target posterior over multi-round trajectories $\tau = (G_0, H_1, G_1, H_2, \dots, G_T, H_T)$ is:

<p align="center">
  <img src="https://latex.codecogs.com/png.latex?P%5E%2A%28%5Ctau%29%20%5Cpropto%20%5Cprod_%7Bt%3D1%7D%5ET%20R%28H_t%29%20%5Ccdot%20P%28G_t%20%5Cmid%20G_%7Bt-1%7D%2C%20H_%7Bt-1%7D%29%2C" alt="Trajectory Posterior Equation">
</p>

with $P(\cdot \mid \cdot, \cdot)$ from $M$. The GFlowNet marginal $P_T(\tau) \propto \prod_t R(H_t)$ omits the kernel, yielding Kullback-Leibler divergence:

<p align="center">
  <img src="https://latex.codecogs.com/png.latex?D_%7B%5Cmathrm%7BKL%7D%7D%5Cleft%28P%5E%2A%20%5Cparallel%20P_T%5Cright%29%20%3D%20%5Cmathbb%7BE%7D_%7BP%5E%2A%7D%5Cleft%5B%5Csum_t%20%5Clog%20P%28G_t%20%5Cmid%20G_%7Bt-1%7D%2C%20H_%7Bt-1%7D%29%20%5Cright%5D%20-%20%5Cmathcal%7BH%7D%5BP%5E%2A%5D%20%2B%20%5Cmathcal%7BH%7D%5BP_T%5D" alt="KL divergence">
</p>

Since:

<p align="center">
  <img src="https://latex.codecogs.com/png.latex?%5Cmathcal%7BH%7D%5BP%5E%2A%5D%20%3D%20%5Cmathcal%7BH%7D%5BP_T%5D%20%2B%20%5Csum_t%20%5Cmathbb%7BE%7D_%7BP%5E%2A%7D%20%5B%5Cmathcal%7BH%7D%5BP%28G_t%20%5Cmid%20%5Ccdot%29%5D%5D" alt="Entropy Decomposition Equation">
</p>

(by chain rule, with extra entropy from transitions), and the expectation term is positive unless $M$ is deterministic (impossible given Poisson/Bernoulli variability), **$D_{\text{KL}} > 0$**. This gap scales with branching: higher $\mathcal{H}[P(\cdot \mid \cdot)]$ biases $P_T$ toward low-variance modes, as unmodeled stochasticity penalizes exploratory paths with dispersed downstream rewards. In economic terms, this collapses to greedy, short-sighted policies, underweighting matchings that preserve pool diversity for future welfare gains—e.g., sparing hard-to-match pairs yields $\mathbb{E}[\sum R_t \mid H]$ inflated by replenishment, but variance suppresses sampling.

Gradient variance in TB exacerbates inefficiency:

<p align="center">
  <img src="https://latex.codecogs.com/png.latex?%5Cnabla%20L_%7B%5Ctext%7BTB%7D%7D%20%5Capprox%20%5Csum_%5Ctau%20%5Cnabla%20%5Clog%20P_F%28%5Ctau%29%20%5Ccdot%20%5Cleft%28%20%5Clog%20Z%20%2B%20%5Csum%20%5Clog%20P_F%20-%20%5Clog%20R%20-%20%5Csum%20%5Clog%20P_B%20%5Cright%29%2C" alt="Gradient of TB Loss">
</p>

where unmodeled $M$ injects noise via sampled $G_{t+1}$, with $\text{Var}[\nabla L] = O(T \cdot \bar{b})$ ($T$ rounds, $\bar{b}$ average branches). As $T \to \infty$, variance diverges, inducing mode collapse (Pan et al., 2023, Figs. 5-9), flattening $P_T$ to uniform under exploration ($\alpha \to 1$) or greedy under exploitation.

Time-dependency amplifies this on real data, fracturing the assumed static DAG. In the simulator, graphs accumulate stochasticity over 20 rounds, but single-round episodes ($N=1$) treat states as independent, ignoring propagation: later-round sparsity/edges depend on prior $H_t$ via $M$. Real KEPs exhibit stronger dependency due to demographic biases in cPRA: Black patients are disproportionately sensitized (cPRA >80% in ~25-30% of cases vs. ~15-20% for Whites; Reese et al., 2016, from UNOS data implying higher from disparities in prior exposures), as are Hispanics (~20-25% high cPRA; Pando et al., 2018). Statistically, Black candidates comprise ~32% of the waitlist but receive ~25% of transplants, with higher cPRA causing longer dwell times (OPTN/SRTR 2022 Report: Blacks have 2x ESRD incidence, lower transplant rates). This skews pool composition: minorities accumulate, altering edge densities (fewer outgoing for high-cPRA), making $P(G_t \mid \cdot)$ history-dependent. Proved via conditional entropy: $\mathcal{H}[G_t \mid G_{t-1}, H_{t-1}] < \mathcal{H}[G_t]$ (arrivals conditioned on removals), but cross-round $\mathbb{E}[\text{cPRA}(G_t)] > \mathbb{E}[\text{cPRA}(G_0)]$ due to retention, with racial correlation $\text{Cov}(\text{cPRA}, \text{Race}) > 0$ (e.g., regression coefficients from SRTR: $\beta_{\text{Black}} \approx 0.15-0.20$ for high-cPRA probability). Thus, assuming time-independence biases marginals, under-sampling equity-weighted $H$ for minorities, reducing planner welfare by ~10-20% in disparate access (Mohandas et al., 2022 estimates).

Outside the authors' a priori simulated pipeline—where graphs are generated homogenously without real demographic correlations—the model breaks: real UNOS data introduces persistent time-dependencies, inflating $D_{\text{KL}}$ as simulated embeddings fail to capture evolving distributions, leading to higher variance and collapse (e.g., test KL spikes 2-5x in Pan et al. analogs).

To correct, integrate Stochastic GFlowNets (Pan et al., 2023) via sub-trajectory balance, incorporating the kernel:

<p align="center">
  <img src="https://latex.codecogs.com/png.latex?F%28s%29%20%5Cpi%28a%20%5Cmid%20s%29%20P%28s%27%20%5Cmid%20s%2C%20a%29%20%3D%20F%28s%27%29%20%5Cpi_B%28%28s%2C%20a%29%20%5Cmid%20s%27%29%2C" alt="Sub-trajectory balance equation">
</p>

with losses using $\hat{P} \approx M$ (MLE or Monte Carlo samples from simulator/real traces). This aligns $P_T \to P^*$, driving $D_{\text{KL}} \to 0$, stabilizing gradients, and enabling unbiased multi-round optimization—better serving the planner by capturing full entropy for equitable, long-term welfare maximization.

## KEP Environment Simulation

The core KEP environment simulates a kidney exchange program with incompatible patient-donor pairs and optional altruistic donors. Key features include:

Pair Generation: Incompatible pairs are generated based on blood type distributions (O: 48.14%, A: 33.73%, B: 14.28%, AB: 3.85%) and cPRA levels (low: 0.05 with 70.19% prob, medium: 0.45 with 20%, high: 0.90 with 9.81%). ABO compatibility is checked, with crossmatch failures simulated for compatible pairs.
Graph Construction: A directed graph where nodes represent pairs (patient/donor blood types, cPRA) or altruists (donor blood type only). Edges represent compatibility (ABO match + negative crossmatch).
Multi-Round Simulation: Pairs arrive over rounds via Poisson process (optional; equivalent to single round for static graphs in this setup).
Cycle and Chain Detection: Identifies cycles (up to max length, e.g., 3) and chains starting from altruists (up to max length, e.g., 4).
MIP Solver: Uses integer programming to maximize transplants by selecting disjoint cycles/chains.
Environment Interface: Provides a stateful environment for stepwise actions (select cycle/chain or terminate), tracking remaining graph and matched pairs. Rewards are exponential in matched pairs upon maximal termination.
Example output for a small instance (8 pairs, no altruists):

![KEP Env Figure](img/Figure_2.png)

Nodes table with IDs, types, blood types, cPRA.
Edges table listing source-target compatibilities.
Maximum matched pairs (e.g., via MIP).
Selected cycles/chains.
Visualization shows nodes colored by patient blood type (O: red, A: cyan, B: green, AB: yellow), with directed edges for compatibilities.

## Experiment: Demonstrating Collapse in Stochastic Environments (v1)

To illustrate the problem highlighted in the mathematical analysis, I've added a preliminary experiment inspired by the stochastic chain environment from Pan et al. (2023). This serves as a proof-of-concept to show how standard GFlowNets falter in stochastic settings, while the stochastic variant holds up. It's a simple chain where you start at state 0 and can either "stop" (terminate with reward at current state) or "continue" (move forward with probability p=0.5 or stay put with 0.5). The reward function is bimodal, with peaks around N/4 and 3N/4, making it a good test for capturing multiple modes without collapse.

In this env, the standard GFlowNet treats transitions as deterministic, ignoring the probabilistic branching. As N (chain length) grows, unmodeled stochasticity injects noise into the gradients, leading to higher variance, biased marginals, and eventual mode collapse— the model favors low-entropy paths, missing the true distribution. You see the KL divergence (measuring how far the sampled distribution is from the true posterior) spike for the standard version, while the stochastic one, which explicitly includes P(s' | s, a) in the balance, keeps the divergence low and stable.

This is just v1—a quick iteration to get the ball rolling and validate the critique. I'm building this iteratively, in the spirit of fail-fast-and-iterate: prototype, test, refine, push. It's how we move at speed in a world that's not slowing down—think Musk vs. the inertia of traditional research, where velocity can be intimidating but necessary for progress. No arrogance here, just the drive to expose flaws and fix them before they cost lives in applications like KEPs. For now, this Pan-inspired toy shows why stochasticity breaks the standard approach; I'm finalizing the full KEP simulation on my local machine (with Poisson arrivals, compatibility graphs, etc.) and will push updates in the coming hours or days. Expect polished proofs, more experiments, and iterative improvements as we die and retry to get it right.

Running the code below generate the plots. Here's the performance comparison (KL divergence vs. environment size):

![Performance Comparison: Standard vs Stochastic GFlowNet](img/Figure_1.png)  <!-- Replace with actual image link or embed -->

The env is highly stochastic

In the plot, the blue line (standard) climbs sharply as N increases, showing the collapse, while orange (stochastic) rises gently, staying closer to the truth.

    Code (PyTorch-based, runs locally):

## Future Directions

The next phase involves implementing a GFlowNet model to test on the KEP environment. The goal is to extend the approach by incorporating Stochastic GFlowNets with sub-trajectory balance, evaluating whether this leads to better convergence, reduced bias, and improved handling of stochastic dynamics in KEP simulations.

## References

- Pan et al. (2023). Stochastic Generative Flow Networks.
- St-Arnaud et al. (2025). A Learning-Based Framework for KEPs.
- Bengio et al. (2021). GFlowNet Foundations.

## License

This repository, including all text, mathematical derivations, conceptual simulations, and derived works, is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). You are free to share, copy, distribute, and adapt the material for non-commercial purposes, provided you give appropriate credit to the author, provide a link to the original repository (https://github.com/yourusername/stochastic-kep-critique.git), and indicate if changes were made.

Citation Requirement: Any use of this work must include a citation to the author (please contact the repository owner for the preferred citation format) and the repository URL. For inquiries or to notify the author of use, contact the repository owner directly.

Commercial use is prohibited without explicit written permission from the author. Violations of this license will be pursued to the fullest extent permitted by law.


