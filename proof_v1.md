
## Mathematic Dissection

Standards GFlowNets enforce equilibrium under the veil of determinism, where transitions $T(s,a)=s'$ map uniquely, preserving a pristine directed acyclic graph (DAG). For a trajectory $\tau = (s_0 \to \dots \to s_n =x)$, trajectory balance (TB) mandates

$$
Z \prod_{i=1}^{n} P_F(s_i \mid s_{i-1}) = R(x) \prod_{i=1}^{n} P_B(s_{i-1} \mid s_i)
$$

ensuring marginals $P_T(x) = \sum_{\tau \to x} P_F(\tau) \proto R(x)$, with $Z$ the partition scalar. Detailed balance (DB) localizes this to edges: $F(s) P_F(s' \mid s) = F(s') P_B(s \mid s')$, where state flows $F(s) = \sum_{\tau \ni s} F(\tau)$ aggregate trajectory contributions.

Yet, in KEPs, external transitions $G' \sim M(G, H)$-Poisson arrivals ($\lambda=5$), Bernoulli compatibilities (cPRA-modulated)-shatter this DAG into a probabilitic thicket. Each action (selecting $H$) branches stochastically, yielding multiple $G'$ with probabilities $P(G' \mid (G,H)) > 0$ for $|G'| > 1$. The assumption $T(s,a) = s'$ unique fractures: marginals $P_T(H) \propto R(H)$ remain tethered to local rewards, oblivious to $\mathbb{E}[\sum_t R_t \mid \pi, M]$, the true horizon under Markovian evol...

To unearth the absurdity, consider the flow inconsistency. In standard GFlowNets, the forward marginal over exchanges $H$ is $P_T(H) = \int P_F(\tau_H) d\tau_H$, where $\tau_H$ builds $H$ deterministically. Post-matching, the next state $s' = (G', \emptyset) \sim M$ introduce unmodeled multiplicity: the effective flow to downstream rewards $R_{t+1}(H')$ disperses across branches, yet upstream $P_F$ receives no backpropagation of this variance. Formally, the true target distribution over trajectories span...

$$
D_{KL}(P^* \parallel P_T) = \mathbb{E}_{P^*} \left[ \log \frac{\prod_t R(H_t) \prod_t P(s_{t+1} \mid (s_t, H_t))}{\prod_t R(H_t)} \right]
= \mathbb{E}_{P^*} \left[ \sum_t \log P(s_{t+1} \mid (s_t, H_t)) \right] > 0,
$$

unless $P(\cdot \mid (s,H)) = \delta$ (Dirac, deterministic)-an impossibility in KEPs. This $D_{KL} > 0$ grows with stochasticity: as branching factor $|{s' \mid P(s' \mid (s, H)) > 0}|$ increases (e.g., via cPRA variance), entropy $\mathbb{H}[P(\cdot \mid (s, H))]$ inflates the gap, biaising $P_T$ toward entropy-minimizing greed (high local $R$), at the expense of expected global utility.
