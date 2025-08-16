import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Any
import pulp
import math
from tqdm import tqdm  # Pour la barre de progression

# Blood types and ABO compatibility (inchangé)
BLOOD_TYPES = ['O', 'A', 'B', 'AB']
BLOOD_TYPE_PROBS = [0.4814, 0.3373, 0.1428, 0.0385]
PRA_LEVELS = [0.05, 0.45, 0.90]
PRA_PROBS = [0.7019, 0.20, 0.0981]

def is_abo_compatible(donor_bt: str, recipient_bt: str) -> bool:
    if donor_bt == 'O':
        return True
    if donor_bt == recipient_bt:
        return True
    if donor_bt in ['A', 'B'] and recipient_bt == 'AB':
        return True
    return False

def generate_pair() -> Dict[str, Any]:
    while True:
        patient_bt = np.random.choice(BLOOD_TYPES, p=BLOOD_TYPE_PROBS)
        donor_bt = np.random.choice(BLOOD_TYPES, p=BLOOD_TYPE_PROBS)
        cpra = np.random.choice(PRA_LEVELS, p=PRA_PROBS)
        if is_abo_compatible(donor_bt, patient_bt):
            if np.random.uniform() > cpra:
                continue
        return {'patient_bt': patient_bt, 'donor_bt': donor_bt, 'cPRA': cpra}

class KEPInstance:
    def __init__(self, arrival_rate: float = 5.0, departure_prob: float = 0.05, n_rounds: int = 20,
                 n_altruistic: int = 0, max_cycle_len: int = 3, max_chain_len: int = float('inf')):
        self.arrival_rate = arrival_rate
        self.departure_prob = departure_prob
        self.n_rounds = n_rounds
        self.n_altruistic = n_altruistic
        self.max_cycle_len = max_cycle_len
        self.max_chain_len = max_chain_len
        self.pair_nodes = []
        self.altru_nodes = []
        self.remaining_nodes = []
        self.graph = self._generate_graph()

    def _generate_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        node_id = 0
        if self.n_rounds == 1:
            num_pairs = int(np.random.poisson(self.arrival_rate))
            pairs = [generate_pair() for _ in range(num_pairs)]
            for p in pairs:
                g.add_node(node_id, type='pair', **p)
                self.pair_nodes.append(node_id)
                self.remaining_nodes.append(node_id)
                node_id += 1
            self._add_edges(g, self.remaining_nodes)
        else:
            for _ in range(self.n_rounds):
                to_remove = [n for n in self.remaining_nodes if np.random.uniform() < self.departure_prob]
                for n in to_remove:
                    g.remove_node(n)
                    self.remaining_nodes.remove(n)
                    if n in self.pair_nodes:
                        self.pair_nodes.remove(n)
                num_new = np.random.poisson(self.arrival_rate)
                new_pairs = [generate_pair() for _ in range(num_new)]
                new_nodes = []
                for p in new_pairs:
                    g.add_node(node_id, type='pair', **p)
                    self.pair_nodes.append(node_id)
                    self.remaining_nodes.append(node_id)
                    new_nodes.append(node_id)
                    node_id += 1
                self._add_edges(g, new_nodes)
        return g

    def _add_edges(self, g: nx.DiGraph, new_nodes: List[int], is_altru: bool = False):
        all_pairs = self.remaining_nodes if not is_altru else [n for n in g.nodes if g.nodes[n]['type'] == 'pair']
        if not is_altru:
            sources = all_pairs
            targets = new_nodes
            self._add_compat_edges(g, sources, targets)
            sources = new_nodes
            targets = all_pairs
            self._add_compat_edges(g, sources, targets)
        else:
            sources = new_nodes
            targets = all_pairs
            self._add_compat_edges(g, sources, targets)

    def _add_compat_edges(self, g: nx.DiGraph, sources: List[int], targets: List[int]):
        for s in sources:
            donor_bt = g.nodes[s]['donor_bt']
            for t in targets:
                if s == t:
                    continue
                if g.nodes[t]['type'] != 'pair':
                    continue
                patient_bt = g.nodes[t]['patient_bt']
                cpra = g.nodes[t]['cPRA']
                if is_abo_compatible(donor_bt, patient_bt):
                    if np.random.uniform() < 1 - cpra:
                        g.add_edge(s, t)

class KEPEnv:
    def __init__(self, instance: KEPInstance):
        self.instance = instance
        self.graph = instance.graph
        self.max_cycle_len = instance.max_cycle_len
        self.max_chain_len = instance.max_chain_len
        self.pair_nodes = instance.pair_nodes
        self.altru_nodes = instance.altru_nodes
        self.reset()

    def reset(self):
        self.current_remaining = self.graph.copy()
        self.current_matched = 0

    def _find_cycles(self, g: nx.DiGraph, max_len: int) -> List[List[int]]:
        cycles = []
        # Find cycles of length 2
        for u in g.nodes:
            for v in g.nodes:
                if u != v and g.has_edge(u, v) and g.has_edge(v, u):
                    cycle = [u, v] if u < v else [v, u]
                    if cycle not in cycles:
                        cycles.append(cycle)
        # Find cycles of length 3
        if max_len >= 3:
            for u in g.nodes:
                for v in g.successors(u):
                    if v == u:
                        continue
                    for w in g.successors(v):
                        if w in [u, v]:
                            continue
                        if g.has_edge(w, u):
                            cycle = sorted([u, v, w])
                            if cycle not in cycles:
                                cycles.append(cycle)
        return cycles

    def _find_chains(self, g: nx.DiGraph, max_len: int) -> List[List[int]]:
        chains = []
        altru_in_g = [n for n in self.altru_nodes if n in g.nodes]
        for a in altru_in_g:
            for length in range(1, max_len + 1):
                paths = list(nx.all_simple_paths(g, source=a, cutoff=length))
                for path in paths:
                    if len(path) == length + 1 and all(g.nodes[n]['type'] == 'pair' for n in path[1:]):
                        chains.append(path)
        return chains

    def solve_with_mip(self) -> Tuple[int, List[Any]]:
        cycles = self._find_cycles(self.graph, self.max_cycle_len)
        chains = self._find_chains(self.graph, self.max_chain_len)
        prob = pulp.LpProblem("KEP", pulp.LpMaximize)
        x_cycles = {i: pulp.LpVariable(f"cycle_{i}", cat='Binary') for i in range(len(cycles))}
        x_chains = {i: pulp.LpVariable(f"chain_{i}", cat='Binary') for i in range(len(chains))}
        obj = 0
        for i, c in enumerate(cycles):
            obj += len(c) * x_cycles[i]
        for i, ch in enumerate(chains):
            obj += (len(ch) - 1) * x_chains[i]
        prob += obj
        vertex_to_components = {v: [] for v in self.graph.nodes}
        for i, c in enumerate(cycles):
            for v in c:
                vertex_to_components[v].append(('cycle', i))
        for i, ch in enumerate(chains):
            for v in ch:
                vertex_to_components[v].append(('chain', i))
        for v, comps in vertex_to_components.items():
            constr = 0
            for typ, idx in comps:
                if typ == 'cycle':
                    constr += x_cycles[idx]
                else:
                    constr += x_chains[idx]
            if len(comps) > 0:
                prob += constr <= 1
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        matched = int(pulp.value(prob.objective) or 0)
        selected = []
        for i in x_cycles:
            if pulp.value(x_cycles[i]) == 1:
                selected.append(cycles[i])
        for i in x_chains:
            if pulp.value(x_chains[i]) == 1:
                selected.append(chains[i])
        return matched, selected

def add_compat_edges(g, sources, targets):
    for s in sources:
        donor_bt = g.nodes[s]['donor_bt']
        for t in targets:
            if s == t:
                continue
            patient_bt = g.nodes[t]['patient_bt']
            cpra = g.nodes[t]['cPRA']
            if is_abo_compatible(donor_bt, patient_bt):
                if np.random.uniform() < 1 - cpra:
                    g.add_edge(s, t)

def simulate_horizon(policy='none', n_rounds=20, arrival_rate=5.0, dep_prob=0.05, max_cycle_len=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    g = nx.DiGraph()
    remaining = []
    node_id_global = 0
    cumulative_matched = 0
    round_stats = []
    for r in range(n_rounds):
        # Departures
        to_remove = [n for n in remaining if np.random.uniform() < dep_prob]
        for n in to_remove:
            g.remove_node(n)
            remaining.remove(n)
        # Arrivals
        num_new = np.random.poisson(arrival_rate)
        new_nodes = []
        for _ in range(num_new):
            p = generate_pair()
            g.add_node(node_id_global, type='pair', **p)
            remaining.append(node_id_global)
            new_nodes.append(node_id_global)
            node_id_global += 1
        # Add edges
        all_current = remaining.copy()
        add_compat_edges(g, all_current, new_nodes)
        add_compat_edges(g, new_nodes, all_current)
        # Collect stats
        num_nodes = len(remaining)
        if num_nodes > 0:
            prop_O = sum(1 for n in remaining if g.nodes[n]['patient_bt'] == 'O') / num_nodes
            avg_degree = sum(g.degree(n) for n in remaining) / num_nodes
        else:
            prop_O = 0
            avg_degree = 0
        round_stats.append({'num_nodes': num_nodes, 'prop_O': prop_O, 'avg_degree': avg_degree})
        # Apply policy
        if policy == 'mip':
            # Create sub_g
            sub_g = g.subgraph(remaining).copy()
            # Create instance with no generation
            instance = KEPInstance(arrival_rate=0, n_rounds=1, n_altruistic=0, max_cycle_len=max_cycle_len)
            instance.graph = sub_g
            instance.pair_nodes = list(sub_g.nodes)
            instance.altru_nodes = []
            instance.max_chain_len = float('inf')
            env = KEPEnv(instance)
            matched, selected = env.solve_with_mip()
            cumulative_matched += matched
            # Remove matched
            matched_nodes = set()
            for s in selected:
                matched_nodes.update(s)
            for n in matched_nodes:
                g.remove_node(n)
                remaining.remove(n)
    return round_stats, cumulative_matched

# Monte Carlo simulation to quantify bias
np.random.seed(42)
num_sims = 200  # Réduit pour vitesse (augmente si besoin pour précision)
n_rounds = 20  # Suggestion : réduit à 10 pour tests plus rapides
arrival_rate = 5.0  # Suggestion : réduit à 2.0 pour graphes plus petits/faster MIP
policies = ['none', 'mip']

sim_results = {p: {'num_nodes': [], 'prop_O': [], 'avg_degree': [], 'cum_matched': []} for p in policies}

for policy in policies:
    print(f"Simulations pour policy '{policy}'...")
    for sim in tqdm(range(num_sims)):
        seed = 42 + sim
        stats, cum_matched = simulate_horizon(policy=policy, n_rounds=n_rounds, arrival_rate=arrival_rate, seed=seed)
        for r_stat in stats:
            sim_results[policy]['num_nodes'].append(r_stat['num_nodes'])
            sim_results[policy]['prop_O'].append(r_stat['prop_O'])
            sim_results[policy]['avg_degree'].append(r_stat['avg_degree'])
        sim_results[policy]['cum_matched'].append(cum_matched)

# Compute averages
for policy in policies:
    print(f"Policy: {policy}")
    print(f"Average cumulative matched: {np.mean(sim_results[policy]['cum_matched']):.2f}")
    print(f"Average num_nodes over all rounds/sims: {np.mean(sim_results[policy]['num_nodes']):.2f}")
    print(f"Average prop_O: {np.mean(sim_results[policy]['prop_O']):.2f}")
    print(f"Average avg_degree: {np.mean(sim_results[policy]['avg_degree']):.2f}")
    print("---")

# Compute KL divergence for distributions (proof of bias)
def kl_divergence(p_hist, q_hist, eps=1e-10):
    p = p_hist / np.sum(p_hist + eps)
    q = q_hist / np.sum(q_hist + eps)
    return np.sum(p * np.log((p + eps) / (q + eps)))

bins = 50
hist_none_nodes, _ = np.histogram(sim_results['none']['num_nodes'], bins=bins)
hist_mip_nodes, _ = np.histogram(sim_results['mip']['num_nodes'], bins=bins)
kl_nodes = kl_divergence(hist_none_nodes, hist_mip_nodes)
print(f"KL divergence (none || mip) for num_nodes distribution: {kl_nodes:.4f}")

hist_none_prop, _ = np.histogram(sim_results['none']['prop_O'], bins=bins)
hist_mip_prop, _ = np.histogram(sim_results['mip']['prop_O'], bins=bins)
kl_prop = kl_divergence(hist_none_prop, hist_mip_prop)
print(f"KL divergence (none || mip) for prop_O distribution: {kl_prop:.4f}")

hist_none_deg, _ = np.histogram(sim_results['none']['avg_degree'], bins=bins)
hist_mip_deg, _ = np.histogram(sim_results['mip']['avg_degree'], bins=bins)
kl_deg = kl_divergence(hist_none_deg, hist_mip_deg)
print(f"KL divergence (none || mip) for avg_degree distribution: {kl_deg:.4f}")

# Estimation rough du shortfall (15-25%) : compare cum_matched mip à total arrivées attendues (n_rounds * arrival_rate = 100)
avg_total_arrivals = n_rounds * arrival_rate
shortfall_pct = 100 * (avg_total_arrivals - np.mean(sim_results['mip']['cum_matched'])) / avg_total_arrivals
print(f"Rough welfare shortfall (% of max possible transplants): {shortfall_pct:.2f}%")
