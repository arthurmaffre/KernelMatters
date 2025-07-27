import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Any
import pulp  # For the solver
import torch  # For potential GFlowNet, but not implemented here
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
import math

# Blood types and ABO compatibility
BLOOD_TYPES = ['O', 'A', 'B', 'AB']
BLOOD_TYPE_PROBS = [0.4814, 0.3373, 0.1428, 0.0385]  # From Saidman et al. (more precise)

# PRA levels (positive crossmatch probs) and their distribution
PRA_LEVELS = [0.05, 0.45, 0.90]  # Low, medium, high
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
    """
    Generate an incompatible patient-donor pair.
    - Sample blood types for patient and donor.
    - Sample cPRA (PRA level) for the patient.
    - If ABO incompatible: always incompatible, keep.
    - If ABO compatible: incompatible with probability cPRA (positive crossmatch), keep only then.
    """
    while True:
        patient_bt = np.random.choice(BLOOD_TYPES, p=BLOOD_TYPE_PROBS)
        donor_bt = np.random.choice(BLOOD_TYPES, p=BLOOD_TYPE_PROBS)
        cpra = np.random.choice(PRA_LEVELS, p=PRA_PROBS)
        if is_abo_compatible(donor_bt, patient_bt):
            if np.random.uniform() > cpra:  # Compatible (negative crossmatch), discard
                continue
        # Else: incompatible (either ABO or positive crossmatch), keep
        return {'patient_bt': patient_bt, 'donor_bt': donor_bt, 'cPRA': cpra}

def generate_altruistic() -> Dict[str, Any]:
    bt = np.random.choice(BLOOD_TYPES, p=BLOOD_TYPE_PROBS)
    return {'donor_bt': bt}

class KEPInstance:
    def __init__(self, n_pairs: int = 50, n_altruistic: int = 0, max_cycle_len: int = 3, max_chain_len: int = 4, n_rounds: int = 1):
        """
        Generate the KEP graph, optionally in multiple rounds to simulate arrivals (as in the paper).
        If n_rounds > 1, distribute the n_pairs over rounds with Poisson arrivals.
        But since no departures, equivalent to single round for static graph.
        """
        self.n_pairs = n_pairs
        self.n_altruistic = n_altruistic
        self.max_cycle_len = max_cycle_len
        self.max_chain_len = max_chain_len
        self.n_rounds = n_rounds
        self.graph = self._generate_graph()

    def _generate_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        node_id = 0
        self.pair_nodes = []
        self.altru_nodes = []

        # If multiple rounds, simulate arrivals
        if self.n_rounds > 1:
            arrival_rate = self.n_pairs / self.n_rounds  # \varsigma = n_pairs / N
            for _ in range(self.n_rounds):
                num_new = np.random.poisson(arrival_rate)
                new_pairs = [generate_pair() for _ in range(num_new)]
                new_nodes = []
                for p in new_pairs:
                    g.add_node(node_id, type='pair', **p)
                    self.pair_nodes.append(node_id)
                    new_nodes.append(node_id)
                    node_id += 1
                # Add edges involving new nodes
                self._add_edges(g, new_nodes)
        else:
            # Single round: all at once
            pairs = [generate_pair() for _ in range(self.n_pairs)]
            for p in pairs:
                g.add_node(node_id, type='pair', **p)
                self.pair_nodes.append(node_id)
                node_id += 1
            self._add_edges(g, self.pair_nodes)

        # Add altruistic donors (if any, though paper doesn't use them)
        altruistics = [generate_altruistic() for _ in range(self.n_altruistic)]
        new_nodes = []
        for a in altruistics:
            g.add_node(node_id, type='altruistic', **a)
            self.altru_nodes.append(node_id)
            new_nodes.append(node_id)
            node_id += 1
        self._add_edges(g, new_nodes, is_altru=True)

        return g

    def _add_edges(self, g: nx.DiGraph, new_nodes: List[int], is_altru: bool = False):
        """
        Add compatibility edges involving the new nodes.
        For pairs: from old to new, new to old, new to new.
        For altru: from altru to all pairs, since no incoming to altru.
        """
        all_pairs = [n for n in g.nodes if g.nodes[n]['type'] == 'pair']
        if not is_altru:
            sources = all_pairs  # All can point to new
            targets = new_nodes  # Old to new
            self._add_compat_edges(g, sources, targets)
            sources = new_nodes  # New to all pairs (including new)
            targets = all_pairs
            self._add_compat_edges(g, sources, targets)
        else:
            # For altru: only outgoing to all pairs
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
                    continue  # Targets are pairs
                patient_bt = g.nodes[t]['patient_bt']
                cpra = g.nodes[t]['cPRA']
                if is_abo_compatible(donor_bt, patient_bt):
                    if np.random.uniform() < 1 - cpra:  # Negative crossmatch
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

    def get_state(self) -> nx.DiGraph:
        return self.current_remaining.copy()  # For embedding in GFlowNet

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

    def actions(self) -> List[Any]:
        cycles = self._find_cycles(self.current_remaining, self.max_cycle_len)
        chains = self._find_chains(self.current_remaining, self.max_chain_len)
        return cycles + chains + ['terminate']

    def step(self, action: Any) -> Tuple[nx.DiGraph, float, bool]:
        done = False
        reward = 0.0
        if action == 'terminate':
            done = True
            # Reward only if maximal (no more actions possible except terminate)
            if len(self.actions()) > 1:  # More than just 'terminate'
                reward = 0.0
            else:
                reward = np.exp(self.current_matched)
        else:
            # action is list of nodes (cycle or chain)
            is_chain = action[0] in self.altru_nodes if self.altru_nodes else False
            transplants = len(action) if not is_chain else len(action) - 1
            self.current_matched += transplants
            # Remove nodes and incident edges
            for node in action:
                self.current_remaining.remove_node(node)
        return self.get_state(), reward, done

    def solve_with_mip(self) -> Tuple[int, List[Any]]:
        # Enumerate all possible cycles and chains
        cycles = self._find_cycles(self.graph, self.max_cycle_len)
        chains = self._find_chains(self.graph, self.max_chain_len)

        # Create PuLP problem
        prob = pulp.LpProblem("KEP", pulp.LpMaximize)
        x_cycles = {i: pulp.LpVariable(f"cycle_{i}", cat='Binary') for i in range(len(cycles))}
        x_chains = {i: pulp.LpVariable(f"chain_{i}", cat='Binary') for i in range(len(chains))}

        # Objective: maximize transplants
        obj = 0
        for i, c in enumerate(cycles):
            obj += len(c) * x_cycles[i]
        for i, ch in enumerate(chains):
            obj += (len(ch) - 1) * x_chains[i]
        prob += obj

        # Constraints: each vertex used at most once
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

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        matched = int(pulp.value(prob.objective) or 0)

        # Get selected
        selected = []
        for i in x_cycles:
            if pulp.value(x_cycles[i]) == 1:
                selected.append(cycles[i])
        for i in x_chains:
            if pulp.value(x_chains[i]) == 1:
                selected.append(chains[i])

        return matched, selected

if __name__ == "__main__":
    import pandas as pd
    np.random.seed(44)  # Pour reproductibilité
    instance = KEPInstance(n_pairs=8, n_altruistic=0, max_cycle_len=3, max_chain_len=4)  # Augmenté à 7 pour matcher l'exemple
    g = instance.graph
    nodes_df = pd.DataFrame([{'node': n, **d} for n, d in g.nodes(data=True)])
    print("Tableau des nœuds :")
    print(nodes_df.to_string(index=False))
    edges_df = pd.DataFrame(g.edges(), columns=['source', 'target'])
    print("\nTableau des arêtes :")
    print(edges_df.to_string(index=False))
    env = KEPEnv(instance)
    matched, selected = env.solve_with_mip()
    print(f"\nNombre maximum de paires appariées : {matched}")
    print("Cycles/chaînes sélectionnés :")
    for s in selected:
        print(s)

    # Visualisation améliorée pour publication : flèches arrêtées au bord des nœuds, marges élargies, design clean
    print("\nGénération du graphe visuel...")
    plt.rcParams['font.family'] = 'sans-serif'  # Police sans-serif pour un look moderne
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']  # Priorité à Arial (Apple-like)
    plt.rcParams['font.size'] = 12  # Taille de police de base

    pos = nx.spring_layout(g, seed=42, k=0.6, iterations=100)  # Augmenté k pour plus d'espacement

    patient_color_map = {
        'O': '#FF6B6B',  # Corail doux
        'A': '#4ECDC4',  # Bleu-vert clair
        'B': '#45B7AF',  # Vert menthe
        'AB': '#FFD97D'  # Jaune doux
    }
    node_colors = [patient_color_map.get(data['patient_bt'], '#CCCCCC') for node, data in g.nodes(data=True)]
    labels = {n: f"{n}\n{data['patient_bt']}/{data['donor_bt']}\ncPRA: {data['cPRA']:.2f}" for n, data in g.nodes(data=True)}

    fig, ax = plt.subplots(figsize=(16, 12))  # Taille augmentée pour plus d'espace et éviter l'écrasement
    ax.margins(0.15)  # Marges élargies pour un design aéré

    # Calcul du rayon des nœuds (node_size est l'aire en points^2)
    node_size = 3500
    radius = math.sqrt(node_size / math.pi)

    # Dessin manuel des arêtes avec FancyArrowPatch pour shrinkB = radius (flèches s'arrêtent au bord)
    for u, v in g.edges():
        # Déterminer la courbure seulement si arête mutuelle pour séparer les flèches et visualiser les cycles
        rad = 0.0
        if g.has_edge(v, u):
            rad = 0.25 if u < v else -0.25  # Courbure opposée pour arêtes mutuelles
        conn_style = f'arc3,rad={rad}' if rad != 0 else None

        arrow = FancyArrowPatch(pos[u], pos[v],
                                connectionstyle=conn_style,
                                arrowstyle='->',
                                mutation_scale=25,  # Taille de la flèche
                                linewidth=1.5,
                                color='#333333',
                                alpha=0.8,
                                shrinkA=0,  # Pas de rétrécissement au départ
                                shrinkB=radius + 2,  # Rétrécissement à l'arrivée pour arrêter au bord (+ marge)
                                )
        ax.add_patch(arrow)

    # Dessin des nœuds par-dessus les arêtes (mais comme shrinkB est appliqué, pas besoin de couvrir)
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_size, alpha=0.95, edgecolors='white', linewidths=2, ax=ax)
    nx.draw_networkx_labels(g, pos, labels, font_size=11, font_weight='bold', ax=ax)

    # Légende pour les couleurs (positionnée en bas à droite, sans cadre pour minimalisme)
    legend_elements = [Patch(facecolor=patient_color_map[bt], label=f'{bt}') for bt in patient_color_map]
    ax.legend(handles=legend_elements, title='Type sanguin du patient', loc='lower right', frameon=False, fontsize=11, title_fontsize=12)

    ax.set_title("Graphe d'Échange de Reins (KEP)\nNœuds colorés par type sanguin du patient", fontsize=18, fontweight='bold', pad=30)
    ax.axis('off')  # Pas d'axes pour un look minimaliste
    fig.patch.set_facecolor('white')  # Fond blanc
    plt.tight_layout(pad=3.0)  # Espacement élargi pour éviter l'écrasement
    plt.show()