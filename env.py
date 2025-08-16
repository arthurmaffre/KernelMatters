import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Any
import pulp
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
import math

# Blood types and ABO compatibility (inchangé)
BLOOD_TYPES = ['O', 'A', 'B', 'AB']
BLOOD_TYPE_PROBS = [0.4814, 0.3373, 0.1428, 0.0385]
PRA_LEVELS = [0.05, 0.45, 0.90]
PRA_PROBS = [0.7019, 0.20, 0.0981]

def is_abo_compatible(donor_bt: str, recipient_bt: str) -> bool:
    # Inchangé
    if donor_bt == 'O':
        return True
    if donor_bt == recipient_bt:
        return True
    if donor_bt in ['A', 'B'] and recipient_bt == 'AB':
        return True
    return False

def generate_pair() -> Dict[str, Any]:
    # Inchangé : génère seulement des paires incompatibles
    while True:
        patient_bt = np.random.choice(BLOOD_TYPES, p=BLOOD_TYPE_PROBS)
        donor_bt = np.random.choice(BLOOD_TYPES, p=BLOOD_TYPE_PROBS)
        cpra = np.random.choice(PRA_LEVELS, p=PRA_PROBS)
        if is_abo_compatible(donor_bt, patient_bt):
            if np.random.uniform() > cpra:  # Compatible, discard
                continue
        return {'patient_bt': patient_bt, 'donor_bt': donor_bt, 'cPRA': cpra}

def generate_altruistic() -> Dict[str, Any]:
    # Inchangé
    bt = np.random.choice(BLOOD_TYPES, p=BLOOD_TYPE_PROBS)
    return {'donor_bt': bt}

class KEPInstance:
    def __init__(self, arrival_rate: float = 5.0, departure_prob: float = 0.05, n_rounds: int = 20,
                 n_altruistic: int = 0, max_cycle_len: int = 3, max_chain_len: int = float('inf'),
                 n_pairs: int = None):  # n_pairs optionnel pour compatibilité backward
        """
        Améliorations :
        - arrival_rate=5.0 (λ du papier), departure_prob=0.05.
        - n_rounds=20 par défaut.
        - Si n_rounds >1, simule départs + arrivées Poisson pur (sans fixer n_pairs total).
        - Si n_pairs donné et n_rounds>1, set arrival_rate = n_pairs / n_rounds (comme avant).
        - Stabilise la taille à ~100 nodes.
        """
        self.arrival_rate = arrival_rate if n_pairs is None else n_pairs / n_rounds
        self.departure_prob = departure_prob
        self.n_rounds = n_rounds
        self.n_altruistic = n_altruistic
        self.max_cycle_len = max_cycle_len
        self.max_chain_len = max_chain_len
        self.pair_nodes = []
        self.altru_nodes = []
        self.remaining_nodes = []  # Nouveau : track des nodes actuels pour départs
        self.graph = self._generate_graph()

    def _generate_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        node_id = 0
        if self.n_rounds == 1:
            # Single round : comme avant, pas de départs
            num_pairs = int(np.random.poisson(self.arrival_rate)) if self.arrival_rate else self.n_pairs
            pairs = [generate_pair() for _ in range(num_pairs)]
            for p in pairs:
                g.add_node(node_id, type='pair', **p)
                self.pair_nodes.append(node_id)
                self.remaining_nodes.append(node_id)
                node_id += 1
            self._add_edges(g, self.remaining_nodes)
        else:
            # Multi-rounds : départs + arrivées par round
            for _ in range(self.n_rounds):
                # Départs : remove avec prob 0.05 chaque paire actuelle
                to_remove = [n for n in self.remaining_nodes if np.random.uniform() < self.departure_prob]
                for n in to_remove:
                    g.remove_node(n)
                    self.remaining_nodes.remove(n)
                    if n in self.pair_nodes:
                        self.pair_nodes.remove(n)
                # Arrivées
                num_new = np.random.poisson(self.arrival_rate)
                new_pairs = [generate_pair() for _ in range(num_new)]
                new_nodes = []
                for p in new_pairs:
                    g.add_node(node_id, type='pair', **p)
                    self.pair_nodes.append(node_id)
                    self.remaining_nodes.append(node_id)
                    new_nodes.append(node_id)
                    node_id += 1
                # Ajout edges pour nouveaux (vers/depuis remaining, incl. new)
                self._add_edges(g, new_nodes)

        # Altruistes (si any, ajoutés à la fin, comme avant)
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
        # Inchangé, mais utilise self.remaining_nodes pour all_pairs si pas altru
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
        # Inchangé
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
    # Inchangé pour le core, mais ajout d'une note pour extension
    def __init__(self, instance: KEPInstance):
        self.instance = instance
        self.graph = instance.graph
        self.max_cycle_len = instance.max_cycle_len
        self.max_chain_len = instance.max_chain_len
        self.pair_nodes = instance.pair_nodes
        self.altru_nodes = instance.altru_nodes
        self.reset()

    # ... (reset, get_state, _find_cycles, _find_chains, actions, step, solve_with_mip inchangés)

class MultiRoundKEPEnv(KEPEnv):
    """
    Nouvelle sous-classe pour simuler multi-rounds avec matchings (adresse la critique).
    - Chaque 'step' : un round complet.
    - Arrivées Poisson, départs sur unmatched avant/after matching, choix d'action (matching ou terminate per round).
    - Reward cumulé sur horizon.
    - Done après n_rounds.
    """
    def __init__(self, instance: KEPInstance):
        super().__init__(instance)
        self.current_round = 0
        self.cumulative_matched = 0

    def reset(self):
        super().reset()
        self.current_round = 0
        self.cumulative_matched = 0
        self.current_remaining = nx.DiGraph()  # Start vide, build over rounds

    def step(self, action: Any) -> Tuple[nx.DiGraph, float, bool]:
        done = self.current_round >= self.instance.n_rounds
        if done:
            return self.get_state(), 0.0, True

        reward = 0.0
        # Départs pre-round sur remaining
        to_remove = [n for n in list(self.current_remaining.nodes) if np.random.uniform() < self.instance.departure_prob]
        for n in to_remove:
            self.current_remaining.remove_node(n)

        # Arrivées : add new nodes/edges comme dans _generate_graph
        num_new = np.random.poisson(self.instance.arrival_rate)
        new_pairs = [generate_pair() for _ in range(num_new)]
        new_nodes = []
        node_id = max(self.current_remaining.nodes) + 1 if self.current_remaining.nodes else 0
        for p in new_pairs:
            self.current_remaining.add_node(node_id, type='pair', **p)
            new_nodes.append(node_id)
            node_id += 1
        # Add edges pour new (similaire à _add_edges)
        all_current = [n for n in self.current_remaining if self.current_remaining.nodes[n]['type'] == 'pair']
        self.instance._add_compat_edges(self.current_remaining, all_current, new_nodes)  # old+new -> new
        self.instance._add_compat_edges(self.current_remaining, new_nodes, all_current)  # new -> old+new

        # Choix d'action : matching sur current_remaining
        if action != 'terminate':
            transplants = len(action) if action[0] not in self.altru_nodes else len(action) - 1
            self.cumulative_matched += transplants
            for node in action:
                self.current_remaining.remove_node(node)
            reward = np.exp(transplants)  # Per-round reward

        # Départs post-matching sur remaining
        to_remove = [n for n in list(self.current_remaining.nodes) if np.random.uniform() < self.instance.departure_prob]
        for n in to_remove:
            self.current_remaining.remove_node(n)

        self.current_round += 1
        done = self.current_round >= self.instance.n_rounds
        return self.get_state(), reward, done

# Exemple usage (adapté)
if __name__ == "__main__":
    np.random.seed(44)
    instance = KEPInstance(arrival_rate=5.0, n_rounds=20, departure_prob=0.05, n_altruistic=0)
    print(f"Nombre de nœuds générés : {len(instance.graph.nodes)}")  # ~100 expected
    env = KEPEnv(instance)
    matched, selected = env.solve_with_mip()
    print(f"Nombre maximum de paires appariées : {matched}")
    # Pour multi-round : multi_env = MultiRoundKEPEnv(instance), then loop on steps
