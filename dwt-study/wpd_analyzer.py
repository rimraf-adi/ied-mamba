"""
Wavelet Packet Decomposition (WPD) and Coifman-Wickerhauser Best Basis Engine for EEG Analysis.

Unlike standard DWT (which only decomposes low-frequency approximation sub-bands),
WPD recursively decomposes both approximation and detail sub-bands, producing a full
balanced binary tree of 2^J uniform frequency sub-bands.

For fs = 250 Hz (Nyquist = 125 Hz) at Level 5:
- 32 uniform sub-bands with fine frequency resolution of 3.90625 Hz each.
"""

import numpy as np
import pywt
from typing import Dict, List, Tuple, Optional


def compute_shannon_entropy(data: np.ndarray) -> float:
    """
    Computes Coifman-Wickerhauser additive energy entropy cost:
    Cost(v) = - sum(v_i^2 * log(v_i^2 + eps)).
    Satisfies additivity across orthogonal wavelet packet tree nodes.
    """
    eps = 1e-12
    e = data ** 2
    return float(-np.sum(e * np.log(e + eps)))


def compute_log_energy_entropy(data: np.ndarray) -> float:
    """
    Computes Log Energy entropy cost: sum(log(data^2 + eps)).
    """
    eps = 1e-12
    return float(np.sum(np.log(data ** 2 + eps)))


def compute_wpd_uniform_spectrum(
    signal: np.ndarray,
    wavelet: str = 'db4',
    level: int = 5
) -> Tuple[np.ndarray, List[Tuple[float, float]], List[str]]:
    """
    Decomposes signal into 2^level uniform frequency sub-bands.
    Returns:
    - subband_energies: array of relative energies (percentage of total energy)
    - freq_ranges: list of (low_hz, high_hz) frequency tuples
    - node_paths: list of node path strings in natural frequency order
    """
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=level)
    
    # Get all nodes at level in natural (frequency) order
    nodes = wp.get_level(level, order='freq')
    
    nyquist = 125.0  # for fs = 250 Hz
    bw = nyquist / len(nodes)
    
    energies = []
    freq_ranges = []
    node_paths = []
    
    for idx, node in enumerate(nodes):
        low_f = idx * bw
        high_f = (idx + 1) * bw
        e = np.sum(node.data ** 2)
        energies.append(e)
        freq_ranges.append((low_f, high_f))
        node_paths.append(node.path)
        
    energies = np.array(energies)
    total_e = np.sum(energies) + 1e-12
    rel_energies = energies / total_e
    
    return rel_energies, freq_ranges, node_paths


def compute_coifman_wickerhauser_best_basis(
    signal: np.ndarray,
    wavelet: str = 'db4',
    max_level: int = 5,
    cost_func=compute_shannon_entropy
) -> Dict:
    """
    Executes the Coifman-Wickerhauser Best Basis tree pruning algorithm.
    Finds the optimal orthogonal wavelet packet basis that minimizes total entropy cost.
    """
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
    
    # Step 1: Compute cost for all nodes in the tree
    node_costs = {}
    
    def calculate_costs(node):
        node_costs[node.path] = cost_func(node.data)
        if node.level < max_level:
            calculate_costs(node['a'])
            calculate_costs(node['d'])
            
    calculate_costs(wp)
    
    # Step 2: Bottom-up pruning to determine optimal basis nodes
    best_basis_nodes = []
    
    def prune_tree(node) -> Tuple[float, List[str]]:
        if node.level == max_level:
            return node_costs[node.path], [node.path]
            
        left_cost, left_nodes = prune_tree(node['a'])
        right_cost, right_nodes = prune_tree(node['d'])
        
        children_cost = left_cost + right_cost
        parent_cost = node_costs[node.path]
        
        if parent_cost <= children_cost:
            # Parent is more compact than children
            return parent_cost, [node.path]
        else:
            # Children are more compact
            return children_cost, left_nodes + right_nodes
            
    optimal_cost, optimal_nodes = prune_tree(wp)
    
    # Extract node details
    basis_details = []
    nyquist = 125.0
    for path in optimal_nodes:
        node = wp[path]
        level = node.level
        bw = nyquist / (2 ** level)
        basis_details.append({
            'path': path,
            'level': level,
            'node_len': len(node.data),
            'energy': float(np.sum(node.data ** 2)),
            'cost': float(node_costs[path])
        })
        
    return {
        'optimal_cost': float(optimal_cost),
        'root_cost': float(node_costs['']),
        'cost_reduction_pct': float(max(0.0, (1.0 - optimal_cost / (node_costs[''] + 1e-12)) * 100.0)),
        'num_basis_nodes': len(optimal_nodes),
        'optimal_paths': optimal_nodes,
        'basis_details': basis_details
    }
