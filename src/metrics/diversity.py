"""
Métricas de diversidad entre conjuntos de niveles generados.
"""
import numpy as np
from itertools import combinations


def pairwise_l1(levels: list) -> float:
    """Distancia L1 promedio entre todos los pares de niveles."""
    if len(levels) < 2:
        return 0.0
    dists = [
        np.abs(np.array(a) - np.array(b)).mean()
        for a, b in combinations(levels, 2)
    ]
    return float(np.mean(dists))


def tile_distribution(level: np.ndarray) -> np.ndarray:
    """Retorna distribución de frecuencia de los 10 tipos de tiles."""
    counts = np.bincount(level.flatten(), minlength=10)
    return counts / counts.sum()


def distribution_diversity(levels: list) -> float:
    """Diversidad basada en la varianza de las distribuciones de tiles."""
    dists = np.array([tile_distribution(np.array(l)) for l in levels])
    return float(dists.var(axis=0).mean())
