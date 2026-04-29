"""
Función de fitness F3: jugabilidad + coherencia estructural + dificultad.
Diseñada para integrarse al bucle CMA-ES del proyecto NeuralPlumber.
"""
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from metrics.structural import structural_score

LEVEL_LENGTH = 200   # columnas totales del nivel de Mario
MAX_JUMPS    = 150   # referencia para normalizar el componente de dificultad


def f3(level: np.ndarray, agent_info, weights: tuple = (0.40, 0.35, 0.25)) -> float:
    """
    Fitness combinado para minimización (valores más negativos = mejor nivel).

    Args:
        level:       np.ndarray (14, 28) con IDs de tiles [0-9]
        agent_info:  objeto EvaluationInfo de MarioAI con atributos:
                         .distance_passed  (float)
                         .jumps_performed  (int)
        weights:     tupla (w_jugabilidad, w_estructura, w_dificultad)
                     deben sumar 1.0

    Returns:
        Escalar. Más negativo = nivel más deseable para CMA-ES.
    """
    w_play, w_struct, w_diff = weights

    # 1. Jugabilidad: fracción del nivel completada por el agente A*
    progress    = agent_info.distance_passed / LEVEL_LENGTH
    playability = -progress

    # 2. Coherencia estructural: promedio de las 4 métricas
    struct     = structural_score(level)
    structural = -struct["structural_avg"]

    # 3. Dificultad: saltos normalizados (solo si el nivel fue completado)
    if progress >= 1.0:
        difficulty = -min(agent_info.jumps_performed / MAX_JUMPS, 1.0)
    else:
        difficulty = 0.0

    return w_play * playability + w_struct * structural + w_diff * difficulty


def f3_static(level: np.ndarray, weights: tuple = (0.55, 0.45)) -> float:
    """
    Versión de F3 sin simulación del agente (solo métricas estáticas).
    Útil para desarrollo, debugging y experimentos sin el framework Java.

    Args:
        level:   np.ndarray (14, 28)
        weights: (w_suelo, w_estructura)

    Returns:
        Escalar a minimizar. Más negativo = más coherente estructuralmente.
    """
    w_g, w_s = weights
    struct = structural_score(level)
    ground    = -struct["ground_continuity"]
    structure = -struct["structural_avg"]
    return w_g * ground + w_s * structure


def f1_original(agent_info) -> float:
    """
    F1 del paper base (Volz et al.): maximizar saltos en niveles completados.
    Reproducción exacta para comparación en Experimento A.
    """
    progress = agent_info.distance_passed / LEVEL_LENGTH
    if progress < 1.0:
        return -progress
    return -progress - agent_info.jumps_performed


def f2_original(agent_info) -> float:
    """
    F2 del paper base (Volz et al.): minimizar saltos en niveles completados.
    Reproducción exacta para comparación en Experimento A.
    """
    OFFSET = 60
    progress = agent_info.distance_passed / LEVEL_LENGTH
    if progress < 1.0:
        return -progress + OFFSET
    return -progress + agent_info.jumps_performed
