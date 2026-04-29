"""
Métricas de coherencia estructural para niveles de Mario.
Todas reciben un array numpy (14, 28) con IDs de tiles [0-9].
Todas retornan un escalar en [0, 1] donde 1 es el mejor valor posible.

IDs de tiles (DagstuhlGAN):
    0=Ground, 1=Breakable, 2=Empty, 3=Q_full, 4=Q_empty,
    5=Enemy, 6=Pipe_TL, 7=Pipe_TR, 8=Pipe_L, 9=Pipe_R
"""
import numpy as np

GROUND    = 0
BREAKABLE = 1
EMPTY     = 2
Q_FULL    = 3
Q_EMPTY   = 4
ENEMY     = 5
PIPE_TL   = 6
PIPE_TR   = 7
PIPE_L    = 8
PIPE_R    = 9

SOLID_TILES = {GROUND, BREAKABLE, Q_FULL, Q_EMPTY, PIPE_TL, PIPE_TR, PIPE_L, PIPE_R}
PIPE_TILES  = {PIPE_TL, PIPE_TR, PIPE_L, PIPE_R}


def pipe_completeness(level: np.ndarray) -> float:
    """
    Fracción de columnas con tubería que tienen estructura válida.
    Válido: columna i tiene PIPE_TL y columna i+1 tiene PIPE_TR en la misma fila,
    con al menos una fila de cuerpo (PIPE_L / PIPE_R) debajo.
    """
    rows, cols = level.shape
    valid_pipes = 0
    total_pipe_cols = 0

    for col in range(cols - 1):
        col_has_pipe = bool(set(level[:, col].tolist()) & PIPE_TILES)
        if col_has_pipe:
            total_pipe_cols += 1
            for row in range(rows - 1):
                if level[row, col] == PIPE_TL and level[row, col + 1] == PIPE_TR:
                    if level[row + 1, col] in {PIPE_L, PIPE_R}:
                        valid_pipes += 1
                    break

    return valid_pipes / max(total_pipe_cols, 1)


def ground_continuity(level: np.ndarray) -> float:
    """
    Fracción de tiles sólidos en las últimas 2 filas del nivel.
    Indica si el nivel tiene una base caminable continua.
    """
    bottom = level[-2:, :]
    solid_count = sum(1 for v in bottom.flatten() if v in SOLID_TILES)
    return solid_count / bottom.size


def gap_traversability(level: np.ndarray, max_jump_width: int = 4) -> float:
    """
    Fracción de huecos NO saltables sobre el total de huecos detectados.
    Un hueco es una secuencia de columnas sin tiles sólidos en la fila inferior.
    Retorna 0 si todos los huecos son saltables (valor deseable).
    """
    bottom_row = level[-1, :]
    is_gap = np.array([int(t not in SOLID_TILES) for t in bottom_row])

    gaps = []
    in_gap, gap_len = False, 0
    for g in is_gap:
        if g:
            in_gap = True
            gap_len += 1
        elif in_gap:
            gaps.append(gap_len)
            in_gap, gap_len = False, 0
    if in_gap:
        gaps.append(gap_len)

    if not gaps:
        return 0.0

    unjumpable = sum(1 for g in gaps if g > max_jump_width)
    return unjumpable / len(gaps)


def enemy_placement(level: np.ndarray) -> float:
    """
    Fracción de enemigos correctamente ubicados sobre alguna superficie sólida.
    Un enemigo es válido si hay al menos un tile sólido en alguna fila debajo.
    """
    rows, cols = level.shape
    enemy_positions = [(r, c) for r in range(rows) for c in range(cols)
                       if level[r, c] == ENEMY]

    if not enemy_positions:
        return 1.0

    valid = 0
    for row, col in enemy_positions:
        for r in range(row + 1, rows):
            if level[r, col] in SOLID_TILES:
                valid += 1
                break

    return valid / len(enemy_positions)


def structural_score(level: np.ndarray) -> dict:
    """
    Calcula todas las métricas estructurales de un nivel.

    Returns:
        dict con claves: pipe_completeness, ground_continuity,
        gap_traversability (invertido), enemy_placement, structural_avg.
    """
    metrics = {
        "pipe_completeness":  pipe_completeness(level),
        "ground_continuity":  ground_continuity(level),
        "gap_traversability": 1.0 - gap_traversability(level),  # invertido: mayor = mejor
        "enemy_placement":    enemy_placement(level),
    }
    metrics["structural_avg"] = float(np.mean(list(metrics.values())))
    return metrics


if __name__ == "__main__":
    # Test rápido con un nivel dummy
    level = np.full((14, 28), EMPTY, dtype=int)
    level[-1, :] = GROUND        # suelo completo
    level[10, 5] = ENEMY         # enemigo sobre suelo
    level[-2, 5] = GROUND
    print(structural_score(level))
