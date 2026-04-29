# NeuralPlumber — Proyecto INF398
## Generación Procedural de Niveles de Super Mario Bros mediante GANs y Optimización Evolutiva

> **Curso:** INF398 — Introducción al Aprendizaje Automático
> **Equipo:** Carlos Ramírez, Nicolás Rogel, Carlos Saavedra
> **Fecha propuesta:** 27 de Marzo 2026

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Análisis de la Propuesta](#2-análisis-de-la-propuesta)
3. [Contribución Técnica — El CÓMO](#3-contribución-técnica--el-cómo)
4. [Estructura del Proyecto en 4 Bloques](#4-estructura-del-proyecto-en-4-bloques)
5. [Implementación Detallada](#5-implementación-detallada)
6. [Métricas de Evaluación](#6-métricas-de-evaluación)
7. [División de Trabajo](#7-división-de-trabajo)
8. [Riesgos y Decisiones de Diseño](#8-riesgos-y-decisiones-de-diseño)
9. [Cronograma](#9-cronograma)

---

## 1. Resumen Ejecutivo

El proyecto extiende el trabajo de Volz et al. (2018) — *Evolving Mario Levels in the Latent Space of a DCGAN* — con el objetivo concreto de **mejorar la coherencia estructural** de los niveles generados.

El paper base falla en dos aspectos específicos y medibles:
1. **Estructuras rotas**: tuberías incompletas, plataformas sin continuidad
2. **Dataset mínimo**: entrenado sobre un único nivel de Mario (173 ventanas)

La propuesta ataca ambos frentes con dos contribuciones técnicas complementarias:
- **Expansión del dataset** de 1 a 15 niveles del VGLC
- **Nuevas funciones de fitness estructural** integradas al bucle CMA-ES

Todo sobre la base del repositorio existente (DagstuhlGAN), sin cambiar la arquitectura del GAN.

---

## 2. Análisis de la Propuesta

### 2.1 Fortalezas del enfoque base (Volz et al.)

| Aspecto | Descripción |
|---|---|
| GAN + CMA-ES | Separa aprendizaje de distribución (GAN) de optimización de objetivos (CMA-ES) |
| Espacio latente compacto | 32 dimensiones → manejable para algoritmos evolutivos |
| Agente A* como oráculo | Evaluación de jugabilidad sin necesidad de jugadores humanos |
| Repositorio completo | Código, modelos preentrenados y datos disponibles |

### 2.2 Limitaciones identificadas

```
LIMITACIÓN 1 — Dataset mínimo
  Entrenamiento sobre mario-1-1.txt únicamente
  → 173 ventanas de entrenamiento
  → El GAN no aprende variedad estructural
  → Genera siempre variaciones del mismo patrón

LIMITACIÓN 2 — Fitness ignora estructura
  F1/F2 solo miden: fracción completada + saltos
  → No penaliza tuberías incompletas
  → No penaliza plataformas inalcanzables
  → No penaliza enemigos mal ubicados
  → Niveles con estructuras rotas pueden tener fitness alto

LIMITACIÓN 3 — Evaluación subjetiva
  El paper no define métricas cuantitativas de coherencia
  → Difícil comparar objetivamente distintos métodos
```

### 2.3 Por qué NO cambiar la arquitectura del GAN

Cambiar la arquitectura (e.g. StyleGAN, Transformer) introduciría:
- Tiempo de implementación que supera el alcance del curso
- Riesgo alto de no converger con datasets pequeños
- Dificultad para aislar qué mejora viene de qué cambio

La propuesta mantiene el DCGAN/WGAN y actúa sobre **inputs** (datos) y **outputs** (fitness), que son las capas más controlables.

---

## 3. Contribución Técnica — El CÓMO

### 3.1 Expansión del dataset

**Antes (paper base):**
```
mario-1-1.txt → 173 ventanas 28×14 → example.json
```

**Propuesta:**
```
mario-1-1.txt  ┐
mario-1-2.txt  │
mario-1-3.txt  │  → ~2500 ventanas 28×14 → dataset_full.json
     ...       │
mario-8-1.txt  ┘
```

Los 15 niveles del VGLC cubren mundos distintos con estructuras diferentes:
- Mundo 1: nivel de introducción, suelo continuo
- Mundo 2: plataformas suspendidas, más enemigos
- Mundo 3: acuático (diferente distribución de tiles)
- Mundo 4-8: progresivamente más complejos

Este cambio solo requiere un **parser** que lea los `.txt` del VGLC y los convierta al formato JSON del repositorio.

### 3.2 Métricas de coherencia estructural

Se definen 4 métricas nuevas, todas computables directamente desde la representación de tiles (sin simulación):

#### Métrica 1 — Completitud de tuberías (`pipe_completeness`)

Una tubería válida tiene exactamente 4 tiles en la configuración correcta:
```
< >    (top-left, top-right)
[ ]    (left, right)
[ ]    (puede repetirse hacia abajo)
```

Una tubería es "rota" si tiene tiles de tubería sin la configuración válida.

```python
def pipe_completeness(level):
    """
    Retorna fracción de tuberías completas sobre el total de tiles de tubería.
    Rango: [0, 1]. 1 = todas las tuberías son estructuralmente válidas.
    """
    pipe_tiles = {6: '<', 7: '>', 8: '[', 9: ']'}
    # Detectar columnas con tiles de tubería
    # Verificar que cada columna con '<' tenga '>' adyacente y '[',']' debajo
    ...
```

#### Métrica 2 — Continuidad del suelo (`ground_continuity`)

En niveles de Mario, las últimas 1-2 filas deben ser mayoritariamente suelo (`X`, ID=0). Un nivel sin suelo continuo es esencialmente injugable.

```python
def ground_continuity(level):
    """
    Fracción de tiles de suelo en las últimas 2 filas del nivel.
    Rango: [0, 1]. Objetivo: > 0.7
    """
    bottom_rows = level[-2:, :]    # últimas 2 filas
    ground_count = (bottom_rows == 0).sum()
    return ground_count / bottom_rows.size
```

#### Métrica 3 — Traversabilidad de huecos (`gap_traversability`)

Un hueco es un conjunto de columnas consecutivas sin suelo. En Mario, el salto máximo alcanzable es de aproximadamente 4 tiles horizontales. Huecos mayores hacen el nivel injugable incluso con salto perfecto.

```python
def gap_traversability(level, max_jump=4):
    """
    Penaliza huecos que superen el salto máximo posible.
    Retorna fracción de huecos no jugables sobre total de huecos.
    Rango: [0, 1]. 0 = todos los huecos son saltables.
    """
    ground_row = level[-1, :]
    gaps = detect_gaps(ground_row)       # segmentos sin suelo
    unjumpable = [g for g in gaps if g > max_jump]
    return len(unjumpable) / max(len(gaps), 1)
```

#### Métrica 4 — Posición de enemigos (`enemy_placement`)

Un enemigo flotando en el aire (sin plataforma debajo) es una anomalía estructural. Los enemigos deben estar sobre alguna superficie sólida.

```python
def enemy_placement(level):
    """
    Fracción de enemigos correctamente ubicados sobre superficie sólida.
    Rango: [0, 1]. 1 = todos los enemigos tienen superficie debajo.
    """
    enemy_positions = np.argwhere(level == 5)   # tile ID 5 = enemigo
    valid = 0
    for row, col in enemy_positions:
        # Buscar surface debajo del enemigo
        below = level[row+1:, col]
        if any(t in [0, 1, 3, 4] for t in below):  # suelo, rompible, bloque ?
            valid += 1
    return valid / max(len(enemy_positions), 1)
```

### 3.3 Función de fitness combinada (F3)

La nueva función de fitness integra jugabilidad (del paper original) con coherencia estructural (contribución propia):

```python
def F3(level, agent_info, weights=(0.4, 0.35, 0.25)):
    """
    Fitness combinado: jugabilidad + coherencia estructural + dificultad.

    Parámetros:
        level       : array (14, 28) de IDs de tiles
        agent_info  : resultado de simulación A* (EvaluationInfo de MarioAI)
        weights     : (w_play, w_struct, w_diff)

    Retorna: escalar a minimizar (más negativo = mejor)
    """
    w_play, w_struct, w_diff = weights

    # --- Componente 1: Jugabilidad (del paper base) ---
    progress = agent_info.distance_passed / LEVEL_LENGTH
    playability = -progress  # minimizar negativo = maximizar progreso

    # --- Componente 2: Coherencia estructural (NUEVO) ---
    pipe_score    = pipe_completeness(level)       # [0,1], mayor es mejor
    ground_score  = ground_continuity(level)       # [0,1], mayor es mejor
    gap_score     = 1 - gap_traversability(level)  # invertido: mayor es mejor
    enemy_score   = enemy_placement(level)         # [0,1], mayor es mejor

    structural = -(pipe_score + ground_score + gap_score + enemy_score) / 4

    # --- Componente 3: Dificultad (del paper base, solo si completó) ---
    if progress >= 1.0:
        difficulty = -agent_info.jumps_performed / MAX_JUMPS
    else:
        difficulty = 0

    return w_play * playability + w_struct * structural + w_diff * difficulty
```

**Justificación de pesos:** El componente de jugabilidad tiene mayor peso porque un nivel injugable con estructura perfecta sigue siendo un nivel fallido. Los pesos son hiperparámetros a explorar.

---

## 4. Estructura del Proyecto en 4 Bloques

```
BLOQUE 1          BLOQUE 2          BLOQUE 3          BLOQUE 4
Reproducción  →   Datos             Fitness           Evaluación
del baseline      expandidos        estructural       comparativa
(semana 1-2)      (semana 2-3)      (semana 3-4)      (semana 4-5)

Resultados:       Resultados:       Resultados:       Resultados:
• Métricas        • GAN re-         • CMA-ES con      • Tablas
  base medidas      entrenado         F3 funcionando    comparativas
• Pipeline        • dataset_full    • Métricas        • Visualiza-
  funcional         .json listo       nuevas            ciones
                                      calculadas      • Conclusiones
```

---

## 5. Implementación Detallada

### 5.1 Estructura de archivos del proyecto

```
NeuralPlumber/
├── README.md                      # Setup y ejecución
├── proyecto_inf398.md             # Este documento
│
├── src/
│   ├── data/
│   │   ├── vglc_parser.py         # NUEVO: lee .txt del VGLC → JSON
│   │   └── dataset_builder.py     # NUEVO: construye dataset_full.json
│   │
│   ├── metrics/
│   │   ├── structural.py          # NUEVO: pipe_completeness, ground_continuity, etc.
│   │   ├── playability.py         # NUEVO: wraps del agente A*
│   │   └── diversity.py           # NUEVO: distancia entre niveles generados
│   │
│   ├── fitness/
│   │   ├── f1_f2.py               # Del paper base (reproducción)
│   │   └── f3_combined.py         # NUEVO: F3 con coherencia estructural
│   │
│   ├── evolution/
│   │   └── cmaes_runner.py        # NUEVO: wrapper Python del CMA-ES
│   │
│   ├── visualization/
│   │   ├── level_renderer.py      # NUEVO: renderiza nivel como imagen
│   │   └── metrics_plotter.py     # NUEVO: gráficas de comparación
│   │
│   └── baseline/
│       └── reproduce_volz.py      # NUEVO: reproduce experimentos del paper
│
├── models/
│   └── (symlink o copia de netG_epoch_5000.pth)
│
├── data/
│   ├── raw/                       # .txt del VGLC (symlink a Datasets/TheVGLC)
│   ├── example.json               # Dataset original del paper (173 ventanas)
│   └── dataset_full.json          # NUEVO: dataset con 15 niveles (~2500 ventanas)
│
├── experiments/
│   ├── baseline/                  # Resultados reproducción Volz et al.
│   ├── expanded_data/             # Resultados con dataset completo
│   └── structural_fitness/        # Resultados con F3
│
├── notebooks/
│   ├── 01_baseline_analysis.ipynb
│   ├── 02_dataset_expansion.ipynb
│   ├── 03_structural_metrics.ipynb
│   └── 04_results_comparison.ipynb
│
└── requirements.txt
```

### 5.2 `src/data/vglc_parser.py`

```python
"""
Parser: convierte niveles .txt del VGLC al formato JSON de DagstuhlGAN.
Uso: python vglc_parser.py --input_dir ../../Datasets/TheVGLC/Super\ Mario\ Bros/Processed/
                           --output data/dataset_full.json
                           --window_width 28 --window_height 14
"""
import json
import numpy as np
from pathlib import Path

# Mapeo VGLC → ID numérico (mismo que usa DagstuhlGAN)
TILE_MAP = {
    'X': 0,  # Solid/Ground
    'S': 1,  # Breakable
    '-': 2,  # Empty (passable)
    '?': 3,  # Full question block
    'Q': 4,  # Empty question block
    'E': 5,  # Enemy
    '<': 6,  # Top-left pipe
    '>': 7,  # Top-right pipe
    '[': 8,  # Left pipe
    ']': 9,  # Right pipe
    # Tiles del VGLC no usados por el paper → mapear a vacío
    'o': 2,  # Coin → empty
    'B': 0,  # Cannon top → solid
    'b': 0,  # Cannon bottom → solid
}

def parse_level_file(path: Path) -> np.ndarray:
    """Lee un .txt del VGLC y retorna array (14, N) de IDs."""
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 14, f"Nivel {path.name} tiene {len(lines)} filas, esperaba 14"
    return np.array([[TILE_MAP.get(c, 2) for c in line] for line in lines])

def sliding_windows(level: np.ndarray, width=28, step=1) -> list:
    """Extrae ventanas deslizantes de ancho `width` con paso `step`."""
    _, total_width = level.shape
    windows = []
    for start in range(0, total_width - width + 1, step):
        windows.append(level[:, start:start + width].tolist())
    return windows

def build_dataset(input_dir: str, output_path: str, width=28, step=1):
    input_dir = Path(input_dir)
    txt_files = sorted(input_dir.glob("mario-*.txt"))
    print(f"Encontrados {len(txt_files)} niveles")

    all_windows = []
    for f in txt_files:
        level = parse_level_file(f)
        windows = sliding_windows(level, width, step)
        all_windows.extend(windows)
        print(f"  {f.name}: {level.shape[1]} cols → {len(windows)} ventanas")

    print(f"\nTotal ventanas: {len(all_windows)}")
    Path(output_path).write_text(json.dumps(all_windows))
    print(f"Guardado en {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", default="data/dataset_full.json")
    parser.add_argument("--window_width", type=int, default=28)
    parser.add_argument("--window_step", type=int, default=1)
    args = parser.parse_args()
    build_dataset(args.input_dir, args.output, args.window_width, args.window_step)
```

### 5.3 `src/metrics/structural.py`

```python
"""
Métricas de coherencia estructural para niveles de Mario.
Todas reciben un array numpy (14, 28) con IDs de tiles [0-9].
Todas retornan un escalar en [0, 1] donde 1 es el mejor valor posible.
"""
import numpy as np

# IDs de tiles (según DagstuhlGAN)
GROUND    = 0
BREAKABLE = 1
EMPTY     = 2
Q_FULL    = 3
Q_EMPTY   = 4
ENEMY     = 5
PIPE_TL   = 6  # top-left  '<'
PIPE_TR   = 7  # top-right '>'
PIPE_L    = 8  # left      '['
PIPE_R    = 9  # right     ']'

SOLID_TILES = {GROUND, BREAKABLE, Q_FULL, Q_EMPTY, PIPE_TL, PIPE_TR, PIPE_L, PIPE_R}
PIPE_TILES  = {PIPE_TL, PIPE_TR, PIPE_L, PIPE_R}


def pipe_completeness(level: np.ndarray) -> float:
    """
    Fracción de columnas con tubería que tienen estructura válida.
    Una tubería válida: columna i tiene PIPE_TL y columna i+1 tiene PIPE_TR
    en la misma fila, con PIPE_L y PIPE_R en las filas siguientes.
    """
    rows, cols = level.shape
    valid_pipes = 0
    total_pipe_cols = 0

    for col in range(cols - 1):
        col_tiles = set(level[:, col])
        if col_tiles & PIPE_TILES:
            total_pipe_cols += 1
            # Verificar estructura: buscar fila donde empieza la tubería
            for row in range(rows - 1):
                if level[row, col] == PIPE_TL and level[row, col + 1] == PIPE_TR:
                    # Verificar que la fila siguiente tiene cuerpo de tubería
                    if row + 1 < rows and level[row + 1, col] in {PIPE_L, PIPE_R}:
                        valid_pipes += 1
                    break

    return valid_pipes / max(total_pipe_cols, 1)


def ground_continuity(level: np.ndarray) -> float:
    """
    Fracción de tiles de suelo en las últimas 2 filas.
    Indica si el nivel tiene una base sólida caminable.
    """
    bottom = level[-2:, :]
    solid_count = np.isin(bottom, list(SOLID_TILES)).sum()
    return solid_count / bottom.size


def gap_traversability(level: np.ndarray, max_jump_width: int = 4) -> float:
    """
    Fracción de huecos NO saltables sobre el total de huecos.
    Un hueco es una secuencia de columnas sin tiles sólidos en la fila inferior.
    Retorna 0 si todos los huecos son saltables (valor deseado).
    """
    bottom_row = level[-1, :]
    is_gap = ~np.isin(bottom_row, list(SOLID_TILES))

    # Detectar segmentos de huecos consecutivos
    gaps = []
    in_gap = False
    gap_len = 0
    for tile in is_gap:
        if tile:
            in_gap = True
            gap_len += 1
        elif in_gap:
            gaps.append(gap_len)
            in_gap = False
            gap_len = 0
    if in_gap:
        gaps.append(gap_len)

    if not gaps:
        return 0.0

    unjumpable = sum(1 for g in gaps if g > max_jump_width)
    return unjumpable / len(gaps)


def enemy_placement(level: np.ndarray) -> float:
    """
    Fracción de enemigos sobre superficie sólida.
    Un enemigo es válido si hay al menos un tile sólido directamente debajo.
    """
    rows, cols = level.shape
    enemy_positions = np.argwhere(level == ENEMY)

    if len(enemy_positions) == 0:
        return 1.0  # Sin enemigos = sin problemas de placement

    valid = 0
    for row, col in enemy_positions:
        # Buscar superficie sólida en las filas siguientes
        for r in range(row + 1, rows):
            if level[r, col] in SOLID_TILES:
                valid += 1
                break

    return valid / len(enemy_positions)


def structural_score(level: np.ndarray) -> dict:
    """
    Calcula todas las métricas estructurales de un nivel.
    Retorna dict con cada métrica y el promedio global.
    """
    metrics = {
        "pipe_completeness":   pipe_completeness(level),
        "ground_continuity":   ground_continuity(level),
        "gap_traversability":  1 - gap_traversability(level),  # invertido
        "enemy_placement":     enemy_placement(level),
    }
    metrics["structural_avg"] = np.mean(list(metrics.values()))
    return metrics
```

### 5.4 `src/fitness/f3_combined.py`

```python
"""
Función de fitness F3: jugabilidad + coherencia estructural + dificultad.
Integra las métricas estructurales al bucle CMA-ES.
"""
import numpy as np
from metrics.structural import structural_score

LEVEL_LENGTH = 200   # columnas del nivel completo de Mario
MAX_JUMPS    = 150   # referencia para normalizar dificultad


def f3(level: np.ndarray, agent_info, weights=(0.40, 0.35, 0.25)) -> float:
    """
    Fitness combinado para minimización (valores más negativos = mejor).

    Args:
        level:       array (14, 28), IDs de tiles
        agent_info:  EvaluationInfo de MarioAI (distancia, saltos)
        weights:     (w_jugabilidad, w_estructura, w_dificultad)

    Returns:
        Escalar. Más negativo = nivel más deseable.
    """
    w_play, w_struct, w_diff = weights

    # 1. Jugabilidad
    progress    = agent_info.distance_passed / LEVEL_LENGTH
    playability = -progress

    # 2. Coherencia estructural
    struct  = structural_score(level)
    structural = -struct["structural_avg"]

    # 3. Dificultad (solo si el nivel fue completado)
    if progress >= 1.0:
        difficulty = -min(agent_info.jumps_performed / MAX_JUMPS, 1.0)
    else:
        difficulty = 0.0

    return w_play * playability + w_struct * structural + w_diff * difficulty


def f3_static(level: np.ndarray, weights=(0.6, 0.4)) -> float:
    """
    Versión de F3 sin simulación (solo métricas estáticas).
    Útil para desarrollo y debugging sin el framework Java.

    Args:
        level:   array (14, 28)
        weights: (w_ground, w_structure)
    """
    w_g, w_s = weights
    struct = structural_score(level)

    ground    = -struct["ground_continuity"]
    structure = -struct["structural_avg"]

    return w_g * ground + w_s * structure
```

### 5.5 `src/baseline/reproduce_volz.py`

```python
"""
Reproduce los experimentos de Volz et al. (2018) y mide métricas base.
Requiere: netG_epoch_5000.pth en models/
"""
import torch
import numpy as np
import json
from pathlib import Path

# Ajustar path según estructura del proyecto
import sys
sys.path.append("../../clone/DagstuhlGAN/pytorch")
from models.dcgan import DCGAN_G


def load_generator(model_path: str, nz=32, nc=10, ngf=64) -> DCGAN_G:
    gen = DCGAN_G(32, nz, nc, ngf, 0, 0)
    gen.load_state_dict(torch.load(model_path, map_location="cpu"))
    gen.eval()
    return gen


def generate_level(generator: DCGAN_G, z: np.ndarray = None) -> np.ndarray:
    """Genera un nivel a partir de un vector latente (aleatorio si z=None)."""
    if z is None:
        z = torch.randn(1, 32, 1, 1)
    else:
        z = torch.FloatTensor(z).view(1, 32, 1, 1)
    with torch.no_grad():
        out = generator(z)                    # (1, 10, 32, 32)
    level = out[0, :, :14, :28].argmax(0)    # (14, 28)
    return level.numpy()


def sample_levels(generator: DCGAN_G, n: int = 100) -> list:
    """Genera n niveles aleatorios."""
    return [generate_level(generator) for _ in range(n)]


if __name__ == "__main__":
    from metrics.structural import structural_score

    gen = load_generator("../../clone/DagstuhlGAN/pytorch/netG_epoch_5000.pth")
    levels = sample_levels(gen, n=100)

    # Medir métricas estructurales del baseline
    all_metrics = [structural_score(l) for l in levels]
    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

    print("=== Métricas baseline (Volz et al., n=100) ===")
    for k, v in avg.items():
        print(f"  {k:25s}: {v:.4f}")
```

---

## 6. Métricas de Evaluación

Se definen métricas para comparar **baseline** (Volz et al.) vs **propuesta** (NeuralPlumber):

### 6.1 Tabla de métricas

| Métrica | Descripción | Rango | Mejor | Cómo medir |
|---|---|---|---|---|
| **Tasa de jugabilidad** | % de niveles que A* completa | [0,1] | Alto | Simulación MarioAI |
| **Saltos promedio (F1)** | Promedio de saltos en niveles completados | ≥0 | Alto (F1) / Bajo (F2) | Simulación MarioAI |
| **Completitud de tuberías** | % de tuberías estructuralmente válidas | [0,1] | Alto | `pipe_completeness` |
| **Continuidad de suelo** | % de tiles sólidos en fila inferior | [0,1] | Alto | `ground_continuity` |
| **Traversabilidad** | % de huecos saltables | [0,1] | Alto | `gap_traversability` |
| **Placement de enemigos** | % de enemigos sobre superficie | [0,1] | Alto | `enemy_placement` |
| **Score estructural** | Promedio de las 4 anteriores | [0,1] | Alto | `structural_score` |
| **Diversidad** | Distancia media entre niveles generados | ≥0 | Alto | Distancia L1 en tiles |

### 6.2 Configuración de experimentos

Se proponen 3 configuraciones a comparar:

```
Experimento A — Baseline
  Modelo:   netG_epoch_5000.pth (1 nivel de entrenamiento)
  Fitness:  F1/F2 original (jugabilidad + saltos)
  Runs:     40 (como el paper)

Experimento B — Dataset expandido
  Modelo:   netG_15levels.pth (15 niveles de entrenamiento)
  Fitness:  F1/F2 original
  Runs:     40

Experimento C — Dataset expandido + Fitness estructural
  Modelo:   netG_15levels.pth
  Fitness:  F3 (jugabilidad + estructura + dificultad)
  Runs:     40
```

Comparar A vs B aísla el efecto de los datos.
Comparar B vs C aísla el efecto del fitness estructural.

---

## 7. División de Trabajo

### Carlos Ramírez
- `src/data/vglc_parser.py` — parser VGLC → JSON
- `src/data/dataset_builder.py` — construcción de `dataset_full.json`
- Re-entrenamiento del GAN con dataset expandido
- Notebook `02_dataset_expansion.ipynb`

### Nicolás Rogel
- `src/metrics/structural.py` — las 4 métricas estructurales
- `src/metrics/playability.py` — wrapper del agente A*
- `src/metrics/diversity.py` — diversidad entre niveles
- Notebook `03_structural_metrics.ipynb`

### Carlos Saavedra
- `src/fitness/f3_combined.py` — integración en CMA-ES
- `src/baseline/reproduce_volz.py` — reproducción del baseline
- `src/visualization/` — renderizado y gráficas comparativas
- Notebook `04_results_comparison.ipynb`

---

## 8. Riesgos y Decisiones de Diseño

### Por qué NO cambiar la arquitectura del GAN
Cambiar a StyleGAN, Transformer u otra arquitectura requiere:
- Tiempo de implementación que supera el alcance del curso
- Mayor riesgo de no converger con datasets pequeños de tiles discretos
- Dificultad para aislar la causa de mejoras en la evaluación

**Decisión:** Mantener DCGAN/WGAN. Actuar sobre datos (input) y fitness (output).

### Por qué NO implementar RL (Shu et al. 2021)
El paper de referencia [1] usa RL como optimizador en lugar de CMA-ES. Es un paradigma distinto que requeriría re-diseñar todo el componente de optimización.

**Decisión:** RL queda como trabajo futuro / comparación teórica en la discusión.

### Riesgo: GAN no generaliza con más datos
Con 15 niveles, el GAN podría colapsar o producir mezclas sin coherencia si los mundos son muy diferentes (e.g., mundo 3 acuático vs mundo 1 terrestre).

**Mitigación:** Entrenar también un modelo por mundo (mundo 1-1 a 1-4, etc.) y comparar. Si el modelo unificado es peor, se puede usar el modelo original de Volz et al. para el Experimento C.

### Riesgo: Pesos de F3 arbitrarios
Los pesos `(0.4, 0.35, 0.25)` de F3 son iniciales. Cambiarlos puede afectar significativamente los resultados.

**Mitigación:** Reportar resultados con al menos 2 configuraciones de pesos y discutir la sensibilidad.

### Compatibilidad de PyTorch
El código de DagstuhlGAN usa PyTorch ~0.4 (API antigua con `Variable()`). En versiones modernas puede requerir ajustes menores.

**Mitigación:** Crear un entorno conda con PyTorch 1.x o adaptar las llamadas deprecadas.

---

## 9. Cronograma

| Semana | Bloque | Tareas | Entregable |
|---|---|---|---|
| 1 | Baseline | Setup entorno, ejecutar DagstuhlGAN, medir métricas base | Resultados baseline documentados |
| 2 | Datos | Parser VGLC, construir dataset_full.json | `dataset_full.json` listo |
| 3 | Datos + Métricas | Re-entrenar GAN, implementar métricas estructurales | `netG_15levels.pth` + `structural.py` testeado |
| 4 | Fitness | Integrar F3 en CMA-ES, ejecutar Experimento C | Resultados Exp. C |
| 5 | Evaluación | Comparativa A vs B vs C, visualizaciones, conclusiones | Tablas + gráficas finales |

---

## Referencias

```
[1] Volz, V. et al. "Evolving Mario Levels in the Latent Space of a Deep Convolutional 
    Generative Adversarial Network." GECCO '18. Kyoto, Japan: ACM, 2018.
    https://doi.org/10.1145/3205455.3205517

[2] Shu, T., Liu, J., Yannakakis, G. N. "Experience-Driven PCG via Reinforcement Learning:
    A Super Mario Bros Study." IEEE Conference on Games (CoG), 2021.
    https://doi.org/10.1109/CoG52621.2021.9619124

[3] Summerville, A. et al. "The VGLC: The Video Game Level Corpus."
    7th Workshop on Procedural Content Generation, 2016.

[4] Vapnik, V. N. "An Overview of Statistical Learning Theory."
    IEEE Transactions on Neural Networks, Vol. 10, No. 5, 1999.

[5] Goodfellow, I. et al. "Generative Adversarial Nets."
    Advances in Neural Information Processing Systems, 2014.

[6] Arjovsky, M., Chintala, S., Bottou, L. "Wasserstein Generative Adversarial Networks."
    ICML, 2017.

[7] Hansen, N., Müller, S. D., Koumoutsakos, P. "Reducing the Time Complexity of the
    Derandomized Evolution Strategy with Covariance Matrix Adaptation (CMA-ES)."
    Evolutionary Computation, 2003.
```
