"""
Parser: convierte niveles .txt del VGLC al formato JSON de DagstuhlGAN.
Uso:
    python vglc_parser.py
        --input_dir ../../Datasets/TheVGLC/Super\ Mario\ Bros/Processed/
        --output    ../../NeuralPlumber/data/dataset_full.json
        --window_width 28
        --window_step  1
"""
import json
import numpy as np
from pathlib import Path

# Mapeo VGLC símbolo → ID numérico (mismo que usa DagstuhlGAN)
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
    # Tiles del VGLC no usados por el paper → mapear a vacío o sólido
    'o': 2,  # Coin      → empty
    'B': 0,  # Cannon top → solid
    'b': 0,  # Cannon bottom → solid
}


def parse_level_file(path: Path) -> np.ndarray:
    """Lee un .txt del VGLC y retorna array (14, N) de IDs."""
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) != 14:
        raise ValueError(f"{path.name}: esperaba 14 filas, encontró {len(lines)}")
    return np.array([[TILE_MAP.get(c, 2) for c in line] for line in lines])


def sliding_windows(level: np.ndarray, width: int = 28, step: int = 1) -> list:
    """Extrae ventanas deslizantes de ancho `width` con paso `step`."""
    _, total_width = level.shape
    return [
        level[:, start:start + width].tolist()
        for start in range(0, total_width - width + 1, step)
    ]


def build_dataset(input_dir: str, output_path: str, width: int = 28, step: int = 1):
    input_dir = Path(input_dir)
    txt_files = sorted(input_dir.glob("mario-*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No se encontraron archivos mario-*.txt en {input_dir}")

    print(f"Encontrados {len(txt_files)} niveles en {input_dir}")
    all_windows = []

    for f in txt_files:
        level = parse_level_file(f)
        windows = sliding_windows(level, width, step)
        all_windows.extend(windows)
        print(f"  {f.name}: {level.shape[1]:4d} cols → {len(windows):4d} ventanas")

    print(f"\nTotal ventanas: {len(all_windows)}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(all_windows), encoding="utf-8")
    print(f"Guardado en: {output_path}")
    return all_windows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Construye dataset JSON desde VGLC")
    parser.add_argument("--input_dir", required=True,
                        help="Directorio con archivos mario-*.txt del VGLC")
    parser.add_argument("--output", default="../../data/dataset_full.json",
                        help="Ruta de salida del JSON")
    parser.add_argument("--window_width", type=int, default=28,
                        help="Ancho de la ventana deslizante (default: 28)")
    parser.add_argument("--window_step", type=int, default=1,
                        help="Paso de la ventana (default: 1)")
    args = parser.parse_args()
    build_dataset(args.input_dir, args.output, args.window_width, args.window_step)
