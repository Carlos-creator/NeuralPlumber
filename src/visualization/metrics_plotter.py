"""
Gráficas comparativas de métricas entre experimentos A, B y C.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


EXPERIMENT_LABELS = {
    "baseline":          "A — Baseline\n(1 nivel, F1/F2)",
    "expanded_data":     "B — Datos\n(15 niveles, F1/F2)",
    "structural_fitness": "C — Full\n(15 niveles, F3)",
}

METRIC_LABELS = {
    "pipe_completeness":  "Completitud\nde tuberías",
    "ground_continuity":  "Continuidad\ndel suelo",
    "gap_traversability": "Traversabilidad\nde huecos",
    "enemy_placement":    "Placement\nde enemigos",
    "structural_avg":     "Score\nestructural",
}


def load_metrics(experiments_dir: str) -> dict:
    """Carga métricas guardadas de cada experimento."""
    results = {}
    exp_dir = Path(experiments_dir)
    for exp_name in EXPERIMENT_LABELS:
        metrics_file = exp_dir / exp_name / "metrics_baseline.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            results[exp_name] = data["metrics"]
    return results


def plot_metrics_comparison(results: dict, save_path: str = None):
    """
    Gráfica de barras agrupadas comparando métricas entre experimentos.
    """
    metric_keys = [k for k in METRIC_LABELS if k != "structural_avg"]
    exp_keys    = list(results.keys())

    x    = np.arange(len(metric_keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, (exp_key, color) in enumerate(zip(exp_keys, colors)):
        values = [results[exp_key].get(k, 0) for k in metric_keys]
        bars = ax.bar(x + i * width, values, width, label=EXPERIMENT_LABELS[exp_key],
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([METRIC_LABELS[k] for k in metric_keys], fontsize=9)
    ax.set_ylabel("Score (0-1, mayor es mejor)")
    ax.set_title("Comparación de métricas estructurales entre experimentos")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_structural_avg(results: dict, save_path: str = None):
    """Gráfica de barras simple del score estructural promedio por experimento."""
    exp_keys = list(results.keys())
    values   = [results[k].get("structural_avg", 0) for k in exp_keys]
    labels   = [EXPERIMENT_LABELS[k] for k in exp_keys]
    colors   = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors[:len(values)], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Score estructural promedio")
    ax.set_title("Score estructural global por experimento")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", default="../../experiments")
    parser.add_argument("--save_dir", default="../../experiments")
    args = parser.parse_args()

    results = load_metrics(args.experiments_dir)
    if results:
        plot_metrics_comparison(results,
            save_path=f"{args.save_dir}/metrics_comparison.png")
        plot_structural_avg(results,
            save_path=f"{args.save_dir}/structural_avg.png")
    else:
        print("No se encontraron resultados. Ejecuta primero reproduce_volz.py")
