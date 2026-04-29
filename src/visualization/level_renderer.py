"""
Renderiza niveles de Mario como imágenes con colores por tipo de tile.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Colores por ID de tile
TILE_COLORS = {
    0: "#8B4513",  # Ground       — marrón
    1: "#CD853F",  # Breakable    — marrón claro
    2: "#87CEEB",  # Empty        — cielo
    3: "#FFD700",  # Q_full       — dorado
    4: "#DAA520",  # Q_empty      — dorado oscuro
    5: "#FF4500",  # Enemy        — rojo
    6: "#228B22",  # Pipe_TL      — verde
    7: "#228B22",  # Pipe_TR      — verde
    8: "#2E8B57",  # Pipe_L       — verde medio
    9: "#2E8B57",  # Pipe_R       — verde medio
}

TILE_NAMES = {
    0: "Ground", 1: "Breakable", 2: "Empty",
    3: "Q-Block(full)", 4: "Q-Block(empty)", 5: "Enemy",
    6: "Pipe TL", 7: "Pipe TR", 8: "Pipe L", 9: "Pipe R",
}


def render_level(level: np.ndarray, title: str = "", show: bool = True,
                 save_path: str = None, figsize: tuple = (14, 4)):
    """
    Renderiza un nivel (14, 28) como imagen con colores por tipo de tile.

    Args:
        level:     np.ndarray (14, 28) con IDs [0-9]
        title:     título del gráfico
        show:      si True, muestra la imagen
        save_path: si se especifica, guarda la imagen en esa ruta
        figsize:   tamaño de la figura matplotlib
    """
    rows, cols = level.shape
    img = np.zeros((rows, cols, 3))

    for r in range(rows):
        for c in range(cols):
            tile_id = int(level[r, c])
            hex_color = TILE_COLORS.get(tile_id, "#000000")
            rgb = tuple(int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5))
            img[r, c] = rgb

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")
    ax.set_xticks(range(0, cols, 4))
    ax.set_yticks(range(rows))

    # Leyenda
    patches = [
        mpatches.Patch(color=TILE_COLORS[i], label=TILE_NAMES[i])
        for i in sorted(set(level.flatten()))
        if i in TILE_COLORS
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=7, framealpha=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def render_comparison(levels: list, titles: list = None, save_path: str = None):
    """
    Renderiza múltiples niveles en una grilla para comparación visual.

    Args:
        levels:    lista de np.ndarray (14, 28)
        titles:    lista de strings con títulos
        save_path: ruta de guardado opcional
    """
    n = len(levels)
    if titles is None:
        titles = [f"Nivel {i+1}" for i in range(n)]

    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    for ax, level, title in zip(axes, levels, titles):
        rows, cols = np.array(level).shape
        img = np.zeros((rows, cols, 3))
        for r in range(rows):
            for c in range(cols):
                tile_id = int(level[r][c] if isinstance(level[r], list) else level[r, c])
                hex_c = TILE_COLORS.get(tile_id, "#000000")
                img[r, c] = tuple(int(hex_c[i:i+2], 16) / 255 for i in (1, 3, 5))
        ax.imshow(img, aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
