"""Figuras estáticas da triagem de categorias, para o texto do TCC.

Lê `data/interim/screening/_payload.json`, produzido por `src/data/screening.py`,
e emite cada figura em PDF vetorial (impressão) e PNG 300 dpi (rascunho).

Convenções adotadas, discutidas em `reports/tcc_text/03_selecao_da_categoria.md`:

- Largura de 15 cm, a área útil de uma página A4 com margens ABNT.
- Liberation Serif, metricamente compatível com Times New Roman.
- Legibilidade em escala de cinza: onde há cor, ela é redundante com posição,
  forma do marcador ou rótulo direto. Nenhuma informação depende só de cor.
- Nenhum limiar é desenhado nas figuras (decisão D6). Linhas de referência,
  quando existem, marcam ajuste de reta ou mediana, não corte de decisão.

Uso:
    python3 src/reports/figuras_triagem.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CM = 1 / 2.54
LARGURA = 15 * CM

AZUL = "#2a78d6"
LARANJA = "#eb6834"
VERDE = "#1baf7a"
TINTA = "#1a1f26"
TINTA2 = "#5a6472"
GRADE = "#d9dee5"


def configurar() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": TINTA2,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.5,
        "xtick.color": TINTA2,
        "ytick.color": TINTA2,
        "text.color": TINTA,
        "axes.labelcolor": TINTA,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def salvar(fig, destino: Path, nome: str) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino / f"{nome}.pdf")
    fig.savefig(destino / f"{nome}.png", dpi=300)
    plt.close(fig)
    print(f"  {nome}.pdf e {nome}.png")


def _correlacao(x, y) -> float:
    """Correlação de postos de Spearman, sem depender do SciPy."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _rotular(ax, pontos, cor=TINTA2, dy=0.014, evitar=()):
    """Rotula pontos evitando sobreposição, desistindo quando não há espaço.

    Melhor omitir um nome do que empilhar dois textos ilegíveis. As posições
    candidatas são acima, abaixo, à direita e à esquerda do ponto.

    `evitar` recebe artistas já desenhados (uma legenda, por exemplo) cujas
    caixas devem contar como ocupadas: sem isso o rótulo passa por cima da
    legenda, que o algoritmo não teria como enxergar.
    """
    fig = ax.figure
    fig.canvas.draw()
    ocupados = [a.get_window_extent().extents for a in evitar]

    def colide(a, b):
        return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

    for x, y, texto in pontos:
        for dx_, dy_, ha, va in ((0, dy, "center", "bottom"),
                                 (0, -dy, "center", "top"),
                                 (0.004, 0, "left", "center"),
                                 (-0.004, 0, "right", "center")):
            t = ax.text(x + dx_, y + dy_, texto, ha=ha, va=va,
                        fontsize=7.2, color=cor)
            fig.canvas.draw()
            cx = t.get_window_extent().extents
            if any(colide(cx, o) for o in ocupados):
                t.remove()
                continue
            ocupados.append(cx)
            break


# --------------------------------------------------------------------------
# Figura 1: posicionamento nos três critérios
# --------------------------------------------------------------------------

def figura_posicionamento(cats, destino):
    fig, ax = plt.subplots(figsize=(LARGURA, LARGURA * 0.66))
    x = np.array([c["cv_preco_p50"] for c in cats])
    y = np.array([c["regularidade_p50"] for c in cats])
    p = np.array([c["promo_preco_p50"] for c in cats])

    tam = 18 + 320 * (p / p.max())
    ax.scatter(x, y, s=tam, facecolor=AZUL, alpha=0.42,
               edgecolor=AZUL, linewidth=0.7, zorder=3)

    ax.set_xlabel("Variância de preço (coeficiente de variação mediano)")
    ax.set_ylabel("Regularidade mediana")
    ax.set_xlim(0.02, 0.14)
    ax.set_ylim(0, 0.92)

    # Referência de tamanho, para o leitor decodificar a terceira dimensão.
    for valor, rotulo in ((0.05, "5%"), (0.15, "15%"), (0.23, "23%")):
        ax.scatter([], [], s=18 + 320 * (valor / p.max()), facecolor=AZUL,
                   alpha=0.42, edgecolor=AZUL, linewidth=0.7,
                   label=f"{rotulo} das semanas")
    # Canto superior esquerdo: variância baixa com regularidade alta é
    # combinação que nenhuma categoria apresenta, então a região está livre.
    leg = ax.legend(title="Frequência promocional", loc="upper left",
                    frameon=True, labelspacing=1.0, borderpad=0.7,
                    handletextpad=1.1)
    leg.get_frame().set_edgecolor(GRADE)
    leg.get_frame().set_linewidth(0.5)
    leg.get_title().set_fontsize(8)

    # Rotular só depois da legenda, para que ela conte como área ocupada.
    destaque = {"sdr", "cso", "frj", "fre", "tti", "che", "tbr", "sha",
                "cig", "bat", "frd"}
    _rotular(ax, [(c["cv_preco_p50"], c["regularidade_p50"], c["nome"])
                  for c in cats if c["categoria"] in destaque],
             dy=0.028, evitar=[leg])

    salvar(fig, destino, "fig1_posicionamento")


# --------------------------------------------------------------------------
# Figura 2: redundância entre os critérios 1 e 2
# --------------------------------------------------------------------------

def figura_redundancia(cats, destino):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARGURA, LARGURA * 0.44))
    cv = np.array([c["cv_preco_p50"] for c in cats])
    pr = np.array([c["promo_preco_p50"] for c in cats])
    rg = np.array([c["regularidade_p50"] for c in cats])

    for ax, eixo, rotulo, cor in ((a1, pr, "Frequência promocional", LARANJA),
                                  (a2, rg, "Regularidade", VERDE)):
        ax.scatter(cv, eixo, s=22, facecolor=cor, alpha=0.65,
                   edgecolor=cor, linewidth=0.6, zorder=3)
        rho = _correlacao(cv, eixo)
        ax.set_xlabel("Variância de preço")
        ax.set_ylabel(rotulo)
        ax.set_title(f"$\\rho_{{Spearman}} = {rho:.3f}$", pad=6)
        ax.set_xlim(0.02, 0.14)

    salvar(fig, destino, "fig2_redundancia")


# --------------------------------------------------------------------------
# Figura 3: estabilidade contra tamanho do sortimento
# --------------------------------------------------------------------------

def figura_sortimento(cats, destino):
    fig, ax = plt.subplots(figsize=(LARGURA, LARGURA * 0.52))
    n = np.array([c["n_upcs"] for c in cats], dtype=float)
    rg = np.array([c["regularidade_p50"] for c in cats])

    ax.scatter(n, rg, s=24, facecolor=VERDE, alpha=0.7,
               edgecolor=VERDE, linewidth=0.6, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Número de SKUs na categoria (escala logarítmica)")
    ax.set_ylabel("Regularidade mediana")
    ax.set_title(f"$\\rho_{{Spearman}} = {_correlacao(n, rg):.3f}$", pad=6)

    destaque = {"sha", "sdr", "gro", "cso", "tti", "oat", "frj", "cig"}
    _rotular(ax, [(c["n_upcs"], c["regularidade_p50"], c["nome"])
                  for c in cats if c["categoria"] in destaque], dy=0.03)
    salvar(fig, destino, "fig3_sortimento")


# --------------------------------------------------------------------------
# Figura 4: distribuição da variância, todas as categorias
# --------------------------------------------------------------------------

def figura_distribuicao_cv(cats, destino):
    """Caixas construídas a partir dos quantis já apurados na triagem.

    Responde à pergunta de como apresentar 27 categorias sem 27 figuras: uma
    caixa por categoria, ordenadas pela mediana, mostra distribuição inteira e
    ordenação no mesmo gráfico.
    """
    ordenadas = sorted(cats, key=lambda c: c["cv_preco_p50"])
    fig, ax = plt.subplots(figsize=(LARGURA, LARGURA * 0.95))

    caixas = [{
        "label": c["nome"],
        "whislo": c["cv_preco_p10"], "q1": c["cv_preco_p25"],
        "med": c["cv_preco_p50"],
        "q3": c["cv_preco_p75"], "whishi": c["cv_preco_p90"],
        "fliers": [],
    } for c in ordenadas]

    ax.bxp(caixas, vert=False, showfliers=False, widths=0.62,
           boxprops=dict(facecolor=AZUL, alpha=0.32, edgecolor=AZUL, linewidth=0.8),
           medianprops=dict(color=TINTA, linewidth=1.3),
           whiskerprops=dict(color=AZUL, linewidth=0.8),
           capprops=dict(color=AZUL, linewidth=0.8),
           patch_artist=True)
    ax.set_xlabel("Coeficiente de variação do preço unitário, por série (SKU × loja)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.24)
    salvar(fig, destino, "fig4_distribuicao_cv")


# --------------------------------------------------------------------------
# Figura 5: curvas de sobrevivência da regularidade
# --------------------------------------------------------------------------

def figura_sobrevivencia(cats, destino, siglas=("cso", "che", "frj", "sdr", "sha")):
    """Proporcao de SKUs que sobrevive a cada limiar possivel de regularidade.

    Normalizar pela contagem inicial e o que torna comparaveis categorias de
    tamanhos muito diferentes no mesmo eixo (xampus tem 3.179 SKUs, papel
    higienico tem 128).

    Legenda em vez de rotulo em ponta de linha: todas as curvas convergem para
    zero no limiar 1, entao rotulos no fim se sobrepoem por construcao.
    """
    fig, ax = plt.subplots(figsize=(LARGURA, LARGURA * 0.58))
    xs = np.linspace(0, 1, 21)
    porc = {c["categoria"]: c for c in cats}
    estilos = ["-", (0, (5, 1.5)), (0, (1, 1.3)), (0, (4, 1.4, 1, 1.4)), (0, (8, 2))]

    for i, s in enumerate(siglas):
        c = porc[s]
        y = np.array(c["surv_reg"], dtype=float) / c["surv_reg"][0]
        # linestyle vai por palavra-chave: o argumento posicional de formato
        # do matplotlib so aceita string, nao a tupla de tracejado.
        ax.plot(xs, y, linestyle=estilos[i % len(estilos)], color=TINTA,
                linewidth=1.25, label=f"{c['nome']} ({c['n_upcs']} SKUs)")

    ax.set_xlabel("Limiar de regularidade")
    ax.set_ylabel("Proporção de SKUs que permanecem")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    leg = ax.legend(loc="lower left", frameon=True, handlelength=3.4)
    leg.get_frame().set_edgecolor(GRADE)
    leg.get_frame().set_linewidth(0.5)
    salvar(fig, destino, "fig5_sobrevivencia")


def main() -> None:
    configurar()
    payload = json.loads(
        Path("data/interim/screening/_payload.json").read_text()
    )
    cats = payload["categorias"]
    destino = Path("reports/figures")
    print(f"gerando figuras de {len(cats)} categorias em {destino}/")
    figura_posicionamento(cats, destino)
    figura_redundancia(cats, destino)
    figura_sortimento(cats, destino)
    figura_distribuicao_cv(cats, destino)
    figura_sobrevivencia(cats, destino)


if __name__ == "__main__":
    main()
