"""Duas medições que fecham o item 6 e que não cabem numa varredura.

**8.4, ρ intra-célula sobre o resíduo da REDE.** A pendência 4.1 foi corrigida
com ρ medido sobre o resíduo de uma árvore, e ficou declarado que ele não é
invariante ao modelo. Aqui ele sai sobre a rede adotada e sobre a rede sem
restrição, que é o que fecha o resíduo daquela pendência. A rede sem restrição
dá o número menos confundido pela forma imposta.

**2.8, a identidade de agregação, agora como CONTA e não estimativa.** A
restrição é

    |ε_agregada|  ≤  |média ponderada das próprias|

e ela existe porque, para substitutos, as cruzadas positivas compensam parte do
efeito próprio: a cesta inteira é menos elástica que a média dos seus itens. Com
a matriz da rede a conta é direta, para cada célula, com `sᵢ = V̂ᵢ / Σ V̂`:

    ε_agregada = Σᵢ sᵢ Σⱼ εᵢⱼ          (subida uniforme de 1% em todos os preços)
    própria ponderada = Σᵢ sᵢ εᵢᵢ

O que se reporta é a mediana das duas, a folga entre elas e a **fração de
células em que a desigualdade é violada**. Violação não é erro de conta: é a
rede dizendo que a cesta é mais elástica que seus itens, o que é sinal de
cruzada com sinal errado em peso suficiente.

Uso:
    python3 src/experiments/fechamento_item6.py frj --sementes 0-9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src" / "models"))
sys.path.insert(0, str(RAIZ / "src" / "experiments"))

import rede  # noqa: E402
from baseline_arvores import metricas  # noqa: E402
from ruido_semente import faixa  # noqa: E402
from unidade_de_observacao import icc_balanceado  # noqa: E402
from varredura_rede import MELHOR  # noqa: E402


# A identidade de agregação mora em `rede.identidade_de_agregacao`, ponto único.
# Ela estava escrita aqui também, com outros nomes de chave, e fórmula duplicada
# é fórmula que diverge.
from rede import identidade_de_agregacao  # noqa: E402,F401

def uma_semente(dados: dict, cfg: dict) -> dict:
    saida = rede.treinar(dados, cfg)
    T = saida["tensores"]
    te = dados["teste"]
    y = te["alvo"]
    p_te = saida["pred_teste"]
    p_tr = saida["pred_treino"]

    grupo = np.repeat(np.arange(len(y)), y.shape[1])
    mu = np.maximum(p_te, 1e-9)
    pearson = ((y - mu) / np.sqrt(mu)).ravel()
    icc = icc_balanceado(pearson, grupo)

    with torch.no_grad():
        v = torch.tensor(p_te, dtype=torch.float32)
    eps = rede.elasticidades_por_autograd(
        saida["modelo"], T["ctx_teste"], T["u_teste"], T["loja_teste"]).detach()

    return {
        "wmape_treino": float(metricas(dados["treino"]["alvo"].ravel(),
                                       p_tr.ravel())["wmape_pct"]),
        "wmape_teste": float(metricas(y.ravel(), p_te.ravel())["wmape_pct"]),
        "rho_intra_celula": icc["rho"],
        "efeito_de_desenho": icc["efeito_de_desenho"],
        "n_efetivo": icc["n_efetivo"],
        "razao_n_efetivo_sobre_n_celulas": icc["razao_n_efetivo_sobre_n_celulas"],
        **identidade_de_agregacao(eps, v),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--sementes", default="0-9")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--saida", default="reports")
    a = ap.parse_args()

    dados = rede.preparar(a.categoria, Path(a.painel))
    registro = {"categoria": a.categoria, "configuracao": MELHOR, "redes": {}}
    for rotulo, restrita in (("restrita", True), ("irrestrita", False)):
        linhas = [uma_semente(dados, dict(MELHOR, restrita=restrita, semente=s))
                  for s in faixa(a.sementes)]
        registro["redes"][rotulo] = {
            "n_sementes": len(linhas),
            **{k: {"media": float(np.mean([x[k] for x in linhas])),
                   "erro_padrao_da_media":
                       float(np.std([x[k] for x in linhas], ddof=1)
                             / np.sqrt(len(linhas))) if len(linhas) > 1 else None}
               for k in linhas[0]},
        }

    destino = Path(a.saida) / f"fechamento_item6_{a.categoria}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(registro, indent=2, ensure_ascii=False))

    print(f"{'':12s} {'WMAPE tr':>9} {'WMAPE te':>9} {'rho':>8} {'deff':>7} "
          f"{'n efet.':>10} {'x cel':>7} {'e_agr':>8} {'propria':>8} "
          f"{'folga':>8} {'%viol':>7}")
    for rot, d in registro["redes"].items():
        g = lambda k: d[k]["media"]  # noqa: E731
        print(f"{rot:12s} {g('wmape_treino'):8.2f}% {g('wmape_teste'):8.2f}% "
              f"{g('rho_intra_celula'):8.4f} {g('efeito_de_desenho'):7.2f} "
              f"{g('n_efetivo'):10.0f} {g('razao_n_efetivo_sobre_n_celulas'):7.2f} "
              f"{g('elast_agregada_mediana'):8.4f} "
              f"{g('propria_ponderada_mediana'):8.4f} "
              f"{g('folga_mediana'):8.4f} {g('pct_celulas_com_violacao'):6.1f}%")
    print(f"\ngravado em {destino}")


if __name__ == "__main__":
    main()
