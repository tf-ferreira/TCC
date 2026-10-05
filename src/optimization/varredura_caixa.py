"""Varredura do raio da caixa: a coerencia do modelo piora no otimo?

## A pergunta, e por que ela mudou de enunciado

A primeira versao deste script, de 17/09/2026, partia de uma premissa FALSA: que
o otimizador estava explorando incoerencia do modelo, porque na caixa de +-15% ele
subia o "preco medio" e o volume total subia junto em 73% das celulas sob margem.

Aquela medida confundia incoerencia com COMPOSICAO. Media de variacoes de preco
sem ponderacao nao e indice de preco, e volume total responde a um vetor de vinte
precos que se movem em direcoes diferentes: baixar o preco de um SKU de giro alto e
subir dezenove de giro baixo faz a media subir e o volume subir, e isso e
substituicao funcionando, nao defeito. Trocada a medida pela identidade da
pendencia 2.8, que fala de subida UNIFORME de 1% em todos os precos e por isso e
imune a composicao, a leitura se inverte. Ver `sonda_item8.rodar`.

A pergunta que sobrevive e mais estreita e ainda vale medir: **a coerencia do
modelo piora no ponto que o solver escolhe?** `gamma` e constante (D36) e foi
ajustado sobre co-movimento historico; o solver escolhe combinacoes de canto que o
painel nao contem, e nada garante que a identidade continue valendo ali.

## O criterio, declarado antes do resultado

A caixa admissivel seria a maior em que a violacao da identidade no otimo nao
excede a violacao no ponto historico. Isto NAO e escolher a regua depois de ver o
numero, que e o erro que D21 nomeia: o criterio compara o otimo contra a BASE, e
nao contra o ganho.

## O custo de encolher a caixa, e ele e real

Raio menor e menos ganho por construcao, porque o conjunto viavel esta contido no
maior. A varredura mede os dois lado a lado, de modo que a escolha e entre ganho e
coerencia com os dois numeros na mesa.

Uso:

    python3 src/optimization/varredura_caixa.py frj
    python3 src/optimization/varredura_caixa.py frj --raios 0.01,0.02,0.05,0.10,0.15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
for sub in ("models", "data", "experiments", "optimization"):
    sys.path.insert(0, str(RAIZ / "src" / sub))

import problema as P  # noqa: E402
import rede  # noqa: E402
from sonda_item8 import OBJETIVOS, carregar_modelo, rodar  # noqa: E402

RAIOS = (0.01, 0.02, 0.05, 0.10, 0.15)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--artefato", default=None)
    ap.add_argument("--raios", default=",".join(str(r) for r in RAIOS))
    ap.add_argument("--celulas", type=int, default=0,
                    help="0 usa TODAS as células do escopo de D38")
    ap.add_argument("--semente-amostra", type=int, default=0)
    ap.add_argument("--metodo", default="SLSQP")
    ap.add_argument("--saida", default="reports")
    a = ap.parse_args()

    raios = [float(x) for x in a.raios.split(",")]
    if any(not 0 < r < 1 for r in raios):
        raise SystemExit(f"raios têm de estar em (0, 1): {raios}")

    painel = Path(a.painel)
    dados = rede.preparar(a.categoria, painel)
    arq = Path(a.artefato or RAIZ / "models_artifacts" / f"rede_{a.categoria}.pt")
    modelo, cfg, centro = carregar_modelo(arq, dados)
    esc = P.escopo(dados, painel, a.categoria, "teste")

    total = len(esc["u"])
    if a.celulas and a.celulas < total:
        rng = np.random.default_rng(a.semente_amostra)
        indices = np.sort(rng.choice(total, size=a.celulas, replace=False))
        amostra = {"tipo": "aleatoria_declarada", "n": int(a.celulas),
                   "semente": a.semente_amostra}
    else:
        indices = np.arange(total)
        amostra = {"tipo": "todas", "n": total}

    linhas = []
    for r in raios:
        limites = (1.0 - r, 1.0 + r)
        for obj in OBJETIVOS:
            saida = rodar(modelo, centro, esc, obj, indices, a.metodo, limites)
            saida["raio"] = r
            linhas.append(saida)

    bloco = {"categoria": a.categoria, "artefato": arq.name,
             "configuracao": cfg, "escopo": esc["contagens"],
             "amostra": amostra, "raios": raios, "linhas": linhas}
    destino = Path(a.saida) / f"varredura_caixa_item8_{a.categoria}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(bloco, indent=2, ensure_ascii=False))

    print(f"escopo: {esc['contagens']['n_celulas_com_custo_completo']} células, "
          f"amostra {amostra['tipo']} de {amostra['n']}\n")
    for obj in OBJETIVOS:
        print(f"== {obj}")
        print(f"{'raio':>6} {'ganho ag.':>10} {'viol. base':>11} "
              f"{'viol. otimo':>12} {'e agr. base':>12} {'e agr. otimo':>13} "
              f"{'interior':>9}")
        for L in (x for x in linhas if x["objetivo"] == obj):
            print(f"{100 * L['raio']:5.0f}% {L['ganho_pct_agregado']:9.2f}% "
                  f"{L['pct_violacao_identidade_base']:10.2f}% "
                  f"{L['pct_violacao_identidade_otimo']:11.2f}% "
                  f"{L['elast_agregada_uniforme_base']:12.3f} "
                  f"{L['elast_agregada_uniforme_otimo']:13.3f} "
                  f"{L['pct_coord_interior']:8.2f}%")
        print()
    print(f"-> {destino}")


if __name__ == "__main__":
    main()
