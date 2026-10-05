"""Carregador único das previsões do item 7, sem retreinar.

As 50 redes do ponto adotado já foram treinadas por `espelho_d16.py`, que gravou
`ln V̂` por semente em `data/interim/espelho_d16/`, e por
`decomposicao_espelho_d16.py`, que gravou os três termos aditivos em
`componentes/`. Os scripts de 9.2 a 9.4 leem esses fragmentos. Se algum faltar,
este módulo **para** em vez de treinar em silêncio uma rede que ninguém conferiu.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
for pasta in ("data", "models", "experiments"):
    sys.path.insert(0, str(RAIZ / "src" / pasta))

CANAIS = ("contexto", "cruzado", "proprio")


def carregar(categoria: str, painel: Path, sementes: int, fragmentos: Path,
             componentes: bool = False) -> dict:
    import rede
    from espelho_d16 import impressao_digital, linhas
    from varredura_rede import MELHOR

    dados = rede.preparar(categoria, painel)
    cel = pd.read_parquet(painel / f"{categoria}_celulas.parquet")
    base = dict(MELHOR)
    base.setdefault("gama_contextual", False)

    z = {"treino": [], "teste": []}
    # Os componentes entram só como MÉDIA acumulada: guardar 3 × 50 × linhas
    # estourava a memória de uma máquina de 4 GB. A média basta, porque o IV de
    # cada canal é linear em y e a decomposição é lida sobre a média das sementes.
    comp = {p: {c: None for c in CANAIS} for p in z}
    faltando = []
    for s in range(sementes):
        dig = impressao_digital(dict(base, semente=s))
        arq = fragmentos / f"semente_{s:02d}_{dig}.npz"
        if not arq.exists():
            faltando.append(arq.name)
            continue
        f = np.load(arq)
        for p in z:
            z[p].append(f[p])
        if componentes:
            arq_c = fragmentos / "componentes" / f"semente_{s:02d}_{dig}.npz"
            if not arq_c.exists():
                faltando.append("componentes/" + arq_c.name)
                continue
            g = np.load(arq_c)
            for p in z:
                for c in CANAIS:
                    v = g[f"{p}_{c}"].astype("float64") / sementes
                    comp[p][c] = v if comp[p][c] is None else comp[p][c] + v
    if faltando:
        raise SystemExit(f"{len(faltando)} fragmentos ausentes (ex.: {faltando[0]}). "
                         "Rode espelho_d16.py e decomposicao_espelho_d16.py antes.")
    saida = {"dados": dados, "n": dados["n"], "base": base,
             "linhas": {"treino": linhas(dados["treino"], cel),
                        "teste": linhas(dados["teste"], cel)},
             "ln_prev": {p: np.column_stack(z[p]) for p in z}}
    if componentes:
        saida["componentes_media"] = comp
    return saida
