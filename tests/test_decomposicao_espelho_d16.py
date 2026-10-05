"""A decomposição do espelho é exata: 2SLS é linear em y.

Planta três canais com IVs conhecidos e confere que cada canal devolve o seu e
que os três somam o total até erro de máquina.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
for pasta in ("data", "experiments"):
    sys.path.insert(0, str(RAIZ / "src" / pasta))

from decomposicao_espelho_d16 import decompor  # noqa: E402


def test_tres_canais_devolvem_cada_um_o_seu_e_somam_o_total():
    rng = np.random.default_rng(0)
    S, T = 40, 100
    loja = np.repeat(np.arange(S), T)
    semana = np.tile(np.arange(T), S)
    serie = loja
    n = S * T
    lnc = rng.normal(0, 0.1, n)
    lnp = 0.8 * lnc + rng.normal(0, 0.05, n)
    lag = 0.6 * lnc + rng.normal(0, 0.05, n)          # defasagem que o custo arrasta
    outro = 0.3 * lnc + rng.normal(0, 0.05, n)
    partes = {"proprio": np.column_stack([-2.0 * lnp] * 2),
              "cruzado": np.column_stack([0.5 * outro] * 2),
              "contexto": np.column_stack([-1.0 * lag + rng.normal(0, 0.1, n)
                                           for _ in range(2)])}
    y = sum(p[:, 0] for p in partes.values())
    r = decompor(y, partes, lnp, lnc, serie, semana, loja)["resumo"]
    assert abs(r["proprio"]["media"] - (-2.0)) < 0.05
    assert abs(r["cruzado"]["media"] - 0.5 * 0.3 / 0.8) < 0.05
    assert abs(r["contexto"]["media"] - (-1.0 * 0.6 / 0.8)) < 0.05
    assert r["maior_erro_de_soma"] < 1e-9
