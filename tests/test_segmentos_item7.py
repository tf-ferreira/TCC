"""Testes de 9.2 a 9.4, sem torch."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
for pasta in ("data", "experiments"):
    sys.path.insert(0, str(RAIZ / "src" / pasta))

from erro_por_segmento_item7 import nivel_por_grupo, wmape_por_grupo  # noqa: E402
from espelho_por_sku_d16 import espelho_por_sku  # noqa: E402


def _painel(semente=0, S=40, T=120, N=3):
    rng = np.random.default_rng(semente)
    loja = np.repeat(np.arange(S), N * T)
    sku = np.tile(np.repeat(np.arange(N), T), S)
    semana = np.tile(np.arange(T), S * N)
    serie = loja * 100 + sku
    L = len(loja)
    fe = rng.normal(0, 1, S * 100)[serie] + rng.normal(0, 1, T)[semana]
    lnc = rng.normal(0, 0.1, L)
    choque = rng.normal(0, 0.1, L)
    lnp = 0.8 * lnc + 0.5 * choque + rng.normal(0, 0.05, L)
    eps = np.array([-1.0, -2.0, -3.0])[sku]
    y = fe + eps * lnp + choque + rng.normal(0, 0.05, L)
    return y, lnp, lnc, serie, semana, sku, fe, eps


def test_espelho_por_sku_acha_o_sku_que_a_rede_erra():
    y, lnp, lnc, serie, semana, sku, fe, eps = _painel()
    errada = eps.copy(); errada[sku == 2] = -2.0          # rede erra só o SKU 2
    rng = np.random.default_rng(1)
    H = np.column_stack([fe + errada * lnp + rng.normal(0, 0.01, len(y)) for _ in range(3)])
    canais = {"proprio": errada * lnp, "cruzado": 0 * y, "contexto": fe}
    r = espelho_por_sku(y, H, lnp, lnc, serie, semana, sku, 3, canais)
    p = r["por_sku"]
    assert abs(p[0]["diferenca"]) < 0.08 and abs(p[1]["diferenca"]) < 0.08
    assert abs(p[2]["diferenca"] - (-1.0)) < 0.1
    for q in p:
        assert abs(q["diferenca"] - (q["obs"] - q["espelho"])) < 1e-8
        assert not q["instrumento_fraco"]
    assert abs(p[2]["canais"]["proprio"] - (-2.0)) < 0.05


def test_wmape_por_grupo_bate_com_a_conta_a_mao():
    y = np.array([10., 20., 30., 40.])
    Yh = np.array([[12., 11.], [18., 22.], [30., 27.], [50., 40.]])
    g = np.array([0, 0, 1, 1])
    _, w, dp = wmape_por_grupo(y, Yh, g)
    w0 = [100 * (2 + 2) / 30, 100 * (1 + 2) / 30]
    w1 = [100 * (0 + 10) / 70, 100 * (3 + 0) / 70]
    assert np.allclose(w, [np.mean(w0), np.mean(w1)])
    assert np.allclose(dp, [np.std(w0, ddof=1), np.std(w1, ddof=1)])


def test_contribuicoes_ao_deficit_somam_um():
    rng = np.random.default_rng(0)
    y = rng.uniform(1, 100, 300)
    Yh = y[:, None] * rng.uniform(0.6, 1.0, (300, 4))
    g = rng.integers(0, 7, 300)
    r = nivel_por_grupo(y, Yh, g)
    assert abs(r["contribuicao_deficit"].sum() - 1) < 1e-12
    assert abs(r["participacao_volume"].sum() - 1) < 1e-12


def test_contribuicao_estavel_quando_o_nivel_cruza_1_entre_sementes():
    """Regressão do erro de 17/09: sementes com nível acima e abaixo de 1."""
    y = np.array([100., 100.])
    Yh = np.array([[80., 125.], [110., 70.]])      # semente 0: 190; semente 1: 195
    Yh = np.column_stack([Yh, np.array([120., 85.])])  # semente 2: 205 (acima)
    r = nivel_por_grupo(y, Yh, np.array([0, 1]))
    sh = Yh.mean(1)
    assert np.allclose(r["contribuicao_deficit"], (y - sh) / (y.sum() - sh.sum()))
    # grupo 0 subprevisto na média, grupo 1 superprevisto: sinais opostos
    assert np.sign(r["contribuicao_deficit"][0]) != np.sign(r["contribuicao_deficit"][1])
