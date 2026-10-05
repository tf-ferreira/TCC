"""Testes de `alinhamento_d16.py`.

Dois mecanismos plantados. O primeiro é o viés de cruzado omitido sob IV: a
versão sem cruzados tem de errar **pelo tanto que a fórmula prevê**,
`β = ε₁₁ + γ₁₂·δ`. O segundo é o defeito da primeira versão do script: custo
que varia só por SKU × semana, igual em todas as lojas. Um efeito fixo de SKU ×
semana o absorve; o empilhado com semana comum não.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
for pasta in ("data", "experiments"):
    sys.path.insert(0, str(RAIZ / "src" / pasta))

from alinhamento_d16 import (estimar_inclinacoes, matrizes, resumir,  # noqa: E402
                             variancia_custo_sku_semana)
from elasticidades import centralizar  # noqa: E402

EPS, GAMA = -2.0, 0.5
UPCS = [10, 20]


def _longo(semente=0, S=60, T=150, var_comum=1.0, custo_so_no_tempo=False):
    """Dois SKUs; ln V₁ = −2·u₁ + 0,5·u₂ + choque, preço endógeno ao choque."""
    rng = np.random.default_rng(semente)
    lojas, semanas = np.repeat(np.arange(S), T), np.tile(np.arange(T), S)
    n = S * T
    a_loja = np.repeat(rng.normal(0, 1, S), T)
    a_sem = np.tile(rng.normal(0, 1, T), S)
    if custo_so_no_tempo:
        comum = np.tile(rng.normal(0, np.sqrt(var_comum) * 0.1, T), S)
        lnc = [comum + np.tile(rng.normal(0, 0.1, T), S) for _ in range(2)]
    else:
        comum = rng.normal(0, np.sqrt(var_comum) * 0.1, n)
        lnc = [comum + rng.normal(0, 0.1, n) for _ in range(2)]
    choque = rng.normal(0, 0.1, n)
    lnp = [0.8 * lnc[j] + 0.5 * choque + rng.normal(0, 0.05, n) for j in range(2)]
    lnv = [4 + a_loja + 0.3 * a_sem + EPS * lnp[0] + GAMA * lnp[1] + choque
           + rng.normal(0, 0.05, n),
           4 + a_loja + 0.3 * a_sem + EPS * lnp[1] + GAMA * lnp[0] + choque
           + rng.normal(0, 0.05, n)]
    partes = [pd.DataFrame({"store": lojas, "week": semanas, "upc": UPCS[j],
                            "preco": np.exp(lnp[j]), "custo": np.exp(lnc[j]),
                            "volume": np.exp(lnv[j])}) for j in range(2)]
    return pd.concat(partes, ignore_index=True)


def _estimar(longo, cruz):
    P, C, _ = matrizes(longo, UPCS)
    return estimar_inclinacoes(longo, P, C, UPCS, cruz)


def test_cruzado_exogeno_nao_corrige_o_vies_do_choque_comum():
    """O preço do outro também responde ao choque: tratá-lo como exógeno não basta.

    Por isso E4 é uma medida do eixo, não a estimativa certa das cruzadas.
    """
    s = _estimar(_longo(), "exogenos")["por_sku"][0]["iv"]
    assert abs(s["media_outros"]["coef"] - GAMA) > 0.1


def test_sem_cruzados_erra_pelo_tanto_que_a_formula_preve():
    longo = _longo()
    r = _estimar(longo, "sem")
    d = longo.pivot_table(index=["store", "week"], columns="upc", values=["preco", "custo"])
    idx = d.index
    M = centralizar(np.column_stack([np.log(d["preco"][10]), np.log(d["preco"][20]),
                                     np.log(d["custo"][10])]),
                    [idx.get_level_values(0).to_numpy(), idx.get_level_values(1).to_numpy()])
    delta = (M[:, 1] @ M[:, 2]) / (M[:, 0] @ M[:, 2])
    assert delta > 0.3
    b = r["por_sku"][0]["iv"]["propria"]["coef"]
    assert abs(b - (EPS + GAMA * delta)) < 0.08
    assert abs(b - EPS) > 0.1


def test_sem_insumo_comum_o_vies_some():
    r = _estimar(_longo(var_comum=0.0), "sem")
    assert abs(r["por_sku"][0]["iv"]["propria"]["coef"] - EPS) < 0.08


def test_custo_so_no_tempo_sobrevive_no_empilhado_e_e_detectado():
    longo = _longo(custo_so_no_tempo=True)
    v = variancia_custo_sku_semana(longo)
    assert v["ln_custo"] > 0.9                      # o diagnóstico enxerga
    r = _estimar(longo, "sem")
    assert r["por_sku"][0]["r2_parcial_custo_proprio"] > 0.05
    assert np.isfinite(r["por_sku"][0]["iv"]["propria"]["ep_agrupado"])


def test_media_dos_outros_custos_perde_um_posto():
    """Por que não há E5: custo invariante entre lojas, instrumentos com posto menor que 2N."""
    longo = _longo(custo_so_no_tempo=True)
    P, C, _ = matrizes(longo, UPCS)
    d = longo.merge(P.reset_index()[["store", "week"]], on=["store", "week"])
    chave = pd.MultiIndex.from_arrays([d.store, d.week])
    LC = np.log(C.loc[chave].to_numpy())
    pos = d.upc.map({10: 0, 20: 1}).to_numpy()
    c_prop = LC[np.arange(len(d)), pos]
    c_out = LC.sum(axis=1) - c_prop
    D = np.eye(2)[pos]
    Z = np.column_stack([c_prop[:, None] * D, c_out[:, None] * D])
    serie = d.store.to_numpy() * 100 + pos
    Zc = centralizar(Z, [serie, d.week.to_numpy()])
    sv = np.linalg.svd(Zc, compute_uv=False)
    assert sv[-1] / sv[0] < 1e-8          # posto menor que 2N: não identificado


def test_mqo_e_viesado_pelo_choque_e_iv_nao():
    s = _estimar(_longo(var_comum=0.0), "sem")["por_sku"][0]
    assert s["ols"]["propria"]["coef"] > EPS + 0.1
    assert abs(s["iv"]["propria"]["coef"] - EPS) < 0.08


def test_resumo_separa_mediana_de_media_ponderada():
    por = [{"volume_total": w, "ols": {"propria": {"coef": b}},
            "iv": {"propria": {"coef": b}}} for b, w in ((-1, 1), (-2, 1), (-4, 8))]
    s = resumir(por)["iv"]
    assert s["mediana"] == -2 and abs(s["media_ponderada_volume"] - (-3.5)) < 1e-12


def test_matrizes_ficam_alinhadas_quando_uma_celula_nao_tem_custo_nenhum():
    """Regressão do erro de 17/09: 32.255 linhas de preço contra 32.253 de custo."""
    longo = pd.DataFrame({
        "store": [1, 1, 1, 1], "week": [1, 1, 2, 2], "upc": [10, 20, 10, 20],
        "preco": [1.0, 2.0, 1.1, 2.1], "custo": [0.5, 1.0, np.nan, np.nan],
        "volume": [5, 6, 7, 8]})
    P, C, V = matrizes(longo, [10, 20])
    assert P.index.equals(C.index) and P.index.equals(V.index)
    assert len(P) == 2 and C.loc[(1, 2)].isna().all()
