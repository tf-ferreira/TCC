"""Testes da função de controle do custo (D42), antes de usar com dado real.

O que se protege aqui é o que NÃO aparece no resultado: o alinhamento do custo
com a posição do vetor, o ajuste que não pode ver a outra parte (D28) e o valor
neutro quando o custo falta. Um erro em qualquer um deles deixa `v̂` com o
tamanho certo e a rede aprendendo outra coisa.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src" / "data"))

import funcao_de_controle as FC  # noqa: E402


def _painel(n_lojas=4, n_semanas=60, n=3, pi=(0.6, 0.3, 0.0), semente=0):
    """Painel sintético: ln p = a_loja + π·ln c + ruído, e o cel correspondente."""
    rng = np.random.default_rng(semente)
    linhas, cel = [], []
    for s in range(n_lojas):
        for w in range(n_semanas):
            reg = {"store": 10 + s, "week": 100 + w}
            for i in range(n):
                lnc = rng.normal(0.0, 0.2)
                lnp = 0.1 * s + pi[i] * lnc + rng.normal(0.0, 0.05) + 1.0
                reg[f"custo_{i:02d}"] = float(np.exp(lnc))
                linhas.append({"store": 10 + s, "week": 100 + w, "sku": i,
                               "preco_proprio": float(np.exp(lnp)),
                               "sen_1": np.sin(w / 8), "cos_1": np.cos(w / 8),
                               "sen_2": np.sin(w / 4), "cos_2": np.cos(w / 4),
                               "tempo": w / n_semanas})
            cel.append(reg)
    return pd.DataFrame(linhas), pd.DataFrame(cel)


def test_custo_longo_alinha_com_a_posicao():
    cel = pd.DataFrame({"store": [1], "week": [7], "custo_00": [0.5],
                        "custo_01": [0.7], "custo_02": [0.9]})
    c = FC.custo_longo(cel, 3).sort_values("sku")
    assert c["sku"].tolist() == [0, 1, 2]
    assert np.allclose(c["custo"], [0.5, 0.7, 0.9])


def test_recupera_o_repasse_num_dado_sintetico():
    longo, cel = _painel()
    coefs = FC.ajustar(longo, FC.custo_longo(cel, 3))
    assert abs(coefs[0]["pi"] - 0.6) < 0.03
    assert abs(coefs[1]["pi"] - 0.3) < 0.03
    assert abs(coefs[2]["pi"]) < 0.03
    # O SKU sem repasse tem F baixo: é o diagnóstico de instrumento fraco.
    assert coefs[2]["f_parcial"] < 10 < coefs[0]["f_parcial"]


def test_o_residuo_tem_media_zero_por_serie_no_ajuste():
    """Com efeito de loja por SKU, MQO zera a soma do resíduo em cada série."""
    longo, cel = _painel()
    fora, diag = FC.adicionar(longo, cel=cel, n=3)
    medias = fora.groupby(["store", "sku"])["v_cf"].mean()
    assert np.abs(medias).max() < 1e-10
    assert len(diag["pi"]) == 3


def test_o_ajuste_nao_ve_a_outra_parte():
    """D28: mudar o preço da parte de APLICAÇÃO não muda o ajuste."""
    longo, cel = _painel()
    ajuste = longo[longo.week < 140].copy()
    teste = longo[longo.week >= 140].copy()
    a1, t1, _ = FC.adicionar(ajuste, teste, cel=cel, n=3)
    teste2 = teste.copy()
    teste2["preco_proprio"] *= 1.5
    a2, t2, _ = FC.adicionar(ajuste, teste2, cel=cel, n=3)
    assert np.allclose(a1["v_cf"], a2["v_cf"])
    # E na parte de aplicação o resíduo muda exatamente por ln(1,5).
    assert np.allclose(t2["v_cf"] - t1["v_cf"], np.log(1.5))


def test_custo_ausente_vira_zero_com_indicador():
    longo, cel = _painel()
    cel.loc[cel.index[0], "custo_01"] = np.nan
    fora, _ = FC.adicionar(longo, cel=cel, n=3)
    s0, w0 = cel.loc[cel.index[0], ["store", "week"]]
    linha = fora[(fora.store == s0) & (fora.week == w0) & (fora.sku == 1)]
    assert float(linha["v_cf"].iloc[0]) == 0.0
    assert float(linha["falta_v_cf"].iloc[0]) == 1.0
    assert float(fora["falta_v_cf"].sum()) == 1.0


def test_loja_fora_do_ajuste_recebe_a_media_dos_efeitos():
    longo, cel = _painel()
    ajuste = longo[longo.store != 13].copy()
    nova = longo[longo.store == 13].copy()
    _, aplicada, _ = FC.adicionar(ajuste, nova, cel=cel, n=3)
    # A loja 13 tem efeito 0,3 no gerador e o ajuste lhe dá a média das outras
    # (0,1 = média de 0; 0,1; 0,2): o resíduo absorve a diferença, cerca de 0,2.
    assert abs(float(aplicada["v_cf"].mean()) - 0.2) < 0.03


def test_a_ordem_das_linhas_nao_importa():
    longo, cel = _painel()
    embaralhado = longo.sample(frac=1.0, random_state=3)
    a, _ = FC.adicionar(longo, cel=cel, n=3)
    b, _ = FC.adicionar(embaralhado, cel=cel, n=3)
    assert np.allclose(a["v_cf"].to_numpy(), b.loc[a.index, "v_cf"].to_numpy())
