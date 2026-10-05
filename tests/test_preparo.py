"""Testes do preparo das entradas da rede (seção 6).

O teste central escreve **D25 como invariante executável**: o valor com que uma
defasagem ausente é preenchida não pode mudar quando o preço da própria semana
muda. É a propriedade, e não a lista de preenchimentos proibidos, o que faz o
teste pegar um preenchimento novo que ninguém revisou.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src" / "data"))

from preparo import (INDICE_RESERVA, ajustar_escalonador,  # noqa: E402
                     aplicar_escalonador, codificar, medianas_de_treino,
                     preencher_defasagens, tempo_truncado, vocabulario)


def _painel() -> pd.DataFrame:
    return pd.DataFrame({
        "store": [1, 1, 1, 2, 2],
        "sku": [0, 0, 0, 0, 0],
        "week": [1, 2, 3, 1, 2],
        "preco_proprio": [4.0, 3.0, 5.0, 8.0, 7.0],
        "preco_lag1": [np.nan, 4.0, 3.0, np.nan, 8.0],
    })


def test_o_preenchimento_nao_muda_quando_o_preco_decidido_muda():
    """D25 como invariante. Preencher com o preço da própria semana passaria em
    qualquer teste de formato e falha aqui, que é o ponto."""
    painel = _painel()
    med = medianas_de_treino(painel)
    a = preencher_defasagens(painel, med, ["preco_lag1"])

    mexido = painel.copy()
    mexido["preco_proprio"] = mexido["preco_proprio"] * 3.0  # o otimizador agiu
    b = preencher_defasagens(mexido, med, ["preco_lag1"])

    faltavam = painel["preco_lag1"].isna()
    assert np.allclose(a.loc[faltavam, "preco_lag1"], b.loc[faltavam, "preco_lag1"])


def test_o_indicador_distingue_ausencia_de_coincidencia():
    """Uma defasagem que por acaso vale a mediana não pode ficar indistinguível
    de uma que não existia."""
    painel = _painel()
    med = medianas_de_treino(painel)
    d = preencher_defasagens(painel, med, ["preco_lag1"])
    assert d["falta_preco_lag1"].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]
    assert d["preco_lag1"].notna().all()


def test_a_mediana_e_por_serie_e_so_do_treino():
    painel = _painel()
    treino = painel[painel.week <= 2]
    med = medianas_de_treino(treino)
    assert med.loc[(1, 0)] == 3.5      # mediana de 4,0 e 3,0
    assert med.loc[(2, 0)] == 7.5      # a loja 2 não empresta da loja 1
    # A semana 3, que é "teste" aqui, não entra no cálculo.
    assert med.loc[(1, 0)] != painel[painel.store == 1]["preco_proprio"].median()


def test_o_tempo_congela_fora_da_janela_de_treino():
    """89 semanas de extrapolação no painel real. Fora da janela a entrada
    temporal não pode continuar crescendo."""
    t = tempo_truncado([1, 50, 100, 150, 200], semana_min_treino=1,
                       semana_max_treino=100)
    assert t[0] == 0.0 and t[2] == 1.0
    assert t[3] == 1.0 and t[4] == 1.0          # congelado
    assert 0.0 < t[1] < 1.0                      # tendência dentro da janela


def test_o_escalonador_nao_ve_o_teste():
    treino = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    teste = pd.DataFrame({"x": [1000.0, 2000.0]})
    p = ajustar_escalonador(treino, ["x"])
    assert p["x"]["media"] == 2.0
    # Aplicar no teste não muda os parâmetros nem os recalcula.
    fora = aplicar_escalonador(teste, p)
    assert fora["x"].iloc[0] == (1000.0 - 2.0) / p["x"]["desvio"]
    assert ajustar_escalonador(treino, ["x"]) == p


def test_o_vocabulario_reserva_o_indice_zero():
    vocab = vocabulario(_painel(), "store")
    assert INDICE_RESERVA not in vocab.values()
    assert min(vocab.values()) == 1


def test_loja_so_do_teste_cai_na_reserva_com_vocabulario_do_treino():
    """Pendência 9.6. Com o vocabulário do painel inteiro este teste falha: a
    loja 3 ganha código próprio e a reserva nunca é usada."""
    painel = pd.concat([_painel(), pd.DataFrame({
        "store": [3], "sku": [0], "week": [3], "preco_proprio": [6.0],
        "preco_lag1": [np.nan]})], ignore_index=True)
    treino, teste = painel[painel.week <= 2], painel[painel.week == 3]
    vocab = vocabulario(treino, "store")
    cod_teste = codificar(teste.store.to_numpy(), vocab)
    assert cod_teste[teste.store.to_numpy() == 3].tolist() == [INDICE_RESERVA]
    # e a mesma loja tem o mesmo código nas duas partes (o erro de F2)
    assert cod_teste[teste.store.to_numpy() == 1][0] == vocab[1]


def test_categoria_nunca_vista_cai_no_indice_de_reserva():
    vocab = vocabulario(_painel(), "store")
    assert codificar([1, 2, 99], vocab).tolist() == [vocab[1], vocab[2], INDICE_RESERVA]
