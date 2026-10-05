"""Testes da regra de seleção do sortimento (D12).

Cada caso usa uma matriz de presença minúscula cuja resposta é contável à
mão. O módulo existe porque a regra esteve implementada em dois lugares com
critérios diferentes sem que nada detectasse; teste é a segunda linha de
defesa contra a mesma classe de erro.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src" / "data"))

from sortimento import (PISO_AMPLITUDE, PISO_LONGEVIDADE, agregar_direto,
                        agregar_dois_estagios, elegiveis, guloso,
                        matriz_presenca, semente_padrao)


def _presenca(colunas: list[list[int]], n_celulas: int) -> np.ndarray:
    """Matriz células x candidatos a partir da lista de células de cada um."""
    M = np.zeros((n_celulas, len(colunas)), dtype=bool)
    for j, celulas in enumerate(colunas):
        M[celulas, j] = True
    return M


CASO = _presenca([
    list(range(0, 10)),   # 0: em todas as 10 células
    list(range(0, 8)),    # 1: nas oito primeiras
    list(range(0, 7)),    # 2: nas sete primeiras
    list(range(5, 10)),   # 3: nas cinco últimas
    list(range(0, 5)),    # 4: nas cinco primeiras
], 10)


def test_semente_e_o_mais_presente():
    assert semente_padrao(CASO) == 0


def test_guloso_escolhe_a_maior_intersecao():
    escolhidos, completas = guloso(CASO, 3, semente=0)
    assert escolhidos == [0, 1, 2]
    assert completas == 7


def test_completude_nao_cresce_com_n():
    contagens = [guloso(CASO, n, semente=0)[1] for n in range(1, 6)]
    assert contagens == sorted(contagens, reverse=True)
    assert contagens[0] == 10


def test_guloso_depende_da_semente_e_pode_perder():
    """A heurística não é ótima, e o teste fixa o contraexemplo.

    O candidato 0 é o mais presente e portanto a semente padrão, mas a
    interseção que ele forma com qualquer outro é menor que a interseção entre
    1 e 2. Partindo da semente padrão o resultado é pior.
    """
    M = _presenca([
        list(range(0, 8)),                 # 0: oito células, semente padrão
        [0, 1, 2, 3, 8, 9],                # 1
        [0, 1, 2, 3, 8, 9],                # 2
    ], 10)
    assert semente_padrao(M) == 0
    assert guloso(M, 2, semente=0)[1] == 4
    assert guloso(M, 2, semente=1)[1] == 6


def test_piso_de_suporte_e_inclusivo_e_exige_as_duas_dimensoes():
    cobertura = pd.DataFrame({
        "upc": [1, 2, 3, 4, 5],
        "regularidade": [0.9, 0.9, 0.9, 0.9, np.nan],
        "longevidade": [PISO_LONGEVIDADE, PISO_LONGEVIDADE - 0.01, 0.99, 0.99, 0.99],
        "amplitude": [PISO_AMPLITUDE, 0.99, PISO_AMPLITUDE - 0.01, 0.99, 0.99],
    })
    assert elegiveis(cobertura) == [1, 4]


def test_matriz_presenca_marca_onde_ha_preco():
    longo = pd.DataFrame({
        "store": [1, 1, 1, 2, 2],
        "week": [1, 1, 2, 1, 1],
        "upc": [10, 20, 10, 10, 20],
        "preco": [1.0, 2.0, 1.0, 1.0, 2.0],
    })
    M = matriz_presenca(longo, [10, 20])
    # três células: (1,1), (1,2), (2,1). O SKU 20 falta em (1,2).
    assert M.shape == (3, 2)
    assert M[:, 0].sum() == 3
    assert M[:, 1].sum() == 2


def test_dois_estagios_difere_de_direto_quando_ha_desequilibrio_de_lojas():
    """O SKU presente em muitas lojas domina a mediana direta e não a de dois
    estágios, que dá um voto por SKU. É a razão declarada em D3 para a
    agregação em dois estágios ser a oficial."""
    series = pd.DataFrame({
        "upc": [1] * 9 + [2],
        "cv": [0.10] * 9 + [0.50],
    })
    assert agregar_direto(series, [1, 2], "cv") == 0.10
    assert agregar_dois_estagios(series, [1, 2], "cv") == 0.30
