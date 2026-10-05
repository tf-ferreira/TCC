"""Testes da decomposição de dois eixos de `varredura_perda.decompor`.

A decomposição atribui parte de uma diferença ao deslocamento de **nível** e o
resto à **ponderação entre SKUs**, usando como taxa de câmbio o par que move só
o nível. É uma atribuição aproximada, e o risco que estes testes cobrem é o de
ela parecer exata: um número como "63,6% vem do nível" é citável e por isso
precisa de estimador testado contra caso de resposta conhecida.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src" / "experiments"))

from varredura_perda import decompor  # noqa: E402


def _comparacao(nivel: dict, wmape: dict, elast: dict, invertido=()) -> dict:
    """Monta o formato de `ruido_semente.comparar`.

    `invertido` grava o par com os nomes na ordem trocada e o sinal invertido,
    que é como `comparar` de fato pode gravá-lo, já que ele ordena os nomes
    alfabeticamente. É o caso que o ajudante `_dif` existe para absorver.
    """
    eixos = {}
    for eixo, valores in (("vies_de_nivel", nivel), ("wmape", wmape),
                          ("elast_h10", elast)):
        pares = {}
        for (a, b), v in valores.items():
            if (a, b) in invertido:
                pares[f"{b}__menos__{a}"] = {"diferenca_media": -v}
            else:
                pares[f"{a}__menos__{b}"] = {"diferenca_media": v}
        eixos[eixo] = {"pares": pares}
    return {"eixos": eixos}


def test_diferenca_inteiramente_de_nivel_da_cem_por_cento():
    """Taxa de 10 por unidade de nível, deslocamento total o dobro do puro: se
    a diferença total for exatamente 10 x 0,2, nada sobra para a ponderação."""
    c = _comparacao(
        nivel={("msle", "msle_smearing"): 0.1, ("msle", "poisson"): 0.2},
        wmape={("msle", "msle_smearing"): 1.0, ("msle", "poisson"): 2.0},
        elast={("msle", "msle_smearing"): 0.0, ("msle", "poisson"): 0.0})
    d = decompor(c)["eixos"]["wmape"]
    assert abs(d["taxa_por_unidade_de_nivel"] - 10.0) < 1e-12
    assert abs(d["atribuido_ao_nivel"] - 2.0) < 1e-12
    assert abs(d["residuo_de_ponderacao"]) < 1e-12
    assert abs(d["pct_explicado_pelo_nivel"] - 100.0) < 1e-9


def test_eixo_insensivel_ao_nivel_atribui_tudo_a_ponderacao():
    """É o caso da derivada: a taxa é zero porque um fator multiplicativo
    constante não muda ∂ln V / ∂ln p, então a diferença toda é ponderação."""
    c = _comparacao(
        nivel={("msle", "msle_smearing"): 0.1, ("msle", "poisson"): 0.2},
        wmape={("msle", "msle_smearing"): 1.0, ("msle", "poisson"): 2.0},
        elast={("msle", "msle_smearing"): 0.0, ("msle", "poisson"): -0.5})
    d = decompor(c)["eixos"]["elast_h10"]
    assert d["taxa_por_unidade_de_nivel"] == 0.0
    assert d["atribuido_ao_nivel"] == 0.0
    assert abs(d["residuo_de_ponderacao"] + 0.5) < 1e-12
    assert d["pct_explicado_pelo_nivel"] == 0.0


def test_metade_e_metade():
    c = _comparacao(
        nivel={("msle", "msle_smearing"): 0.1, ("msle", "poisson"): 0.1},
        wmape={("msle", "msle_smearing"): 1.0, ("msle", "poisson"): 2.0},
        elast={("msle", "msle_smearing"): 0.0, ("msle", "poisson"): 0.0})
    d = decompor(c)["eixos"]["wmape"]
    assert abs(d["pct_explicado_pelo_nivel"] - 50.0) < 1e-9
    assert abs(d["residuo_de_ponderacao"] - 1.0) < 1e-12


def test_par_gravado_na_ordem_trocada_da_o_mesmo_resultado():
    base = dict(
        nivel={("msle", "msle_smearing"): 0.1, ("msle", "poisson"): 0.2},
        wmape={("msle", "msle_smearing"): 1.0, ("msle", "poisson"): 3.0},
        elast={("msle", "msle_smearing"): 0.0, ("msle", "poisson"): -0.5})
    direto = decompor(_comparacao(**base))
    trocado = decompor(_comparacao(**base, invertido=(("msle", "poisson"),)))
    assert direto == trocado


def test_sem_deslocamento_no_par_puro_falha_em_vez_de_inventar():
    c = _comparacao(
        nivel={("msle", "msle_smearing"): 0.0, ("msle", "poisson"): 0.2},
        wmape={("msle", "msle_smearing"): 0.0, ("msle", "poisson"): 2.0},
        elast={("msle", "msle_smearing"): 0.0, ("msle", "poisson"): 0.0})
    with pytest.raises(ValueError):
        decompor(c)
