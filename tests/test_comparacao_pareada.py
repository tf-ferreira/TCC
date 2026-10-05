"""Testes do teste pareado de D26, em `ruido_semente.comparar`.

O erro que estes testes existem para impedir é o que D26 nasceu combatendo, e
que reapareceu na decisão da perda: a comparação era feita **à mão** a cada
varredura, e três coisas dela erram com facilidade silenciosa. O pareamento
(as configurações correm sobre as mesmas sementes, e a variância de semente é
comum), o limiar corrigido por C(k,2) e não por k, e a cobertura dos três
eixos.

O caso que prova o pareamento é construído de propósito: variância de semente
grande e diferença entre configurações pequena e constante. Sem parear, o sinal
some dentro do ruído comum; pareando, ele aparece.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src" / "experiments"))

from ruido_semente import comparar  # noqa: E402


def _consolidado(series: dict[str, np.ndarray]) -> dict:
    """Monta o formato que `consolidar` grava, só com o eixo do WMAPE."""
    return {"configuracoes": {
        nome: {"wmape_por_semente": {str(i): float(v) for i, v in enumerate(vals)}}
        for nome, vals in series.items()}}


def test_o_pareamento_recupera_sinal_que_o_nao_pareado_perde():
    rng = np.random.default_rng(0)
    semente = rng.normal(0.0, 5.0, size=50)   # ruído COMUM às duas
    a = semente + rng.normal(0.0, 0.01, 50)
    b = semente + 0.20 + rng.normal(0.0, 0.01, 50)

    par = comparar(_consolidado({"a": a, "b": b}))["eixos"]["wmape"]["pares"]
    z_pareado = abs(par["a__menos__b"]["z"])

    ep_nao_pareado = np.sqrt(a.std(ddof=1) ** 2 / 50 + b.std(ddof=1) ** 2 / 50)
    z_nao_pareado = abs(a.mean() - b.mean()) / ep_nao_pareado

    assert z_pareado > 50, z_pareado
    assert z_nao_pareado < 1.0, z_nao_pareado


def test_o_limiar_corrige_por_pares_e_nao_por_configuracoes():
    rng = np.random.default_rng(1)
    s = {n: rng.normal(size=30) for n in ("a", "b", "c")}
    tres = comparar(_consolidado(s))
    dois = comparar(_consolidado({k: v for k, v in s.items() if k != "c"}))

    assert tres["n_pares"] == 3 and dois["n_pares"] == 1
    assert abs(dois["limiar_z_corrigido"]
               - statistics.NormalDist().inv_cdf(0.975)) < 1e-9
    assert abs(tres["limiar_z_corrigido"]
               - statistics.NormalDist().inv_cdf(1 - 0.05 / 6)) < 1e-9
    # E o limiar de tres pares tem de ser MAIOR: é isso que a correção faz.
    assert tres["limiar_z_corrigido"] > dois["limiar_z_corrigido"]


def test_diferenca_constante_sem_variancia_pareada_passa():
    a = np.arange(20, dtype=float)
    par = comparar(_consolidado({"a": a, "b": a + 1.0}))["eixos"]["wmape"]["pares"]
    d = par["a__menos__b"]
    assert abs(d["diferenca_media"] + 1.0) < 1e-12
    assert d["erro_padrao_pareado"] == 0.0
    assert d["passa_limiar"] is True


def test_configuracoes_identicas_nao_passam():
    rng = np.random.default_rng(2)
    a = rng.normal(size=40)
    par = comparar(_consolidado({"a": a, "b": a.copy()}))["eixos"]["wmape"]["pares"]
    assert par["a__menos__b"]["diferenca_media"] == 0.0
    assert par["a__menos__b"]["passa_limiar"] is False


def test_eixo_ausente_no_artefato_e_omitido_em_vez_de_inventado():
    """Um consolidado antigo não tem as séries por semente do nível e da
    elasticidade. O certo é omitir o eixo, não devolver zero."""
    rng = np.random.default_rng(3)
    saida = comparar(_consolidado({"a": rng.normal(size=10),
                                   "b": rng.normal(size=10)}))
    assert "wmape" in saida["eixos"]
    assert "vies_de_nivel" not in saida["eixos"]
    assert "elast_h10" not in saida["eixos"]
