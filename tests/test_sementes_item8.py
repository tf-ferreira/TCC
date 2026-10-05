"""Testes do resumo por sementes (D26)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
for sub in ("models", "data", "experiments", "optimization"):
    sys.path.insert(0, str(RAIZ / "src" / sub))

pytest.importorskip("torch")
import sementes_item8 as S  # noqa: E402


def test_erro_padrao_e_dp_sobre_raiz_de_n_e_nao_o_dp():
    """D26 pede erro padrão da MÉDIA. Reportar o desvio entre sementes responderia
    outra pergunta, e a diferença é um fator de `sqrt(n)`."""
    v = np.array([1.0, 2.0, 3.0, 4.0])
    r = S.resumo(v)
    assert r["media"] == pytest.approx(2.5)
    assert r["desvio"] == pytest.approx(np.std(v, ddof=1))
    assert r["erro_padrao"] == pytest.approx(r["desvio"] / 2.0)
    assert r["erro_padrao"] < r["desvio"]


def test_resumo_usa_ddof_1_e_nao_o_populacional():
    v = np.array([10.0, 12.0, 14.0])
    assert S.resumo(v)["desvio"] == pytest.approx(2.0)          # ddof=1
    assert S.resumo(v)["desvio"] != pytest.approx(np.std(v))    # populacional


def test_uma_semente_nao_inventa_dispersao():
    r = S.resumo(np.array([7.0]))
    assert r["desvio"] == 0.0 and r["erro_padrao"] == 0.0 and r["n"] == 1


def test_amplitude_e_max_menos_min():
    v = np.array([3.0, 9.0, 5.0])
    r = S.resumo(v)
    assert r["amplitude"] == pytest.approx(6.0)
    assert (r["min"], r["max"]) == (3.0, 9.0)
