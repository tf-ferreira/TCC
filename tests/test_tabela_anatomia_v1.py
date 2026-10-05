"""Testes das tabelas da anatomia (`src/reports/tabela_anatomia_v1.py`, D47)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "reports"))

import tabela_anatomia_v1 as T  # noqa: E402

PASSADAS = ("estatica", "sequencial", "estatica_com_defasagem_propria",
            "sequencial_com_defasagem_historica")


def _semente(base_rede, base_rob, canal, cal=0.95):
    """Ganho = base + deslocamento por mundo; o sequencial soma `canal` ao ganho."""
    desl = {"mu=0": -4.0, "mu=0.5": 0.0, "mu=1": 4.0, "so_substitutos": 8.0}
    def pol(base):
        fora = {}
        for q in PASSADAS:
            extra = canal if q in ("sequencial", "estatica_com_defasagem_propria") else 0.0
            fora[q] = {}
            for w, d in desl.items():
                g = base + d + extra
                fora[q][w] = {"ganho_pct": g, "objetivo5_pct": 100 * ((1 + g / 100) * cal - 1)}
        return fora
    return {obj: {"rede": pol(base_rede), "robusta": pol(base_rob)}
            for obj in ("margem", "receita")}


def test_principal_canal_e_pareada():
    b = {"especificacao": "v1", "por_semente": [_semente(10, 12, 5), _semente(6, 7, 3)]}
    t = T.montar(b)
    m = t["margem"]
    assert m["calibracao"]["media"] == pytest.approx(0.95)
    # o canal, com a decisão fixa, é exatamente o deslocamento posto no sequencial
    assert m["canal_das_defasagens"]["rede"]["mu=1"]["media"] == pytest.approx(4.0)
    # a principal (estática) não tem o canal: objetivo 5 em μ = 1 da rede
    esperado = [100 * ((1 + (10 + 4) / 100) * 0.95 - 1), 100 * ((1 + (6 + 4) / 100) * 0.95 - 1)]
    assert m["principal"]["rede"]["mu=1"]["objetivo5"]["media"] == pytest.approx(sum(esperado) / 2)
    # robusta − rede: +2 e +1 de ganho, vezes a calibração, em todo mundo
    assert m["principal"]["robusta_menos_rede"]["mu=0"]["media"] == pytest.approx(1.5 * 0.95)


def test_calibracao_inconsistente_levanta():
    s = _semente(10, 12, 5)
    s["receita"]["robusta"]["sequencial"]["mu=0"]["objetivo5_pct"] += 1.0
    with pytest.raises(ValueError):
        T.montar({"especificacao": "v1", "por_semente": [s, _semente(6, 7, 3)]})


def test_markdown():
    t = T.montar({"especificacao": "v1", "por_semente": [_semente(10, 12, 5), _semente(6, 7, 3)]})
    md = T.markdown(t)
    assert md.count("| robusta − rede |") == 4
    assert "canal das defasagens" in md
