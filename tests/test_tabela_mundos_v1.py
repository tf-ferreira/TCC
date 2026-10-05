"""Testes da tabela política × mundo (`src/reports/tabela_mundos_v1.py`)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "reports"))

import tabela_mundos_v1 as T  # noqa: E402


def _linha(base_rede, base_rob, cal=0.95):
    """Uma semente sintética: objetivo 5 = base + deslocamento por mundo."""
    desl = {"mu=0": -4.0, "mu=0.5": 0.0, "mu=1": 4.0, "so_substitutos": 8.0}
    def pol(base):
        return {w: {"objetivo5_contra_o_observado_pct": base + d,
                    "ganho_modelo_contra_modelo_pct": base + d + 1.0,
                    "calibracao_no_historico": cal} for w, d in desl.items()}
    return {obj: {"rede": pol(base_rede), "robusta": pol(base_rob)}
            for obj in ("margem", "receita")}


def test_pareada_e_pior_caso():
    bloco = {"especificacao": "v1", "por_semente": [_linha(10.0, 12.0), _linha(6.0, 7.0)]}
    t = T.montar(bloco)
    m = t["margem"]
    assert m["rede"]["mu=1"]["objetivo5"]["media"] == pytest.approx(12.0)
    # o pior dos três mundos da média é μ = 0 em cada semente: 6 e 2 na rede
    assert m["rede"]["pior_dos_tres"]["media"] == pytest.approx(4.0)
    # robusta − rede: +2 e +1 em todo mundo, pareado
    d = m["robusta_menos_rede"]["mu=0.5"]
    assert d["media"] == pytest.approx(1.5)
    assert d["erro_padrao"] == pytest.approx(0.5)
    assert d["z"] == pytest.approx(3.0)


def test_calibracao_que_varia_entre_mundos_levanta():
    l = _linha(10.0, 12.0)
    l["receita"]["robusta"]["mu=0"]["calibracao_no_historico"] = 0.90
    with pytest.raises(ValueError):
        T.montar({"especificacao": "v1", "por_semente": [l, _linha(6.0, 7.0)]})


def test_markdown_tem_as_duas_tabelas():
    t = T.montar({"especificacao": "v1", "por_semente": [_linha(10.0, 12.0), _linha(6.0, 7.0)]})
    md = T.markdown(t)
    assert md.count("| robusta − rede (pareado) |") == 2
    assert "pior dos três" in md
