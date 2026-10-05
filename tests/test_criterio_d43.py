"""O critério de D43 aplicado a saídas sintéticas das duas rodadas."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "reports"))

from criterio_d43 import MUNDOS, OBJETIVOS, POLITICAS, montar  # noqa: E402


def _sementes(desloc: dict, sementes=range(4)):
    """Saída no formato de `sementes_v1.py`; `desloc[obj]` soma ao objetivo 5."""
    return {"por_semente": [
        {"semente": s, **{obj: {pol: {w: {"objetivo5_contra_o_observado_pct": 10.0 + s + desloc.get(obj, 0.0),
                                          "ganho_modelo_contra_modelo_pct": 20.0 + s + desloc.get(obj, 0.0),
                                          "calibracao_no_historico": 0.9}
                                      for w in MUNDOS} for pol in POLITICAS} for obj in OBJETIVOS}}
        for s in sementes]}


def _anatomia(desloc: dict, sementes=range(4), seq_de=None, canal=10.0):
    """Saída no formato de `anatomia_v1.py`; `canal` é o ganho que passa pelas defasagens."""
    L = []
    for s in sementes:
        l = {"semente": s}
        for obj in OBJETIVOS:
            l[obj] = {}
            for pol in POLITICAS:
                l[obj][pol] = {
                    "estatica": {w: {"objetivo5_pct": 5.0 + s + desloc.get(obj, 0.0), "ganho_pct": 20.0}
                                 for w in MUNDOS},
                    "sequencial": {w: {"objetivo5_pct": (10.0 + s if seq_de is None else seq_de),
                                       "ganho_pct": 20.0 + canal} for w in MUNDOS},
                    "sequencial_com_defasagem_historica": {w: {"objetivo5_pct": 0.0, "ganho_pct": 20.0}
                                                           for w in MUNDOS}}
        L.append(l)
    return {"por_semente": L}


def test_pareia_por_semente_e_aplica_o_limiar():
    v1 = _sementes({})
    cod = _sementes({"margem": -2.5, "receita": -0.5}, sementes=range(3))  # 3 sementes comuns
    t = montar(v1, cod, None, None)
    seq = t["bases"]["sequencial"]
    assert seq["margem"]["robusta"]["mu=1"]["n"] == 3
    assert abs(seq["margem"]["robusta"]["mu=1"]["media"] + 2.5) < 1e-12
    assert seq["criterio"]["leitura"] == "cenario"
    assert "pendente" in t["bases"]["estatica"]
    dec = seq["decomposicao_robusta_mu1"]["margem"]
    assert abs(dec["ganho_pontos"]["media"] + 2.5) < 1e-12 and dec["calibracao"]["media"] == 0.0


def test_base_estatica_e_conferencia_da_sequencial():
    v1s, cods = _sementes({}), _sementes({"margem": -3.0})
    v1a, coda = _anatomia({}), _anatomia({"margem": -1.0, "receita": 1.9}, canal=4.0)
    t = montar(v1s, cods, v1a, coda)
    est = t["bases"]["estatica"]
    assert abs(est["margem"]["robusta"]["mu=1"]["media"] + 1.0) < 1e-12
    assert est["criterio"]["leitura"] == "sensibilidade"      # |−1,0| e |1,9| < 2
    assert t["bases"]["sequencial"]["criterio"]["leitura"] == "cenario"
    assert t["anatomia_reproduz_r3_max_dif_pontos"] == 0.0
    # o canal das defasagens com e sem o código, nas mesmas sementes
    c = t["canal_das_defasagens"]["receita"]["robusta"]["mu=1"]
    assert c["v1"]["media"] == 10.0 and c["v1_codigo"]["media"] == 4.0
    assert c["codigo_menos_v1"]["media"] == -6.0
    # uma anatomia que não reproduz a R3 aparece na conferência
    t2 = montar(v1s, cods, _anatomia({}, seq_de=0.0), coda)
    assert t2["anatomia_reproduz_r3_max_dif_pontos"] > 1.0
