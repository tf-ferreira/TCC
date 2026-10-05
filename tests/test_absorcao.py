"""Testes da absorção de efeitos fixos por centralização alternada (F4).

`test_estimacao.py` já verifica que a fórmula está certa, com painel balanceado
de 12 por 9 e sem ruído, onde uma passada resolve. Esse teste não podia pegar o
risco que F4 apontou, que é de **convergência**: o laço rodava 25 vezes e não
verificava nada, e convergência incompleta atenua o coeficiente na direção do
MQO, em silêncio.

Os testes abaixo separam as duas coisas: um painel bem conectado, onde o antigo
25 de fato bastava, e um painel mal conectado, onde não basta.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
for pasta in ("data", "experiments"):
    sys.path.insert(0, str(RAIZ / "src" / pasta))

from elasticidades import centralizar  # noqa: E402


def _coef(M: np.ndarray) -> float:
    b, *_ = np.linalg.lstsq(M[:, 1:2], M[:, 0], rcond=None)
    return float(b[0])


def _cadeia(n_blocos: int = 12, por_bloco: int = 6):
    """Painel mal conectado: blocos ligados em cadeia por uma observação só.

    É o caso extremo do que governa a velocidade da centralização alternada. Os
    dois espaços de efeito fixo são quase ortogonais quando o painel é bem
    conectado, e quase paralelos quando cada bloco toca o vizinho por um único
    ponto. Aí alternar projeções anda devagar, e um número fixo de passadas
    para antes da hora.
    """
    i, t = [], []
    for b in range(n_blocos):
        for a in range(por_bloco):
            i.append(b)
            t.append(b * por_bloco + a)
    for b in range(n_blocos - 1):
        i.append(b)
        t.append((b + 1) * por_bloco)
    return np.array(i), np.array(t)


def _dados(i, t, semente=1, beta=2.0):
    rng = np.random.default_rng(semente)
    alfa = rng.normal(size=i.max() + 1)[i]
    gama = rng.normal(size=t.max() + 1)[t]
    x = rng.normal(size=len(i))
    return np.column_stack([alfa + gama + beta * x, x])


def test_painel_bem_conectado_converge_em_poucas_passadas():
    """O caso do painel real: 93 lojas por 396 semanas, densamente cruzadas."""
    rng = np.random.default_rng(4)
    n_i, n_t = 40, 30
    i = np.repeat(np.arange(n_i), n_t)
    t = np.tile(np.arange(n_t), n_i)
    M0 = _dados(i, t)

    d: dict = {}
    centralizar(M0, [i, t], diagnostico=d)
    assert d["convergiu"]
    assert d["passadas"] <= 25, (
        "se passar de 25 aqui, o valor fixo antigo era insuficiente ate no "
        "caso facil, e a conclusao de F4 muda")


def test_painel_mal_conectado_nao_converge_em_25_passadas():
    """O risco que F4 apontou, exibido num caso onde ele se realiza."""
    i, t = _cadeia()
    M0 = _dados(i, t)

    curto: dict = {}
    M_curto = centralizar(M0, [i, t], it=25, tol=1e-12, diagnostico=curto)
    assert not curto["convergiu"]
    assert curto["variacao_final"] > 1e-4

    longo: dict = {}
    M_longo = centralizar(M0, [i, t], it=4000, tol=1e-12, diagnostico=longo)
    assert longo["passadas"] > 25

    # E a diferença aparece no que importa, que é o coeficiente.
    assert abs(_coef(M_curto) - _coef(M_longo)) > 1e-3


def test_o_diagnostico_e_gravado_e_nao_apenas_calculado():
    """O ponto de F4 não é iterar mais, é deixar de supor. O número de passadas
    tem que chegar ao JSON de saída para poder ser auditado."""
    rng = np.random.default_rng(9)
    i = np.repeat(np.arange(10), 10)
    t = np.tile(np.arange(10), 10)
    d: dict = {}
    centralizar(_dados(i, t), [i, t], diagnostico=d)
    assert set(d) == {"passadas", "variacao_final", "tolerancia", "convergiu"}
    assert d["passadas"] >= 1
    assert d["variacao_final"] >= 0.0


def test_centralizar_nao_modifica_a_matriz_recebida():
    """Efeito colateral silencioso seria pior que o problema original: o
    chamador reusa `M` para montar outras especificações."""
    i = np.repeat(np.arange(6), 5)
    t = np.tile(np.arange(5), 6)
    M0 = _dados(i, t)
    copia = M0.copy()
    centralizar(M0, [i, t])
    assert np.array_equal(M0, copia)
