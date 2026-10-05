"""Testes do calendário (seção 2 da especificação de atributos).

Os dois primeiros testes são os dois defeitos do `week % 52`, escritos como
contraexemplo: eles falham com a representação antiga e passam com a nova. É
deliberado que a versão antiga apareça no teste, e não só a nova, porque o que
se quer fixar é a **diferença** entre as duas.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src" / "data"))

from calendario import (  # noqa: E402
    SEMANA_1, data_da_semana, feriados, fracao_do_ano, harmonicos,
    pascoa, _n_esima_quinta)


def test_ancora_e_extensao_do_painel():
    """Semana 1 é 14/09/1989 e a 399 cai em 01/05/1997, dentro do período
    documentado pela Kilts Center (setembro de 1989 a maio de 1997)."""
    assert data_da_semana([1])[0].date() == SEMANA_1
    assert data_da_semana([399])[0].date() == dt.date(1997, 5, 1)
    assert (data_da_semana([2])[0] - data_da_semana([1])[0]).days == 7


def test_seno_e_cosseno_nao_tem_salto_na_virada_do_ano():
    """Defeito 1. Em `week % 52`, as semanas 51 e 0 ficam a 51 de distância
    sendo vizinhas. No círculo, todo par de vizinhas fica à mesma distância."""
    semanas = np.arange(1, 400)
    h = harmonicos(data_da_semana(semanas), 1).to_numpy()
    passos = np.linalg.norm(np.diff(h, axis=0), axis=1)
    # Nenhum passo destoa: o maior é no máximo 1,05 vez o menor.
    assert passos.max() / passos.min() < 1.05

    modulo = (semanas % 52).astype(float)
    saltos = np.abs(np.diff(modulo))
    # A representação antiga falha exatamente este teste, uma vez por ano.
    assert saltos.max() == 51.0
    assert (saltos > 1).sum() == 7


def test_a_data_real_elimina_a_deriva_que_o_modulo_acumula():
    """Defeito 2. O mesmo dia do calendário devolve a mesma posição no ano em
    qualquer ano; o mesmo valor de `week % 52` não."""
    mesmas_datas = pd.to_datetime([f"{ano}-07-04" for ano in range(1990, 1998)])
    fracs = fracao_do_ano(mesmas_datas)
    assert fracs.max() - fracs.min() < 2.0 / 365.0

    # E a deriva do módulo, medida: o mesmo contador, sete ciclos depois, cai
    # 1,2 a 1,4 semana adiante no calendário.
    a = fracao_do_ano(data_da_semana([3]))[0]
    b = fracao_do_ano(data_da_semana([3 + 7 * 52]))[0]
    deriva_semanas = abs(b - a) * 365.25 / 7.0
    assert 1.1 < deriva_semanas < 1.5


def test_harmonicas_sao_ortogonais_e_crescem_de_duas_em_duas():
    h1 = harmonicos(data_da_semana(np.arange(1, 400)), 1)
    h3 = harmonicos(data_da_semana(np.arange(1, 400)), 3)
    assert list(h1.columns) == ["sen_1", "cos_1"]
    assert len(h3.columns) == 6
    # Cada par vive no círculo unitário, em qualquer harmônica.
    for j in (1, 2, 3):
        raio = h3[f"sen_{j}"] ** 2 + h3[f"cos_{j}"] ** 2
        assert np.allclose(raio, 1.0)


def test_pascoa_em_datas_conhecidas():
    """Valores publicados, para que o computus não seja reimplementado de
    memória sem conferência."""
    assert pascoa(1990) == dt.date(1990, 4, 15)
    assert pascoa(1993) == dt.date(1993, 4, 11)
    assert pascoa(1997) == dt.date(1997, 3, 30)
    assert pascoa(2000) == dt.date(2000, 4, 23)


def test_acao_de_gracas_e_a_quarta_quinta_de_novembro():
    for ano, dia in ((1990, 22), (1995, 23), (1997, 27)):
        d = _n_esima_quinta(ano, 11, 4)
        assert d == dt.date(ano, 11, dia)
        assert d.weekday() == 3


def test_indicador_marca_a_semana_que_contem_o_feriado():
    """A semana é [data, data + 6 dias]. O 4 de julho de 1990 é uma quarta; a
    semana do painel que o contém é a que começa em 28/06/1990."""
    semanas = np.arange(1, 400)
    f = feriados(data_da_semana(semanas))
    datas = data_da_semana(semanas)
    marcadas = datas[f["fer_quatro_de_julho"] == 1.0]
    for inicio in marcadas:
        assert any((dt.date(ano, 7, 4) - inicio.date()).days in range(0, 7)
                   for ano in range(1989, 1998))
    # Um indicador por ano do painel, e nenhum ano sem ele.
    assert f["fer_quatro_de_julho"].sum() == 7
