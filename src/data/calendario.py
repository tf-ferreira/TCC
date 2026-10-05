"""Calendário do Dominick's: da semana corrida para data, ângulo e feriado.

Ponto único onde a semana do DFF vira data de calendário. Existe porque o
`week % 52` que o modelo de referência usava tem dois defeitos independentes,
medidos e registrados na seção 2 de `docs/especificacao_atributos.md`:

**Salto.** `week % 52` põe as semanas 51 e 0 a 51 unidades de distância, sendo
vizinhas no calendário. Uma rede com entrada contínua precisa gastar capacidade
aprendendo que as duas pontas se encontram. Resolvido trocando um número por
dois, seno e cosseno do ângulo do ano, que ficam a distância igual entre
quaisquer semanas vizinhas.

**Deriva.** O ano tem 52,18 semanas e o contador reseta a cada 52, de modo que o
mesmo valor cai em datas diferentes ao longo do painel: 1,37 semana de diferença
entre a primeira e a última ocorrência, nos 7,6 anos. Resolvido só pela data
real, não pelo seno e cosseno.

Âncora: **a semana 1 do DFF é 14/09/1989**, e a semana é o intervalo de sete
dias que começa nessa data. As semanas do painel vão de 1 a 399, o que devolve
14/09/1989 a 01/05/1997, compatível com o período documentado pela Kilts Center.

Notação do projeto: estas funções produzem parte do contexto `x_{s,t}` que
acompanha o vetor de preços `p_{s,t}` na entrada da rede. Nenhuma delas depende
de preço, o que é o que D25 exige de todo atributo.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

SEMANA_1 = dt.date(1989, 9, 14)
DIAS_NA_SEMANA = 7


def data_da_semana(week) -> pd.Series:
    """Data do primeiro dia da semana corrida do DFF.

    `week` é 1-indexado: a semana 1 devolve 14/09/1989.
    """
    w = np.asarray(week, dtype="int64")
    origem = np.datetime64(SEMANA_1)
    return pd.Series(origem + (w - 1) * np.timedelta64(DIAS_NA_SEMANA, "D"))


def _dias_no_ano(anos: np.ndarray) -> np.ndarray:
    bissexto = ((anos % 4 == 0) & (anos % 100 != 0)) | (anos % 400 == 0)
    return np.where(bissexto, 366.0, 365.0)


def fracao_do_ano(datas) -> np.ndarray:
    """Posição no ano, em [0, 1), medida no **meio** da semana.

    O meio e não a ponta: a semana cobre sete dias, e usar o primeiro deles
    jogaria sistematicamente a representação para trás em três dias e meio. O
    denominador é 366 em ano bissexto, de modo que não sobra deriva nenhuma,
    que é justamente o defeito que esta função existe para eliminar.
    """
    d = pd.to_datetime(pd.Series(datas).reset_index(drop=True))
    meio = d + pd.Timedelta(days=(DIAS_NA_SEMANA - 1) / 2)
    ano = meio.dt.year.to_numpy()
    dia = meio.dt.dayofyear.to_numpy().astype(float)
    return (dia - 1.0) / _dias_no_ano(ano)


def harmonicos(datas, k: int = 1) -> pd.DataFrame:
    """Seno e cosseno do ângulo do ano, até a k-ésima harmônica.

    `k = 1` representa exatamente um pico e um vale por ano. Cada harmônica
    adicional dobra o número de extremos que a forma pode ter, e é por isso que
    `k` é escolhido por medição e não por gosto: mais harmônicas ajustam melhor
    e sobreajustam mais cedo.
    """
    if k < 1:
        raise ValueError("k deve ser >= 1")
    ang = 2.0 * np.pi * fracao_do_ano(datas)
    saida = {}
    for j in range(1, k + 1):
        saida[f"sen_{j}"] = np.sin(j * ang)
        saida[f"cos_{j}"] = np.cos(j * ang)
    return pd.DataFrame(saida)


def pascoa(ano: int) -> dt.date:
    """Domingo de Páscoa pelo algoritmo gregoriano anônimo (Meeus/Jones/Butcher)."""
    a = ano % 19
    b, c = divmod(ano, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    mes, dia = divmod(h + ll - 7 * m + 114, 31)
    return dt.date(ano, mes, dia + 1)


def _n_esima_quinta(ano: int, mes: int, n: int) -> dt.date:
    """N-ésima quinta-feira do mês. Ação de Graças é a quarta de novembro."""
    primeiro = dt.date(ano, mes, 1)
    deslocamento = (3 - primeiro.weekday()) % 7  # 3 = quinta
    return primeiro + dt.timedelta(days=deslocamento + 7 * (n - 1))


FERIADOS = ("natal_ano_novo", "acao_de_gracas", "pascoa", "quatro_de_julho")


def _datas_de_feriado(anos: range) -> dict[str, set[dt.date]]:
    tabela: dict[str, set[dt.date]] = {nome: set() for nome in FERIADOS}
    for ano in anos:
        tabela["natal_ano_novo"].update({dt.date(ano, 12, 25), dt.date(ano, 1, 1)})
        tabela["acao_de_gracas"].add(_n_esima_quinta(ano, 11, 4))
        tabela["pascoa"].add(pascoa(ano))
        tabela["quatro_de_julho"].add(dt.date(ano, 7, 4))
    return tabela


def feriados(datas) -> pd.DataFrame:
    """Indicador por feriado, verdadeiro na semana que **contém** a data.

    A semana é o intervalo fechado [data, data + 6 dias]. Natal e Ano Novo saem
    num indicador só porque caem a sete dias um do outro e, em dado semanal, a
    compra que antecede os dois é a mesma.

    Estes indicadores não são função do preço, então D25 os permite. O que eles
    disputam com as harmônicas é o mesmo sinal: se a harmônica já captura o bico
    de dezembro, o indicador é redundante, e é por isso que os dois entram na
    mesma varredura.
    """
    d = pd.to_datetime(pd.Series(datas).reset_index(drop=True))
    inicio = d.dt.date
    anos = range(int(d.dt.year.min()) - 1, int(d.dt.year.max()) + 2)
    tabela = _datas_de_feriado(anos)
    saida = {}
    for nome in FERIADOS:
        alvos = tabela[nome]
        saida[f"fer_{nome}"] = np.array(
            [any(0 <= (alvo - i).days < DIAS_NA_SEMANA for alvo in alvos)
             for i in inicio], dtype=float)
    return pd.DataFrame(saida)
