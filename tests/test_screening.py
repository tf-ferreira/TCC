"""Testes da triagem de categorias.

Cada teste usa um painel sintético minúsculo cuja resposta correta é
calculável à mão. Isso importa mais aqui do que em código descartável: um
erro na divisão por `qty` ou no filtro de `ok` não levanta exceção, produz um
número plausível e errado, e só apareceria muito depois, contaminado pela
rede neural inteira.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.screening import AcumuladorCategoria, preco_unitario


def _painel(linhas: list[dict]) -> pd.DataFrame:
    """Monta um bloco no formato do arquivo de movimento do DFF."""
    padrao = {"STORE": 1, "UPC": 100, "WEEK": 1, "MOVE": 10.0,
              "PRICE": 2.0, "QTY": 1.0, "SALE": None, "OK": 1}
    return pd.DataFrame([{**padrao, **linha} for linha in linhas])


# --------------------------------------------------------------------- D1

def test_preco_unitario_divide_pelo_tamanho_do_pacote():
    """Decisão D1: 3 latas por $6 é $2 por unidade, não $6."""
    resultado = preco_unitario(np.array([6.0]), np.array([3.0]))
    assert resultado[0] == pytest.approx(2.0)


def test_preco_unitario_trata_qty_invalido_como_um():
    resultado = preco_unitario(np.array([2.0, 2.0]), np.array([0.0, np.nan]))
    assert resultado == pytest.approx([2.0, 2.0])


def test_promocao_de_pacote_nao_vira_aumento_de_preco():
    """O erro que D1 existe para evitar.

    Série com preço regular $2 que entra em promoção "3 por $4,50"
    (unitário $1,50). Se o código usasse PRICE cru, a média subiria; usando
    preço unitário, ela cai.
    """
    acumulador = AcumuladorCategoria()
    acumulador.atualizar(_painel([
        {"WEEK": 1, "PRICE": 2.0, "QTY": 1.0},
        {"WEEK": 2, "PRICE": 2.0, "QTY": 1.0},
        {"WEEK": 3, "PRICE": 4.5, "QTY": 3.0},
    ]))
    series = acumulador.estatisticas_por_serie()
    assert series.loc[0, "media"] < 2.0


# --------------------------------------------------------------------- D5

def test_linhas_com_ok_zero_sao_excluidas():
    acumulador = AcumuladorCategoria()
    acumulador.atualizar(_painel([
        {"WEEK": 1, "PRICE": 2.0, "OK": 1},
        {"WEEK": 2, "PRICE": 99.0, "OK": 0},
    ]))
    series = acumulador.estatisticas_por_serie()
    assert series.loc[0, "n_semanas"] == 1
    assert series.loc[0, "media"] == pytest.approx(2.0)


# --------------------------------------------------------------------- D2

def test_linha_sem_atividade_nao_conta_como_presenca():
    """Decisão D2: o painel vem preenchido, price == 0 significa ausência."""
    acumulador = AcumuladorCategoria()
    acumulador.atualizar(_painel([
        {"WEEK": 1, "PRICE": 2.0, "MOVE": 5.0},
        {"WEEK": 2, "PRICE": 0.0, "MOVE": 0.0},
        {"WEEK": 3, "PRICE": 0.0, "MOVE": 0.0},
        {"WEEK": 4, "PRICE": 2.0, "MOVE": 5.0},
    ]))
    cobertura = acumulador.cobertura_por_upc()
    # 1 loja × 4 semanas = 4 células possíveis, 2 com atividade
    assert cobertura.iloc[0]["cobertura_total"] == pytest.approx(0.5)


# --------------------------------------------------------------------- D8

def test_decomposicao_da_cobertura_e_exata():
    """Os quatro fatores têm que reproduzir a cobertura bruta.

    Se a identidade quebrar, algum denominador está inconsistente e os
    fatores deixam de ser interpretáveis como partição do mesmo fenômeno.
    """
    acumulador = AcumuladorCategoria()
    acumulador.atualizar(_painel([
        {"UPC": 100, "STORE": s, "WEEK": w, "PRICE": 2.0}
        for s in (1, 2) for w in (1, 2, 3)
    ] + [
        {"UPC": 200, "STORE": 3, "WEEK": 4, "PRICE": 5.0},
        {"UPC": 200, "STORE": 3, "WEEK": 5, "PRICE": 5.0},
    ]))
    cobertura = acumulador.cobertura_por_upc()
    produto = (cobertura["amplitude"] * cobertura["longevidade"]
               * cobertura["disponibilidade"] * cobertura["regularidade"])
    assert produto.to_numpy() == pytest.approx(
        cobertura["cobertura_total"].to_numpy()
    )


# -------------------------------------------------------------------- D10

def test_loja_que_abriu_tarde_nao_derruba_a_regularidade_do_sku():
    """O erro que D10 corrige.

    Painel de 10 semanas. A loja 1 opera o tempo todo; a loja 2 só existe da
    semana 8 em diante. Um SKU presente em todas as semanas de ambas as lojas
    é impecável: esteve lá sempre que foi possível estar. Sem o fator de
    disponibilidade, ele seria punido pelas 7 semanas em que a loja 2 sequer
    havia aberto, e a regularidade cairia para cerca de 0,65.
    """
    linhas = [{"UPC": 100, "STORE": 1, "WEEK": w, "PRICE": 2.0} for w in range(1, 11)]
    linhas += [{"UPC": 100, "STORE": 2, "WEEK": w, "PRICE": 2.0} for w in range(8, 11)]

    acumulador = AcumuladorCategoria()
    acumulador.atualizar(_painel(linhas))
    linha = acumulador.cobertura_por_upc().iloc[0]

    assert linha["regularidade"] == pytest.approx(1.0)
    # 13 células ativas sobre um retângulo de 2 lojas x 10 semanas
    assert linha["disponibilidade"] == pytest.approx(13 / 20)


def test_disponibilidade_e_um_quando_todas_as_lojas_operam_o_periodo_todo():
    """Sem entrada nem saída de loja, o fator novo é neutro e a regularidade
    volta a ser exatamente o que D8 media."""
    linhas = [{"UPC": 100, "STORE": s, "WEEK": w, "PRICE": 2.0}
              for s in (1, 2) for w in range(1, 5)]
    linhas += [{"UPC": 200, "STORE": s, "WEEK": w, "PRICE": 3.0}
               for s in (1, 2) for w in (1, 3)]

    acumulador = AcumuladorCategoria()
    acumulador.atualizar(_painel(linhas))
    cob = acumulador.cobertura_por_upc().set_index("upc")

    assert cob.loc[100, "disponibilidade"] == pytest.approx(1.0)
    assert cob.loc[200, "disponibilidade"] == pytest.approx(1.0)
    # O SKU 200 vive das semanas 1 a 3, não 1 a 4: a semana 4 fica fora da
    # vida dele e por isso não entra no denominador (é justamente o que D8
    # corrige). Logo 4 células ativas sobre 2 lojas x 3 semanas.
    assert cob.loc[200, "regularidade"] == pytest.approx(4 / 6)


def test_longevidade_nao_penaliza_produto_que_ainda_nao_existia():
    """O erro que D8 corrige.

    Dois SKUs com o mesmo número de células ativas: um existiu o painel
    todo, mas de forma esparsa; o outro entrou tarde e foi impecável desde
    então. A cobertura bruta os iguala; a regularidade os separa.
    """
    esparso = [{"UPC": 100, "STORE": 1, "WEEK": w, "PRICE": 2.0}
               for w in (1, 3, 5, 7)]
    tardio = [{"UPC": 200, "STORE": 1, "WEEK": w, "PRICE": 2.0}
              for w in (5, 6, 7, 8)]
    preenchimento = [{"UPC": 300, "STORE": 1, "WEEK": w, "PRICE": 1.0}
                     for w in range(1, 9)]

    acumulador = AcumuladorCategoria()
    acumulador.atualizar(_painel(esparso + tardio + preenchimento))
    cobertura = acumulador.cobertura_por_upc().set_index("upc")

    assert cobertura.loc[100, "cobertura_total"] == pytest.approx(
        cobertura.loc[200, "cobertura_total"]
    )
    assert cobertura.loc[200, "regularidade"] == pytest.approx(1.0)
    assert cobertura.loc[100, "regularidade"] < 0.7


# --------------------------------------------- robustez a dado malformado

def test_linhas_com_chave_ausente_sao_contadas_e_descartadas():
    """Algumas categorias trazem NA em STORE/UPC/WEEK/OK.

    O comportamento correto é contar e seguir, não abortar a leitura nem
    ignorar em silêncio.
    """
    acumulador = AcumuladorCategoria()
    bloco = _painel([
        {"WEEK": 1, "PRICE": 2.0},
        {"WEEK": 2, "PRICE": 2.0},
    ])
    bloco.loc[1, "STORE"] = np.nan
    acumulador.atualizar(bloco)

    assert acumulador.n_chave_ausente == 1
    assert acumulador.estatisticas_por_serie().loc[0, "n_semanas"] == 1


# --------------------------------------------------------------------- D3

def test_variancia_e_medida_dentro_da_serie_e_nao_agregada():
    """O teste que pega o erro mais caro da triagem.

    Duas lojas, mesmo SKU, cada uma com preço constante no tempo, mas em
    níveis diferentes ($2 e $4). A variação é inteiramente entre lojas, zero
    dentro de cada série. O CV correto de cada série é 0. Um cálculo
    agregado daria CV alto e classificaria a categoria como boa candidata
    quando ela não oferece nenhuma variação útil para identificar
    elasticidade.
    """
    acumulador = AcumuladorCategoria()
    acumulador.atualizar(_painel([
        {"STORE": 1, "WEEK": 1, "PRICE": 2.0},
        {"STORE": 1, "WEEK": 2, "PRICE": 2.0},
        {"STORE": 2, "WEEK": 1, "PRICE": 4.0},
        {"STORE": 2, "WEEK": 2, "PRICE": 4.0},
    ]))
    series = acumulador.estatisticas_por_serie()
    assert len(series) == 2
    assert series["cv"].max() == pytest.approx(0.0, abs=1e-9)


def test_cv_captura_variacao_temporal_real():
    acumulador = AcumuladorCategoria()
    acumulador.atualizar(_painel([
        {"WEEK": 1, "PRICE": 1.0},
        {"WEEK": 2, "PRICE": 3.0},
    ]))
    series = acumulador.estatisticas_por_serie()
    # média 2, desvio populacional 1, CV = 0,5
    assert series.loc[0, "cv"] == pytest.approx(0.5)


# --------------------------------------------------------------------- D4

def test_preco_regular_e_a_moda_nao_a_media():
    """Decisão D4: média e mediana são puxadas pelas próprias promoções."""
    acumulador = AcumuladorCategoria()
    acumulador.atualizar(_painel([
        {"WEEK": w, "PRICE": 10.0} for w in range(1, 9)
    ] + [
        {"WEEK": 9, "PRICE": 6.0},
        {"WEEK": 10, "PRICE": 6.0},
    ]))
    series = acumulador.estatisticas_por_serie()
    assert series.loc[0, "preco_regular"] == pytest.approx(10.0)
    # 2 de 10 semanas abaixo de 90% de $10
    assert series.loc[0, "frac_promo_preco"] == pytest.approx(0.2)


def test_queda_pequena_nao_conta_como_promocao():
    """$9,50 é 95% de $10, acima do limiar de 90%."""
    acumulador = AcumuladorCategoria()
    acumulador.atualizar(_painel([
        {"WEEK": w, "PRICE": 10.0} for w in range(1, 5)
    ] + [{"WEEK": 5, "PRICE": 9.5}]))
    series = acumulador.estatisticas_por_serie()
    assert series.loc[0, "frac_promo_preco"] == pytest.approx(0.0)


def test_sale_declarado_e_contado_por_tipo():
    acumulador = AcumuladorCategoria()
    acumulador.atualizar(_painel([
        {"WEEK": 1, "SALE": "B"},
        {"WEEK": 2, "SALE": "C"},
        {"WEEK": 3, "SALE": None},
        {"WEEK": 4, "SALE": "B"},
    ]))
    assert acumulador.contagem_sale["B"] == 2
    assert acumulador.contagem_sale["C"] == 1
    assert acumulador.n_sale_preenchido == 3


def test_as_duas_medidas_de_promocao_compartilham_denominador():
    """Decisão D4: só faz sentido comparar as duas vias se a base for a mesma.

    Painel com 10 semanas válidas. Quatro têm código de promoção, mas só
    duas dessas tiveram queda real de preço. As duas frações precisam sair
    sobre as mesmas 10 semanas, senão a divergência entre elas mede
    diferença de denominador em vez de inconsistência do campo `sale`.
    """
    linhas = [{"WEEK": w, "PRICE": 10.0} for w in range(1, 7)]
    linhas += [{"WEEK": 7, "PRICE": 6.0, "SALE": "B"},
               {"WEEK": 8, "PRICE": 6.0, "SALE": "B"},
               {"WEEK": 9, "PRICE": 10.0, "SALE": "C"},
               {"WEEK": 10, "PRICE": 10.0, "SALE": "C"}]

    acumulador = AcumuladorCategoria()
    acumulador.atualizar(_painel(linhas))
    serie = acumulador.estatisticas_por_serie().iloc[0]

    assert serie["n_semanas"] == 10
    assert serie["frac_sale_declarado"] == pytest.approx(0.4)
    assert serie["frac_promo_preco"] == pytest.approx(0.2)
    # Cupom (C) não muda preço de gôndola, então a divergência é negativa.
    assert serie["divergencia_promo"] == pytest.approx(-0.2)


def test_linhas_invalidas_nao_entram_na_promocao_declarada():
    """Semana com ok=0 ou sem atividade não conta em nenhuma das duas vias."""
    acumulador = AcumuladorCategoria()
    acumulador.atualizar(_painel([
        {"WEEK": 1, "PRICE": 10.0},
        {"WEEK": 2, "PRICE": 10.0},
        {"WEEK": 3, "PRICE": 6.0, "SALE": "B", "OK": 0},
        {"WEEK": 4, "PRICE": 0.0, "SALE": "B"},
    ]))
    serie = acumulador.estatisticas_por_serie().iloc[0]
    assert serie["n_semanas"] == 2
    assert serie["frac_sale_declarado"] == pytest.approx(0.0)


# ------------------------------------------------- acumulação entre blocos

def test_resultado_independe_da_divisao_em_blocos():
    """Ler em 1 bloco ou em 3 tem que dar exatamente o mesmo resultado.

    É a garantia de que o acumulador é associativo, sem a qual o tamanho do
    chunk viraria um parâmetro que altera o resultado silenciosamente.
    """
    linhas = [{"WEEK": w, "PRICE": 1.0 + (w % 4)} for w in range(1, 13)]

    inteiro = AcumuladorCategoria()
    inteiro.atualizar(_painel(linhas))

    partido = AcumuladorCategoria()
    for inicio in range(0, 12, 4):
        partido.atualizar(_painel(linhas[inicio:inicio + 4]))

    a = inteiro.estatisticas_por_serie().iloc[0]
    b = partido.estatisticas_por_serie().iloc[0]
    assert a["n_semanas"] == b["n_semanas"]
    assert a["media"] == pytest.approx(b["media"])
    assert a["cv"] == pytest.approx(b["cv"])
