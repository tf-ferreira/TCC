"""Defasagens do painel: preço e volume da mesma série, k semanas atrás.

Ponto único onde defasagem existe. A seção 3 da especificação de atributos
explica por que ela é diferente de todos os outros atributos do projeto, e o
resumo é este: **D25 olha uma célula por vez, e aí o preço defasado passa**,
porque mudar o preço que está sendo decidido agora não altera o preço da semana
passada. O problema da defasagem não é o atributo, é a **avaliação entre
células**: quando a simulação contrafactual reprecifica semanas consecutivas, a
defasagem da semana seguinte tem de ser alimentada com o preço que o otimizador
recomendou, e não com o histórico. Alimentada com o histórico, a simulação
superestima a receita **sempre no sentido favorável**.

## Por que a junção é por semana e não por posição

O painel de células completas tem buracos: exigir os vinte SKUs simultaneamente
presentes faz a mediana das lojas perder 66 semanas dentro do próprio intervalo
de operação. Defasar por **posição na série** (um `shift` sobre linhas ordenadas)
comparia semanas que não são vizinhas no calendário, e o erro seria silencioso:
nenhuma linha ficaria vazia, todas ficariam erradas.

A junção aqui é por chave `(loja, sku, semana − k)`. Onde a semana anterior não
existe no painel, a defasagem fica **ausente** e é contada, nunca preenchida com
sentinela, que é a regra de D9.

## Vazamento

A partição temporal usa zona morta de 8 semanas (`particionar` em
`baseline_arvores.py`), que existe exatamente para isto. Defasagens de até 8
semanas não atravessam a fronteira. Acima de 8 a zona morta teria de crescer, e
a decisão de tamanho tem de levar isso em conta.
"""

from __future__ import annotations

import pandas as pd

CHAVE = ["store", "sku", "week"]


def adicionar_defasagens(longo: pd.DataFrame, ks_preco: tuple[int, ...] = (1,),
                         ks_volume: tuple[int, ...] = ()) -> pd.DataFrame:
    """Acrescenta `preco_lagK` e `volume_lagK` por junção de calendário.

    `ks_volume` fica vazio por padrão: o volume defasado é o próprio alvo de uma
    semana anterior, e a seção 3 o rejeita por dois motivos (não existe volume
    contrafactual na simulação sem propagar a previsão semana a semana, e ele
    tende a absorver variância e encolher a derivada em relação ao preço). O
    parâmetro existe para que essa rejeição seja **medida** e não afirmada.
    """
    saida = longo.copy()
    base = longo[CHAVE + ["preco_proprio", "alvo"]]
    for k in sorted(set(ks_preco) | set(ks_volume)):
        direita = base.rename(columns={"preco_proprio": f"preco_lag{k}",
                                       "alvo": f"volume_lag{k}"}).copy()
        direita["week"] = direita["week"] + k
        colunas = []
        if k in ks_preco:
            colunas.append(f"preco_lag{k}")
        if k in ks_volume:
            colunas.append(f"volume_lag{k}")
        saida = saida.merge(direita[CHAVE + colunas], on=CHAVE, how="left")
    return saida


def cobertura(longo: pd.DataFrame, colunas: list[str]) -> dict:
    """Fração de linhas em que cada defasagem está definida.

    Reportar isto é obrigatório: uma defasagem ausente em metade do painel não é
    o mesmo atributo que uma definida em 95%, e a comparação preditiva entre
    configurações fica contaminada se a cobertura não for declarada junto.
    """
    return {c: float(longo[c].notna().mean()) for c in colunas}
