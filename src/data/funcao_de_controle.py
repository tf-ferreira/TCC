"""Função de controle do custo (D42): o primeiro estágio e o resíduo `v̂ᵢ`.

## O que resolve

A rede aprende de toda a variação de preço do painel, que no Dominick's é
dominada por promoções definidas para a rede inteira e que vêm junto com
encarte, exposição e antecipação de compra. A derivada que o otimizador usa
fica associativa: parte do volume das semanas de promoção é creditada ao preço.

A função de controle (Petrin e Train, 2010) separa as duas coisas. No primeiro
estágio, por SKU e só na janela de ajuste,

    ln pᵢ = a_loja,i + πᵢ·ln cᵢ + b_i'·harmônicos + δᵢ·tempo + vᵢ

o resíduo `v̂ᵢ` é a parte do preço que o custo não explica, e é nela que mora o
choque de demanda que anda junto com a promoção. Com `v̂ᵢ` no contexto de cada
SKU, isto é, em `gᵢ(x)` e **nunca** em `mᵢ(uᵢ)`, a variação de `uᵢ` que resta para
identificar `mᵢ` é a movida pelo custo, que é a hipótese de identificação de D18.

## A emenda a D25, e por que ela não é exceção de conveniência

D25 proíbe atributo que seja **função do preço que o otimizador decide**. `v̂ᵢ` é
função do preço **observado**, e no contrafactual ele fica **congelado** no valor
histórico da célula, porque representa o choque de demanda daquela célula, que
não muda quando o preço muda. Recalcular `v̂ᵢ` com o preço decidido seria
exatamente o erro que D25 condena, e o contrafactual do item 9 nunca faz isso
(`contrafactual_item9.ctx_com_lags` só reescreve as defasagens; o teste
`test_o_contrafactual_nunca_reescreve_a_funcao_de_controle` fixa isso).

A diferença com o indicador de promoção montado a partir do preço, que D25
condena com razão: aquele é uma função não linear do preço que a rede vê, e
rouba o efeito do preço por construção. `v̂ᵢ` também é função do preço, mas é
construído para ser a parte que o instrumento **não** explica, de modo que
condicionar nele deixa a variação do instrumento identificando a derivada. É o
argumento de variáveis instrumentais escrito como controle.

## O que invalidaria esta decisão

- **Exclusão violada.** Se o custo tiver efeito direto sobre a demanda (acordo
  comercial com exigência de encarte), a função de controle herda o viés do IV
  (F2 do diagnóstico, sonda 12).
- **Instrumento fraco por SKU.** Três SKUs quase não repassam custo (π perto de
  zero). Neles `v̂ᵢ` é quase o preço inteiro e a identificação é fraca; o
  diagnóstico `f_parcial` por SKU vai para o texto.

## D28

O primeiro estágio é ajustado **só** no conjunto de ajuste do pipeline (treino
no de reporte, treino reduzido no de seleção) e aplicado com os coeficientes do
ajuste nas outras partes. Loja sem linha no ajuste recebe a média dos efeitos de
loja, que é o análogo do índice de reserva do embedding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUNAS = ["v_cf", "falta_v_cf"]
REGRESSORES = ["sen_1", "cos_1", "sen_2", "cos_2", "tempo"]


def custo_longo(cel: pd.DataFrame, n: int) -> pd.DataFrame:
    """O custo `kᵢ` do painel de células em formato longo (store, week, sku).

    A coluna `custo_ii` corresponde à posição `i` do vetor, a mesma de
    `preco_ii`; `test_custo_longo_alinha_com_a_posicao` fixa isso.
    """
    colunas = [f"custo_{i:02d}" for i in range(n)]
    c = cel[["store", "week"] + colunas].melt(id_vars=["store", "week"],
                                              var_name="coluna", value_name="custo")
    c["sku"] = c["coluna"].str[-2:].astype(int)
    return c.drop(columns="coluna")


def _com_custo(longo: pd.DataFrame, custos: pd.DataFrame) -> pd.DataFrame:
    fora = longo.merge(custos, on=["store", "week", "sku"], how="left",
                       validate="many_to_one")
    if len(fora) != len(longo):
        raise ValueError("a junção com o custo mudou o número de linhas")
    fora.index = longo.index
    return fora


def _desenho(d: pd.DataFrame, lojas: np.ndarray) -> np.ndarray:
    """Efeitos de loja (média dos efeitos para loja fora do ajuste), ln c e controles."""
    mapa = {l: j for j, l in enumerate(lojas)}
    j = d["store"].map(mapa)
    tem = j.notna().to_numpy()
    D = np.zeros((len(d), len(lojas)))
    D[np.flatnonzero(tem), j[tem].astype(int).to_numpy()] = 1.0
    D[~tem] = 1.0 / len(lojas)
    outros = np.column_stack([np.log(d["custo"].to_numpy(float))]
                             + [d[c].to_numpy(float) for c in REGRESSORES])
    return np.column_stack([D, outros])


def ajustar(ajuste: pd.DataFrame, custos: pd.DataFrame) -> dict:
    """Primeiro estágio por SKU, estimado só em `ajuste`.

    Devolve, por SKU, as lojas do ajuste, os coeficientes e dois diagnósticos:
    `pi` (o repasse do custo ao preço) e `f_parcial` (o t² de `pi` sob erro
    homoscedástico, o F de um instrumento só). O F é diagnóstico de força, não
    inferência: o erro padrão que vai para o texto é o agrupado de D24.
    """
    d = _com_custo(ajuste, custos)
    coefs = {}
    for sku, g in d.groupby("sku"):
        ok = g["custo"].notna() & (g["custo"] > 0)
        g = g[ok]
        lojas = np.sort(g["store"].unique())
        X = _desenho(g, lojas)
        y = np.log(g["preco_proprio"].to_numpy(float))
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ b
        gl = max(len(y) - np.linalg.matrix_rank(X), 1)
        s2 = float(e @ e) / gl
        XtX_inv = np.linalg.pinv(X.T @ X)
        k = len(lojas)
        ep = float(np.sqrt(s2 * XtX_inv[k, k]))
        coefs[int(sku)] = {"lojas": lojas, "b": b, "pi": float(b[k]),
                           "f_parcial": float((b[k] / ep) ** 2) if ep > 0 else float("inf"),
                           "n": int(len(y))}
    return coefs


def aplicar(longo: pd.DataFrame, custos: pd.DataFrame, coefs: dict) -> pd.DataFrame:
    """Escreve `v_cf` e `falta_v_cf` em `longo` com os coeficientes do ajuste.

    Custo ausente ou não positivo: `v_cf = 0` e `falta_v_cf = 1`. Zero é o valor
    neutro, o do choque médio, e o indicador deixa a rede distinguir "choque
    médio" de "sem informação", como `falta_preco_lagK` faz nas defasagens.
    """
    d = _com_custo(longo, custos).reset_index(drop=True)
    v = np.zeros(len(d))
    falta = np.ones(len(d))
    sku = d["sku"].to_numpy()
    ok = (d["custo"].notna() & (d["custo"] > 0)).to_numpy()
    for s in np.unique(sku):
        if int(s) not in coefs:
            raise ValueError(f"SKU {s} sem primeiro estágio ajustado")
        c = coefs[int(s)]
        pos = np.flatnonzero((sku == s) & ok)
        g = d.iloc[pos]
        v[pos] = np.log(g["preco_proprio"].to_numpy(float)) - _desenho(g, c["lojas"]) @ c["b"]
        falta[pos] = 0.0
    fora = longo.copy()
    fora["v_cf"] = v
    fora["falta_v_cf"] = falta
    return fora


def adicionar(ajuste: pd.DataFrame, *partes: pd.DataFrame, cel: pd.DataFrame,
              n: int) -> tuple:
    """Ajusta em `ajuste` e aplica em `ajuste` e em cada uma das `partes`.

    Devolve `(ajuste, *partes, diagnostico)`, com o diagnóstico por SKU (`pi`,
    `f_parcial`, `n`) pronto para o registro do item 6.
    """
    custos = custo_longo(cel, n)
    coefs = ajustar(ajuste, custos)
    saidas = [aplicar(p, custos, coefs) for p in (ajuste, *partes)]
    diag = {"pi": [coefs[i]["pi"] for i in sorted(coefs)],
            "f_parcial": [coefs[i]["f_parcial"] for i in sorted(coefs)],
            "n_ajuste": [coefs[i]["n"] for i in sorted(coefs)]}
    return (*saidas, diag)
