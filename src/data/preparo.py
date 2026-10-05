"""Preparo das entradas da rede (seção 6 da especificação de atributos).

Ponto único das quatro operações que transformam o painel em entrada de rede.
Elas são pequenas, e é justamente por isso que moram juntas: cada uma esconde
uma armadilha diferente, e as quatro armadilhas são silenciosas.

## As duas regras, e elas pegam coisas diferentes

| regra | o que impede |
|---|---|
| **D25** | atributo que é **função do preço decidido**, que atenua ∂V̂/∂p |
| **ajuste só no treino** | usar dado do período de **teste** para construir uma entrada |

Um preenchimento pode passar numa e falhar na outra. A mediana da série calculada
sobre o painel inteiro não é função do preço decidido, e mesmo assim é
inaceitável, porque usa o teste. O preço da própria semana não usa o teste, e
mesmo assim é inaceitável, porque corta a derivada pela metade nas linhas
preenchidas.

## Por que o preenchimento é decisão e não detalhe

Medido no brinquedo de `volume = 300 − 40·preço + 20·preco_lag1`: com a defasagem
presente e fixa em 4,00, subir o preço de 3,00 para 4,00 derruba o volume em 40.
Com a defasagem **preenchida com o preço da própria semana**, a mesma subida
derruba 20. A derivada cai pela metade, e cai só nas 12,8% de linhas preenchidas,
o que é pior que cair em todas: o viés fica escondido numa subamostra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SERIE = ["store", "sku"]
INDICE_RESERVA = 0


def medianas_de_treino(treino: pd.DataFrame, coluna: str = "preco_proprio",
                       serie: list[str] | None = None) -> pd.Series:
    """Mediana de `coluna` por série `(loja, SKU)`, **só sobre o treino**.

    É o valor de preenchimento adotado na seção 6. Ele é fixo por série e
    calculado em dado que o teste não vê, de modo que satisfaz as duas regras:
    não muda quando o preço decidido muda (D25) e não usa o período de teste.
    """
    return treino.groupby(serie or SERIE, observed=True)[coluna].median()


def preencher_defasagens(longo: pd.DataFrame, medianas: pd.Series,
                         colunas: list[str], serie: list[str] | None = None
                         ) -> pd.DataFrame:
    """Preenche defasagens ausentes e **declara a ausência** num indicador.

    O indicador não é redundante com o valor preenchido: sem ele a rede não
    consegue distinguir uma série cuja defasagem de fato igualou a mediana de
    uma em que a defasagem não existia. É a mesma regra de D9, declarar em vez
    de silenciar, aplicada à entrada da rede em vez de à leitura do dado.

    Séries que não aparecem no treino não têm mediana. Elas ficam com o valor
    global do treino, e o indicador continua marcando a ausência, de modo que a
    rede pode aprender a desconfiar dessas linhas.
    """
    saida = longo.copy()
    chave = serie or SERIE
    alvo = pd.MultiIndex.from_frame(saida[chave]) if len(chave) > 1 else saida[chave[0]]
    referencia = pd.Series(medianas.reindex(alvo).to_numpy(), index=saida.index)
    global_ = float(medianas.median())
    referencia = referencia.fillna(global_)
    for c in colunas:
        saida[f"falta_{c}"] = saida[c].isna().astype("float64")
        saida[c] = saida[c].fillna(referencia)
    return saida


def tempo_truncado(week, semana_min_treino: int, semana_max_treino: int) -> np.ndarray:
    """`week` normalizado na janela de treino e **truncado** nos extremos dela.

    Dentro da janela é tendência; fora dela congela. O motivo é numérico e
    específico deste painel: o treino termina na semana 310 e o teste vai até a
    399, isto é, **89 semanas de extrapolação**. Uma árvore não extrapola, ela
    repete a última folha; uma rede aprende tendência linear e a projeta sem
    freio. Uma deriva de 0,3% por semana viraria 23% de deslocamento que nada no
    dado sustenta.

    O truncamento zera a derivada em relação ao tempo fora da janela, e isso não
    é problema: o tempo não é variável de decisão do otimizador.
    """
    w = np.asarray(week, dtype="float64")
    largura = float(semana_max_treino - semana_min_treino) or 1.0
    return np.clip((w - semana_min_treino) / largura, 0.0, 1.0)


def ajustar_escalonador(treino: pd.DataFrame, colunas: list[str]) -> dict:
    """Média e desvio **do treino**, para aplicar no treino e no teste.

    Ajustar no painel inteiro é vazamento: a média usada para normalizar uma
    linha de teste teria sido calculada com aquela linha dentro.
    """
    return {c: {"media": float(treino[c].mean()),
                "desvio": float(treino[c].std(ddof=0)) or 1.0} for c in colunas}


def aplicar_escalonador(quadro: pd.DataFrame, parametros: dict) -> pd.DataFrame:
    saida = quadro.copy()
    for c, p in parametros.items():
        saida[c] = (saida[c] - p["media"]) / p["desvio"]
    return saida


def vocabulario(longo: pd.DataFrame, coluna: str) -> dict:
    """Códigos do embedding, construídos sobre o conjunto de AJUSTE.

    **O índice 0 é reservado** para a categoria de reserva. Sete das 93 lojas
    aparecem só no teste, e uma tabela de embedding indexada por código as faria
    ler uma linha que nunca recebeu gradiente. A linha 0 é treinada de propósito,
    sorteando uma fração dos acessos para ela durante o treino, de modo que a
    loja nunca vista cai num vetor aprendido em vez de num vetor aleatório.

    **Quem chama passa o TREINO, e o mesmo vocabulário codifica as duas partes.**
    A versão anterior a 17/09/2026 dizia "do painel e não da partição", e as duas
    frases eram incompatíveis: com o painel inteiro toda loja tem código, e a
    reserva nunca é usada. Medido na pendência 9.6: **0 linhas** de teste no
    índice 0, e as 7 lojas só do teste com WMAPE 64,4 contra 47,9.

    O erro que motivou a frase antiga (F2) era outro, e continua evitado: F2
    construía um vocabulário POR partição e codificava o teste com o dele, de modo
    que 73 de 86 códigos apontavam para outra loja. Aqui há um vocabulário só,
    ajustado no treino e aplicado às duas.
    """
    valores = np.sort(longo[coluna].unique())
    return {v: i + 1 for i, v in enumerate(valores)}


def codificar(valores, vocab: dict) -> np.ndarray:
    """Valor para índice, com o não visto caindo no índice de reserva."""
    return np.array([vocab.get(v, INDICE_RESERVA) for v in valores], dtype="int64")
