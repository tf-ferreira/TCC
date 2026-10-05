"""Seleção do sortimento alvo. Ponto único onde a regra de D12 existe.

Este módulo existe porque a regra de seleção estava implementada em dois
lugares com critérios **diferentes**: `painel.py` já usava completude conjunta
(D12 vigente) e `nucleo.py` ainda usava regularidade individual (D12
superada), sem que nada no projeto detectasse a divergência. Os números de
`frj_endogeneidade.json` foram produzidos pela versão superada.

Regra vigente, em dois movimentos:

1. **Piso de suporte.** O SKU precisa ter existido em ao menos metade do
   período da categoria e ter sido vendido em ao menos metade das lojas. O
   piso não é estético: regularidade é uma razão entre células ativas e
   células disponíveis, então um SKU presente numa única semana de uma única
   loja tem regularidade exatamente 1, com denominador 1. Ordenar sem piso
   seleciona a estimativa mais ruidosa que teve sorte, que é a maldição do
   vencedor.

2. **Completude conjunta.** Entre os aprovados, escolhem-se os N que
   maximizam o número de células (loja, semana) em que **todos** estão
   simultaneamente presentes, por busca gulosa progressiva. O critério não é
   assiduidade individual porque a elasticidade cruzada exige o vetor de
   preços concorrentes inteiro na mesma célula: uma célula em que um único
   SKU falta não serve para nenhuma das N funções de demanda.

Uma consequência conceitual que a versão superada escondia. Sob a regra
antiga, o limiar de qualidade era a N-ésima maior regularidade, isto é, uma
estatística de ordem calculada em vez de escolhida, e era esse o argumento da
"inversão" (fixar N e deixar o limiar ser consequência). **Sob a regra vigente
esse argumento não descreve mais a seleção.** O único limiar que resta é o
piso de suporte, fixo em 0,50, justificado pela variância de uma razão e não
pela forma da curva. A regularidade mínima do sortimento escolhido continua
sendo reportada, agora como estatística **descritiva** do conjunto, não como
critério que o produziu.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from screening import DTYPES, COLUNAS_CHAVE, preco_unitario

PISO_LONGEVIDADE = 0.50
PISO_AMPLITUDE = 0.50

COLUNAS = ["STORE", "UPC", "WEEK", "MOVE", "PRICE", "QTY", "SALE", "OK", "PROFIT"]
DTYPES_LOCAL = {**DTYPES, "PROFIT": "float64"}

GRANDEZAS = ["preco", "volume", "custo", "promo"]


def elegiveis(cobertura: pd.DataFrame) -> list[int]:
    """SKUs que passam no piso de suporte de D12."""
    apto = cobertura.dropna(subset=["regularidade"])
    apto = apto[(apto["longevidade"] >= PISO_LONGEVIDADE)
                & (apto["amplitude"] >= PISO_AMPLITUDE)]
    return sorted(apto["upc"].tolist())


def extrair_longo(caminho_zip: Path, upcs: set[int],
                  bloco: int = 2_000_000) -> pd.DataFrame:
    """Percorre o arquivo de movimento e retém apenas os SKUs informados.

    Filtra cedo, de modo que a memória fique ligada ao tamanho do conjunto
    retido e não ao da categoria. Aplica D5 (dado suspeito sai), D2 (presença
    exige preço positivo) e D1 (preço unitário é price/qty).
    """
    partes = []
    leitor = pd.read_csv(caminho_zip, compression="zip", usecols=COLUNAS,
                         dtype=DTYPES_LOCAL, chunksize=bloco)
    for parte in leitor:
        parte = parte.loc[~parte[COLUNAS_CHAVE].isna().any(axis=1)]
        if parte.empty:
            continue
        parte = parte.astype({"STORE": "int32", "UPC": "int64", "WEEK": "int32"})
        sel = parte[(parte["OK"] == 1) & (parte["PRICE"] > 0)
                    & parte["UPC"].isin(upcs)]
        if sel.empty:
            continue
        preco = preco_unitario(sel["PRICE"].to_numpy(), sel["QTY"].to_numpy())
        margem = sel["PROFIT"].to_numpy()
        # Margem fora de (0, 100) é registro corrompido: o custo derivado dela
        # seria negativo ou superior ao preço. Mantém-se a linha, perde-se o
        # custo, e a ausência é reportada.
        margem_valida = np.where((margem > 0) & (margem < 100), margem, np.nan)
        marcado = sel["SALE"].notna() & (sel["SALE"].astype(str).str.strip() != "")
        partes.append(pd.DataFrame({
            "store": sel["STORE"].to_numpy(),
            "week": sel["WEEK"].to_numpy(),
            "upc": sel["UPC"].to_numpy(),
            "preco": preco,
            "volume": sel["MOVE"].to_numpy(),
            "custo": preco * (1.0 - margem_valida / 100.0),
            "promo": marcado.to_numpy().astype("float64"),
        }))
    return pd.concat(partes, ignore_index=True)


def matriz_presenca(longo: pd.DataFrame, candidatos: list[int]) -> np.ndarray:
    """Matriz booleana células x candidatos, verdadeira onde há preço."""
    largo = longo.pivot_table(index=["store", "week"], columns="upc",
                              values="preco", aggfunc="first")
    return largo.reindex(columns=candidatos).notna().to_numpy()


def guloso(presente: np.ndarray, n: int, semente: int) -> tuple[list[int], int]:
    """Busca gulosa a partir de uma semente. Devolve índices e células completas.

    O conjunto ativo é uma interseção, e interseção só encolhe: a contagem de
    células completas é não crescente em N **por construção**, não por
    evidência. É isso que permite escolher N depois de ver a curva.
    """
    escolhidos = [semente]
    ativo = presente[:, semente].copy()
    for _ in range(n - 1):
        pontos = presente[ativo].sum(axis=0).astype(np.int64)
        pontos[escolhidos] = -1
        j = int(np.argmax(pontos))
        escolhidos.append(j)
        ativo &= presente[:, j]
    return escolhidos, int(ativo.sum())


def semente_padrao(presente: np.ndarray) -> int:
    """O SKU presente no maior número de células. Regra adotada, medida em
    `robustez_selecao.py`: com N = 20 nenhuma das 76 sementes a supera."""
    return int(np.argmax(presente.sum(axis=0)))


def selecionar_por_completude(longo: pd.DataFrame, candidatos: list[int],
                              n: int) -> tuple[list[int], list[dict]]:
    """Os n SKUs de maior completude conjunta, mais o histórico da curva.

    A busca é gulosa e não garante o ótimo do problema combinatório, que teria
    C(76, 20) = 1,09e18 candidatos. Maximizar a interseção equivale a
    minimizar a união dos complementos, e para minimização de união sob
    cardinalidade fixa não existe garantia de fator constante, de modo que a
    defesa da heurística é empírica e está em `robustez_selecao.py`.
    """
    presente = matriz_presenca(longo, candidatos)
    volume = longo.groupby("upc")["volume"].sum().reindex(candidatos).to_numpy()
    volume_total = float(longo["volume"].sum())

    indices = [semente_padrao(presente)]
    ativo = presente[:, indices[0]].copy()
    historico = []
    while len(indices) < n:
        pontos = presente[ativo].sum(axis=0).astype(np.int64)
        pontos[indices] = -1
        j = int(np.argmax(pontos))
        indices.append(j)
        ativo &= presente[:, j]
        historico.append({
            "n": len(indices),
            "celulas_completas": int(ativo.sum()),
            "observacoes": int(ativo.sum()) * len(indices),
            # Denominador: volume dos elegíveis, não da categoria. Ver
            # `cobertura_volume_categoria` em nucleo.py para o outro
            # denominador, que é o que vai para o texto.
            "cobertura_volume_elegiveis_pct":
                100.0 * float(volume[indices].sum()) / volume_total,
        })
    return sorted(candidatos[j] for j in indices), historico


def carregar_sortimento(categoria: str,
                        painel: str | Path = "data/interim/painel") -> list[int]:
    """Lê o sortimento vigente do artefato gravado por `painel.py`.

    Todo script que precise dos N SKUs escolhidos deve chamar isto, e nunca
    reexecutar uma regra de seleção própria. Foi a duplicação da regra que
    produziu números publicados a partir de um sortimento superado.
    """
    destino = Path(painel) / f"{categoria}_upcs.json"
    if not destino.exists():
        raise FileNotFoundError(
            f"{destino} não existe. Rode `python3 src/data/painel.py {categoria} "
            f"--n <N>` antes, porque o sortimento é artefato e não constante.")
    return list(json.load(open(destino))["ordem"])


def agregar_dois_estagios(series: pd.DataFrame, upcs, coluna: str) -> float:
    """Mediana por SKU sobre as séries, depois mediana entre SKUs.

    É a agregação que D3 define e a que os documentos reportam. A agregação
    direta sobre todas as séries (mediana única sobre o conjunto indistinto)
    dá valores próximos e diferentes, porque pondera cada SKU pelo número de
    lojas em que ele aparece. As duas são reportadas lado a lado, e a de dois
    estágios é a oficial, porque a pergunta é sobre o SKU típico do
    sortimento e não sobre a série típica.
    """
    s = series[series.upc.isin(set(upcs))]
    return float(s.groupby("upc")[coluna].median().median())


def agregar_direto(series: pd.DataFrame, upcs, coluna: str) -> float:
    """Mediana única sobre todas as séries do conjunto."""
    return float(series[series.upc.isin(set(upcs))][coluna].median())
