"""Construção do painel no nível de célula (loja, semana).

Primeira etapa da fase de estimação. Transforma o arquivo de movimento, que é
longo (uma linha por loja, SKU e semana), no formato que a rede consome: uma
linha por célula, com um vetor de N posições para cada grandeza.

A unidade é a célula e não o produto isolado porque o otimizador decide um
vetor de N preços simultaneamente. Modelar produto a produto exigiria N
avaliações da rede por iteração do solver e não garantiria consistência entre
as funções.

A regra de seleção do sortimento **não** mora aqui. Ela mora em
`sortimento.py`, que é o ponto único onde D12 existe, e este módulo apenas a
consome. Ter a regra em dois lugares foi o que produziu números publicados a
partir de um sortimento superado.

Grandezas extraídas por SKU:
    preco    preço unitário praticado (decisão D1)
    volume   unidades vendidas
    custo    custo unitário de aquisição, price*(1-profit/100)/qty
    promo    indicador de promoção declarada no campo `sale`

Uso:
    python3 src/data/painel.py frj --n 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sortimento import (GRANDEZAS, elegiveis, extrair_longo,
                        selecionar_por_completude)


def montar_celulas(longo: pd.DataFrame, upcs: list[int]) -> pd.DataFrame:
    """Pivota para uma linha por célula, com uma coluna por SKU e grandeza.

    A ordem das colunas segue a ordem de `upcs`, gravada junto com o painel:
    a posição no vetor **é** a identidade do produto para a rede, de modo que
    trocar a ordem entre execuções invalidaria os pesos treinados.
    """
    quadros = []
    for grandeza in GRANDEZAS:
        largo = longo.pivot_table(index=["store", "week"], columns="upc",
                                  values=grandeza, aggfunc="first")
        largo = largo.reindex(columns=upcs)
        largo.columns = [f"{grandeza}_{i:02d}" for i in range(len(upcs))]
        quadros.append(largo)
    return pd.concat(quadros, axis=1).reset_index()


def diagnosticar(celulas: pd.DataFrame, n: int) -> dict:
    precos = celulas[[f"preco_{i:02d}" for i in range(n)]]
    presentes = precos.notna().sum(axis=1)
    por_sku = precos.notna().mean(axis=0)
    return {
        "n_celulas": int(len(celulas)),
        "n_celulas_completas": int((presentes == n).sum()),
        "n_lojas": int(celulas["store"].nunique()),
        "n_semanas": int(celulas["week"].nunique()),
        "observacoes_completas": int((presentes == n).sum()) * n,
        "presenca_por_sku_min": float(por_sku.min()),
        "presenca_por_sku_mediana": float(por_sku.median()),
        "pct_custo_ausente": float(
            celulas[[f"custo_{i:02d}" for i in range(n)]].isna().to_numpy().mean() * 100
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("categoria")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--raiz", default="data/raw")
    parser.add_argument("--screening", default="data/interim/screening")
    parser.add_argument("--saida", default="data/interim/painel")
    args = parser.parse_args()

    cobertura = pd.read_csv(Path(args.screening) / f"{args.categoria}_cobertura.csv")
    candidatos = elegiveis(cobertura)
    pasta_raw = Path(args.raiz) / args.categoria
    zips = sorted(pasta_raw.glob("w*.zip"))
    if len(zips) != 1:
        raise SystemExit(
            f"esperava exatamente um w*.zip em {pasta_raw}, encontrei "
            f"{len(zips)}. Ler so o primeiro deslocaria todos os numeros.")

    # Primeira passada sobre os elegíveis, para a seleção; segunda sobre os
    # escolhidos, para o painel. Duas passadas custam menos que reter em
    # memória o painel largo de todos os candidatos.
    longo_todos = extrair_longo(zips[0], set(candidatos))
    upcs, historico = selecionar_por_completude(longo_todos, candidatos, args.n)
    longo = longo_todos[longo_todos["upc"].isin(set(upcs))]
    celulas = montar_celulas(longo, upcs)

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    celulas.to_parquet(saida / f"{args.categoria}_celulas.parquet", index=False)
    (saida / f"{args.categoria}_upcs.json").write_text(json.dumps(
        {"n": args.n, "criterio": "completude_conjunta", "ordem": upcs,
         "n_elegiveis": len(candidatos), "historico_selecao": historico}, indent=2))
    diag = {"categoria": args.categoria, "n_skus": args.n,
            "n_elegiveis": len(candidatos), **diagnosticar(celulas, args.n)}
    (saida / f"{args.categoria}_diagnostico.json").write_text(json.dumps(diag, indent=2))
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
