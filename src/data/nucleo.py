"""Comparação entre categorias sob o critério vigente de D12.

Responde: **fixado N, que sortimento cada categoria entrega e que painel
sobra?** É a etapa que compara as candidatas depois que a triagem já mediu os
três critérios do projeto.

> **Reescrito em 09/09/2026.** A versão anterior expunha
> `selecionar_sortimento(cobertura, n)`, que ordenava por regularidade
> individual, isto é, a D12 **superada**. `painel.py` já usava completude
> conjunta, e a divergência entre os dois módulos passou despercebida porque
> nada no projeto comparava um com o outro. Números publicados que vieram da
> versão antiga: `tau = 0,934`, `cv = 0,131` e `836.096 observações` na seção
> 3.1 do texto, e o arquivo `frj_endogeneidade.json` inteiro.
>
> A função antiga não sobrevive como API. Ela existe apenas dentro de
> `src/experiments/robustez_selecao.py`, onde é **objeto de comparação** e não
> caminho de produção, para que a evidência de D12 continue reproduzível sem
> que ninguém possa chamá-la por engano.

Duas grandezas com o mesmo nome e denominadores diferentes, que já causaram
confusão de leitura e por isso saem separadas e rotuladas:

    cobertura_volume_categoria_pct   volume do sortimento / volume dos 175
    cobertura_volume_elegiveis_pct   volume do sortimento / volume dos 76

Em sucos congelados com N = 20 elas valem 55,2% e 60,98%, e a razão entre
elas é o peso dos elegíveis na categoria, 90,57%. O número que vai para o
texto é o primeiro, por ser o mais conservador e o que responde à pergunta
que um leitor faz.

Uso:
    python3 src/data/nucleo.py frj --n 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from screening import COLUNAS_CHAVE
from sortimento import (COLUNAS, DTYPES_LOCAL, PISO_AMPLITUDE, PISO_LONGEVIDADE,
                        agregar_direto, agregar_dois_estagios, elegiveis,
                        extrair_longo, selecionar_por_completude)


def volume_por_upc(caminho_zip: Path, bloco: int = 2_000_000) -> pd.Series:
    """Volume total por SKU na categoria inteira, em passada leve.

    Passada separada e sem pivô porque o denominador da cobertura de volume é
    a categoria (175 SKUs), enquanto a seleção só precisa dos elegíveis (76).
    Reter o painel largo dos 175 para obter uma soma seria desperdício.
    """
    total = None
    leitor = pd.read_csv(caminho_zip, compression="zip", usecols=COLUNAS,
                         dtype=DTYPES_LOCAL, chunksize=bloco)
    for parte in leitor:
        parte = parte.loc[~parte[COLUNAS_CHAVE].isna().any(axis=1)]
        if parte.empty:
            continue
        sel = parte[(parte["OK"] == 1) & (parte["PRICE"] > 0)]
        if sel.empty:
            continue
        parcial = sel.groupby(sel["UPC"].astype("int64"))["MOVE"].sum()
        total = parcial if total is None else total.add(parcial, fill_value=0.0)
    return total


def resumir(categoria: str, n: int, raiz: Path, screening: Path) -> dict:
    cobertura = pd.read_csv(screening / f"{categoria}_cobertura.csv")
    series = pd.read_csv(screening / f"{categoria}_series.csv")
    candidatos = elegiveis(cobertura)
    zips = sorted((raiz / categoria).glob("w*.zip"))
    if not zips:
        raise FileNotFoundError(f"sem arquivo de movimento para {categoria}")

    longo = extrair_longo(zips[0], set(candidatos))
    upcs, historico = selecionar_por_completude(longo, candidatos, n)

    volume = volume_por_upc(zips[0])
    vol_categoria = float(volume.sum())
    vol_elegiveis = float(volume.reindex(candidatos).fillna(0.0).sum())
    vol_sortimento = float(volume.reindex(upcs).fillna(0.0).sum())

    escolhidos = cobertura[cobertura["upc"].isin(set(upcs))]
    celulas = historico[-1]["celulas_completas"] if historico else None

    return {
        "categoria": categoria,
        "n": n,
        "criterio": "completude_conjunta",           # D12 vigente
        "piso_longevidade": PISO_LONGEVIDADE,
        "piso_amplitude": PISO_AMPLITUDE,
        "n_upcs_categoria": int(len(cobertura)),
        "n_elegiveis": len(candidatos),
        "n_selecionados": len(upcs),
        "upcs": upcs,
        # Descritiva do conjunto escolhido, NÃO o critério que o produziu.
        # Sob a regra vigente o limiar de regularidade não seleciona nada.
        "regularidade_minima": float(escolhidos["regularidade"].min()),
        "regularidade_mediana": float(escolhidos["regularidade"].median()),
        "longevidade_mediana": float(escolhidos["longevidade"].median()),
        "amplitude_mediana": float(escolhidos["amplitude"].median()),
        "celulas_com_algum_sku": int(len(longo.groupby(["store", "week"]))),
        "celulas_completas": celulas,
        "observacoes_completas": celulas * len(upcs) if celulas else None,
        "cv_dois_estagios": agregar_dois_estagios(series, upcs, "cv"),
        "cv_direto": agregar_direto(series, upcs, "cv"),
        "promo_preco_dois_estagios":
            agregar_dois_estagios(series, upcs, "frac_promo_preco"),
        "promo_preco_direto": agregar_direto(series, upcs, "frac_promo_preco"),
        "cv_categoria_dois_estagios":
            agregar_dois_estagios(series, series["upc"].unique(), "cv"),
        "promo_categoria_dois_estagios":
            agregar_dois_estagios(series, series["upc"].unique(), "frac_promo_preco"),
        "cobertura_volume_categoria_pct": 100.0 * vol_sortimento / vol_categoria,
        "cobertura_volume_elegiveis_pct": 100.0 * vol_sortimento / vol_elegiveis,
        "peso_elegiveis_na_categoria_pct": 100.0 * vol_elegiveis / vol_categoria,
        "historico_selecao": historico,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("categoria")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--raiz", default="data/raw")
    parser.add_argument("--screening", default="data/interim/screening")
    args = parser.parse_args()

    resultado = resumir(args.categoria, args.n,
                        Path(args.raiz), Path(args.screening))
    destino = Path(args.screening) / f"{args.categoria}_nucleo{args.n}.json"
    destino.write_text(json.dumps(resultado, indent=2))
    resumo = {k: v for k, v in resultado.items()
              if k not in ("upcs", "historico_selecao")}
    print(json.dumps(resumo, indent=2))


if __name__ == "__main__":
    main()
