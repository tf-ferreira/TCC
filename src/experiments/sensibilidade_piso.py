"""Duas medições que sustentam a redação vigente de D12.

**1. A curva de custo marginal em células.** Sob a regra vigente, o limiar que
"vira consequência de N" não é mais a N-ésima maior regularidade, e sim o
custo em células do último produto admitido pela busca gulosa. Este script
imprime a curva inteira, para que a escolha de N seja leitura e não decreto,
e para verificar se existe joelho.

**2. Sensibilidade ao piso de suporte.** O piso (longevidade e amplitude
>= 0,50) é o único limiar que resta na regra vigente. Ele não se justifica
pela forma de nenhuma curva, e sim pela variância de uma razão: um SKU visto
numa semana de uma loja tem regularidade 1 com denominador 1. Como é um número
escolhido, o texto precisa mostrar que o resultado não depende dele. Aqui se
repete a seleção com pisos 0,40, 0,50 e 0,60 e compara-se o conjunto obtido.

Uso:
    python3 src/experiments/sensibilidade_piso.py frj --n 20
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from sortimento import (extrair_longo, guloso, matriz_presenca,          # noqa: E402
                        semente_padrao, agregar_dois_estagios)


def elegiveis_com_piso(cobertura: pd.DataFrame, piso: float) -> list[int]:
    apto = cobertura.dropna(subset=["regularidade"])
    apto = apto[(apto["longevidade"] >= piso) & (apto["amplitude"] >= piso)]
    return sorted(apto["upc"].tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pisos", default="0.40,0.50,0.60")
    ap.add_argument("--raiz", default="data/raw")
    ap.add_argument("--screening", default="data/interim/screening")
    ap.add_argument("--saida", default="reports")
    a = ap.parse_args()

    cob = pd.read_csv(Path(a.screening) / f"{a.categoria}_cobertura.csv")
    series = pd.read_csv(Path(a.screening) / f"{a.categoria}_series.csv")
    pisos = [float(x) for x in a.pisos.split(",")]
    zips = sorted((Path(a.raiz) / a.categoria).glob("w*.zip"))

    # Uma única extração, com o piso mais frouxo, cobre todos os cenários.
    cand_amplo = elegiveis_com_piso(cob, min(pisos))
    longo = extrair_longo(zips[0], set(cand_amplo))

    out = {"categoria": a.categoria, "n": a.n, "pisos": {}}
    referencia = None
    for piso in pisos:
        cand = elegiveis_com_piso(cob, piso)
        P = matriz_presenca(longo, cand)
        idx, celulas = guloso(P, a.n, semente_padrao(P))
        upcs = sorted(cand[j] for j in idx)
        if referencia is None:
            referencia = set(upcs)
        # curva de custo marginal: quantas células o k-ésimo produto destrói
        curva, ativo, escolhidos = [], None, []
        for k, j in enumerate(idx, start=1):
            ativo = P[:, j].copy() if ativo is None else (ativo & P[:, j])
            escolhidos.append(j)
            curva.append({"n": k, "celulas": int(ativo.sum()),
                          "custo_marginal": (None if k == 1 else
                                             curva[-1]["celulas"] - int(ativo.sum()))})
        out["pisos"][f"{piso:.2f}"] = {
            "n_elegiveis": len(cand),
            "celulas_completas": celulas,
            "upcs": upcs,
            "iguais_ao_piso_050": len(set(upcs) & referencia),
            "cv_dois_estagios": agregar_dois_estagios(series, upcs, "cv"),
            "curva": curva,
        }
        print(f"piso {piso:.2f}: {len(cand):3d} elegíveis, {celulas} células, "
              f"{len(set(upcs) & referencia)}/{a.n} SKUs em comum com o piso 0,50, "
              f"CV {out['pisos'][f'{piso:.2f}']['cv_dois_estagios']:.4f}")

    base = out["pisos"][f"{0.50:.2f}"]["curva"]
    print(f"\ncurva de custo marginal em células (piso 0,50):")
    print(f"  {'N':>3} {'células':>9} {'custo do N-ésimo':>17}")
    for r in base:
        cm = "" if r["custo_marginal"] is None else f"{r['custo_marginal']:d}"
        print(f"  {r['n']:3d} {r['celulas']:9d} {cm:>17}")

    destino = Path(a.saida) / f"sensibilidade_piso_{a.categoria}_n{a.n}.json"
    destino.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\ngravado em {destino}")


if __name__ == "__main__":
    main()
