"""Sonda 19: quanto da variação de preço movida pelo custo é própria de cada SKU.

Pergunta. Com a função de controle de D42, os vinte resíduos `v̂ⱼ` entram no
contexto comum que alimenta todos os `gᵢ(x)`. Condicionando neles, a variação de
cada `uⱼ` que sobra para identificar a derivada é a movida pelo custo. A própria
`mᵢ` e as cruzadas `γᵢⱼ` só se separam se essa variação de um produto não for a
mesma dos outros dezenove.

Medida, na janela de treino e nas células completas, por SKU `j`:

1. `zⱼ = π̂ⱼ·r(ln cⱼ)`, a parte do log preço movida pelo custo, com `π̂ⱼ` o repasse
   do primeiro estágio de `funcao_de_controle.ajustar` e `r(·)` o resíduo da
   regressão nos mesmos controles do primeiro estágio (efeito de loja, harmônicos,
   tendência);
2. `R²ⱼ` de `zⱼ` contra os outros dezenove `z`, por MQO: a fração da variação de
   custo de `j` que é comum aos outros;
3. o mesmo `R²` para o log preço residualizado nos controles, que é a variação que
   a rede sem função de controle usa, como contraste.

`1 − R²ⱼ` é a fração da variação de custo de `j` que identifica, em separado, a
coluna `j` da matriz. É descritiva: a rede condiciona também nas defasagens e no
resto da categoria, e o número não é o F de um primeiro estágio conjunto.

Uso: `cd sondas_diagnostico && python3 s19_independencia_do_custo.py`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
for sub in ("models", "data"):
    sys.path.insert(0, str(RAIZ / "src" / sub))

import funcao_de_controle as fc  # noqa: E402
import rede  # noqa: E402


def residualizar(g: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Resíduo de `y` nos controles do primeiro estágio, dentro de um SKU."""
    lojas = np.sort(g["store"].unique())
    X = fc._desenho(g.assign(custo=np.exp(0.0)), lojas)
    k = len(lojas)
    X = np.delete(X, k, axis=1)          # tira a coluna de ln c (constante aqui)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def r2_contra_os_outros(Z: np.ndarray) -> np.ndarray:
    out = np.empty(Z.shape[1])
    for j in range(Z.shape[1]):
        y = Z[:, j] - Z[:, j].mean()
        X = np.delete(Z, j, axis=1)
        X = X - X.mean(0)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ b
        out[j] = 1.0 - float(e @ e) / float(y @ y) if float(y @ y) > 0 else np.nan
    return out


def main() -> None:
    painel = RAIZ / "data" / "interim" / "painel"
    n = json.loads((painel / "frj_upcs.json").read_text())["n"]
    cel = pd.read_parquet(painel / "frj_celulas.parquet")
    fora = pd.read_parquet(painel / "frj_bem_externo.parquet")
    longo = rede.construir(cel, fora, n)
    treino, *_ = rede.preparar_treino_teste(longo)
    custos = fc.custo_longo(cel, n)
    coefs = fc.ajustar(treino, custos)
    d = fc._com_custo(treino, custos)
    d = d[d["custo"].notna() & (d["custo"] > 0)].copy()

    z = np.full(len(d), np.nan)
    rp = np.full(len(d), np.nan)
    for s, g in d.groupby("sku"):
        pos = d.index.get_indexer(g.index)
        z[pos] = coefs[int(s)]["pi"] * residualizar(g, np.log(g["custo"].to_numpy(float)))
        rp[pos] = residualizar(g, np.log(g["preco_proprio"].to_numpy(float)))
    d["z"], d["rp"] = z, rp
    Z = d.pivot_table(index=["store", "week"], columns="sku", values="z").dropna()
    P = d.pivot_table(index=["store", "week"], columns="sku", values="rp").loc[Z.index]

    r2_z = r2_contra_os_outros(Z.to_numpy())
    r2_p = r2_contra_os_outros(P.to_numpy())
    pi = np.array([coefs[i]["pi"] for i in sorted(coefs)])
    dp_z = Z.std().to_numpy()
    dp_p = P.std().to_numpy()
    fortes = pi > 0.2
    res = {
        "celulas": int(len(Z)),
        "pi": pi.tolist(),
        "r2_custo_contra_os_outros": r2_z.tolist(),
        "r2_preco_contra_os_outros": r2_p.tolist(),
        "dp_z": dp_z.tolist(), "dp_preco_residual": dp_p.tolist(),
        "resumo": {
            "mediana_r2_custo_skus_com_repasse": float(np.median(r2_z[fortes])),
            "faixa_r2_custo_skus_com_repasse": [float(r2_z[fortes].min()), float(r2_z[fortes].max())],
            "mediana_r2_preco": float(np.median(r2_p)),
            "faixa_r2_preco": [float(r2_p.min()), float(r2_p.max())],
            "mediana_fracao_da_variacao_do_preco_movida_pelo_custo":
                float(np.median((dp_z / dp_p)[fortes] ** 2)),
            "skus_sem_repasse": [int(i) for i in np.flatnonzero(~fortes)],
        },
    }
    print(f"células completas do treino com custo nos 20: {len(Z)}")
    print(" sku    π̂    R² custo  R² preço  (dp z / dp preço)²")
    for j in range(len(pi)):
        print(f"  {j:2d} {pi[j]:+.3f}   {r2_z[j]:.3f}    {r2_p[j]:.3f}     {(dp_z[j]/dp_p[j])**2:.3f}")
    print(json.dumps(res["resumo"], indent=1))
    saida = Path(__file__).with_suffix(".json")
    saida.write_text(json.dumps(res, indent=1))
    print(f"-> {saida}")


if __name__ == "__main__":
    main()
