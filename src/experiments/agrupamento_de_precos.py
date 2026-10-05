"""Agrupamento de preços entre lojas, e o que a demografia de loja consegue.

Existe por dois motivos, e o segundo é uma correção.

**Primeiro**, a seção 4 da especificação de atributos precisa decidir se algum
atributo de loja vale a pena **dado que a rede já tem um embedding de loja**. Um
embedding representa qualquer constante por loja, então todo atributo de loja
invariante no tempo é redundante com ele **dentro das lojas que o treino viu**. O
único trabalho que sobra para esses atributos é generalizar para loja **sem
embedding treinado**, e este script mede se eles conseguem.

**Segundo**, os números que D23 cita sobre esse agrupamento (`zone` explica 50,7%,
`scluster` explica 11,2%) **não têm produtor**: nenhum script do repositório
menciona `zone` ou `scluster`. É o mesmo caso da elasticidade de −1,112 que
motivou `src/reports/numeros_oficiais.py`, e a correção é a mesma: medir, gravar
em artefato e aposentar o valor sem procedência.

Uso:
    python3 src/experiments/agrupamento_de_precos.py frj
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src" / "models"))
sys.path.insert(0, str(RAIZ / "src" / "data"))

from baseline_arvores import montar_longo, particionar  # noqa: E402

CAMPOS = ("zone", "scluster")


def ler_demografia(caminho: Path) -> pd.DataFrame:
    z = zipfile.ZipFile(caminho)
    nome = [n for n in z.namelist() if n.lower().endswith(".dta")][0]
    d = pd.read_stata(io.BytesIO(z.read(nome)))
    return d.drop_duplicates("store").set_index("store")


def variancia_explicada(longo: pd.DataFrame, rotulo: pd.Series) -> dict:
    """Fração da variância de log de preço ENTRE lojas que o rótulo explica.

    A decomposição é dentro de cada par (SKU, semana), que é onde "entre lojas"
    faz sentido: fixado o produto e a semana, o que resta é diferença de loja.

    As lojas sem o rótulo saem da conta, e o número de lojas usadas vai no
    resultado. Mantê-las como uma categoria "sem rótulo" inflaria a explicação,
    porque essa categoria agrupa lojas que de fato se parecem entre si.
    """
    q = longo[["store", "sku", "week", "preco_proprio"]].copy()
    q["g"] = q.store.map(rotulo)
    q = q[q.g.notna()].copy()
    q["lp"] = np.log(q.preco_proprio)
    q["desvio"] = q.lp - q.groupby(["sku", "week"])["lp"].transform("mean")
    dentro = q["desvio"] - q.groupby(["sku", "week", "g"])["desvio"].transform("mean")
    total = float((q["desvio"] ** 2).mean())
    return {
        "pct_explicada": float(100.0 * (1.0 - float((dentro ** 2).mean()) / total)),
        "n_lojas": int(q.store.nunique()),
        "n_grupos": int(q.g.nunique()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--demografia",
                    default="data/raw/store_demographics/demo_stata.zip")
    ap.add_argument("--saida", default="reports")
    ap.add_argument("--frac-teste", type=float, default=0.20)
    ap.add_argument("--zona", type=int, default=8)
    a = ap.parse_args()

    pa = Path(a.painel)
    n = json.loads((pa / f"{a.categoria}_upcs.json").read_text())["n"]
    longo = montar_longo(pd.read_parquet(pa / f"{a.categoria}_celulas.parquet"),
                         pd.read_parquet(pa / f"{a.categoria}_bem_externo.parquet"), n)
    demo = ler_demografia(Path(a.demografia))

    lojas = set(longo.store.unique().tolist())
    sem_demo = sorted(lojas - set(demo.index.tolist()))
    treino, teste, _ = particionar(longo, a.frac_teste, a.zona)
    so_teste = sorted(set(teste.store.unique()) - set(treino.store.unique()))

    # Quantos preços distintos existem para um SKU numa semana, e quantas lojas
    # estão ativas. É a medida direta de "a precificação é agrupada".
    g = longo.groupby(["sku", "week"])["preco_proprio"]
    distintos = g.nunique()
    ativas = g.size()

    faixa = (demo.priclow.astype("Int64").astype(str)
             + demo.pricmed.astype("Int64").astype(str)
             + demo.prichigh.astype("Int64").astype(str))
    faixa = faixa.where(demo.priclow.notna())

    saida = {
        "categoria": a.categoria,
        "lojas_no_painel": len(lojas),
        "lojas_sem_linha_na_demografia": len(sem_demo),
        "lojas_so_no_teste": len(so_teste),
        "sem_demografia_sao_as_mesmas_do_teste": sem_demo == so_teste,
        "precos_distintos_por_sku_semana_mediana": float(distintos.median()),
        "lojas_ativas_por_sku_semana_mediana": float(ativas.median()),
        "variancia_entre_lojas": {
            **{c: variancia_explicada(longo, demo[c]) for c in CAMPOS},
            "faixa_de_preco": variancia_explicada(longo, faixa),
        },
    }
    destino = Path(a.saida) / f"agrupamento_precos_{a.categoria}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))

    print(f"lojas no painel: {saida['lojas_no_painel']}, "
          f"sem linha na demografia: {saida['lojas_sem_linha_na_demografia']}")
    print(f"as sem demografia são exatamente as que só aparecem no teste: "
          f"{saida['sem_demografia_sao_as_mesmas_do_teste']}")
    print(f"preços distintos por (SKU, semana), mediana: "
          f"{saida['precos_distintos_por_sku_semana_mediana']:.0f} "
          f"entre {saida['lojas_ativas_por_sku_semana_mediana']:.0f} lojas ativas")
    for c, v in saida["variancia_entre_lojas"].items():
        print(f"  {c:16s} {v['pct_explicada']:5.1f}%  "
              f"({v['n_grupos']} grupos, {v['n_lojas']} lojas)")
    print(f"gravado em {destino}")


if __name__ == "__main__":
    main()
