"""Diagnóstico da partição, do ponto de vista das entradas da rede (seção 6).

A árvore esconde três problemas que a rede não esconde, e este script mede os
três em vez de deixá-los como suposição:

1. **Lojas nunca vistas.** O LightGBM manda categoria desconhecida para o ramo
   padrão. Um embedding é tabela indexada por código: a loja que só aparece no
   teste lê uma linha que nunca recebeu gradiente.
2. **Extrapolação temporal.** Uma árvore não extrapola, ela repete o valor da
   última folha. Uma rede aprende tendência linear e a projeta para fora da
   amostra sem freio, e o quanto ela projeta depende de quantas semanas separam
   o fim do treino do fim do teste.
3. **Cobertura das defasagens por partição.** Uma defasagem definida em 87% do
   treino e em outra fração do teste não é o mesmo atributo nos dois lados.

Uso:
    python3 src/experiments/diagnostico_particao.py frj
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src" / "experiments"))
sys.path.insert(0, str(RAIZ / "src" / "models"))
sys.path.insert(0, str(RAIZ / "src" / "data"))

from baseline_arvores import particionar  # noqa: E402
from varredura_defasagens import preparar  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--saida", default="reports")
    ap.add_argument("--frac-teste", type=float, default=0.20)
    ap.add_argument("--zona", type=int, default=8)
    a = ap.parse_args()

    pa = Path(a.painel)
    n = json.loads((pa / f"{a.categoria}_upcs.json").read_text())["n"]
    longo = preparar(pd.read_parquet(pa / f"{a.categoria}_celulas.parquet"),
                     pd.read_parquet(pa / f"{a.categoria}_bem_externo.parquet"), n)
    treino, teste, corte = particionar(longo, a.frac_teste, a.zona)

    lt = set(treino.store.unique().tolist())
    lte = set(teste.store.unique().tolist())
    so_teste = sorted(lte - lt)
    marca = teste.store.isin(so_teste)

    saida = {
        "categoria": a.categoria,
        "particao": {"semana_de_corte": int(corte), "zona_morta_semanas": a.zona,
                     "n_treino": int(len(treino)), "n_teste": int(len(teste))},
        "lojas": {
            "no_treino": len(lt), "no_teste": len(lte),
            "so_no_teste": len(so_teste), "so_no_treino": len(lt - lte),
            "pct_linhas_teste_em_loja_nunca_vista": float(100.0 * marca.mean()),
            "pct_volume_teste_em_loja_nunca_vista":
                float(100.0 * teste.loc[marca, "alvo"].sum() / teste["alvo"].sum()),
        },
        "tempo": {
            "semana_max_treino": int(treino.week.max()),
            "semana_max_teste": int(teste.week.max()),
            "semanas_de_extrapolacao": int(teste.week.max() - treino.week.max()),
        },
        "cobertura_defasagem": {
            f"preco_lag{k}": {
                "treino": float(treino[f"preco_lag{k}"].notna().mean()),
                "teste": float(teste[f"preco_lag{k}"].notna().mean()),
            } for k in (1, 2)
        },
    }
    destino = Path(a.saida) / f"diagnostico_particao_{a.categoria}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))

    lj = saida["lojas"]
    print(f"lojas: {lj['no_treino']} no treino, {lj['no_teste']} no teste, "
          f"{lj['so_no_teste']} só no teste")
    print(f"  {lj['pct_linhas_teste_em_loja_nunca_vista']:.2f}% das linhas de teste "
          f"e {lj['pct_volume_teste_em_loja_nunca_vista']:.2f}% do volume")
    print(f"extrapolação: {saida['tempo']['semanas_de_extrapolacao']} semanas "
          f"além do fim do treino")
    for k, v in saida["cobertura_defasagem"].items():
        print(f"  {k}: treino {v['treino']:.1%}, teste {v['teste']:.1%}")
    print(f"gravado em {destino}")


if __name__ == "__main__":
    main()
