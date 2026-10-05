"""Varredura de sazonalidade: como o calendário entra no modelo (seção 2).

Mede, e não supõe, a escolha entre quatro representações do tempo dentro do ano,
com todo o resto igual: mesma partição temporal, mesmos demais atributos, mesma
semente, mesmo modelo de referência de D20.

    modulo52   `week % 52`, a representação herdada do piso de D20
    harm1..3   seno e cosseno do ângulo do ano sobre a data real, k harmônicas
    +feriados  a melhor das anteriores mais os quatro indicadores de feriado

Por que medir e não decidir por argumento. O salto e a deriva de `week % 52`
são defeitos reais e demonstrados (ver `src/data/calendario.py`), mas eles
machucam uma **rede**, que recebe a variável como número contínuo. Uma **árvore**
corta onde quiser e captura qualquer forma com divisões suficientes, de modo que
para ela a troca pode até custar ajuste. Como D20 obriga os dois modelos a
receber as mesmas variáveis, a troca precisa ser barata na árvore, e isto aqui é
o que verifica se é.

Todas as execuções já seguem D25: `promo_proprio` está fora de todas elas, de
modo que a diferença entre linhas mede sazonalidade e nada mais.

Uso:
    python3 src/experiments/sazonalidade.py frj
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(RAIZ / "src" / "models"))
sys.path.insert(0, str(RAIZ / "src" / "data"))

from baseline_arvores import (colunas_atributos, fixar_categorias,  # noqa: E402
                              metricas, montar_longo, particionar)
from calendario import FERIADOS, data_da_semana, feriados, harmonicos  # noqa: E402

CATEGORICAS = ["sku", "store"]
K_MAX = 3


def preparar(cel: pd.DataFrame, fora: pd.DataFrame, n: int) -> pd.DataFrame:
    """Painel longo com TODAS as colunas de calendário disponíveis.

    As colunas são construídas uma vez só e as configurações escolhem entre
    elas, para que nenhuma diferença venha de recomputar o painel.
    """
    longo = montar_longo(cel, fora, n).reset_index(drop=True)
    datas = data_da_semana(longo["week"].to_numpy())
    longo = pd.concat([longo, harmonicos(datas, K_MAX), feriados(datas)], axis=1)
    return fixar_categorias(longo, CATEGORICAS)


def configuracoes(k_melhor: int | None = None) -> dict[str, list[str]]:
    cfg = {"modulo52": ["semana_do_ano"]}
    for k in range(1, K_MAX + 1):
        cfg[f"harm{k}"] = [f"{p}_{j}" for j in range(1, k + 1) for p in ("sen", "cos")]
    if k_melhor is not None:
        base = cfg[f"harm{k_melhor}"]
        cfg[f"harm{k_melhor}_feriados"] = base + [f"fer_{f}" for f in FERIADOS]
    return cfg


def rodar(lgb, treino, teste, base: list[str], sazonais: list[str],
          arvores: int) -> dict:
    colunas = base + sazonais
    modelo = lgb.LGBMRegressor(
        objective="poisson", n_estimators=arvores, learning_rate=0.05,
        num_leaves=127, min_child_samples=40, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    modelo.fit(treino[colunas], treino["alvo"], categorical_feature=CATEGORICAS)
    imp = dict(zip(colunas, modelo.feature_importances_.tolist()))
    total = float(sum(imp.values())) or 1.0
    return {
        "atributos_sazonais": sazonais,
        "n_atributos": len(colunas),
        "treino": metricas(treino["alvo"], modelo.predict(treino[colunas])),
        "teste": metricas(teste["alvo"], modelo.predict(teste[colunas])),
        "share_importancia_sazonal_pct":
            100.0 * sum(imp[c] for c in sazonais) / total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--saida", default="reports")
    ap.add_argument("--frac-teste", type=float, default=0.20)
    ap.add_argument("--zona", type=int, default=8)
    ap.add_argument("--arvores", type=int, default=800)
    a = ap.parse_args()
    import lightgbm as lgb

    pa = Path(a.painel)
    n = json.loads((pa / f"{a.categoria}_upcs.json").read_text())["n"]
    cel = pd.read_parquet(pa / f"{a.categoria}_celulas.parquet")
    fora = pd.read_parquet(pa / f"{a.categoria}_bem_externo.parquet")
    longo = preparar(cel, fora, n)
    treino, teste, corte = particionar(longo, a.frac_teste, a.zona)

    # D25 já vale: promo_proprio fora de todas as execuções.
    base = [c for c in colunas_atributos(n, sem_promo_proprio=True)
            if c != "semana_do_ano"]

    resultados = {}
    for nome, sazonais in configuracoes().items():
        resultados[nome] = rodar(lgb, treino, teste, base, sazonais, a.arvores)

    harms = {k: v["teste"]["wmape_pct"] for k, v in resultados.items()
             if k.startswith("harm")}
    k_melhor = int(min(harms, key=harms.get).removeprefix("harm"))
    nome, sazonais = list(configuracoes(k_melhor).items())[-1]
    resultados[nome] = rodar(lgb, treino, teste, base, sazonais, a.arvores)

    melhor = min(resultados, key=lambda k: resultados[k]["teste"]["wmape_pct"])
    saida = {
        "categoria": a.categoria,
        "particao": {"semana_de_corte": corte, "zona_morta_semanas": a.zona,
                     "n_treino": int(len(treino)), "n_teste": int(len(teste))},
        "nota": ("Todas as execuções já seguem D25 (promo_proprio fora). A "
                 "referência da linha modulo52 é a especificação herdada do "
                 "piso de D20, e a diferença contra ela mede só a mudança de "
                 "representação do tempo."),
        "k_harmonicas_melhor": k_melhor,
        "configuracao_melhor": melhor,
        "resultados": resultados,
    }
    destino = Path(a.saida) / f"sazonalidade_{a.categoria}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))

    print(f"partição: teste a partir da semana {corte}, zona morta de {a.zona}")
    print(f"{'configuração':22s} {'RMSE':>8} {'MAE':>8} {'RMSE log':>9} "
          f"{'WMAPE':>8} {'importância sazonal':>20}")
    for nome, r in resultados.items():
        t = r["teste"]
        print(f"{nome:22s} {t['rmse']:8.2f} {t['mae']:8.2f} {t['rmse_log']:9.4f} "
              f"{t['wmape_pct']:7.2f}% {r['share_importancia_sazonal_pct']:19.2f}%")
    print(f"\nmelhor: {melhor}")
    print(f"gravado em {destino}")


if __name__ == "__main__":
    main()
