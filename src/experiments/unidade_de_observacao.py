"""A unidade de observação do treino, medida: evidência de D31.

Duas perguntas que a abertura do item 6 fez, e que só número responde.

**1. O tamanho efetivo da amostra.** A pendência 4.1 afirmava que o número de
exemplos de treino é o número de células, 25.234, e não células vezes N. Isso é
uma das duas pontas, não a resposta. As 504.680 linhas do painel longo carregam
entre 25.234 e 504.680 números de informação, e onde exatamente depende de
quanto os 20 produtos de uma mesma célula se movem juntos. É a mesma maquinaria
de E.1 e D24: efeito de desenho sobre observações agrupadas, com o grupo sendo
agora a célula e não a loja.

O estimador é o da análise de variância para grupos balanceados de tamanho m,
que é o caso aqui por construção, porque só células completas entram:

    rho = (MSB - MSW) / (MSB + (m - 1) * MSW)
    deff = 1 + (m - 1) * rho
    n_efetivo = n_linhas / deff

O resíduo é o do modelo de referência de árvores na configuração final do item
5, nas duas partições. Dois resíduos são reportados, e eles medem coisas
diferentes: o **cru** (y - mu) é o que entra no escore da perda de Poisson e é
dominado pelos SKUs de volume alto; o **de Pearson** ((y - mu) / raiz(mu)) é o
padronizado, em que um SKU pequeno pesa o mesmo que um grande. O número citado
em D31 é o de Pearson, porque o de interesse é a correlação do ruído e não a da
escala.

Limite declarado: rho aqui é do resíduo de uma árvore, não da rede, e não é
invariante ao modelo. Um modelo que capture melhor o choque comum da célula
deixa rho menor. O número serve para saber em que ponta do intervalo estamos,
não como constante do painel.

**2. As células incompletas.** O painel de modelagem usa 25.234 das 32.255
células com atividade. A pergunta é se as 7.021 restantes poderiam entrar, e a
medida decide: quantos SKUs faltam em cada uma, se o buraco é só no alvo ou
também na entrada, e como a incompletude se distribui entre as partições.

Uso:
    python3 src/experiments/unidade_de_observacao.py frj
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src" / "models"))
sys.path.insert(0, str(RAIZ / "src" / "data"))

from atributos import atributos, construir, preparar_treino_teste  # noqa: E402


def icc_balanceado(residuo: np.ndarray, grupo: np.ndarray) -> dict:
    """Correlação intra-classe pelo estimador de ANOVA, grupos balanceados.

    Exige que todos os grupos tenham o mesmo tamanho m, que é o caso do painel
    de células completas por construção. Se não for, a função falha em vez de
    devolver um número que parece certo.
    """
    quadro = pd.DataFrame({"r": np.asarray(residuo, float), "g": grupo})
    tamanhos = quadro.groupby("g", observed=True)["r"].size()
    m = int(tamanhos.iloc[0])
    if not bool((tamanhos == m).all()):
        raise ValueError("grupos desbalanceados; o estimador exige tamanho fixo")
    k = int(len(tamanhos))
    media_geral = float(quadro["r"].mean())
    medias = quadro.groupby("g", observed=True)["r"].mean()

    ssb = float(m * ((medias - media_geral) ** 2).sum())
    sst = float(((quadro["r"] - media_geral) ** 2).sum())
    ssw = sst - ssb
    msb = ssb / (k - 1)
    msw = ssw / (k * (m - 1))

    rho = (msb - msw) / (msb + (m - 1) * msw)
    deff = 1.0 + (m - 1) * rho
    n = k * m
    return {
        "n_linhas": n, "n_grupos": k, "m": m,
        "rho": float(rho),
        "fracao_da_variancia_entre_celulas_pct": float(100.0 * ssb / sst),
        "efeito_de_desenho": float(deff),
        "n_efetivo": float(n / deff),
        # As duas pontas do intervalo do degrau 7, para leitura direta: o
        # numero de celulas (k) e o numero de linhas (n). A pendencia 4.1
        # afirmava a ponta de baixo como se fosse a resposta.
        "razao_n_efetivo_sobre_n_celulas": float((n / deff) / k),
        "razao_n_efetivo_sobre_n_linhas": float(1.0 / deff),
    }


def _cobertura(parte: pd.DataFrame, pcols: list[str], vcols: list[str]) -> dict:
    inc = parte[pcols].isna().sum(axis=1) > 0
    vc = float(parte.loc[~inc, vcols].sum().sum())
    vi = float(parte.loc[inc, vcols].sum().sum())
    return {
        "celulas": int(len(parte)),
        "celulas_completas": int((~inc).sum()),
        "pct_incompletas": float(100.0 * inc.mean()),
        "volume_em_completas_pct": float(100.0 * vc / (vc + vi)),
    }


def estrutura_das_incompletas(cel: pd.DataFrame, n: int,
                              corte: int, zona: int) -> dict:
    pcols = [f"preco_{i:02d}" for i in range(n)]
    vcols = [f"volume_{i:02d}" for i in range(n)]
    falta_p = cel[pcols].isna().sum(axis=1)
    falta_v = cel[vcols].isna().sum(axis=1)
    inc = falta_p > 0

    dist = falta_p[inc].value_counts().sort_index()
    treino = cel[cel.week < corte - zona]
    teste = cel[cel.week >= corte]

    return {
        "celulas_com_atividade": int(len(cel)),
        "celulas_completas": int((~inc).sum()),
        "celulas_incompletas": int(inc.sum()),
        # Se isto for verdadeiro, a ausência de um SKU tira a coluna dele do
        # vetor de preços de ENTRADA, e não só o alvo. É o número que decide,
        # porque a entrada é compartilhada pelas 20 linhas da célula nas duas
        # arquiteturas: nenhuma das duas absorve a célula incompleta de graça.
        "buraco_de_preco_igual_ao_de_volume": bool((falta_p == falta_v).all()),
        "distribuicao_skus_faltando": {int(k): int(v) for k, v in dist.items()},
        "pct_incompletas_com_ate_2_faltando":
            float(100.0 * dist.loc[dist.index <= 2].sum() / int(inc.sum())),
        "linhas_sku_celula_recuperaveis": int((n - falta_p[inc]).sum()),
        "ganho_relativo_de_linhas_pct":
            float(100.0 * (n - falta_p[inc]).sum() / ((~inc).sum() * n)),
        "volume_nas_completas": float(cel.loc[~inc, vcols].sum().sum()),
        "volume_nas_incompletas": float(cel.loc[inc, vcols].sum().sum()),
        # A cobertura de volume do TESTE e o numero que vira escopo declarado do
        # resultado principal: o contrafactual so existe onde os N precos sao
        # variaveis de decisao, isto e, nas celulas completas.
        "particao": {
            rotulo: _cobertura(parte, pcols, vcols)
            for rotulo, parte in (("treino", treino), ("teste", teste))
        },
        "pct_incompletas_por_sku":
            {c: float(100.0 * cel[c].isna().mean()) for c in pcols},
        "lojas_com_mais_da_metade_incompleta":
            int((cel.assign(i=inc).groupby("store")["i"].mean() > 0.5).sum()),
        "n_lojas": int(cel.store.nunique()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--saida", default="reports")
    ap.add_argument("--arvores", type=int, default=800)
    ap.add_argument("--semente", type=int, default=0)
    a = ap.parse_args()
    import lightgbm as lgb

    pa = Path(a.painel)
    n = json.loads((pa / f"{a.categoria}_upcs.json").read_text())["n"]
    cel = pd.read_parquet(pa / f"{a.categoria}_celulas.parquet")
    fora = pd.read_parquet(pa / f"{a.categoria}_bem_externo.parquet")

    painel = construir(cel, fora, n)
    treino, teste, _, _, meta = preparar_treino_teste(painel)
    colunas = atributos(n)

    modelo = lgb.LGBMRegressor(
        objective="poisson", n_estimators=a.arvores, learning_rate=0.05,
        num_leaves=127, min_child_samples=40, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, random_state=a.semente,
        verbose=-1)
    modelo.fit(treino[colunas], treino["alvo"],
               categorical_feature=["sku", "store"])

    saida = {"categoria": a.categoria, "n_sortimento": n,
             "particao": meta, "semente": a.semente,
             "modelo": {"tipo": "LightGBM", "objetivo": "poisson",
                        "n_estimators": a.arvores, "num_leaves": 127},
             "correlacao_intra_celula": {}}

    for rotulo, parte in (("treino", treino), ("teste", teste)):
        mu = np.maximum(modelo.predict(parte[colunas]), 1e-9)
        y = parte["alvo"].to_numpy(float)
        grupo = (parte["store"].astype(str) + "_" + parte["week"].astype(str)).to_numpy()
        saida["correlacao_intra_celula"][rotulo] = {
            "cru": icc_balanceado(y - mu, grupo),
            "pearson": icc_balanceado((y - mu) / np.sqrt(mu), grupo),
            "alvo_log1p": icc_balanceado(np.log1p(y), grupo),
        }

    saida["celulas_incompletas"] = estrutura_das_incompletas(
        cel, n, meta["semana_de_corte"], meta["zona_morta_semanas"])

    destino = Path(a.saida) / f"unidade_de_observacao_{a.categoria}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))

    print(f"partição: {meta['n_treino']} linhas de treino, {meta['n_teste']} de teste")
    print(f"\ncorrelação intra-célula, m = {n}:")
    print(f"  {'partição':9} {'resíduo':11} {'rho':>8} {'deff':>7} "
          f"{'n efetivo':>11} {'x células':>10} {'/ linhas':>9}")
    for rot, bloco in saida["correlacao_intra_celula"].items():
        for tipo, g in bloco.items():
            print(f"  {rot:9} {tipo:11} {g['rho']:8.4f} {g['efeito_de_desenho']:7.2f} "
                  f"{g['n_efetivo']:11.0f} {g['razao_n_efetivo_sobre_n_celulas']:10.2f} "
                  f"{g['razao_n_efetivo_sobre_n_linhas']:9.2f}")
    i = saida["celulas_incompletas"]
    print(f"\ncélulas: {i['celulas_completas']} completas, {i['celulas_incompletas']} incompletas")
    print(f"  buraco de preço == buraco de volume: {i['buraco_de_preco_igual_ao_de_volume']}")
    print(f"  {i['pct_incompletas_com_ate_2_faltando']:.1f}% das incompletas têm 1 ou 2 SKUs faltando")
    print(f"  recuperáveis: {i['linhas_sku_celula_recuperaveis']} linhas, "
          f"+{i['ganho_relativo_de_linhas_pct']:.1f}%")
    for rot in ("treino", "teste"):
        g = i["particao"][rot]
        print(f"  {rot}: {g['celulas']} células, {g['pct_incompletas']:.1f}% incompletas, "
              f"{g['volume_em_completas_pct']:.1f}% do volume em células completas")


if __name__ == "__main__":
    main()
