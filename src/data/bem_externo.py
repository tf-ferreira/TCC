"""Bem externo: agregado dos SKUs fora do sortimento (decisão D23).

O sortimento otimizado tem vinte produtos, mas a categoria tem 175. Os 155
restantes continuam na prateleira, com preços que se movem, e continuam
disponíveis ao consumidor. Ignorá-los tem duas consequências:

1. viés de variável omitida na estimação, porque o preço do resto correlaciona
   com o preço dos vinte (calendário promocional comum da categoria);
2. contrafactual otimista, porque sem alternativa externa o modelo só desloca
   demanda **entre** os vinte, e superestima a receita de uma elevação geral
   de preços.

O agregado é construído como **variável de contexto**, não como produto
adicional: ele entra na rede como entrada exógena e permanece fixo durante a
otimização. N continua sendo vinte.

Por que o agregado não reintroduz o problema de completude: ele é definido
mesmo quando membros faltam, porque é um índice sobre quem estiver presente,
com pesos fixos renormalizados. Recupera a informação dos excluídos sem
recuperar a exigência que os excluiu.

Índice de preço: Laspeyres de cesta fixa com pesos renormalizados,

    I_st = sum_{j presente} w_j (p_jst / p_j0) / sum_{j presente} w_j

com p_j0 a mediana do preço unitário de j na janela base e w_j a participação
de j na receita do resto na mesma janela. Relativos de preço são adimensionais,
de modo que a heterogeneidade de tamanho de embalagem não contamina o índice.
A janela base é anterior ao período de teste, para não haver vazamento.

## Robustez a pesos concentrados (critério de invalidação de D23)

D23 registra como condição de invalidação a hipótese de o índice ser dominado
por poucos SKUs do resto com peso alto e comportamento atípico. A verificação
é `--truncar-peso 95`: os pesos acima do percentil 95 são achatados nesse
valor e a cesta é renormalizada, de modo que nenhum SKU possa responder por
mais do que o percentil 95 da distribuição original. Se o coeficiente do
índice não se mover, a concentração não estava governando o resultado.

A saída ganha sufixo (`_p95`), para que o artefato principal não seja
sobrescrito e as duas versões possam ser comparadas lado a lado.

Uso:
    python3 src/data/bem_externo.py frj --janela-base 104
    python3 src/data/bem_externo.py frj --truncar-peso 95
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from sortimento import extrair_longo  # noqa: E402

COLS = ["idx_preco_resto", "idx_custo_resto", "vol_resto", "promo_resto", "n_resto"]


def cesta_base(resto: pd.DataFrame, ate_semana: int,
               truncar_peso: float | None = None) -> pd.DataFrame:
    """Preço de referência e peso de cada SKU do resto, na janela base.

    `truncar_peso` é um percentil: pesos acima dele são achatados no valor do
    percentil e a cesta é renormalizada. Serve à checagem de robustez de D23,
    não à construção principal.
    """
    base = resto[resto.week < ate_semana]
    g = base.groupby("upc")
    ref = pd.DataFrame({
        "p0": g.preco.median(),
        "c0": g.custo.median(),
        "receita": (base.preco * base.volume).groupby(base.upc).sum(),
        "n_obs": g.size(),
    })
    ref = ref[(ref.p0 > 0) & (ref.n_obs >= 30)]
    ref["w"] = ref.receita / ref.receita.sum()
    if truncar_peso is not None:
        teto = float(np.percentile(ref.w, truncar_peso))
        ref["w"] = np.minimum(ref.w, teto)
        ref["w"] = ref.w / ref.w.sum()
    return ref[["p0", "c0", "w"]]


def agregar(resto: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Índice de preço, volume, promoção e contagem do resto, por célula."""
    d = resto.merge(ref, left_on="upc", right_index=True, how="inner")
    d["rel_w"] = (d.preco / d.p0) * d.w
    # O índice de custo usa os mesmos pesos e a mesma base, para que preço e
    # custo do resto sejam comparáveis termo a termo. Onde o custo é ausente
    # (margem corrompida, D9), o SKU sai apenas do índice de custo.
    valido = d.custo.notna() & (d.c0 > 0)
    d["rel_w_custo"] = np.where(valido, (d.custo / d.c0) * d.w, np.nan)
    d["w_custo"] = np.where(valido, d.w, np.nan)
    g = d.groupby(["store", "week"])
    fora = pd.DataFrame({
        "soma_rel_w": g.rel_w.sum(),
        "soma_w": g.w.sum(),
        "soma_rel_w_custo": g.rel_w_custo.sum(),
        "soma_w_custo": g.w_custo.sum(),
        "vol_resto": g.volume.sum(),
        "promo_resto": g.promo.mean(),
        "n_resto": g.size(),
    })
    fora["idx_preco_resto"] = fora.soma_rel_w / fora.soma_w
    fora["idx_custo_resto"] = fora.soma_rel_w_custo / fora.soma_w_custo
    return fora[COLS].reset_index()


def _centralizar(M: np.ndarray, chaves: list[np.ndarray], it: int = 15) -> np.ndarray:
    """Absorve efeitos fixos por centralização alternada (FWL iterado)."""
    M = M.astype(float).copy()
    for _ in range(it):
        for k in chaves:
            df = pd.DataFrame(M)
            M = (df - df.groupby(k).transform("mean")).to_numpy()
    return M


def teste_colinearidade(celulas: pd.DataFrame, fora: pd.DataFrame, n: int) -> dict:
    """R² do índice do resto explicado pelos N preços do sortimento.

    R² alto significa que o agregado quase não carrega variação independente,
    e portanto acrescenta pouco. R² moderado significa o contrário.
    """
    precos = [f"preco_{i:02d}" for i in range(n)]
    d = celulas.merge(fora, on=["store", "week"], how="inner")
    d = d.dropna(subset=precos + ["idx_preco_resto"])
    d = d[(d[precos] > 0).all(axis=1) & (d.idx_preco_resto > 0)]

    y = np.log(d.idx_preco_resto.to_numpy())
    X = np.log(d[precos].to_numpy())

    def r2(yy, XX):
        XX = np.column_stack([np.ones(len(XX)), XX])
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        res = yy - XX @ beta
        return 1.0 - float(res @ res) / float(((yy - yy.mean()) ** 2).sum())

    bruto = r2(y, X)
    chaves = [d.store.to_numpy(), d.week.to_numpy()]
    M = _centralizar(np.column_stack([y, X]), chaves)
    yc, Xc = M[:, 0], M[:, 1:]
    dentro = r2(yc, Xc)

    media = np.log(np.exp(X).mean(axis=1))
    mc = _centralizar(np.column_stack([y, media]), chaves)
    simples = r2(mc[:, 0], mc[:, 1:])

    # Variação que sobra depois de absorver efeitos fixos e os N preços.
    XX = np.column_stack([np.ones(len(Xc)), Xc])
    beta, *_ = np.linalg.lstsq(XX, yc, rcond=None)
    resid = yc - XX @ beta

    return {
        "n_celulas_teste": int(len(d)),
        "r2_sem_efeitos_fixos": bruto,
        "r2_com_efeitos_fixos": dentro,
        "r2_apenas_preco_medio_do_sortimento": simples,
        "variacao_independente_pct": 100.0 * (1.0 - dentro),
        "vif": 1.0 / (1.0 - dentro),
        "dp_ln_idx_intra_pct": 100.0 * float(yc.std(ddof=1)),
        "dp_residual_pct": 100.0 * float(resid.std(ddof=1)),
        "corr_idx_preco_medio_sortimento": float(np.corrcoef(yc, mc[:, 1])[0, 1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("categoria")
    ap.add_argument("--janela-base", type=int, default=104)
    ap.add_argument("--truncar-peso", type=float, default=None,
                    help="percentil em que achatar os pesos da cesta (ex.: 95)")
    ap.add_argument("--raiz", default="data/raw")
    ap.add_argument("--painel", default="data/interim/painel")
    a = ap.parse_args()

    meta = json.loads((Path(a.painel) / f"{a.categoria}_upcs.json").read_text())
    sortimento = set(meta["ordem"])
    n = meta["n"]
    celulas = pd.read_parquet(Path(a.painel) / f"{a.categoria}_celulas.parquet")

    todos = pd.read_csv(Path("data/interim/screening") /
                        f"{a.categoria}_cobertura.csv").upc.tolist()
    zips = sorted((Path(a.raiz) / a.categoria).glob("w*.zip"))
    longo = extrair_longo(zips[0], set(todos))
    resto = longo[~longo.upc.isin(sortimento)]

    ref = cesta_base(resto, a.janela_base, a.truncar_peso)
    fora = agregar(resto, ref)
    sufixo = "" if a.truncar_peso is None else f"_p{int(a.truncar_peso)}"

    vol_tot = float(longo.volume.sum())
    diag = {
        "categoria": a.categoria,
        "n_sortimento": n,
        "n_resto": int(resto.upc.nunique()),
        "n_resto_na_cesta": int(len(ref)),
        "receita_cesta_pct_do_resto": 100.0 * float(
            (resto[resto.upc.isin(ref.index)].preco
             * resto[resto.upc.isin(ref.index)].volume).sum()
            / (resto.preco * resto.volume).sum()),
        "janela_base_semanas": a.janela_base,
        "truncar_peso_percentil": a.truncar_peso,
        "peso_maximo": float(ref.w.max()),
        "peso_p95": float(np.percentile(ref.w, 95)),
        "concentracao_top5_pct": 100.0 * float(ref.w.nlargest(5).sum()),
        "volume_resto_pct_categoria": 100.0 * float(resto.volume.sum()) / vol_tot,
        "celulas_com_resto": int(len(fora)),
        "celulas_painel": int(len(celulas)),
        "cobertura_do_agregado_pct": 100.0 * float(
            celulas.merge(fora, on=["store", "week"], how="left").idx_preco_resto.notna().mean()),
        "idx_preco_p05": float(fora.idx_preco_resto.quantile(0.05)),
        "idx_preco_mediana": float(fora.idx_preco_resto.median()),
        "idx_preco_p95": float(fora.idx_preco_resto.quantile(0.95)),
        "n_resto_mediana": float(fora.n_resto.median()),
        **teste_colinearidade(celulas, fora, n),
    }

    saida = Path(a.painel)
    fora.to_parquet(saida / f"{a.categoria}_bem_externo{sufixo}.parquet", index=False)
    (saida / f"{a.categoria}_bem_externo{sufixo}_diagnostico.json").write_text(
        json.dumps(diag, indent=2))
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
