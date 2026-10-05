"""D16, passo 1: alinhar o estimando de D18 ao objeto que a rede reporta.

## Por que existe

A rede reporta a **mediana**, sobre (célula de teste, SKU), da elasticidade
própria pontual `εᵢᵢ = −mᵢ′(uᵢ)`, com os outros 19 preços **fixos**. D18 é **um
coeficiente** de 2SLS, com o custo próprio como instrumento, efeitos fixos de
série e de semana, **sem** os preços dos outros SKUs, sobre as 396 semanas e
todas as séries com volume e custo válidos. Comparar −1,911 com −1,757 antes de
alinhar compara objetos diferentes.

Este script move D18, **um eixo por vez**, em direção ao objeto da rede. O lado
da rede (regressão-espelho) é o passo seguinte, e não está aqui.

## O eixo que motivou o script: preços cruzados omitidos

Com `ln Vᵢ = εᵢᵢ uᵢ + Σ γᵢⱼ uⱼ + ...` e só `uᵢ` na regressão, o 2SLS converge para

    β = εᵢᵢ + Σⱼ γᵢⱼ · δⱼ,     δⱼ = Cov(uⱼ, zᵢ) / Cov(uᵢ, zᵢ)

depois dos efeitos fixos, com `zᵢ` o custo próprio. O viés depende de quanto o
**instrumento** acompanha os outros preços, e não de quanto os preços se
acompanham entre si. Custo com insumo comum é o caso em que `δⱼ ≠ 0`.

## A retratação de 17/09/2026: efeito fixo de semana POR SKU mata o instrumento

A primeira versão estimava E3 a E5 com uma regressão por SKU, efeitos fixos de
loja e semana. Com um SKU só, "semana" é **SKU × semana**, e o custo de um SKU
numa semana é quase o mesmo em todas as lojas (compra centralizada). O efeito
fixo absorvia o instrumento: R² parcial abaixo de 1% em 13 de 20 SKUs e
Cragg-Donald de 0,048. Ela trocava **dois** eixos de uma vez (estatística e
efeitos fixos) e só declarava um.

A versão atual estima as vinte inclinações **num modelo empilhado**, com os
efeitos fixos de série e de semana **comuns** aos SKUs, exatamente como D18. E
mede a hipótese em vez de supô-la: `variancia_custo_sku_semana` é a fração da
variação intra-série do log do custo explicada por médias de SKU × semana.

## As especificações, na ordem dos eixos

    E0  D18 como está                          reproduz −1,757
    E1  E0 restrita às células completas      eixo: amostra
    E2  E1 restrita à janela de teste          eixo: janela
    E3  E1 com custo válido nos N, 20 inclinações próprias      eixo: estatística
    E4  E3 mais a média dos outros 19 preços, inclinação por SKU, exógena

## Por que NÃO há E5 (média dos outros instrumentada pela média dos outros custos)

Com `c̄ₒᵤₜᵣₒₛ = (Sₛₜ − cᵢ)/(N−1)`, vale `(N−1)·Σₖ c̄ₒᵤₜᵣₒₛ·1[i=k] + Σₖ cᵢ·1[i=k] = Sₛₜ`,
a soma dos N custos da célula. Se o custo não varia entre lojas, `Sₛₜ` só varia
na semana e o efeito fixo de semana o absorve: os 2N instrumentos têm posto
menor que 2N para 2N endógenos, e o modelo **não é identificado**. Com custo quase
invariante entre lojas, é quase não identificado. O teste
`test_media_dos_outros_custos_perde_um_posto` planta isso. Identificar as
cruzadas pelo custo exigiria os N custos interagidos com o SKU (N² colunas),
e fica registrado como pendência, não como especificação.

Uso:
    python3 src/experiments/alinhamento_d16.py frj
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sortimento import carregar_sortimento, extrair_longo            # noqa: E402
from elasticidades import (centralizar, ols, projetar,  # noqa: E402
                           sanduiche, tsls, zip_unico)

FRAC_TESTE, ZONA_MORTA = 0.2, 8                      # D20, a mesma regra da rede
CRUZADOS = ("sem", "exogenos")
BLOCO_COLUNAS = 8                                    # memória da centralização


# ------------------------------------------------------------------ dados --
def matrizes(longo: pd.DataFrame, upcs: list[int]):
    """(store, week) x N: preço, custo e volume, colunas na ordem do sortimento.

    `pivot_table` descarta a linha (store, week) em que TODOS os valores são
    nan. Custo inválido nos N numa célula some da matriz de custo e fica na de
    preço. O índice de preço é o de referência, porque presença é preço (D2).
    """
    def piv(col):
        return longo.pivot_table(index=["store", "week"], columns="upc",
                                 values=col, aggfunc="first").reindex(columns=upcs)
    P = piv("preco")
    return P, piv("custo").reindex(P.index), piv("volume").reindex(P.index)


def corte_de_teste(semanas: np.ndarray) -> int:
    s = np.sort(np.unique(semanas))
    return int(s[int(len(s) * (1 - FRAC_TESTE))])


def centralizar_em_blocos(M: np.ndarray, chaves: list[np.ndarray]) -> np.ndarray:
    """A centralização é linear coluna a coluna: blocos dão o mesmo resultado."""
    return np.column_stack([centralizar(M[:, j:j + BLOCO_COLUNAS], chaves)
                            for j in range(0, M.shape[1], BLOCO_COLUNAS)])


# -------------------------------------------------------------- estimação --
def _resumo(b, se_iid, ag, rotulos):
    return {r: {"coef": float(b[i]), "ep_iid": float(se_iid[i]),
                "ep_agrupado": float(ag["ep"][i])} for i, r in enumerate(rotulos)}


def _r2_parcial(x, Z, j, k):
    """R² parcial da coluna `j` de Z no primeiro estágio de `x`."""
    _, _, r_cheio = ols(x, Z, k)
    if Z.shape[1] == 1:
        r_sem = x
    else:
        _, _, r_sem = ols(x, np.delete(Z, j, axis=1), k)
    return 1.0 - float(r_cheio @ r_cheio) / float(r_sem @ r_sem)


def filtrar_validas(longo: pd.DataFrame) -> pd.DataFrame:
    return longo[(longo.volume > 0) & longo.custo.notna() & (longo.custo > 0)]


def estimar_empilhado(longo: pd.DataFrame, mascara: np.ndarray) -> dict:
    """E0 a E2: a especificação de D18 (`estimar_serie`), numa subamostra."""
    d = filtrar_validas(longo[mascara])
    serie = d.store.astype(np.int64).to_numpy() * 10**12 + d.upc.to_numpy()
    M = centralizar(np.column_stack([np.log(d.volume), np.log(d.preco),
                                     np.log(d.custo)]),
                    [serie, d.week.to_numpy()])
    k = len(np.unique(serie)) + d.week.nunique() + 1
    y, X, Z = M[:, 0], M[:, 1:2], M[:, 2:3]
    b_ols, se_ols, r_ols = ols(y, X, k)
    b_iv, se_iv, r_iv = tsls(y, X, Z, k)
    g = {"serie": serie}
    return {"n": int(len(d)), "n_series": int(len(np.unique(serie))),
            "n_semanas": int(d.week.nunique()),
            "r2_parcial_custo": _r2_parcial(X[:, 0], Z, 0, k),
            "ols": _resumo(b_ols, se_ols, sanduiche(X, r_ols, g, k)["serie"], ["propria"]),
            "iv": _resumo(b_iv, se_iv, sanduiche(projetar(Z, X), r_iv, g, k)["serie"],
                          ["propria"])}


def variancia_custo_sku_semana(longo: pd.DataFrame) -> dict:
    """Fração da variação intra-série de ln custo e ln preço explicada por SKU × semana.

    Perto de 1 quer dizer que o regressor varia quase só no tempo e igual em
    todas as lojas, e então um efeito fixo de SKU × semana o absorve inteiro.
    """
    d = filtrar_validas(longo)
    serie = d.store.astype(np.int64).to_numpy() * 10**12 + d.upc.to_numpy()
    out = {}
    for nome, col in (("ln_custo", d.custo), ("ln_preco", d.preco)):
        r = centralizar(np.log(col.to_numpy())[:, None], [serie])[:, 0]
        m = pd.Series(r).groupby([d.upc.to_numpy(), d.week.to_numpy()]).transform("mean")
        out[nome] = float((m.to_numpy() @ m.to_numpy()) / (r @ r))
    return out


def estimar_inclinacoes(longo: pd.DataFrame, P: pd.DataFrame, C: pd.DataFrame,
                        upcs: list[int], cruzados: str) -> dict:
    """E3 a E5: vinte inclinações num modelo empilhado com os efeitos fixos de D18.

    `ln Vᵢ = Σₖ 1[i=k]·(βₖ uᵢ + θₖ ūₒᵤₜᵣₒₛ) + série + semana`, com `ūₒᵤₜᵣₒₛ` a
    média de ln preço dos outros N−1 na célula. A amostra é a mesma nas três
    versões (custo válido nos N), para que só o conjunto de regressores mude.
    """
    if cruzados not in CRUZADOS:
        raise ValueError(cruzados)
    n = len(upcs)
    celulas = P.index[(P.notna().all(axis=1) & (C > 0).all(axis=1)).to_numpy()]
    d = filtrar_validas(longo)
    d = d[pd.MultiIndex.from_arrays([d.store, d.week]).isin(celulas)
          & d.upc.isin(upcs)]
    chave = pd.MultiIndex.from_arrays([d.store, d.week])
    LP = np.log(P.loc[chave].to_numpy())
    LC = np.log(C.loc[chave].to_numpy())
    pos = d.upc.map({u: i for i, u in enumerate(upcs)}).to_numpy()
    linhas = np.arange(len(d))
    # o preço e o custo próprios vêm da matriz, para bater com os dos outros
    p_prop, c_prop = LP[linhas, pos], LC[linhas, pos]
    p_out = (LP.sum(axis=1) - p_prop) / (n - 1)
    D = np.eye(n)[pos]

    Xp, Zp = p_prop[:, None] * D, c_prop[:, None] * D
    if cruzados == "sem":
        X, Z = Xp, Zp
    else:
        X = np.column_stack([Xp, p_out[:, None] * D])
        Z = np.column_stack([Zp, p_out[:, None] * D])

    serie = d.store.astype(np.int64).to_numpy() * 10**12 + d.upc.to_numpy()
    semana = d.week.to_numpy()
    kx = X.shape[1]
    M = centralizar_em_blocos(np.column_stack([np.log(d.volume.to_numpy()), X, Z]),
                              [serie, semana])
    y, X, Z = M[:, 0], M[:, 1:1 + kx], M[:, 1 + kx:]
    k = len(np.unique(serie)) + len(np.unique(semana)) + kx

    b_ols, se_ols, r_ols = ols(y, X, k)
    b_iv, se_iv, r_iv = tsls(y, X, Z, k)
    g = {"serie": serie}
    se_ag_ols = sanduiche(X, r_ols, g, k)["serie"]["ep"]
    se_ag_iv = sanduiche(projetar(Z, X), r_iv, g, k)["serie"]["ep"]

    vol = pd.Series(d.volume.to_numpy()).groupby(pos).sum()
    por_sku = []
    for i in range(n):
        s = {"sku": i, "upc": int(upcs[i]), "n": int((pos == i).sum()),
             "volume_total": float(vol.get(i, 0.0)),
             "r2_parcial_custo_proprio": _r2_parcial(X[:, i], Z, i, k),
             "ols": {"propria": {"coef": float(b_ols[i]), "ep_agrupado": float(se_ag_ols[i])}},
             "iv": {"propria": {"coef": float(b_iv[i]), "ep_agrupado": float(se_ag_iv[i])}}}
        if cruzados != "sem":
            s["ols"]["media_outros"] = {"coef": float(b_ols[n + i]),
                                        "ep_agrupado": float(se_ag_ols[n + i])}
            s["iv"]["media_outros"] = {"coef": float(b_iv[n + i]),
                                       "ep_agrupado": float(se_ag_iv[n + i])}
        por_sku.append(s)

    saida = {"cruzados": cruzados, "n": int(len(d)), "n_celulas": int(len(celulas)),
             "resumo": resumir(por_sku), "por_sku": por_sku}
    return saida


def resumir(por_sku: list[dict]) -> dict:
    """Os três agregados que separam o eixo 'estatística'."""
    out = {}
    for est in ("ols", "iv"):
        b = np.array([s[est]["propria"]["coef"] for s in por_sku])
        w = np.array([s["volume_total"] for s in por_sku])
        out[est] = {"mediana": float(np.median(b)), "media": float(b.mean()),
                    "media_ponderada_volume": float((b * w).sum() / w.sum()),
                    "min": float(b.min()), "max": float(b.max())}
    return out


# ------------------------------------------------------------------ main --
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--raiz", default="data/raw")
    ap.add_argument("--painel", default="data/interim/painel")
    a = ap.parse_args()
    pa = Path(a.painel)

    upcs = carregar_sortimento(a.categoria, pa)
    longo = extrair_longo(zip_unico(Path(a.raiz) / a.categoria), set(upcs))
    P, C, V = matrizes(longo, upcs)

    completas = P.index[P.notna().all(axis=1)]
    em_completa = pd.MultiIndex.from_arrays([longo.store, longo.week]).isin(completas)
    corte = corte_de_teste(longo.week.to_numpy())

    ref = json.loads((pa / f"{a.categoria}_elasticidades.json").read_text())
    ref = ref.get("estimativas", ref)["serie"]["iv"]["ln_preco_proprio"]["coef"]

    saida = {"corte_teste": corte, "zona_morta": ZONA_MORTA, "n_skus": len(upcs),
             "n_celulas_completas": int(len(completas)), "d18_registrado": ref,
             "variancia_sku_semana": variancia_custo_sku_semana(longo)}
    print("fração intra-série explicada por SKU x semana:",
          json.dumps(saida["variancia_sku_semana"]), flush=True)

    saida["E0_d18"] = estimar_empilhado(longo, np.ones(len(longo), bool))
    saida["E0_d18"]["reproduz_d18"] = bool(
        abs(saida["E0_d18"]["iv"]["propria"]["coef"] - ref) < 1e-6)
    saida["E1_completas"] = estimar_empilhado(longo, em_completa)
    saida["E2_completas_teste"] = estimar_empilhado(
        longo, em_completa & (longo.week.to_numpy() >= corte))
    for e in ("E0_d18", "E1_completas", "E2_completas_teste"):
        x = saida[e]
        print(e, x["n"], "iv", round(x["iv"]["propria"]["coef"], 4),
              "ep", round(x["iv"]["propria"]["ep_agrupado"], 4),
              "ols", round(x["ols"]["propria"]["coef"], 4),
              "r2p", round(x["r2_parcial_custo"], 4), flush=True)

    for rotulo, cruz in (("E3_inclinacoes_sem_cruzados", "sem"),
                         ("E4_inclinacoes_cruzados_exogenos", "exogenos")):
        saida[rotulo] = estimar_inclinacoes(longo, P, C, upcs, cruz)
        r = saida[rotulo]
        r2 = [s["r2_parcial_custo_proprio"] for s in r["por_sku"]]
        print(rotulo, "n", r["n"], "iv", json.dumps(r["resumo"]["iv"]),
              "r2p mediano", round(float(np.median(r2)), 4), flush=True)

    destino = Path("reports") / f"alinhamento_d16_{a.categoria}.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))
    print("reproduz D18:", saida["E0_d18"]["reproduz_d18"], "->", destino)


if __name__ == "__main__":
    main()
