"""D16, passo 2: a regressão-espelho.

## Por que existe

O passo 1 (`alinhamento_d16.py`) tentou mover D18 até o objeto da rede e mostrou
que dois eixos não se movem desse lado: na janela de teste o instrumento
enfraquece (R² parcial 0,048), e condicionar nos preços dos outros SKUs derruba a
precisão por colinearidade. Então a comparação vai na direção oposta: **o
estimador de D18 é aplicado às previsões da rede**, nas mesmas linhas.

    β_obs = 2SLS(ln V   ~ ln p | ln c ; FE série, semana)
    β_esp = 2SLS(ln V̂  ~ ln p | ln c ; FE série, semana)

Mesmo estimador e mesma amostra dos dois lados. O termo `γ·δ` do cruzado omitido
entra nos dois por construção, e a pergunta passa a ser uma só: **a superfície
da rede, lida pelo instrumento, responde a preço como o dado responde?**

## A diferença sai com erro padrão próprio

2SLS é linear em `y`, então `β_obs − β_esp = 2SLS(ln V − ln V̂ ~ ...)`, exatamente.
Estimar a diferença direto dá o erro padrão agrupado **da diferença**, que
aproveita a correlação entre os dois lados na mesma linha. Somar os dois erros
padrão como se fossem independentes superestimaria a incerteza.

## Sementes (D26)

A rede é treinada em 50 sementes no ponto adotado (`varredura_rede.MELHOR`).
Cada semente dá um `β_esp` e uma diferença. A incerteza da média declarada é

    ep_total = √( média(ep_dif²) + var_sementes / S )

o primeiro termo é amostral (o dado), o segundo é de otimização (a rede).

## O que NÃO se lê daqui

Concordância nas linhas de **treino** pode ser ajuste: a rede viu `ln V`
naquelas linhas. Por isso as três amostras saem separadas, `treino`, `teste` e
`ambos`, com o R² parcial de cada uma. No teste o instrumento é fraco (passo 1),
e a leitura principal fica em `ambos`, com essa ressalva escrita.

Uso:
    python3 src/experiments/espelho_d16.py frj [--sementes 50]
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
for pasta in ("data", "models", "experiments"):
    sys.path.insert(0, str(RAIZ / "src" / pasta))
from elasticidades import centralizar, ols, projetar, sanduiche, tsls  # noqa: E402

BLOCO_COLUNAS = 8


# -------------------------------------------------------------- estimação --
def _centralizar_blocos(M, chaves):
    return np.column_stack([centralizar(M[:, j:j + BLOCO_COLUNAS], chaves)
                            for j in range(0, M.shape[1], BLOCO_COLUNAS)])


def comparar_espelho(y_obs, Yhat, lnp, lnc, serie, semana, loja) -> dict:
    """β_obs, e para cada coluna de `Yhat`: β_esp, diferença e ep agrupado da diferença.

    `Yhat` é (linhas, S), uma coluna de `ln V̂` por semente.
    """
    Yhat = np.atleast_2d(np.asarray(Yhat, float).T).T
    S = Yhat.shape[1]
    M = _centralizar_blocos(np.column_stack([lnp, lnc, y_obs, Yhat]), [serie, semana])
    X, Z, y = M[:, 0:1], M[:, 1:2], M[:, 2]
    H = M[:, 3:]
    k = len(np.unique(serie)) + len(np.unique(semana)) + 1
    grupos = {"serie": serie, "loja": loja}
    R = projetar(Z, X)

    def iv(v):
        b, se, r = tsls(v, X, Z, k)
        ag = sanduiche(R, r, grupos, k)
        return {"coef": float(b[0]), "ep_iid": float(se[0]),
                "ep_serie": float(ag["serie"]["ep"][0]),
                "ep_loja": float(ag["loja"]["ep"][0])}

    _, _, r1 = ols(X[:, 0], Z, k)
    r2p = 1.0 - float(r1 @ r1) / float(X[:, 0] @ X[:, 0])
    saida = {"n": int(len(y)), "r2_parcial_custo": r2p, "obs": iv(y),
             "por_semente": []}
    for s in range(S):
        esp, dif = iv(H[:, s]), iv(y - H[:, s])
        saida["por_semente"].append({"espelho": esp, "diferenca": dif})

    b = np.array([p["espelho"]["coef"] for p in saida["por_semente"]])
    d = np.array([p["diferenca"]["coef"] for p in saida["por_semente"]])
    resumo = {"espelho_media": float(b.mean()),
              "diferenca_media": float(d.mean())}
    for g in ("serie", "loja"):
        e2 = np.array([p["diferenca"][f"ep_{g}"] ** 2 for p in saida["por_semente"]])
        var_s = float(d.var(ddof=1)) if S > 1 else 0.0
        ep = float(np.sqrt(e2.mean() + var_s / S))
        resumo[f"ep_total_{g}"] = ep
        resumo[f"z_{g}"] = float(d.mean() / ep) if ep > 0 else float("nan")
    resumo["dp_espelho_entre_sementes"] = float(b.std(ddof=1)) if S > 1 else 0.0
    saida["resumo"] = resumo
    return saida


# ------------------------------------------------------------------ dados --
def linhas(parte: dict, cel: pd.DataFrame) -> dict:
    """De uma parte pivotada da rede (células × N) para linhas (célula, SKU).

    O custo não está no pivô da rede: vem do painel de células, pela chave. As
    linhas com volume 0 ou custo inválido saem, como em D18, e a máscara fica
    para aplicar às previsões.
    """
    ch = parte["chaves"][["store", "week"]]
    n = parte["precos"].shape[1]
    custo = ch.merge(cel[["store", "week"] + [f"custo_{i:02d}" for i in range(n)]],
                     on=["store", "week"], how="left")
    if len(custo) != len(ch):
        raise ValueError("chave duplicada no painel de células")
    C = custo[[f"custo_{i:02d}" for i in range(n)]].to_numpy(float)
    V, P = parte["alvo"], parte["precos"]
    loja = np.repeat(ch.store.to_numpy(), n)
    semana = np.repeat(ch.week.to_numpy(), n)
    sku = np.tile(np.arange(n), len(ch))
    ok = (V.ravel() > 0) & np.isfinite(C.ravel()) & (C.ravel() > 0)
    return {"mascara": ok, "y": np.log(V.ravel()[ok]), "lnp": np.log(P.ravel()[ok]),
            "lnc": np.log(C.ravel()[ok]), "loja": loja[ok], "semana": semana[ok],
            "serie": (loja * 100 + sku)[ok]}


def juntar(a: dict, b: dict) -> dict:
    return {k: np.concatenate([a[k], b[k]]) for k in a if k != "mascara"}


def impressao_digital(cfg: dict) -> str:
    """Configuração mais versão do pipeline (pendência 9.6, 17/09/2026)."""
    from atributos import VERSAO_PIPELINE
    return hashlib.sha1(json.dumps([cfg, VERSAO_PIPELINE], sort_keys=True)
                        .encode()).hexdigest()[:12]


# ------------------------------------------------------------------ main --
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--sementes", type=int, default=50)
    ap.add_argument("--fragmentos", default="data/interim/espelho_d16")
    a = ap.parse_args()

    import rede
    from varredura_rede import MELHOR

    pa = Path(a.painel)
    dados = rede.preparar(a.categoria, pa)
    cel = pd.read_parquet(pa / f"{a.categoria}_celulas.parquet")
    L = {"treino": linhas(dados["treino"], cel), "teste": linhas(dados["teste"], cel)}

    base = dict(MELHOR)
    base.setdefault("gama_contextual", False)
    pasta = Path(a.fragmentos)
    pasta.mkdir(parents=True, exist_ok=True)

    prev = {"treino": [], "teste": []}
    for s in range(a.sementes):
        cfg = dict(base, semente=s)
        dig = impressao_digital(cfg)
        arq = pasta / f"semente_{s:02d}_{dig}.npz"
        if arq.exists():
            f = np.load(arq)
            zt, ze = f["treino"], f["teste"]
        else:
            out = rede.treinar(dados, cfg)
            zt = np.log(out["pred_treino"].ravel()).astype("float64")
            ze = np.log(out["pred_teste"].ravel()).astype("float64")
            np.savez(arq, treino=zt, teste=ze)
            # Os três termos aditivos saem da MESMA rede e vão para onde
            # `decomposicao_espelho_d16.py` os procura: sem isso ela retreinava
            # as 50 redes só para lê-las por dentro.
            from decomposicao_espelho_d16 import CANAIS, componentes
            T = out["tensores"]
            comp = {"treino": componentes(out["modelo"], T["ctx_treino"],
                                          T["u_treino"], T["loja_treino"]),
                    "teste": componentes(out["modelo"], T["ctx_teste"],
                                         T["u_teste"], T["loja_teste"])}
            (pasta / "componentes").mkdir(exist_ok=True)
            np.savez(pasta / "componentes" / arq.name,
                     **{f"{p}_{c}": comp[p][c] for p in comp for c in CANAIS})
        prev["treino"].append(zt[L["treino"]["mascara"]])
        prev["teste"].append(ze[L["teste"]["mascara"]])
        print(f"semente {s:02d} pronta", flush=True)

    saida = {"categoria": a.categoria, "configuracao": base,
             "impressao_digital_base": impressao_digital(base),
             "n_sementes": a.sementes}
    amostras = {
        "treino": (L["treino"], np.column_stack(prev["treino"])),
        "teste": (L["teste"], np.column_stack(prev["teste"])),
        "ambos": (juntar(L["treino"], L["teste"]),
                  np.vstack([np.column_stack(prev["treino"]),
                             np.column_stack(prev["teste"])])),
    }
    for nome, (d, H) in amostras.items():
        r = comparar_espelho(d["y"], H, d["lnp"], d["lnc"], d["serie"],
                             d["semana"], d["loja"])
        saida[nome] = r
        print(nome, "n", r["n"], "r2p", round(r["r2_parcial_custo"], 4),
              "obs", round(r["obs"]["coef"], 4), "ep", round(r["obs"]["ep_serie"], 4),
              json.dumps({k: round(v, 4) for k, v in r["resumo"].items()}), flush=True)

    destino = RAIZ / "reports" / f"espelho_d16_{a.categoria}.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))
    print("->", destino)


if __name__ == "__main__":
    main()
