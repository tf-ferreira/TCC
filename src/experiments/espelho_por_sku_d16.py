"""Item 7, pendência 9.2: a regressão-espelho SKU a SKU.

## Por que existe

O espelho agregado (−0,125 no treino) é uma média sobre 20 produtos cuja
elasticidade medida pelo instrumento vai de −0,16 a −4,83 (E3 de
`alinhamento_d16.py`). Uma média pequena pode esconder SKUs em que a rede erra
muito para os dois lados. Este script faz o espelho com **20 inclinações num
modelo empilhado**, com os efeitos fixos de série e de semana **comuns**, que é a
especificação de E3 (a por-SKU com efeito fixo de semana próprio mata o
instrumento, retratação de 17/09 em `alinhamento_d16.py`).

## O que sai por SKU

- `obs`, `espelho` (média das 50 sementes) e a diferença, estimada direto sobre
  `ln V − ln V̂` (2SLS é linear em y), com erro padrão agrupado por série da
  diferença mais a variação entre sementes;
- a decomposição do espelho nos três canais, exata, sobre a média das sementes;
- a força do instrumento do SKU: R² parcial e F de primeiro estágio
  **agrupado**, `(b/ep)²` do custo próprio. Abaixo de 10, o SKU é marcado como
  fraco e fica fora das contagens.

## O limiar de leitura

Com 20 SKUs testados, o limiar bicaudal a 5% corrigido por Bonferroni é
`z = 2,807`. As contagens saem pelos dois limiares, 1,96 e 2,807. Só a janela de
treino é usada: a de teste não valida derivadas (emenda de D16, seção 4).

Uso:
    python3 src/experiments/espelho_por_sku_d16.py frj
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
for pasta in ("data", "models", "experiments"):
    sys.path.insert(0, str(RAIZ / "src" / pasta))
from elasticidades import centralizar, ols, projetar, sanduiche  # noqa: E402

BLOCO = 8
F_FRACO = 10.0
Z_BONFERRONI_20 = 2.807033768343811


def _centralizar(M, chaves):
    return np.column_stack([centralizar(M[:, j:j + BLOCO], chaves)
                            for j in range(0, M.shape[1], BLOCO)])


def espelho_por_sku(y, H, lnp, lnc, serie, semana, sku, n, canais=None) -> dict:
    """`H` (linhas, S): ln V̂ por semente. `canais`: dict canal -> (linhas,) média.

    Centraliza `H` UMA semente por vez: empilhar tudo antes (20 + 20 + 1 + 50 + 3
    colunas sobre 407 mil linhas, com as cópias do pandas) matou o processo por
    memória numa máquina de 4 GB. O resultado é idêntico, porque a centralização
    é linear coluna a coluna.
    """
    H = np.atleast_2d(np.asarray(H, float).T).T
    S = H.shape[1]
    chaves = [serie, semana]
    D = np.eye(n)[sku]
    X = _centralizar(lnp[:, None] * D, chaves)
    Z = _centralizar(lnc[:, None] * D, chaves)
    del D
    yc = centralizar(y[:, None], chaves)[:, 0]
    k = len(np.unique(serie)) + len(np.unique(semana)) + n
    Xh = projetar(Z, X)
    A_inv = np.linalg.inv(Xh.T @ Xh)
    grupos = {"serie": serie}

    def iv(v):
        b = A_inv @ (Xh.T @ v)
        r = v - X @ b
        return b, sanduiche(Xh, r, grupos, k)["serie"]["ep"]

    b_obs, ep_obs = iv(yc)
    b_esp = np.empty((S, n)); b_dif = np.empty((S, n)); ep_dif = np.empty((S, n))
    for s in range(S):
        hc = centralizar(H[:, s:s + 1], chaves)[:, 0]
        b_esp[s] = A_inv @ (Xh.T @ hc)
        b_dif[s], ep_dif[s] = iv(yc - hc)

    r2p = np.empty(n); F = np.empty(n)
    for i in range(n):
        bi, _, ri = ols(X[:, i], Z, k)
        epi = sanduiche(Z, ri, grupos, k)["serie"]["ep"][i]
        F[i] = (bi[i] / epi) ** 2
        _, _, r_sem = ols(X[:, i], np.delete(Z, i, axis=1), k)
        r2p[i] = 1.0 - float(ri @ ri) / float(r_sem @ r_sem)

    nomes = list(canais) if canais else []
    b_can = {c: A_inv @ (Xh.T @ centralizar(np.asarray(canais[c], float)[:, None],
                                            chaves)[:, 0]) for c in nomes}
    dif = b_dif.mean(0)
    var_s = b_dif.var(0, ddof=1) if S > 1 else np.zeros(n)
    ep_tot = np.sqrt((ep_dif ** 2).mean(0) + var_s / S)
    por = []
    for i in range(n):
        por.append({
            "sku": i, "n_linhas": int((sku == i).sum()),
            "r2_parcial_custo": float(r2p[i]), "F_primeiro_estagio_agrupado": float(F[i]),
            "instrumento_fraco": bool(F[i] < F_FRACO),
            "obs": float(b_obs[i]), "ep_obs": float(ep_obs[i]),
            "espelho": float(b_esp[:, i].mean()),
            "dp_espelho_sementes": float(b_esp[:, i].std(ddof=1)) if S > 1 else 0.0,
            "diferenca": float(dif[i]), "ep_total": float(ep_tot[i]),
            "z": float(dif[i] / ep_tot[i]),
            "canais": {c: float(b_can[c][i]) for c in nomes},
        })
    return {"n_linhas": int(len(y)), "n_sementes": S, "por_sku": por,
            "resumo": resumir(por, sku, n)}


def resumir(por, sku, n) -> dict:
    fortes = [p for p in por if not p["instrumento_fraco"]]
    peso = np.bincount(sku, minlength=n).astype(float)
    out = {"n_fortes": len(fortes), "n_fracos": n - len(fortes),
           "z_bonferroni_20": Z_BONFERRONI_20}
    if fortes:
        z = np.array([p["z"] for p in fortes])
        d = np.array([p["diferenca"] for p in fortes])
        w = np.array([peso[p["sku"]] for p in fortes])
        out.update({
            "fortes_com_z_acima_1_96": int((np.abs(z) > 1.96).sum()),
            "fortes_com_z_acima_bonferroni": int((np.abs(z) > Z_BONFERRONI_20).sum()),
            "fortes_rede_mais_elastica": int((d > 0).sum()),
            "fortes_rede_menos_elastica": int((d < 0).sum()),
            "diferenca_mediana_fortes": float(np.median(d)),
            "diferenca_media_ponderada_fortes": float((d * w).sum() / w.sum()),
            "maior_diferenca_absoluta_fortes": float(d[np.argmax(np.abs(d))]),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--sementes", type=int, default=50)
    ap.add_argument("--fragmentos", default="data/interim/espelho_d16")
    a = ap.parse_args()

    from previsoes_item7 import CANAIS, carregar
    P = carregar(a.categoria, Path(a.painel), a.sementes, Path(a.fragmentos),
                 componentes=True)
    L = P["linhas"]["treino"]
    m = L["mascara"]
    H = P["ln_prev"]["treino"][m]
    canais = {c: P["componentes_media"]["treino"][c][m] for c in CANAIS}
    sku = (L["serie"] % 100).astype(int)

    r = espelho_por_sku(L["y"], H, L["lnp"], L["lnc"], L["serie"], L["semana"],
                        sku, P["n"], canais)
    saida = {"categoria": a.categoria, "janela": "treino", "configuracao": P["base"], **r}
    destino = RAIZ / "reports" / f"espelho_por_sku_d16_{a.categoria}.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))

    print(json.dumps(r["resumo"], ensure_ascii=False))
    print(f'{"sku":>3} {"F":>7} {"obs":>7} {"esp":>7} {"dif":>7} {"z":>6}  '
          f'{"própr":>6} {"cruz":>6} {"ctx":>6}')
    for p in r["por_sku"]:
        c = p["canais"]
        print(f'{p["sku"]:3d} {p["F_primeiro_estagio_agrupado"]:7.1f} {p["obs"]:7.3f} '
              f'{p["espelho"]:7.3f} {p["diferenca"]:7.3f} {p["z"]:6.2f}  '
              f'{c["proprio"]:6.2f} {c["cruzado"]:6.2f} {c["contexto"]:6.2f}'
              f'{"  FRACO" if p["instrumento_fraco"] else ""}')
    print("->", destino)


if __name__ == "__main__":
    main()
