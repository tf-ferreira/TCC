"""D16, passo 3: em qual canal está a diferença do espelho.

## Por que existe

O espelho (`espelho_d16.py`) mede quanto `ln V̂` da rede muda quando o custo
muda, somando todos os canais. A rede restrita é uma SOMA de três termos,

    ln V̂ᵢ = gᵢ(x)  +  Σ_{j≠i} γᵢⱼ uⱼ  −  mᵢ(uᵢ)
            contexto   cruzado          próprio

e 2SLS é linear em `y`. Então o coeficiente do espelho se parte EXATAMENTE:

    β_esp = IV(contexto) + IV(cruzado) + IV(próprio)

Sem aproximação nenhuma. O canal `contexto` inclui as defasagens de preço (D27),
que é a hipótese do passo 2 para o −5,3 do teste. Separar as defasagens DENTRO de
`gᵢ` não é exato, porque `gᵢ` não é linear, e fica fora deste script de propósito.

## Retreino e a conferência que ele exige

Os pesos por semente não foram salvos no passo 2, só as previsões. As 50 redes
são retreinadas com as mesmas sementes, e cada uma passa por uma conferência:
a soma dos três termos tem de bater com o `ln V̂` gravado no fragmento do passo 2.
Se o treino não for reprodutível, a decomposição descreveria outra rede, e o
script **para** em vez de seguir.

Uso:
    python3 src/experiments/decomposicao_espelho_d16.py frj [--sementes 50]
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
for pasta in ("data", "models", "experiments"):
    sys.path.insert(0, str(RAIZ / "src" / pasta))
from espelho_d16 import comparar_espelho, impressao_digital, linhas  # noqa: E402

CANAIS = ("contexto", "cruzado", "proprio")
TOL_REPRODUCAO = 1e-3            # em ln V̂; float32 no treino


def componentes(modelo, ctx, u, loja) -> dict:
    """Os três termos aditivos de `log_demanda`, com o sinal com que entram."""
    import torch
    with torch.no_grad():
        g, gama = modelo.tronco(ctx, loja)
        if gama is not None:
            raise ValueError("decomposição exata só vale com γ constante (D36)")
        cruz = u @ (modelo.gama * modelo.fora_da_diagonal).T
        prop = -modelo.monotona(u)
    return {"contexto": g.numpy().ravel().astype("float64"),
            "cruzado": cruz.numpy().ravel().astype("float64"),
            "proprio": prop.numpy().ravel().astype("float64")}


def decompor(y, partes: dict, lnp, lnc, serie, semana, loja) -> dict:
    """IV de cada canal, por semente, e a conferência de que somam o espelho."""
    total = sum(partes[c] for c in CANAIS)
    saida = {"total": comparar_espelho(y, total, lnp, lnc, serie, semana, loja)}
    for c in CANAIS:
        saida[c] = comparar_espelho(y, partes[c], lnp, lnc, serie, semana, loja)
    S = total.shape[1]
    soma = np.array([sum(saida[c]["por_semente"][s]["espelho"]["coef"] for c in CANAIS)
                     for s in range(S)])
    tot = np.array([saida["total"]["por_semente"][s]["espelho"]["coef"] for s in range(S)])
    resumo = {"obs": saida["total"]["obs"]["coef"],
              "r2_parcial_custo": saida["total"]["r2_parcial_custo"],
              "espelho": float(tot.mean()),
              "maior_erro_de_soma": float(np.max(np.abs(soma - tot)))}
    for c in CANAIS:
        b = np.array([p["espelho"]["coef"] for p in saida[c]["por_semente"]])
        e = np.array([p["espelho"]["ep_serie"] for p in saida[c]["por_semente"]])
        resumo[c] = {"media": float(b.mean()),
                     "dp_sementes": float(b.std(ddof=1)) if S > 1 else 0.0,
                     "ep_total_serie": float(np.sqrt((e ** 2).mean()
                                                     + (b.var(ddof=1) / S if S > 1 else 0)))}
    return {"resumo": resumo,
            "por_semente": {c: [p["espelho"] for p in saida[c]["por_semente"]]
                            for c in ("total",) + CANAIS}}


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
    destino_frag = pasta / "componentes"
    destino_frag.mkdir(parents=True, exist_ok=True)

    acum = {p: {c: [] for c in CANAIS} for p in L}
    reproducao = []
    for s in range(a.sementes):
        cfg = dict(base, semente=s)
        dig = impressao_digital(cfg)
        arq = destino_frag / f"semente_{s:02d}_{dig}.npz"
        if arq.exists():
            f = np.load(arq)
            comp = {p: {c: f[f"{p}_{c}"] for c in CANAIS} for p in L}
        else:
            out = rede.treinar(dados, cfg)
            T = out["tensores"]
            comp = {"treino": componentes(out["modelo"], T["ctx_treino"],
                                          T["u_treino"], T["loja_treino"]),
                    "teste": componentes(out["modelo"], T["ctx_teste"],
                                         T["u_teste"], T["loja_teste"])}
            ref = pasta / f"semente_{s:02d}_{dig}.npz"
            erro = None
            if ref.exists():
                r = np.load(ref)
                erro = max(float(np.max(np.abs(sum(comp[p][c] for c in CANAIS) - r[p])))
                           for p in L)
                if erro > TOL_REPRODUCAO:
                    raise SystemExit(
                        f"semente {s}: soma dos termos difere do ln V̂ do passo 2 em "
                        f"{erro:.2e}. O treino não reproduziu; a decomposição "
                        "descreveria outra rede. Parando.")
            reproducao.append({"semente": s, "erro_max": erro})
            np.savez(arq, **{f"{p}_{c}": comp[p][c] for p in L for c in CANAIS})
        for p in L:
            for c in CANAIS:
                acum[p][c].append(comp[p][c][L[p]["mascara"]])
        print(f"semente {s:02d} pronta", flush=True)

    saida = {"categoria": a.categoria, "configuracao": base,
             "n_sementes": a.sementes, "reproducao": reproducao}
    for p, d in L.items():
        partes = {c: np.column_stack(acum[p][c]) for c in CANAIS}
        r = decompor(d["y"], partes, d["lnp"], d["lnc"], d["serie"], d["semana"], d["loja"])
        saida[p] = r
        z = r["resumo"]
        print(p, "obs", round(z["obs"], 4), "espelho", round(z["espelho"], 4),
              "| contexto", round(z["contexto"]["media"], 4),
              "cruzado", round(z["cruzado"]["media"], 4),
              "proprio", round(z["proprio"]["media"], 4),
              "| erro de soma", f'{z["maior_erro_de_soma"]:.1e}', flush=True)

    destino = RAIZ / "reports" / f"decomposicao_espelho_d16_{a.categoria}.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))
    print("->", destino)


if __name__ == "__main__":
    main()
