"""Item 7, pendências 9.3 e 9.4: onde a rede erra, e quem paga o nível.

## 9.3 Erro de previsão por segmento

WMAPE da rede no **teste**, por SKU, por loja e por semana, como média das 50
sementes (média do WMAPE de cada semente, não WMAPE da previsão média: a
previsão média é um modelo que ninguém treinou). As lojas que só aparecem no
teste caem no índice de reserva do embedding (D28), e o erro delas sai separado.
Na semana, o que se pergunta é se o erro cresce com a distância ao treino.

## 9.4 Nível por SKU

Sob MSLE a previsão é `exp(z)`, e `exp(E[ln y]) < E[y]`: a rede subprevê a média
já em amostra (encolhimento de Jensen, 0,8578 no agregado, 8.22). Por SKU:

    nível_teste = encolhimento_treino × transferência

e a **contribuição** de cada SKU ao déficit agregado de volume,

    cᵢ = (Σ yᵢ − Σ ŷᵢ) / (Σ y − Σ ŷ),    Σ cᵢ = 1

diz se o encolhimento é espalhado ou concentrado. É o que a etapa 9 precisa,
porque ela compara receita projetada com receita histórica.

Uso:
    python3 src/experiments/erro_por_segmento_item7.py frj
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
for pasta in ("data", "models", "experiments"):
    sys.path.insert(0, str(RAIZ / "src" / pasta))

BLOCO_SEMANAS = 20


def wmape_por_grupo(y, Yhat, grupo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`y` (L,), `Yhat` (L, S), `grupo` (L,) inteiro. Devolve (grupos, média, dp) do WMAPE."""
    g, inv = np.unique(grupo, return_inverse=True)
    G = len(g)
    den = np.bincount(inv, weights=y, minlength=G)
    S = Yhat.shape[1]
    w = np.empty((G, S))
    for s in range(S):
        w[:, s] = 100.0 * np.bincount(inv, weights=np.abs(y - Yhat[:, s]), minlength=G) / den
    return g, w.mean(1), (w.std(1, ddof=1) if S > 1 else np.zeros(G))


def nivel_por_grupo(y, Yhat, grupo) -> dict:
    """Σŷ/Σy por grupo e contribuição ao déficit agregado.

    A contribuição é calculada sobre a soma MÉDIA das sementes, e não como média
    das contribuições por semente. A primeira versão fazia a segunda coisa, e o
    denominador `Σy − Σŷ` troca de sinal entre sementes (nível de 0,869 a 1,051
    no teste, 12 das 50 acima de 1): a média de razões com denominador cruzando
    zero não quer dizer nada, e produziu sinais invertidos.
    """
    g, inv = np.unique(grupo, return_inverse=True)
    G = len(g)
    sy = np.bincount(inv, weights=y, minlength=G)
    S = Yhat.shape[1]
    sh = np.column_stack([np.bincount(inv, weights=Yhat[:, s], minlength=G)
                          for s in range(S)])
    niv = sh / sy[:, None]
    sh_m = sh.mean(1)
    return {"grupos": g, "nivel": niv.mean(1),
            "nivel_dp": niv.std(1, ddof=1) if S > 1 else 0 * sy,
            "contribuicao_deficit": (sy - sh_m) / (sy.sum() - sh_m.sum()),
            "participacao_volume": sy / sy.sum()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--sementes", type=int, default=50)
    ap.add_argument("--fragmentos", default="data/interim/espelho_d16")
    a = ap.parse_args()

    from preparo import INDICE_RESERVA
    from previsoes_item7 import carregar
    P = carregar(a.categoria, Path(a.painel), a.sementes, Path(a.fragmentos))
    dados, n = P["dados"], P["n"]

    parte = {}
    for p in ("treino", "teste"):
        d = dados[p]
        c = len(d["alvo"])
        parte[p] = {"y": d["alvo"].ravel().astype(float),
                    "Yhat": np.exp(P["ln_prev"][p]),
                    "sku": np.tile(np.arange(n), c),
                    "loja": np.repeat(d["chaves"]["store"].to_numpy(), n),
                    "semana": np.repeat(d["chaves"]["week"].to_numpy(), n),
                    "reserva": np.repeat(d["loja_idx"] == INDICE_RESERVA, n),
                    "lojas_do_treino": set(dados["treino"]["loja"].tolist())}
    te, tr = parte["teste"], parte["treino"]

    # --- 9.3 --------------------------------------------------------------
    _, w_sku, w_sku_dp = wmape_por_grupo(te["y"], te["Yhat"], te["sku"])
    lojas, w_loja, _ = wmape_por_grupo(te["y"], te["Yhat"], te["loja"])
    # "nunca vista" é definido pela PRESENÇA NO TREINO, não pelo índice de
    # reserva: o vocabulário é construído no painel inteiro (preparo.vocabulario),
    # então nenhuma loja do teste cai no índice 0. Achado de 17/09, pendência 9.5.
    nova = np.array([l not in tr["lojas_do_treino"] for l in te["loja"]])
    res_loja = {int(l): bool(nova[te["loja"] == l][0]) for l in lojas}
    _, w_res, w_res_dp = wmape_por_grupo(te["y"], te["Yhat"], nova.astype(int))
    semanas, w_sem, _ = wmape_por_grupo(te["y"], te["Yhat"], te["semana"])
    bloco = (te["semana"] - te["semana"].min()) // BLOCO_SEMANAS
    blocos, w_bloco, w_bloco_dp = wmape_por_grupo(te["y"], te["Yhat"], bloco)
    vol_sem = np.array([te["y"][te["semana"] == s].sum() for s in semanas])
    inclin = np.polyfit(semanas, w_sem, 1)[0]
    _, w_total, w_total_dp = wmape_por_grupo(te["y"], te["Yhat"], np.zeros(len(te["y"]), int))

    # --- 9.4 --------------------------------------------------------------
    nt = nivel_por_grupo(te["y"], te["Yhat"], te["sku"])
    nr = nivel_por_grupo(tr["y"], tr["Yhat"], tr["sku"])
    ntot_te = nivel_por_grupo(te["y"], te["Yhat"], np.zeros(len(te["y"]), int))
    ntot_tr = nivel_por_grupo(tr["y"], tr["Yhat"], np.zeros(len(tr["y"]), int))
    conc = np.sort(nt["contribuicao_deficit"])[::-1]

    por_sku = [{"sku": i, "participacao_volume_teste": float(nt["participacao_volume"][i]),
                "wmape_teste": float(w_sku[i]), "wmape_teste_dp": float(w_sku_dp[i]),
                "encolhimento_treino": float(nr["nivel"][i]),
                "nivel_teste": float(nt["nivel"][i]),
                "transferencia": float(nt["nivel"][i] / nr["nivel"][i]),
                "contribuicao_deficit_treino": float(nr["contribuicao_deficit"][i]),
                "contribuicao_deficit_teste": float(nt["contribuicao_deficit"][i])}
               for i in range(n)]
    saida = {
        "categoria": a.categoria, "n_sementes": a.sementes, "configuracao": P["base"],
        "wmape_teste_total": float(w_total[0]), "wmape_teste_total_dp": float(w_total_dp[0]),
        "por_sku": por_sku,
        "lojas": {"n": int(len(lojas)), "n_so_no_teste": int(sum(res_loja.values())),
                  "linhas_no_indice_de_reserva": int(te["reserva"].sum()),
                  "participacao_volume_so_no_teste": float(te["y"][nova].sum() / te["y"].sum()),
                  "wmape_mediana": float(np.median(w_loja)),
                  "wmape_p10": float(np.quantile(w_loja, 0.1)),
                  "wmape_p90": float(np.quantile(w_loja, 0.9)),
                  "wmape_vistas": float(w_res[0]), "wmape_vistas_dp": float(w_res_dp[0]),
                  "wmape_so_no_teste": float(w_res[-1]) if len(w_res) > 1 else None,
                  "wmape_so_no_teste_dp": float(w_res_dp[-1]) if len(w_res) > 1 else None,
                  "por_loja": [{"loja": int(l), "so_no_teste": res_loja[int(l)], "wmape": float(w)}
                               for l, w in zip(lojas, w_loja)]},
        "semanas": {"inclinacao_wmape_por_semana": float(inclin),
                    "corr_wmape_volume": float(np.corrcoef(w_sem, vol_sem)[0, 1]),
                    "blocos": [{"bloco": int(b), "wmape": float(w), "dp": float(dp)}
                               for b, w, dp in zip(blocos, w_bloco, w_bloco_dp)],
                    "por_semana": [{"semana": int(s), "wmape": float(w), "volume": float(v)}
                                   for s, w, v in zip(semanas, w_sem, vol_sem)]},
        "nivel": {"encolhimento_treino_total": float(ntot_tr["nivel"][0]),
                  "nivel_teste_total": float(ntot_te["nivel"][0]),
                  "contribuicao_3_maiores_teste": float(conc[:3].sum()),
                  "contribuicao_5_maiores_teste": float(conc[:5].sum()),
                  "encolhimento_min_max": [float(nr["nivel"].min()), float(nr["nivel"].max())]},
    }
    destino = RAIZ / "reports" / f"erro_por_segmento_item7_{a.categoria}.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))

    print("WMAPE teste total:", round(saida["wmape_teste_total"], 3), "dp", round(saida["wmape_teste_total_dp"], 3))
    print("lojas:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in saida["lojas"].items() if k != "por_loja"}))
    print("semanas: inclinação", round(inclin, 4), "pontos/semana; corr com volume",
          round(saida["semanas"]["corr_wmape_volume"], 3),
          "| blocos", [round(b["wmape"], 2) for b in saida["semanas"]["blocos"]])
    print("nível:", json.dumps({k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v])
                                for k, v in saida["nivel"].items()}))
    print(f'{"sku":>3} {"vol%":>5} {"wmape":>6} {"encolh":>6} {"nív te":>6} {"transf":>6} {"déf%":>6}')
    for p in por_sku:
        print(f'{p["sku"]:3d} {100*p["participacao_volume_teste"]:5.1f} {p["wmape_teste"]:6.2f} '
              f'{p["encolhimento_treino"]:6.3f} {p["nivel_teste"]:6.3f} {p["transferencia"]:6.3f} '
              f'{100*p["contribuicao_deficit_teste"]:6.1f}')
    print("->", destino)


if __name__ == "__main__":
    main()
