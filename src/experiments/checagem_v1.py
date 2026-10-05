"""Checagem preditiva da especificação v1 (rodada R1 do plano de 23/09/2026).

Para cada especificação pedida, treina as mesmas sementes e mede, por semente:

- o ajuste no teste (WMAPE e nível, os dois eixos de D26);
- o resumo da matriz de elasticidades no teste (própria mediana e ponderada,
  agregada, soma da linha de γ, dispersão e fração negativa das cruzadas);
- o espelho da CESTA no treino (D45): o mesmo estimador, MQO com efeitos fixos de
  loja e de semana e erro agrupado por loja, aplicado a `ln Σ V` e à previsão da
  rede, em diferença pareada. Com função de controle ele testa o AJUSTE da
  previsão, não a derivada com `v̂` fixo, e é reportado como diagnóstico.

Depois compara especificações PAREADAS por semente, que é o teste de D26, e
avalia os critérios declarados de D42 antes da execução:

1. **estabilidade**: o desvio padrão da própria mediana entre sementes, na v1,
   é no máximo 0,05;
2. **ajuste**: a v1 não piora o WMAPE de teste em 1 ponto ou mais contra a
   `base_v1`, a v1 sem a função de controle. Teste de não inferioridade: a média
   da diferença pareada mais 1,645 erro padrão fica abaixo de 1.

A faixa das referências IV (−1,76 sem os códigos de promoção, −1,27 com) é
**reportada ao lado da própria mediana, e não é critério**: D16 (ponto 1) mostrou
que a comparação direta entre a elasticidade própria da rede e o coeficiente de
D18 é inválida, porque o IV sem preços cruzados estima a própria mais a cruzada
vezes o arrasto do instrumento.

Uso:
    python3 src/experiments/checagem_v1.py frj --sementes 2 --sufixo _sonda
    python3 src/experiments/checagem_v1.py frj --sementes 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
for sub in ("models", "data", "experiments", "optimization"):
    sys.path.insert(0, str(RAIZ / "src" / sub))

import rede  # noqa: E402
from elasticidades import centralizar, ols, sanduiche  # noqa: E402

ESPECIFICACOES_PADRAO = ["adotada", "base_v1", "v1", "v1_codigo"]
PARES = [("base_v1", "adotada", "D41"), ("v1", "base_v1", "D42"),
         ("v1_codigo", "v1", "D43")]
REFERENCIAS_IV = {"sem_codigos": -1.757, "com_codigos": -1.270}
LIMIAR_DP_PROPRIA = 0.05
LIMIAR_WMAPE = 1.0
Z_UNILATERAL = 1.6448536269514722


def _resumo(x) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    dp = float(x.std(ddof=1)) if n > 1 else float("nan")
    return {"media": float(x.mean()), "desvio": dp,
            "erro_padrao": dp / np.sqrt(n) if n > 1 else float("nan"),
            "minimo": float(x.min()), "maximo": float(x.max()), "n": int(n)}


def elasticidades(modelo, parte: dict) -> dict:
    import torch
    ctx = torch.tensor(np.concatenate([parte["ctx_celula"], parte["ctx_sku"]], 1),
                       dtype=torch.float32)
    u = torch.tensor(parte["u"], dtype=torch.float32)
    loja = torch.tensor(parte["loja_idx"], dtype=torch.long)
    with torch.no_grad():
        eps = modelo.elasticidades_fechadas(u)
        v = torch.exp(modelo.log_demanda(ctx, u, loja))
    s = v / v.sum(1, keepdim=True)
    propria = torch.diagonal(eps, dim1=1, dim2=2)
    G = (modelo.gama * modelo.fora_da_diagonal).detach().numpy()
    fora = G[~np.eye(G.shape[0], dtype=bool)]
    return {"propria_mediana": float(propria.median()),
            "propria_ponderada_mediana": float((s * propria).sum(1).median()),
            "agregada_mediana": float((s * eps.sum(2)).sum(1).median()),
            "soma_da_linha_de_gama_media": float(G.sum(1).mean()),
            "cruzada_dp": float(fora.std()),
            "fracao_cruzadas_negativas": float((fora < 0).mean())}


class EspelhoDaCesta:
    """O desenho do espelho da cesta, montado uma vez por especificação.

    Regressores: o índice de preço dos N (média do log preço ponderada pelo
    volume médio do treino, pesos FIXOS: com pesos contemporâneos o índice vira
    função do desfecho, sonda 11b) e os dois controles do resto que a v1 mantém.
    """

    def __init__(self, parte: dict, painel: Path, categoria: str):
        fora = pd.read_parquet(Path(painel) / f"{categoria}_bem_externo.parquet")
        ch = parte["chaves"].reset_index(drop=True).merge(
            fora[["store", "week", "idx_preco_resto", "promo_resto"]],
            on=["store", "week"], how="left", validate="one_to_one")
        V = parte["alvo"]
        self.w = V.mean(0) / V.mean(0).sum()
        X = np.column_stack([(np.log(parte["precos"]) * self.w).sum(1),
                             np.log(ch["idx_preco_resto"].to_numpy(float)),
                             ch["promo_resto"].to_numpy(float)])
        ok = np.isfinite(X).all(1)
        self.ok = ok
        self.chaves = [ch["store"].to_numpy()[ok], ch["week"].to_numpy()[ok]]
        self.Xc = centralizar(X[ok], self.chaves)
        self.k = X.shape[1]
        self.y_dado = np.log(V.sum(1))[ok]

    def coeficiente(self, y) -> tuple:
        yc = centralizar(np.asarray(y, float)[:, None], self.chaves)[:, 0]
        b, _, r = ols(yc, self.Xc, self.k)
        ep = sanduiche(self.Xc, r, {"loja": self.chaves[0]}, self.k)["loja"]["ep"]
        return float(b[0]), float(ep[0])

    def dado(self) -> tuple:
        return self.coeficiente(self.y_dado)

    def diferenca_pareada(self, pred) -> tuple:
        """`(rede − dado)` no coeficiente do índice, com erro agrupado por loja."""
        return self.coeficiente(np.log(pred.sum(1))[self.ok] - self.y_dado)


def uma_semente(dados, cfg, espelho) -> dict:
    from baseline_arvores import metricas
    saida = rede.treinar(dados, cfg)
    modelo = saida["modelo"].eval()
    y_te = dados["teste"]["alvo"].ravel()
    p_te = saida["pred_teste"].ravel()
    y_tr = dados["treino"]["alvo"].ravel()
    p_tr = saida["pred_treino"].ravel()
    dif, ep = espelho.diferenca_pareada(saida["pred_treino"])
    return {"wmape_teste": float(metricas(y_te, p_te)["wmape_pct"]),
            "nivel_teste": float(p_te.sum() / y_te.sum()),
            "nivel_treino": float(p_tr.sum() / y_tr.sum()),
            "elasticidades": elasticidades(modelo, dados["teste"]),
            "espelho_da_cesta_rede_menos_dado": dif,
            "espelho_da_cesta_ep": ep}


def comparar(a: list, b: list, chave) -> dict:
    """Diferença pareada por semente, `a − b`, com erro padrão e z."""
    da = np.array([chave(x) for x in a]); db = np.array([chave(x) for x in b])
    d = da - db
    r = _resumo(d)
    r["z"] = r["media"] / r["erro_padrao"] if r["erro_padrao"] > 0 else float("nan")
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--especificacoes", nargs="+", default=ESPECIFICACOES_PADRAO)
    ap.add_argument("--sementes", type=int, default=50)
    ap.add_argument("--semente-inicial", type=int, default=0)
    ap.add_argument("--saida", default="reports")
    ap.add_argument("--sufixo", default="")
    a = ap.parse_args()
    from varredura_rede import MELHOR

    painel = Path(a.painel)
    por_espec, extra, t0 = {}, {}, time.perf_counter()
    for espec in a.especificacoes:
        rede.usar_especificacao(espec)
        dados = rede.preparar(a.categoria, painel)
        espelho = EspelhoDaCesta(dados["treino"], painel, a.categoria)
        extra[espec] = {"espelho_da_cesta_dado": espelho.dado(),
                        "funcao_de_controle": dados["meta"].get("funcao_de_controle")}
        linhas = []
        for i in range(a.sementes):
            s = a.semente_inicial + i
            cfg = dict(MELHOR, semente=s, restrita=True, especificacao=espec)
            cfg.setdefault("gama_contextual", False)
            linha = uma_semente(dados, cfg, espelho)
            linha["semente"] = s
            linhas.append(linha)
            e = linha["elasticidades"]
            print(f"  {espec:10s} semente {s:2d}  WMAPE {linha['wmape_teste']:6.2f}  "
                  f"nível {linha['nivel_teste']:.3f}  própria {e['propria_mediana']:+.3f}  "
                  f"agregada {e['agregada_mediana']:+.3f}  linha γ "
                  f"{e['soma_da_linha_de_gama_media']:+.3f}  cesta "
                  f"{linha['espelho_da_cesta_rede_menos_dado']:+.3f}", flush=True)
        por_espec[espec] = linhas
    rede.usar_especificacao("adotada")

    agregado = {}
    for espec, linhas in por_espec.items():
        agregado[espec] = {k: _resumo([x[k] for x in linhas])
                           for k in ("wmape_teste", "nivel_teste", "nivel_treino",
                                     "espelho_da_cesta_rede_menos_dado")}
        agregado[espec]["elasticidades"] = {
            k: _resumo([x["elasticidades"][k] for x in linhas])
            for k in linhas[0]["elasticidades"]}
    pares = {}
    for p, q, decisao in PARES:
        if p in por_espec and q in por_espec:
            pares[f"{p}_menos_{q}"] = {
                "decisao": decisao,
                "wmape_teste": comparar(por_espec[p], por_espec[q], lambda x: x["wmape_teste"]),
                "nivel_teste": comparar(por_espec[p], por_espec[q], lambda x: x["nivel_teste"]),
                "propria_mediana": comparar(por_espec[p], por_espec[q],
                                            lambda x: x["elasticidades"]["propria_mediana"]),
                "agregada_mediana": comparar(por_espec[p], por_espec[q],
                                             lambda x: x["elasticidades"]["agregada_mediana"]),
                "soma_da_linha_de_gama_media": comparar(
                    por_espec[p], por_espec[q],
                    lambda x: x["elasticidades"]["soma_da_linha_de_gama_media"])}
    criterios = {}
    if "v1" in agregado:
        dp = agregado["v1"]["elasticidades"]["propria_mediana"]["desvio"]
        criterios["d42_estabilidade"] = {"dp_propria_mediana": dp,
                                         "limiar": LIMIAR_DP_PROPRIA,
                                         "passa": bool(dp <= LIMIAR_DP_PROPRIA)}
        if "v1_menos_base_v1" in pares:
            w = pares["v1_menos_base_v1"]["wmape_teste"]
            sup = w["media"] + Z_UNILATERAL * w["erro_padrao"]
            criterios["d42_ajuste"] = {"diferenca_media": w["media"],
                                       "limite_superior_unilateral_95": sup,
                                       "limiar": LIMIAR_WMAPE,
                                       "passa": bool(sup < LIMIAR_WMAPE)}
        criterios["faixa_das_referencias_iv_reportada_nao_criterio"] = {
            "propria_mediana_v1": agregado["v1"]["elasticidades"]["propria_mediana"]["media"],
            **REFERENCIAS_IV}

    bloco = {"categoria": a.categoria, "configuracao": dict(MELHOR),
             "especificacoes": a.especificacoes, "n_sementes": a.sementes,
             "semente_inicial": a.semente_inicial, "extra": extra,
             "agregado": agregado, "pares": pares, "criterios": criterios,
             "por_especificacao": por_espec,
             "tempo_total_min": (time.perf_counter() - t0) / 60.0}
    destino = Path(a.saida) / f"checagem_v1_{a.categoria}_n{a.sementes}{a.sufixo}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(bloco, indent=2, ensure_ascii=False, default=float))

    print(f"\n{a.sementes} sementes; média (ep)")
    for espec, g in agregado.items():
        e = g["elasticidades"]
        print(f"  {espec:10s} WMAPE {g['wmape_teste']['media']:6.2f} ({g['wmape_teste']['erro_padrao']:.2f})"
              f"  nível {g['nivel_teste']['media']:.3f}  própria {e['propria_mediana']['media']:+.3f}"
              f" (dp {e['propria_mediana']['desvio']:.3f})  agregada {e['agregada_mediana']['media']:+.3f}"
              f"  cesta {g['espelho_da_cesta_rede_menos_dado']['media']:+.3f}")
    for nome, p in pares.items():
        print(f"  {nome:22s} WMAPE {p['wmape_teste']['media']:+.3f} (z {p['wmape_teste']['z']:+.2f})"
              f"  própria {p['propria_mediana']['media']:+.3f}  agregada {p['agregada_mediana']['media']:+.3f}")
    for nome, c in criterios.items():
        print(f"  {nome}: {c}")
    print(f"\ntempo total {bloco['tempo_total_min']:.1f} min\n-> {destino}")


if __name__ == "__main__":
    main()
