"""O resultado do item 8 sobre 50 sementes, como D26 exige.

D26 diz que comparação preditiva é entre **médias sobre sementes com erro padrão**,
nunca entre execuções únicas. O ganho do otimizador não é uma comparação preditiva,
mas depende de **níveis** previstos, que são a parte instável entre sementes: o item 5
mediu amplitude de até 4,8 pontos de WMAPE entre sementes com a derivada estável. Um
ganho reportado de uma semente só herda aquela instabilidade sem declará-la.

Decisão do Thiago, 17/09/2026: **50 sementes**.

## A divisão de trabalho com a multipartida, e ela evita multiplicar por 50

As duas coisas medem ruídos diferentes e por isso não se cruzam:

| | mede | quantas execuções |
|---|---|---|
| 50 sementes (aqui) | ruído de **estimação do modelo** | 50 treinos, **uma partida** cada |
| multipartida (D15) | a **superfície do solver** | 1 treino, 50 partidas |

Bacia de atração é propriedade do problema e não da semente de treino, de modo que
cruzar as duas gastaria 2.500 otimizações por objetivo para medir duas coisas
distintas com um número só. A multipartida roda no artefato canônico e está em
`multipartida.py`.

**A consequência a declarar:** o número desta execução é o ganho com **partida
única**, e a multipartida mostrou que ela subestima a receita em 1,41 ponto e a margem
em 0,06. O erro padrão entre sementes e o viés da partida única são coisas separadas,
e as duas vão para o texto.

## O que é refeito por semente e o que não é

Refeito: os pesos da rede, e só. O painel, a partição, o centro de preço, o
escalonador, o escopo de D38 e as restrições de D39 dependem do **dado**, não da
semente, e refazê-los seria fingir variação que não existe.

Uso:

    python3 src/optimization/sementes_item8.py frj --sementes 50
    python3 src/optimization/sementes_item8.py frj --sementes 5      # sonda de custo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
for sub in ("models", "data", "experiments", "optimization"):
    sys.path.insert(0, str(RAIZ / "src" / sub))

import hierarquia as HIER  # noqa: E402
import problema as P  # noqa: E402
import rede  # noqa: E402
from sonda_item8 import OBJETIVOS, restricoes_do_escopo  # noqa: E402


def resumo(v: np.ndarray) -> dict:
    """Média, desvio e **erro padrão da média**, que é o que D26 pede.

    O erro padrão é `dp / sqrt(n)`, e é ele que diz se duas configurações se separam.
    Reportar só o desvio entre sementes responderia outra pergunta.
    """
    v = np.asarray(v, dtype=float)
    dp = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return {"media": float(v.mean()), "desvio": dp,
            "erro_padrao": dp / np.sqrt(len(v)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()),
            "amplitude": float(v.max() - v.min()), "n": int(len(v))}


def uma_semente(dados: dict, cfg: dict, esc: dict, centro, restr, metodo: str):
    """Treina uma rede e otimiza as células do escopo, uma partida por célula."""
    import torch
    saida = rede.treinar(dados, cfg)
    modelo = saida["modelo"]
    modelo.eval()

    linha = {}
    for objetivo in OBJETIVOS:
        custos = (esc["custos"] if objetivo == "margem"
                  else np.zeros_like(esc["custos"]))
        total_ini = total_fim = 0.0
        interior = inviaveis = 0
        for c in range(len(esc["u"])):
            if restr is not None and restr[2][c]:
                inviaveis += 1
                continue
            r_celula = None if restr is None else (restr[0], restr[1][c])
            prob = P.ProblemaDeCelula(modelo, esc["ctx"][c], int(esc["loja_idx"][c]),
                                      centro, custos[c], esc["u"][c],
                                      restricoes=r_celula)
            r = P.resolver(prob, metodo=metodo)
            total_ini += r["valor_inicial"]
            total_fim += r["valor"]
            interior += r["n_interior"]
        resolvidas = len(esc["u"]) - inviaveis
        linha[objetivo] = {
            "ganho_pct_agregado": 100.0 * (total_fim / total_ini - 1.0),
            "pct_coord_interior": 100.0 * interior / max(resolvidas * esc["n"], 1),
        }
    # O WMAPE de teste vai junto: e ele que liga esta execucao ao item 6 e permite
    # ver se as sementes de ganho alto sao as de ajuste bom.
    from baseline_arvores import metricas
    y = dados["teste"]["alvo"].ravel()
    pred = saida["pred_teste"].ravel()
    linha["wmape_teste"] = float(metricas(y, pred)["wmape_pct"])
    linha["vies_de_nivel_teste"] = float(pred.sum() / y.sum())
    return linha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--bruto", default="data/raw")
    ap.add_argument("--sementes", type=int, default=50)
    ap.add_argument("--semente-inicial", type=int, default=0)
    ap.add_argument("--hierarquia", default="nao_piorar",
                    choices=("nenhuma",) + HIER.FORMAS)
    ap.add_argument("--metodo", default="SLSQP")
    ap.add_argument("--saida", default="reports")
    ap.add_argument("--sufixo", default="")
    a = ap.parse_args()

    from varredura_rede import MELHOR

    painel = Path(a.painel)
    dados = rede.preparar(a.categoria, painel)
    centro = np.asarray(dados["centro_de_preco"], float)
    esc = P.escopo(dados, painel, a.categoria, "teste")

    restr, pares = None, []
    if a.hierarquia != "nenhuma":
        A, B, pares, _, inviavel = restricoes_do_escopo(
            a.categoria, painel, a.bruto, centro, esc, a.hierarquia)
        restr = (A, B, inviavel)

    linhas, t0 = [], time.perf_counter()
    for i in range(a.sementes):
        s = a.semente_inicial + i
        cfg = dict(MELHOR, semente=s, restrita=True)
        cfg.setdefault("gama_contextual", False)
        t = time.perf_counter()
        linha = uma_semente(dados, cfg, esc, centro, restr, a.metodo)
        linha["semente"] = s
        linha["tempo_s"] = time.perf_counter() - t
        linhas.append(linha)
        gasto = time.perf_counter() - t0
        resta = gasto * (a.sementes - i - 1) / (i + 1)
        print(f"  semente {s:2d}  margem {linha['margem']['ganho_pct_agregado']:6.2f}%"
              f"  receita {linha['receita']['ganho_pct_agregado']:6.2f}%"
              f"  WMAPE {linha['wmape_teste']:5.2f}%"
              f"  ({linha['tempo_s']:.0f} s, ~{resta / 60:.0f} min restantes)",
              flush=True)

    agregado = {
        obj: {
            "ganho_pct_agregado": resumo([x[obj]["ganho_pct_agregado"] for x in linhas]),
            "pct_coord_interior": resumo([x[obj]["pct_coord_interior"] for x in linhas]),
        } for obj in OBJETIVOS
    }
    agregado["wmape_teste"] = resumo([x["wmape_teste"] for x in linhas])
    agregado["vies_de_nivel_teste"] = resumo([x["vies_de_nivel_teste"] for x in linhas])

    bloco = {"categoria": a.categoria, "configuracao": dict(MELHOR),
             "hierarquia": a.hierarquia, "n_pares": len(pares),
             "escopo": esc["contagens"], "n_sementes": a.sementes,
             "partidas_por_celula": 1,
             "agregado": agregado, "por_semente": linhas,
             "tempo_total_min": (time.perf_counter() - t0) / 60.0}
    destino = (Path(a.saida) / f"sementes_item8_{a.categoria}"
               f"_{a.hierarquia}_n{a.sementes}{a.sufixo}.json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(bloco, indent=2, ensure_ascii=False))

    print(f"\n{a.sementes} sementes, hierarquia {a.hierarquia}, "
          f"{esc['contagens']['n_celulas_com_custo_completo']} células, "
          f"UMA partida por célula\n")
    print(f"{'':22s} {'média':>9} {'ep':>8} {'dp':>8} {'amplitude':>10}")
    for obj in OBJETIVOS:
        g = agregado[obj]["ganho_pct_agregado"]
        i = agregado[obj]["pct_coord_interior"]
        print(f"ganho {obj:16s} {g['media']:8.2f}% {g['erro_padrao']:7.3f} "
              f"{g['desvio']:7.3f} {g['amplitude']:9.3f}")
        print(f"interior {obj:13s} {i['media']:8.2f}% {i['erro_padrao']:7.3f} "
              f"{i['desvio']:7.3f} {i['amplitude']:9.3f}")
    w = agregado["wmape_teste"]
    print(f"WMAPE de teste         {w['media']:8.2f}% {w['erro_padrao']:7.3f} "
          f"{w['desvio']:7.3f} {w['amplitude']:9.3f}")
    print(f"\ntempo total {bloco['tempo_total_min']:.1f} min")
    print(f"-> {destino}")


if __name__ == "__main__":
    main()
