"""O contrafactual do item 9 sobre 50 sementes, como D26 exige.

Mesma divisão de trabalho de `sementes_item8.py`, pelo mesmo motivo:

| | mede | execuções |
|---|---|---|
| multipartida, `contrafactual_item9.py -k 50` | a superfície do solver | 1 treino, 50 partidas |
| **aqui** | o ruído de **estimação do modelo** | 50 treinos, **uma partida** |
| bootstrap por loja, no `.npz` | a amostragem de **células** | 1 treino, 2.000 reamostras |

São três ruídos distintos, e nenhum deles cobre o quarto, que não é medível: se a
calibração vale nos preços contrafactuais (D40).

## Por que o item 9 precisa disto e não herda do item 8

O item 8 mostrou que a semente canônica é **pessimista** nos dois objetivos. Aqui há
um motivo a mais: a calibração `M̂/R` depende do **nível** previsto, que é a parte
instável entre sementes (o item 5 mediu amplitude de até 4,8 pontos de WMAPE com
derivada estável). O segundo fator da identidade herda essa instabilidade inteira, e
o primeiro só em parte.

## A pergunta que a execução canônica deixou aberta

`C/B − 1` saiu **negativo** nos dois objetivos, e o bootstrap por loja o separa de
zero (IC95 de [−2,25; −1,20] na margem e [−1,31; −0,44] na receita). O bootstrap
mede amostragem de células com o modelo **fixo**. Se o sinal for da semente, ele
some aqui; se for da forma funcional que D31 a D36 fixaram, ele fica.

## O que é refeito por semente e o que não é

Refeito: os pesos da rede, e só. Painel, partição, centro de preço, escalonador,
escopo de D38, restrições de D39 e as **corridas** de D40 dependem do dado.

Uso:
    python3 src/optimization/sementes_item9.py frj --sementes 2 --sufixo _sonda
    python3 src/optimization/sementes_item9.py frj --sementes 50
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

import contrafactual_item9 as C9  # noqa: E402
import diagnostico_item9 as D9  # noqa: E402
import hierarquia as HIER  # noqa: E402
import problema as P  # noqa: E402
import rede  # noqa: E402
from sementes_item8 import resumo  # noqa: E402
from sonda_item8 import OBJETIVOS, restricoes_do_escopo  # noqa: E402

CAMPOS = ["superestimacao_da_avaliacao_ingenua_pct",
          "ganho_da_decisao_sequencial_sobre_a_ingenua_pct",
          "ganho_ingenuo_modelo_contra_modelo_pct",
          "ganho_sequencial_modelo_contra_modelo_pct",
          "calibracao_do_objetivo_no_ponto_historico",
          "ganho_sequencial_contra_o_OBSERVADO_pct"]


def uma_semente(dados, cfg, esc, mapa, centro, restr, metodo) -> dict:
    """Treina uma rede e roda as duas passadas nos dois objetivos, K = 1."""
    saida = rede.treinar(dados, cfg)
    modelo = saida["modelo"]
    modelo.eval()
    linha = {}
    for obj in OBJETIVOS:
        custos = (esc["custos"] if obj == "margem"
                  else np.zeros_like(esc["custos"]))
        ing = C9.rodar_passada(modelo, centro, esc, mapa, custos, restr, 1, 0,
                               metodo, False, progresso=0)
        honesto = C9.ctx_honesto_da_politica(esc, mapa, ing["u"], centro)
        seq = C9.rodar_passada(modelo, centro, esc, mapa, custos, restr, 1, 0,
                               metodo, True, progresso=0)
        r = C9.agregar(modelo, centro, esc, mapa, custos, ing, seq, honesto, obj)
        r.pop("_bruto")
        linha[obj] = {k: r[k] for k in CAMPOS}
    # A resposta à defasagem, POR SEMENTE. A sonda de 2 sementes mostrou o viés
    # ingênuo da receita trocando de sinal entre sementes (−3,1 e +6,8) com WMAPE
    # quase igual; 12.10 diz que o sinal é governado pela resposta a uma alta
    # sustentada. Gravar as duas lado a lado transforma aquela hipótese de UM
    # modelo em teste sobre 50.
    S = mapa["indices_elegiveis"]
    for chave in ("lag1", "lag2"):
        J, V = D9.jacobiana_da_defasagem(modelo, esc, S, chave)
        r = D9.resumo_da_jacobiana(J, V)
        linha[f"defasagem_{chave}_agregada"] = r["agregada_uniforme_ponderada"]
        linha[f"defasagem_{chave}_propria_mediana"] = r["propria"]["mediana"]
    linha["defasagem_sustentada"] = (linha["defasagem_lag1_agregada"]
                                     + linha["defasagem_lag2_agregada"])
    from baseline_arvores import metricas
    y = dados["teste"]["alvo"].ravel()
    pred = saida["pred_teste"].ravel()
    linha["wmape_teste"] = float(metricas(y, pred)["wmape_pct"])
    linha["vies_de_nivel_teste"] = float(pred.sum() / y.sum())
    return linha


def _correlacao(x, y):
    """Pearson, ou `None` quando não há variação para correlacionar."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def agregar_sementes(linhas: list) -> dict:
    fora = {}
    for obj in OBJETIVOS:
        fora[obj] = {k: resumo([x[obj][k] for x in linhas]) for k in CAMPOS}
        dec = np.array([x[obj]["ganho_da_decisao_sequencial_sobre_a_ingenua_pct"]
                        for x in linhas])
        sup = np.array([x[obj]["superestimacao_da_avaliacao_ingenua_pct"]
                        for x in linhas])
        fora[obj]["pct_sementes_decisao_negativa"] = 100.0 * float((dec < 0).mean())
        fora[obj]["pct_sementes_superestima"] = 100.0 * float((sup > 0).mean())
        # A calibração e o ganho no modelo andam juntos entre sementes? Se a
        # correlação for forte, o erro padrão do produto NÃO é a soma em
        # quadratura dos dois, e o texto tem de usar o do produto medido.
        g = np.array([x[obj]["ganho_sequencial_modelo_contra_modelo_pct"]
                      for x in linhas])
        c = np.array([x[obj]["calibracao_do_objetivo_no_ponto_historico"]
                      for x in linhas])
        fora[obj]["correlacao_ganho_calibracao"] = _correlacao(g, c)
    sus = np.array([x.get("defasagem_sustentada", np.nan) for x in linhas])
    if np.isfinite(sus).all():
        fora["defasagem_sustentada"] = resumo(sus)
        fora["defasagem_lag1_agregada"] = resumo(
            [x["defasagem_lag1_agregada"] for x in linhas])
        fora["defasagem_lag2_agregada"] = resumo(
            [x["defasagem_lag2_agregada"] for x in linhas])
        fora["pct_sementes_defasagem_sustentada_negativa"] = \
            100.0 * float((sus < 0).mean())
        # O TESTE de 12.10: com alta sustentada negativa, a receita (que corta)
        # herda defasagem baixa e ganha volume, logo A < B. A previsão é
        # correlação POSITIVA entre a resposta sustentada e o viés da receita, e
        # NEGATIVA com o da margem (que sobe preço).
        for obj in OBJETIVOS:
            sup = np.array([x[obj]["superestimacao_da_avaliacao_ingenua_pct"]
                            for x in linhas])
            fora[obj]["correlacao_vies_ingenuo_defasagem_sustentada"] = \
                _correlacao(sus, sup)
            # Concordância de sinal, que é o que a hipótese de fato afirma.
            # Previsão: sustentada < 0 faz a receita SUBESTIMAR e a margem
            # SUPERESTIMAR.
            prev = sus < 0
            obs = (sup < 0) if obj == "receita" else (sup > 0)
            fora[obj]["pct_sementes_sinal_previsto_pela_defasagem"] = \
                100.0 * float((prev == obs).mean())
    fora["wmape_teste"] = resumo([x["wmape_teste"] for x in linhas])
    fora["vies_de_nivel_teste"] = resumo([x["vies_de_nivel_teste"] for x in linhas])
    return fora


# Limiar de D26 para `C/B` contra zero, DECLARADO antes das 25 sementes extras.
#
# São dois testes (margem e receita), bilaterais a 5%, corrigidos por Bonferroni:
# Φ⁻¹(1 − 0,05/4) = 2,241. É a mesma regra que deu 2,394 para três pares.
#
# A regra de parada também é declarada aqui, porque acrescentar sementes só quando
# o z cai perto do limiar infla o erro do tipo I (parada opcional): 75 sementes é o
# número FINAL, passe ou não passe. Se |z| ≥ 2,241, o texto reporta o sinal; se
# não, reporta só que o efeito é menor que meio ponto, sem sinal.
LIMIAR_CB = 2.241
CB = "ganho_da_decisao_sequencial_sobre_a_ingenua_pct"


def juntar(arquivos: list) -> dict:
    """Consolida execuções de sementes DISJUNTAS numa só, e recusa a mistura.

    Recusa se configuração, hierarquia ou corridas divergirem (seria juntar
    medições de problemas diferentes) e se alguma semente aparecer duas vezes
    (seria contar a mesma rede duas vezes e encolher o erro padrão).
    """
    blocos = [json.loads(Path(f).read_text()) for f in arquivos]
    ref = blocos[0]
    for b in blocos[1:]:
        for chave in ("configuracao", "hierarquia", "corridas", "escopo",
                      "partidas_por_celula"):
            if b[chave] != ref[chave]:
                raise ValueError(f"'{chave}' diverge entre as execuções")
    linhas = [x for b in blocos for x in b["por_semente"]]
    sementes = [x["semente"] for x in linhas]
    if len(set(sementes)) != len(sementes):
        raise ValueError("semente repetida entre as execuções")
    agregado = agregar_sementes(linhas)
    teste = {}
    for obj in OBJETIVOS:
        r = agregado[obj][CB]
        z = r["media"] / r["erro_padrao"] if r["erro_padrao"] > 0 else 0.0
        teste[obj] = {"media": r["media"], "erro_padrao": r["erro_padrao"],
                      "z": z, "limiar": LIMIAR_CB,
                      "passa": bool(abs(z) >= LIMIAR_CB)}
    return {**{k: ref[k] for k in ("categoria", "configuracao", "hierarquia",
                                    "n_pares", "escopo", "corridas",
                                    "partidas_por_celula")},
            "arquivos_juntados": [str(f) for f in arquivos],
            "n_sementes": len(linhas), "sementes": sorted(sementes),
            "agregado": agregado, "teste_cb_contra_zero": teste,
            "por_semente": sorted(linhas, key=lambda x: x["semente"]),
            "tempo_total_min": sum(b["tempo_total_min"] for b in blocos)}


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
    ap.add_argument("--juntar", nargs="+", default=None,
                    help="consolida JSONs de sementes disjuntas e sai")
    a = ap.parse_args()

    if a.juntar:
        bloco = juntar(a.juntar)
        destino = (Path(a.saida) / f"sementes_item9_{bloco['categoria']}"
                   f"_{bloco['hierarquia']}_n{bloco['n_sementes']}{a.sufixo}.json")
        destino.write_text(json.dumps(bloco, indent=2, ensure_ascii=False))
        ag = bloco["agregado"]
        print(f"{bloco['n_sementes']} sementes consolidadas\n")
        for obj in OBJETIVOS:
            print(f"== {obj}")
            for k in CAMPOS:
                g = ag[obj][k]
                print(f"  {k[:48]:48s} {g['media']:9.4f} {g['erro_padrao']:7.3f} "
                      f"{g['desvio']:7.3f}")
            t = bloco["teste_cb_contra_zero"][obj]
            print(f"  C/B contra zero: {t['media']:+.3f} ± {t['erro_padrao']:.3f}, "
                  f"z {t['z']:+.2f}, limiar {t['limiar']}: "
                  f"{'PASSA' if t['passa'] else 'não passa'}")
            print(f"  sementes com C/B < 0: "
                  f"{ag[obj]['pct_sementes_decisao_negativa']:.0f}%   "
                  f"sementes em que a ingênua superestima: "
                  f"{ag[obj]['pct_sementes_superestima']:.0f}%")
        print(f"\n-> {destino}")
        return

    from varredura_rede import MELHOR

    painel = Path(a.painel)
    dados = rede.preparar(a.categoria, painel)
    centro = np.asarray(dados["centro_de_preco"], float)
    esc = P.escopo(dados, painel, a.categoria, "teste")
    esc["escalas_dos_lags"] = C9.escalas_dos_lags(dados, esc["n"])
    mapa = C9.mapa_das_corridas(esc["chaves"])

    restr, pares = None, []
    if a.hierarquia != "nenhuma":
        A, B, pares, _, inviavel = restricoes_do_escopo(
            a.categoria, painel, a.bruto, centro, esc, a.hierarquia)
        restr = (A, B, inviavel)

    print(f"{mapa['contagens']['n_celulas_elegiveis']} elegíveis, "
          f"{mapa['contagens']['n_celulas_resolvidas']} resolvidas por passada, "
          f"UMA partida\n", flush=True)
    linhas, t0 = [], time.perf_counter()
    for i in range(a.sementes):
        s = a.semente_inicial + i
        cfg = dict(MELHOR, semente=s, restrita=True)
        cfg.setdefault("gama_contextual", False)
        t = time.perf_counter()
        linha = uma_semente(dados, cfg, esc, mapa, centro, restr, a.metodo)
        linha["semente"] = s
        linha["tempo_s"] = time.perf_counter() - t
        linhas.append(linha)
        gasto = time.perf_counter() - t0
        resta = gasto * (a.sementes - i - 1) / (i + 1)
        m, r = linha["margem"], linha["receita"]
        print(f"  semente {s:2d}  obj5 margem "
              f"{m['ganho_sequencial_contra_o_OBSERVADO_pct']:6.2f}%  receita "
              f"{r['ganho_sequencial_contra_o_OBSERVADO_pct']:6.2f}%  |  C/B "
              f"{m['ganho_da_decisao_sequencial_sobre_a_ingenua_pct']:+5.2f} "
              f"{r['ganho_da_decisao_sequencial_sobre_a_ingenua_pct']:+5.2f}  |  "
              f"lag {linha['defasagem_sustentada']:+.3f}  "
              f"WMAPE {linha['wmape_teste']:5.2f}  "
              f"({linha['tempo_s']:.0f} s, ~{resta / 60:.0f} min restantes)",
              flush=True)

    agregado = agregar_sementes(linhas)
    bloco = {"categoria": a.categoria, "configuracao": dict(MELHOR),
             "hierarquia": a.hierarquia, "n_pares": len(pares),
             "escopo": esc["contagens"], "corridas": mapa["contagens"],
             "n_sementes": a.sementes, "partidas_por_celula": 1,
             "agregado": agregado, "por_semente": linhas,
             "tempo_total_min": (time.perf_counter() - t0) / 60.0}
    destino = (Path(a.saida) / f"sementes_item9_{a.categoria}"
               f"_{a.hierarquia}_n{a.sementes}{a.sufixo}.json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(bloco, indent=2, ensure_ascii=False))

    print(f"\n{a.sementes} sementes, UMA partida por célula\n")
    print(f"{'':50s} {'média':>9} {'ep':>7} {'dp':>7} {'amplit.':>8}")
    for obj in OBJETIVOS:
        print(f"== {obj}")
        for k in CAMPOS:
            g = agregado[obj][k]
            print(f"  {k[:48]:48s} {g['media']:9.4f} {g['erro_padrao']:7.3f} "
                  f"{g['desvio']:7.3f} {g['amplitude']:8.3f}")
        print(f"  sementes com C/B < 0: "
              f"{agregado[obj]['pct_sementes_decisao_negativa']:.0f}%   "
              f"sementes em que a ingênua superestima: "
              f"{agregado[obj]['pct_sementes_superestima']:.0f}%")
        cor = agregado[obj]["correlacao_ganho_calibracao"]
        if cor is not None:
            print(f"  correlação ganho × calibração entre sementes: {cor:+.3f}")
        if "pct_sementes_sinal_previsto_pela_defasagem" in agregado[obj]:
            c2 = agregado[obj]["correlacao_vies_ingenuo_defasagem_sustentada"]
            print(f"  sinal do viés ingênuo previsto pela defasagem sustentada em "
                  f"{agregado[obj]['pct_sementes_sinal_previsto_pela_defasagem']:.0f}% "
                  f"das sementes"
                  + (f", correlação {c2:+.3f}" if c2 is not None else ""))
    if "defasagem_sustentada" in agregado:
        d = agregado["defasagem_sustentada"]
        print(f"\ndefasagem sustentada (lag1 + lag2): média {d['media']:+.4f}, "
              f"dp {d['desvio']:.4f}, amplitude {d['amplitude']:.4f}, negativa em "
              f"{agregado['pct_sementes_defasagem_sustentada_negativa']:.0f}% "
              f"das sementes")
    print(f"\ntempo total {bloco['tempo_total_min']:.1f} min\n-> {destino}")


if __name__ == "__main__":
    main()
