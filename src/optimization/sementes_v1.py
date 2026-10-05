"""O objetivo 5 da primeira versão escrita: faixa por mundo, política da rede e robusta.

Implementa **D44**. Por semente: treina a rede na especificação pedida (a "v1" de
D41 e D42 por padrão), decide com duas políticas e avalia cada uma em quatro
mundos de resposta cruzada (`mundos.mundos_v1`):

| | decide com | o que é |
|---|---|---|
| `rede` | a rede como está (μ = 1) | a política que acredita em γ |
| `robusta` | a média dos mundos μ ∈ {0; 0,5; 1} | a política que D44 recomenda |

Só a passada **sequencial** de D27 roda: é ela que dá o objetivo 5 (`C/R − 1`),
pela identidade de dois fatores de D40, `C/R = (C/M̂)·(M̂/R)`. A passada ingênua e
os números 1 e 2 de D27 são do item 9 e não mudam de leitura aqui. Cortar a
ingênua é o que deixa as 50 sementes caberem numa noite: a política robusta custa
cerca de três vezes a da rede, porque avalia três mundos em cada passo do solver.

A âncora dos mundos é o preço histórico da célula, o mesmo ponto em que a caixa e
a hierarquia de D39 se ancoram. Por construção todos os mundos preveem igual no
histórico, de modo que `M̂` e a calibração `M̂/R` são **os mesmos** em todos os
mundos, e o que varia entre eles é só o primeiro fator. O código confere isso em
vez de assumir.

Uso:
    python3 src/optimization/sementes_v1.py frj --sementes 2 --sufixo _sonda
    python3 src/optimization/sementes_v1.py frj --sementes 50
    python3 src/optimization/sementes_v1.py frj --especificacao adotada --sementes 20
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
import hierarquia as HIER  # noqa: E402
import mundos as MU  # noqa: E402
import problema as P  # noqa: E402
import rede  # noqa: E402
from sementes_item8 import resumo  # noqa: E402
from sonda_item8 import OBJETIVOS, restricoes_do_escopo  # noqa: E402

POLITICAS = ("rede", "robusta")


def objetivo5_no_mundo(mundo, centro, esc, mapa, custos, seq) -> dict:
    """`C/M̂`, `M̂/R` e `C/R` de uma política sequencial, avaliada num mundo."""
    S = mapa["indices_elegiveis"]
    if len(S) == 0:
        raise ValueError("nenhuma célula elegível")
    loja = esc["loja_idx"][S]
    kk = custos[S]
    W = MU.ancorar(mundo, esc["u"][S])
    C = C9.valores_em_lote(W, seq["ctx"][S], seq["u"][S], loja, centro, kk)
    M = C9.valores_em_lote(W, esc["ctx"][S], esc["u"][S], loja, centro, kk)
    R = C9.valor_observado(esc["precos"][S], esc["alvo"][S], kk)
    sC, sM, sR = float(C.sum()), float(M.sum()), float(R.sum())
    ganho = 100.0 * (sC / sM - 1.0)
    calib = sM / sR
    obj5 = 100.0 * (sC / sR - 1.0)
    if abs(100.0 * ((1 + ganho / 100.0) * calib - 1.0) - obj5) > 1e-6:
        raise ValueError("identidade de dois fatores não fecha")
    return {"ganho_modelo_contra_modelo_pct": ganho,
            "calibracao_no_historico": calib,
            "objetivo5_contra_o_observado_pct": obj5,
            "M_modelo_no_historico": sM}


def elasticidades(modelo, dados) -> dict:
    """Resumo da matriz no teste, o mesmo das sondas: própria, agregada, cruzada."""
    import torch
    te = dados["teste"]
    ctx = torch.tensor(np.concatenate([te["ctx_celula"], te["ctx_sku"]], 1),
                       dtype=torch.float32)
    u = torch.tensor(te["u"], dtype=torch.float32)
    loja = torch.tensor(te["loja_idx"], dtype=torch.long)
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


def uma_semente(dados, cfg, esc, mapa, centro, restr, metodo) -> dict:
    saida = rede.treinar(dados, cfg)
    modelo = saida["modelo"]
    modelo.eval()
    decisores = {"rede": modelo, "robusta": MU.politica_robusta_v1(modelo)}
    avaliadores = MU.mundos_v1(modelo)
    linha = {"elasticidades": elasticidades(modelo, dados)}
    for obj in OBJETIVOS:
        custos = (esc["custos"] if obj == "margem"
                  else np.zeros_like(esc["custos"]))
        linha[obj] = {}
        for pol in POLITICAS:
            seq = C9.rodar_passada(decisores[pol], centro, esc, mapa, custos, restr,
                                   1, 0, metodo, True, progresso=0)
            r = {nome: objetivo5_no_mundo(w, centro, esc, mapa, custos, seq)
                 for nome, w in avaliadores.items()}
            ms = [x["M_modelo_no_historico"] for x in r.values()]
            if max(ms) - min(ms) > 1e-6 * abs(ms[0]):
                raise ValueError("os mundos divergem no ponto histórico; a âncora falhou")
            linha[obj][pol] = {nome: {k: v for k, v in x.items()
                                      if k != "M_modelo_no_historico"}
                               for nome, x in r.items()}
            linha[obj][pol]["pct_solves_com_reserva"] = seq["pct_solves_com_reserva"]
    from baseline_arvores import metricas
    y = dados["teste"]["alvo"].ravel()
    pred = saida["pred_teste"].ravel()
    linha["wmape_teste"] = float(metricas(y, pred)["wmape_pct"])
    linha["vies_de_nivel_teste"] = float(pred.sum() / y.sum())
    return linha


def agregar_sementes(linhas: list) -> dict:
    fora = {"elasticidades": {k: resumo([x["elasticidades"][k] for x in linhas])
                              for k in linhas[0]["elasticidades"]},
            "wmape_teste": resumo([x["wmape_teste"] for x in linhas]),
            "vies_de_nivel_teste": resumo([x["vies_de_nivel_teste"] for x in linhas])}
    for obj in OBJETIVOS:
        fora[obj] = {}
        for pol in POLITICAS:
            fora[obj][pol] = {}
            for mundo in linhas[0][obj][pol]:
                if mundo == "pct_solves_com_reserva":
                    continue
                fora[obj][pol][mundo] = {
                    k: resumo([x[obj][pol][mundo][k] for x in linhas])
                    for k in linhas[0][obj][pol][mundo]}
    return fora


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--bruto", default="data/raw")
    ap.add_argument("--especificacao", default="v1", choices=sorted(rede.ESPECIFICACOES))
    ap.add_argument("--sementes", type=int, default=50)
    ap.add_argument("--semente-inicial", type=int, default=0)
    ap.add_argument("--hierarquia", default="nao_piorar",
                    choices=("nenhuma",) + HIER.FORMAS)
    ap.add_argument("--metodo", default="SLSQP")
    ap.add_argument("--saida", default="reports")
    ap.add_argument("--sufixo", default="")
    a = ap.parse_args()

    from varredura_rede import MELHOR

    rede.usar_especificacao(a.especificacao)
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

    print(f"especificação {a.especificacao}; {mapa['contagens']['n_celulas_elegiveis']} "
          f"elegíveis; políticas {POLITICAS}; mundos {MU.NOMES_V1}\n", flush=True)
    linhas, t0 = [], time.perf_counter()
    for i in range(a.sementes):
        s = a.semente_inicial + i
        cfg = dict(MELHOR, semente=s, restrita=True, especificacao=a.especificacao)
        cfg.setdefault("gama_contextual", False)
        t = time.perf_counter()
        linha = uma_semente(dados, cfg, esc, mapa, centro, restr, a.metodo)
        linha["semente"] = s
        linha["tempo_s"] = time.perf_counter() - t
        linhas.append(linha)
        resta = (time.perf_counter() - t0) * (a.sementes - i - 1) / (i + 1)
        m, r = linha["margem"], linha["receita"]
        print(f"  semente {s:2d}  obj5 robusta margem "
              + " ".join(f"{k}:{m['robusta'][k]['objetivo5_contra_o_observado_pct']:6.2f}"
                         for k in MU.NOMES_V1)
              + "  receita "
              + " ".join(f"{k}:{r['robusta'][k]['objetivo5_contra_o_observado_pct']:6.2f}"
                         for k in MU.NOMES_V1)
              + f"  própria {linha['elasticidades']['propria_mediana']:+.3f}"
              f"  WMAPE {linha['wmape_teste']:5.2f}  ({linha['tempo_s']:.0f} s, "
              f"~{resta / 60:.0f} min restantes)", flush=True)

    bloco = {"categoria": a.categoria, "especificacao": a.especificacao,
             "configuracao": dict(MELHOR), "hierarquia": a.hierarquia,
             "n_pares": len(pares), "escopo": esc["contagens"],
             "corridas": mapa["contagens"], "n_sementes": a.sementes,
             "partidas_por_celula": 1,
             "funcao_de_controle": dados["meta"].get("funcao_de_controle"),
             "agregado": agregar_sementes(linhas), "por_semente": linhas,
             "tempo_total_min": (time.perf_counter() - t0) / 60.0}
    destino = (Path(a.saida) / f"sementes_v1_{a.categoria}_{a.especificacao}"
               f"_{a.hierarquia}_n{a.sementes}{a.sufixo}.json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(bloco, indent=2, ensure_ascii=False))
    ag = bloco["agregado"]
    print(f"\n{a.sementes} sementes; objetivo 5 (C/R − 1), média e ep")
    for obj in OBJETIVOS:
        print(f"== {obj}")
        for pol in POLITICAS:
            print(f"  {pol:8s} " + "  ".join(
                f"{k}: {ag[obj][pol][k]['objetivo5_contra_o_observado_pct']['media']:6.2f}"
                f" ± {ag[obj][pol][k]['objetivo5_contra_o_observado_pct']['erro_padrao']:4.2f}"
                for k in ag[obj][pol]))
    print(f"\ntempo total {bloco['tempo_total_min']:.1f} min\n-> {destino}")


if __name__ == "__main__":
    main()
