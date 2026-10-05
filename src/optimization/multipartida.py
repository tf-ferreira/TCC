"""Multipartida da Parte 2 de D15: limite inferior e dispersão entre bacias.

O método baseado em gradiente encontra **ótimo local**. D15 recusa a forma fraca de
tratar isso ("o método encontra ótimo local, sem garantia de otimalidade global"),
que é honesta e não informa nada, e pede duas coisas mensuráveis. Esta é a segunda:
rodar a otimização de **K pontos de partida** distintos e reportar o **melhor valor
encontrado**, que é um limite inferior garantidamente alcançável, e a **dispersão dos
valores finais**, que é evidência direta de superfície acidentada sem depender de
modelo de referência nenhum.

O conjunto de partidas que converge para o mesmo ótimo local é a **bacia de atração**.
Multipartida não muda o algoritmo, só compra mais bacias, e por isso entrega limite
inferior e nunca garantia de otimalidade.

## Decisões do Thiago, 17/09/2026

- **O resultado do item 8 passa a ser o melhor sobre as K partidas**, como D15
  enuncia. A partida do ponto histórico sozinha fica reportada ao lado, e a diferença
  entre as duas é **o que a multipartida compra**, que é resultado e não ajuste.
- **K = 50**, e a partida do ponto histórico é uma das cinquenta, não uma extra.

## A pergunta nova, que não estava no plano de D15

A pendência 10.10 mostrou que a **fração de coordenadas no interior** varia de 13,87%
a 25,36% conforme a política de escolha entre os dois solvers, **sem que o ganho se
mova**: perto do ótimo o objetivo é plano em algumas direções. A multipartida mede a
dispersão daquela fração entre partidas, e é ela que decide se o número pode ir para
o texto.

## Por que as partidas são sorteadas JÁ VIÁVEIS

Sortear uniforme na caixa e deixar o solver se arrumar custaria caro: com a hierarquia
de D39 ligada, partida inviável dispara o método de reserva, que é 22 vezes mais lento
(ver `problema.resolver`). Com 50 partidas em 4.022 células e dois objetivos, uma taxa
de reserva alta transforma 0,23 h em horas.

O reparo é barato porque os conjuntos são **disjuntos**: nos 12 pares de D39 nenhum
SKU de marca própria aparece como nacional, e vice-versa. Então um passo baixando a
própria e um passo subindo a nacional bastam, cada um seguido de recorte na caixa:

    u_own ← min(u_own, min_k (u_nat(k) + b_k))      recortado em [lo_own, hi_own]
    u_nat ← max(u_nat, max_k (u_own(k) − b_k))      recortado em [lo_nat, hi_nat]

Se depois disso alguma linha ainda violar, aquele sorteio é **descartado** e substituído
pelo ponto histórico, que é viável por construção sob a forma "não piorar". A fração de
descarte é reportada, porque ela alta significaria que a caixa e a hierarquia juntas
deixam pouco espaço, e isso mudaria a leitura da dispersão.

**O sorteio é reprodutível por célula**, com semente derivada do índice da célula, para
que reexecutar o script não mude as partidas e a dispersão medida.
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
from sonda_item8 import (OBJETIVOS, carregar_modelo,  # noqa: E402
                         demanda_e_matriz, restricoes_do_escopo, _percentis)

K_PADRAO = 50


def reparar_para_viavel(U: np.ndarray, A, b, baixo, alto):
    """Empurra cada partida para dentro da hierarquia, em dois passos.

    Vale porque própria e nacional são conjuntos disjuntos nos pares de D39. Com
    `A = None` devolve `U` recortado na caixa e nenhum descarte.
    """
    U = np.clip(np.asarray(U, dtype=float), baixo, alto)
    if A is None:
        return U, np.zeros(len(U), dtype=bool)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    idx_own = [int(np.flatnonzero(linha > 0)[0]) for linha in A]
    idx_nat = [int(np.flatnonzero(linha < 0)[0]) for linha in A]

    for _ in range(2):
        # Passo 1: baixar a propria ate o teto que cada linha impoe.
        teto = np.full(U.shape, np.inf)
        for k, (o, nn) in enumerate(zip(idx_own, idx_nat)):
            teto[:, o] = np.minimum(teto[:, o], U[:, nn] + b[k])
        U = np.clip(np.minimum(U, teto), baixo, alto)
        # Passo 2: subir a nacional ate o piso que cada linha impoe.
        piso = np.full(U.shape, -np.inf)
        for k, (o, nn) in enumerate(zip(idx_own, idx_nat)):
            piso[:, nn] = np.maximum(piso[:, nn], U[:, o] - b[k])
        U = np.clip(np.maximum(U, piso), baixo, alto)

    ruim = ((U @ A.T) > b[None, :] + 1e-12).any(axis=1)
    return U, ruim


def sortear_partidas(u0: np.ndarray, baixo, alto, k: int, semente: int,
                     A=None, b=None):
    """`k` partidas para UMA célula: a histórica mais `k−1` sorteadas e reparadas.

    Uniforme na caixa, em `u`. Uniforme em `u` é uniforme no **log** do preço, não
    no preço, e isso é deliberado: a caixa é multiplicativa (±15%), de modo que
    uniforme em log trata um corte de 15% e um aumento de 15% com o mesmo peso.
    """
    rng = np.random.default_rng(semente)
    U = rng.uniform(baixo, alto, size=(max(k - 1, 0), len(u0)))
    U, ruim = reparar_para_viavel(U, A, b, baixo, alto)
    if ruim.any():
        U[ruim] = u0                      # descartadas viram a historica
    return np.vstack([np.asarray(u0, dtype=float)[None, :], U]), int(ruim.sum())


def rodar(modelo, centro, esc: dict, objetivo: str, indices: np.ndarray,
          k: int, semente: int, metodo: str, restr=None,
          passo_progresso: int = 500) -> dict:
    """K partidas por célula, e o que a dispersão delas diz.

    `passo_progresso` existe porque a execucao completa leva dezenas de minutos e
    a primeira versao deste script nao imprimia nada ate o fim, de modo que nao
    havia como distinguir "calculando" de "travado" sem olhar o `ps`.
    """
    n = esc["n"]
    custos = (esc["custos"] if objetivo == "margem"
              else np.zeros_like(esc["custos"]))
    baixo_t, alto_t = P.caixa_em_u(esc["u"])

    melhor, historico, disp, n_bacias, interior_melhor = [], [], [], [], []
    interior_disp, descartes, reservas, tempos, inviaveis = [], 0, 0, [], 0
    u_melhor = np.empty((len(indices), n), dtype=float)

    for pos, c in enumerate(indices):
        if restr is not None and restr[2][c]:
            inviaveis += 1
            u_melhor[pos] = esc["u"][c]
            continue
        r_celula = None if restr is None else (restr[0], restr[1][c])
        prob = P.ProblemaDeCelula(modelo, esc["ctx"][c], int(esc["loja_idx"][c]),
                                  centro, custos[c], esc["u"][c],
                                  restricoes=r_celula)
        U, ruim = sortear_partidas(esc["u"][c], baixo_t[c], alto_t[c], k,
                                   semente * 1_000_003 + int(c),
                                   *( (r_celula[0], r_celula[1])
                                      if r_celula else (None, None) ))
        descartes += ruim
        t0 = time.perf_counter()
        valores, interiores, us = [], [], []
        for j in range(len(U)):
            r = P.resolver(prob, u_inicial=U[j], metodo=metodo)
            reservas += int(r["usou_reserva"])
            valores.append(r["valor"])
            interiores.append(r["n_interior"])
            us.append(r["u"])
        tempos.append(time.perf_counter() - t0)
        if passo_progresso and (pos + 1) % passo_progresso == 0:
            feito = pos + 1
            gasto = float(np.sum(tempos))
            resta = gasto * (len(indices) - feito) / max(feito, 1)
            print(f"  [{objetivo}] {feito}/{len(indices)} células, "
                  f"{gasto / 60:.1f} min gastos, ~{resta / 60:.1f} min restantes",
                  flush=True)
        v = np.array(valores)
        j_melhor = int(np.argmax(v))
        melhor.append(v[j_melhor])
        historico.append(v[0])                     # a partida 0 e a historica
        u_melhor[pos] = us[j_melhor]
        interior_melhor.append(interiores[j_melhor])
        interior_disp.append(float(np.std(interiores)))
        # Dispersao RELATIVA, porque o valor tem escala de receita da celula.
        disp.append(float((v.max() - v.min()) / abs(v[j_melhor]))
                    if v[j_melhor] != 0 else 0.0)
        # Bacias: quantos valores finais DISTINTOS, a 1e-6 relativo.
        n_bacias.append(int(len(np.unique(np.round(v / abs(v[j_melhor]), 6)))))

    mel, his = np.array(melhor), np.array(historico)
    resolvidas = len(indices) - inviaveis
    return {
        "objetivo": objetivo, "k": k, "semente_partidas": semente,
        "metodo": metodo,
        "n_celulas": int(len(indices)),
        "n_celulas_inviaveis_na_caixa": inviaveis,
        "pct_partidas_descartadas": 100.0 * descartes / max(resolvidas * (k - 1), 1),
        "pct_solves_com_reserva": 100.0 * reservas / max(resolvidas * k, 1),
        "tempo_total_s": float(np.sum(tempos)),
        "tempo_por_celula_s": _percentis(np.array(tempos)),
        # O QUE A MULTIPARTIDA COMPRA, e e o numero central deste script.
        "ganho_pct_agregado_melhor": 100.0 * (mel.sum() / _base(esc, indices, inviaveis, restr, modelo, centro, custos) - 1.0),
        "melhoria_sobre_a_historica_pct": {
            k2: 100.0 * v for k2, v in _percentis(mel / his - 1.0).items()},
        "pct_celulas_em_que_a_historica_JA_era_a_melhor":
            100.0 * float(np.isclose(mel, his, rtol=1e-9).mean()),
        # Dispersao: superficie acidentada ou nao.
        "dispersao_relativa_entre_partidas": _percentis(np.array(disp)),
        "n_bacias_por_celula": _percentis(np.array(n_bacias, dtype=float)),
        "pct_celulas_com_uma_bacia_so":
            100.0 * float((np.array(n_bacias) == 1).mean()),
        # A pergunta de 10.10.
        "interior_do_melhor_por_celula": _percentis(np.array(interior_melhor,
                                                            dtype=float)),
        "pct_coord_interior_do_melhor":
            100.0 * float(np.sum(interior_melhor)) / max(resolvidas * n, 1),
        "desvio_do_interior_entre_partidas": _percentis(np.array(interior_disp)),
        "u_melhor": u_melhor,
    }


def _base(esc, indices, inviaveis, restr, modelo, centro, custos) -> float:
    """Valor total no ponto HISTÓRICO, nas mesmas células que entraram na conta."""
    import torch
    dtype = next(modelo.parameters()).dtype
    usar = [c for c in indices if restr is None or not restr[2][c]]
    with torch.no_grad():
        ctx = torch.as_tensor(esc["ctx"][usar], dtype=dtype)
        loja = torch.as_tensor(esc["loja_idx"][usar].astype("int64"),
                               dtype=torch.long)
        u = torch.as_tensor(esc["u"][usar], dtype=dtype)
        k = torch.as_tensor(custos[usar], dtype=dtype)
        cen = torch.as_tensor(np.asarray(centro), dtype=dtype)
        valor, _ = P.valor_e_gradiente(modelo, ctx, u, loja, cen, k)
        return float(valor.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--bruto", default="data/raw")
    ap.add_argument("--artefato", default=None)
    ap.add_argument("-k", type=int, default=K_PADRAO)
    ap.add_argument("--semente-partidas", type=int, default=0)
    ap.add_argument("--hierarquia", default="nao_piorar",
                    choices=("nenhuma",) + HIER.FORMAS)
    ap.add_argument("--celulas", type=int, default=0,
                    help="0 usa TODAS as células do escopo de D38")
    ap.add_argument("--semente-amostra", type=int, default=0)
    ap.add_argument("--metodo", default="SLSQP")
    ap.add_argument("--saida", default="reports")
    ap.add_argument("--sufixo", default="")
    ap.add_argument("--objetivos", default=",".join(OBJETIVOS),
                    help="subconjunto de " + ",".join(OBJETIVOS) + ". Existe "
                         "para que uma comparacao que so precisa de um objetivo "
                         "nao pague o outro")
    ap.add_argument("--progresso", type=int, default=500,
                    help="imprime andamento a cada N celulas; 0 desliga")
    a = ap.parse_args()

    painel = Path(a.painel)
    dados = rede.preparar(a.categoria, painel)
    arq = Path(a.artefato or RAIZ / "models_artifacts" / f"rede_{a.categoria}.pt")
    modelo, cfg, centro = carregar_modelo(arq, dados)
    esc = P.escopo(dados, painel, a.categoria, "teste")

    restr, pares = None, []
    if a.hierarquia != "nenhuma":
        A, B, pares, _, inviavel = restricoes_do_escopo(
            a.categoria, painel, a.bruto, centro, esc, a.hierarquia)
        restr = (A, B, inviavel)

    total = len(esc["u"])
    if a.celulas and a.celulas < total:
        rng = np.random.default_rng(a.semente_amostra)
        indices = np.sort(rng.choice(total, size=a.celulas, replace=False))
        amostra = {"tipo": "aleatoria_declarada", "n": int(a.celulas),
                   "semente": a.semente_amostra}
    else:
        indices = np.arange(total)
        amostra = {"tipo": "todas", "n": total}

    objetivos = [o.strip() for o in a.objetivos.split(",") if o.strip()]
    desconhecidos = set(objetivos) - set(OBJETIVOS)
    if desconhecidos:
        raise SystemExit(f"objetivo desconhecido: {sorted(desconhecidos)}")

    linhas = []
    for obj in objetivos:
        r = rodar(modelo, centro, esc, obj, indices, a.k, a.semente_partidas,
                  a.metodo, restr, a.progresso)
        r.pop("u_melhor")
        linhas.append(r)

    bloco = {"categoria": a.categoria, "artefato": arq.name,
             "configuracao": cfg, "hierarquia": a.hierarquia,
             "n_pares": len(pares), "escopo": esc["contagens"],
             "amostra": amostra, "k": a.k, "objetivos": objetivos,
             "resultados": linhas}
    # O nome carrega a hierarquia, senao a execucao de comparacao sobrescreve a
    # adotada, que foi o defeito que a sonda ja teve uma vez.
    destino = (Path(a.saida) / f"multipartida_item8_{a.categoria}_k{a.k}"
               f"_{a.hierarquia}{a.sufixo}.json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(bloco, indent=2, ensure_ascii=False))

    print(f"K = {a.k}, hierarquia {a.hierarquia}, amostra {amostra['tipo']} de "
          f"{amostra['n']} células\n")
    for r in linhas:
        print(f"== {r['objetivo']}")
        print(f"  ganho agregado com o MELHOR de K      {r['ganho_pct_agregado_melhor']:8.2f}%")
        print(f"  melhoria sobre a partida histórica     mediana "
              f"{r['melhoria_sobre_a_historica_pct']['mediana']:7.3f}%  "
              f"p90 {r['melhoria_sobre_a_historica_pct']['p90']:7.3f}%")
        print(f"  células em que a histórica já era a melhor "
              f"{r['pct_celulas_em_que_a_historica_JA_era_a_melhor']:6.2f}%")
        print(f"  dispersão relativa entre partidas      mediana "
              f"{r['dispersao_relativa_entre_partidas']['mediana']:.2e}  "
              f"p90 {r['dispersao_relativa_entre_partidas']['p90']:.2e}")
        print(f"  bacias distintas por célula            mediana "
              f"{r['n_bacias_por_celula']['mediana']:5.1f}  "
              f"uma só em {r['pct_celulas_com_uma_bacia_so']:.2f}% das células")
        print(f"  INTERIOR do melhor                     "
              f"{r['pct_coord_interior_do_melhor']:6.2f}% das coordenadas  "
              f"(dp entre partidas, mediana "
              f"{r['desvio_do_interior_entre_partidas']['mediana']:.2f} de {esc['n']})")
        print(f"  partidas descartadas {r['pct_partidas_descartadas']:5.2f}%   "
              f"solves com reserva {r['pct_solves_com_reserva']:5.2f}%   "
              f"tempo {r['tempo_total_s'] / 60:.1f} min\n")
    print(f"-> {destino}")


if __name__ == "__main__":
    main()
