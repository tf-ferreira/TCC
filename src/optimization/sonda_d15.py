"""Sonda da Parte 1 de D15: o gap até o ótimo global, e até onde o certificado escala.

D15 pede dois números que a Parte 2 (multipartida) não dá:

1. o **gap**, diferença percentual entre o valor do método aproximado e o **ótimo
   global**, num modelo de referência deliberadamente pequeno;
2. a **curva de tempo de solução contra número de produtos**, medida em 2, 3, 4 e 5,
   que é o que justifica **por medição própria** a afirmação de que o certificado não
   escala para os 20 produtos do trabalho.

## Por que existe um modelo de referência separado, e as cinco diferenças dele

A rede do item 6 **não** é codificável exatamente: ela usa Softplus (suave, não linear
por partes), ligação log na saída (`V = exp(ln V)`) e sub-rede monótona. A rede desta
sonda é outra, escolhida para ser **exatamente** codificável, e as diferenças são
declaradas porque é delas que vem o limite epistêmico que D15 já registra:

| | rede do item 6 | rede desta sonda |
|---|---|---|
| ativação | Softplus | **ReLU** |
| saída | ligação log, `exp(ln V)` | **nível, com ReLU final** |
| entrada de preço | `u = ln p − c` | **`p` normalizado** |
| monotonicidade | imposta (D33) | nenhuma |
| produtos | 20 | 2 a 5 |

A entrada em **nível** e não em log é o que evita a única transcendental que sobraria:
com `p` como entrada e a saída linear por partes, o problema fica **bilinear com
binárias** e nada mais. Com `u` haveria `p = exp(u + c)` no objetivo, e o SCIP passaria
de MIQP não convexo para MINLP com exponencial, o que mede a dificuldade da
exponencial em vez da dificuldade da rede.

**Um gap de, digamos, 3% medido aqui NÃO é a perda do modelo principal.** São modelos
diferentes, com superfícies diferentes. A transferência é **hipótese declarada**, não
resultado, e a sonda produz **evidência por analogia**. É o que D15 diz, e este
docstring repete para que ninguém leia a tabela de resultados sem ela.

## A codificação exata do ReLU, e por que ela é exata

Para um neurônio com pré-ativação `z` e ativação `a = max(0, z)`, com `z ∈ [z⁻, z⁺]`
conhecidos, e uma binária `d`:

    a ≥ z          a ≥ 0
    a ≤ z − z⁻·(1 − d)          a ≤ z⁺·d

Com `d = 1`: `a ≥ z`, `a ≤ z`, logo `a = z`, e `a ≤ z⁺` força `z ≥ 0` pela primeira.
Com `d = 0`: `a ≤ 0` e `a ≥ 0`, logo `a = 0`, e `a ≤ z − z⁻` força `z ≤ 0`. Não há
relaxação: os dois ramos são o ReLU, e nada além dele é viável.

Os limites `z⁻` e `z⁺` saem de **aritmética de intervalo exata** na camada linear, e
não de um big-M arbitrado. Um big-M grande demais não erra a resposta, mas afrouxa a
relaxação e faz o tempo explodir, o que estragaria exatamente a curva que se quer medir.

## O objetivo, em forma de epígrafo

`Σ pᵢ·Vᵢ` é bilinear, e o SCIP exige objetivo **linear**:

    max t    s.a.    t ≤ Σ pᵢ·Vᵢ

No máximo a restrição fica ativa, de modo que os dois problemas têm o mesmo ótimo.
Medido no ambiente oficial num brinquedo com solução conhecida (pendência 10.14).

Uso:

    python3 src/optimization/sonda_d15.py frj
    python3 src/optimization/sonda_d15.py frj --produtos 2,3,4,5 --limite-s 600
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

import problema as P  # noqa: E402
import rede  # noqa: E402

LIMITE_S_PADRAO = 300.0


# ---------------------------------------------------------------------------
# A rede reduzida: ReLU, saída em nível, entrada em preço
# ---------------------------------------------------------------------------

def treinar_reduzida(dados: dict, n_prod: int, h: int, epocas: int,
                     semente: int) -> dict:
    """MLP de uma camada oculta ReLU, `p` normalizado na entrada, `V` na saída.

    Treinada nos MESMOS dados, nos `n_prod` primeiros SKUs da ordem do sortimento
    (que é a de `<cat>_upcs.json`, e é declarada e não escolhida por conveniência).
    Perda quadrática em nível, porque a saída é em nível.
    """
    import torch
    from torch import nn

    torch.manual_seed(semente)
    tr, te = dados["treino"], dados["teste"]
    escala = tr["precos"][:, :n_prod].mean(axis=0)

    def entrada(parte):
        p = parte["precos"][:, :n_prod] / escala
        ctx = np.concatenate([parte["ctx_celula"], parte["ctx_sku"]], axis=1)
        return torch.tensor(np.concatenate([p, ctx], axis=1), dtype=torch.float32)

    x_tr, x_te = entrada(tr), entrada(te)
    y_tr = torch.tensor(tr["alvo"][:, :n_prod], dtype=torch.float32)

    modelo = nn.Sequential(nn.Linear(x_tr.shape[1], h), nn.ReLU(),
                           nn.Linear(h, n_prod), nn.ReLU())
    with torch.no_grad():                      # viés de saída na média do alvo
        modelo[2].bias.copy_(torch.tensor(tr["alvo"][:, :n_prod].mean(axis=0),
                                          dtype=torch.float32))
    otim = torch.optim.Adam(modelo.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(semente)
    for _ in range(epocas):
        ordem = torch.randperm(len(x_tr), generator=g)
        for k in range(0, len(x_tr), 256):
            idx = ordem[k:k + 256]
            perda = ((modelo(x_tr[idx]) - y_tr[idx]) ** 2).mean()
            otim.zero_grad()
            perda.backward()
            otim.step()
    modelo.eval()
    with torch.no_grad():
        erro = float(((modelo(x_te) - torch.tensor(te["alvo"][:, :n_prod],
                                                   dtype=torch.float32)) ** 2)
                     .mean().sqrt())
    return {
        "n_prod": n_prod, "h": h, "escala": escala,
        "W1": modelo[0].weight.detach().numpy().astype(float),
        "b1": modelo[0].bias.detach().numpy().astype(float),
        "W2": modelo[2].weight.detach().numpy().astype(float),
        "b2": modelo[2].bias.detach().numpy().astype(float),
        "rmse_teste": erro,
    }


def dobrar_contexto(pesos: dict, ctx: np.ndarray) -> tuple:
    """Absorve o contexto FIXO da célula no viés da primeira camada.

    Numa célula, o contexto é constante, de modo que `W1·[p, x] + b1` é
    `W1_p·p + (W1_x·x + b1)`. Dobrar isso no viés deixa o modelo a codificar com
    `n_prod` variáveis em vez de `n_prod + dim_ctx`, e é exato.
    """
    n = pesos["n_prod"]
    W1p, W1x = pesos["W1"][:, :n], pesos["W1"][:, n:]
    return W1p, pesos["b1"] + W1x @ np.asarray(ctx, dtype=float)


def avaliar(pesos: dict, W1p, b1e, p: np.ndarray) -> np.ndarray:
    """`V(p)` da rede reduzida, em numpy, com o contexto já dobrado."""
    z = W1p @ (np.asarray(p, dtype=float) / pesos["escala"]) + b1e
    a = np.maximum(z, 0.0)
    return np.maximum(pesos["W2"] @ a + pesos["b2"], 0.0)


def receita(pesos, W1p, b1e, p) -> float:
    p = np.asarray(p, dtype=float)
    return float(p @ avaliar(pesos, W1p, b1e, p))


def limites_das_preativacoes(W: np.ndarray, b: np.ndarray, lo: np.ndarray,
                             hi: np.ndarray) -> tuple:
    """`[z⁻, z⁺]` exatos de `W·v + b` com `v ∈ [lo, hi]`, por aritmética de intervalo.

    Exato porque cada coordenada entra linearmente e independentemente: o mínimo põe
    `v_j` no limite inferior onde `W_ij > 0` e no superior onde `W_ij < 0`.
    """
    Wp, Wn = np.clip(W, 0, None), np.clip(W, None, 0)
    return Wp @ lo + Wn @ hi + b, Wp @ hi + Wn @ lo + b


# ---------------------------------------------------------------------------
# A codificação no SCIP
# ---------------------------------------------------------------------------

def resolver_global(pesos: dict, ctx: np.ndarray, p0: np.ndarray,
                    limites=P.LIMITES_CAIXA, limite_s: float = LIMITE_S_PADRAO,
                    silencioso: bool = True) -> dict:
    """O ótimo global da receita na caixa, com certificado, ou o gap no limite."""
    from pyscipopt import Model

    n = pesos["n_prod"]
    W1p, b1e = dobrar_contexto(pesos, ctx)
    esc = pesos["escala"]
    lo_p, hi_p = limites[0] * p0[:n], limites[1] * p0[:n]
    # A camada come `p / escala`, entao os limites dela sao os de `p` divididos.
    z1_lo, z1_hi = limites_das_preativacoes(W1p, b1e, lo_p / esc, hi_p / esc)
    a1_lo, a1_hi = np.maximum(z1_lo, 0.0), np.maximum(z1_hi, 0.0)
    z2_lo, z2_hi = limites_das_preativacoes(pesos["W2"], pesos["b2"], a1_lo, a1_hi)
    v_lo, v_hi = np.maximum(z2_lo, 0.0), np.maximum(z2_hi, 0.0)

    # DIAGNOSTICO que explica o tempo: um ReLU cuja pre-ativacao nao troca de sinal
    # dentro da caixa esta FIXO, e o presolve o elimina. Se quase todos estiverem
    # fixos, a parte combinatoria colapsa e o tempo nao mede a rede, mede a largura
    # da caixa.
    fixos1 = int(np.sum((z1_lo >= 0) | (z1_hi <= 0)))
    fixos2 = int(np.sum((z2_lo >= 0) | (z2_hi <= 0)))

    m = Model("sonda_d15")
    if silencioso:
        m.hideOutput()
    m.setParam("limits/time", float(limite_s))

    p = [m.addVar(f"p{i}", lb=float(lo_p[i]), ub=float(hi_p[i])) for i in range(n)]
    a1, d1 = [], []
    for j in range(pesos["h"]):
        z = sum(float(W1p[j, i]) / float(esc[i]) * p[i] for i in range(n)) + float(b1e[j])
        aj = m.addVar(f"a1_{j}", lb=0.0, ub=float(max(a1_hi[j], 0.0)))
        dj = m.addVar(f"d1_{j}", vtype="B")
        # ReLU exato. Ver o docstring: com d=1 vale a=z, com d=0 vale a=0.
        m.addCons(aj >= z)
        m.addCons(aj <= z - float(min(z1_lo[j], 0.0)) * (1 - dj))
        m.addCons(aj <= float(max(z1_hi[j], 0.0)) * dj)
        a1.append(aj)
        d1.append(dj)

    v = []
    for i in range(n):
        z = sum(float(pesos["W2"][i, j]) * a1[j]
                for j in range(pesos["h"])) + float(pesos["b2"][i])
        vi = m.addVar(f"v{i}", lb=0.0, ub=float(max(v_hi[i], 0.0)))
        di = m.addVar(f"dv{i}", vtype="B")
        m.addCons(vi >= z)
        m.addCons(vi <= z - float(min(z2_lo[i], 0.0)) * (1 - di))
        m.addCons(vi <= float(max(z2_hi[i], 0.0)) * di)
        v.append(vi)

    # Epigrafo: o objetivo do SCIP tem de ser LINEAR, e a bilinearidade vai para a
    # restricao. No maximo ela fica ativa, de modo que t = soma p_i v_i.
    t = m.addVar("t", lb=None)
    m.addCons(t <= sum(p[i] * v[i] for i in range(n)))
    m.setObjective(t, "maximize")

    t0 = time.perf_counter()
    m.optimize()
    dt = time.perf_counter() - t0
    status = m.getStatus()
    ok = m.getNSols() > 0
    return {
        "status": status,
        "tempo_s": dt,
        "fechou": status == "optimal",
        "valor": float(m.getObjVal()) if ok else float("nan"),
        "limite_superior": float(m.getDualbound()),
        "gap_scip": float(m.getGap()) if ok else float("inf"),
        "p": [float(m.getVal(x)) for x in p] if ok else None,
        "n_binarias": pesos["h"] + n,
        "n_relus_fixos_na_caixa": fixos1 + fixos2,
        "pct_relus_fixos_na_caixa": 100.0 * (fixos1 + fixos2) / (pesos["h"] + n),
        "n_variaveis": m.getNVars(),
        "n_restricoes": m.getNConss(),
    }


# ---------------------------------------------------------------------------
# O método aproximado, no MESMO modelo reduzido
# ---------------------------------------------------------------------------

def resolver_aproximado(pesos: dict, ctx: np.ndarray, p0: np.ndarray,
                        limites=P.LIMITES_CAIXA, k: int = 50,
                        semente: int = 0) -> dict:
    """Multipartida com gradiente no modelo reduzido, para o gap ser comparável.

    Tem de ser o **mesmo modelo**, senão a diferença mediria modelo e não método. O
    gradiente é analítico pela regra da cadeia através do ReLU, que existe em quase
    todo ponto; nos pontos de dobra o subgradiente serve, e é o que o SLSQP usa de
    fato quando encontra um.
    """
    from scipy.optimize import minimize

    n = pesos["n_prod"]
    W1p, b1e = dobrar_contexto(pesos, ctx)
    esc = pesos["escala"]
    lo, hi = limites[0] * p0[:n], limites[1] * p0[:n]

    def neg_e_jac(pv):
        pv = np.asarray(pv, dtype=float)
        z1 = W1p @ (pv / esc) + b1e
        a1 = np.maximum(z1, 0.0)
        z2 = pesos["W2"] @ a1 + pesos["b2"]
        v = np.maximum(z2, 0.0)
        # dV_i/dp_j = 1[z2_i>0] * sum_h W2[i,h] * 1[z1_h>0] * W1p[h,j]/esc_j
        J = ((z2 > 0)[:, None] * (pesos["W2"] @ ((z1 > 0)[:, None] * W1p / esc)))
        return -float(pv @ v), -(v + J.T @ pv)

    rng = np.random.default_rng(semente)
    partidas = np.vstack([p0[:n][None, :], rng.uniform(lo, hi, size=(k - 1, n))])
    valores = []
    t0 = time.perf_counter()
    for ini in partidas:
        r = minimize(neg_e_jac, np.clip(ini, lo, hi), jac=True, method="SLSQP",
                     bounds=list(zip(lo.tolist(), hi.tolist())))
        valores.append(-float(r.fun))
    v = np.array(valores)
    return {"melhor": float(v.max()), "historico": float(v[0]),
            "dispersao_relativa": float((v.max() - v.min()) / abs(v.max()))
            if v.max() != 0 else 0.0,
            "k": k, "tempo_s": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--produtos", default="2,3,4,5")
    ap.add_argument("--h", type=int, default=12,
                    help="unidades na camada oculta; D15 pede 8 a 16")
    ap.add_argument("--epocas", type=int, default=15)
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--celulas", type=int, default=5,
                    help="células por tamanho, sorteadas com semente declarada")
    ap.add_argument("--limite-s", type=float, default=LIMITE_S_PADRAO)
    ap.add_argument("--raio", type=float, default=None,
                    help="raio da caixa; o padrao e o do projeto (0,15). Existe para "
                         "testar a hipotese de que o tempo baixo vem da caixa estreita "
                         "e nao da facilidade do problema")
    ap.add_argument("-k", type=int, default=50, help="partidas do método aproximado")
    ap.add_argument("--saida", default="reports")
    ap.add_argument("--sufixo", default="")
    a = ap.parse_args()

    tamanhos = [int(x) for x in a.produtos.split(",")]
    caixa = (P.LIMITES_CAIXA if a.raio is None
             else (1.0 - a.raio, 1.0 + a.raio))
    painel = Path(a.painel)
    dados = rede.preparar(a.categoria, painel)
    te = dados["teste"]
    ctx_te = np.concatenate([te["ctx_celula"], te["ctx_sku"]], axis=1)
    rng = np.random.default_rng(a.semente)
    celulas = np.sort(rng.choice(len(te["precos"]), size=a.celulas, replace=False))

    linhas = []
    for n_prod in tamanhos:
        pesos = treinar_reduzida(dados, n_prod, a.h, a.epocas, a.semente)
        W1p, b1e_ref = dobrar_contexto(pesos, ctx_te[celulas[0]])
        por_celula = []
        for c in celulas:
            ctx = ctx_te[c]
            p0 = te["precos"][c]
            glob = resolver_global(pesos, ctx, p0, limites=caixa,
                                   limite_s=a.limite_s)
            aprox = resolver_aproximado(pesos, ctx, p0, limites=caixa, k=a.k,
                                        semente=a.semente)
            base = receita(pesos, *dobrar_contexto(pesos, ctx), p0[:n_prod])
            gap = (float("nan") if not np.isfinite(glob["valor"]) or glob["valor"] == 0
                   else 100.0 * (glob["valor"] - aprox["melhor"]) / abs(glob["valor"]))
            por_celula.append({
                "celula": int(c), "global": glob, "aproximado": aprox,
                "receita_no_historico": base,
                "gap_pct_do_aproximado": gap,
                "ganho_global_pct": 100.0 * (glob["valor"] / base - 1.0)
                if base else float("nan"),
            })
            print(f"  n={n_prod}  célula {c:5d}  {glob['status']:12s} "
                  f"{glob['tempo_s']:7.2f} s  global {glob['valor']:10.2f}  "
                  f"aprox {aprox['melhor']:10.2f}  gap {gap:6.3f}%", flush=True)
        tempos = np.array([x["global"]["tempo_s"] for x in por_celula])
        gaps = np.array([x["gap_pct_do_aproximado"] for x in por_celula])
        linhas.append({
            "n_prod": n_prod, "h": a.h, "rmse_teste_reduzida": pesos["rmse_teste"],
            "n_binarias": pesos["h"] + n_prod,
            "pct_fechou": 100.0 * float(np.mean([x["global"]["fechou"]
                                                 for x in por_celula])),
            "tempo_s": {"mediana": float(np.median(tempos)),
                        "min": float(tempos.min()), "max": float(tempos.max())},
            "gap_pct": {"mediana": float(np.nanmedian(gaps)),
                        "max": float(np.nanmax(gaps))},
            "pct_relus_fixos_na_caixa": float(np.mean(
                [x["global"]["pct_relus_fixos_na_caixa"] for x in por_celula])),
            "por_celula": por_celula,
        })

    bloco = {"categoria": a.categoria, "tamanhos": tamanhos, "h": a.h,
             "epocas": a.epocas, "semente": a.semente,
             "celulas": celulas.tolist(), "limite_s": a.limite_s,
             "k_aproximado": a.k, "caixa": list(caixa),
             "resultados": linhas}
    destino = (Path(a.saida)
               / f"sonda_d15_{a.categoria}_h{a.h}{a.sufixo}.json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(bloco, indent=2, ensure_ascii=False))

    print(f"\nrede reduzida ReLU, h = {a.h}, {a.celulas} células por tamanho, "
          f"limite de {a.limite_s:.0f} s por resolução\n")
    print(f"caixa: {caixa[0]:.2f} a {caixa[1]:.2f}\n")
    print(f"{'n':>3} {'binárias':>9} {'ReLU fixos':>11} {'fechou':>8} "
          f"{'t mediana':>10} {'t max':>9} {'gap mediana':>12} {'RMSE red.':>10}")
    for L in linhas:
        print(f"{L['n_prod']:3d} {L['n_binarias']:9d} "
              f"{L['pct_relus_fixos_na_caixa']:10.1f}% {L['pct_fechou']:7.0f}% "
              f"{L['tempo_s']['mediana']:9.2f}s {L['tempo_s']['max']:8.2f}s "
              f"{L['gap_pct']['mediana']:11.3f}% {L['rmse_teste_reduzida']:10.2f}")
    if len(linhas) >= 2:
        t = [L["tempo_s"]["mediana"] for L in linhas]
        razoes = [t[i + 1] / t[i] if t[i] > 0 else float("inf")
                  for i in range(len(t) - 1)]
        print("\nrazão de tempo entre tamanhos consecutivos: "
              + ", ".join(f"{r:.2f}x" for r in razoes))
    print(f"\n-> {destino}")


if __name__ == "__main__":
    main()
