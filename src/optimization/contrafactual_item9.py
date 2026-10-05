"""O contrafactual do item 9: sequencial em D27, e contra a receita REAL.

Este módulo existe porque o item 8 **não** respondeu o objetivo específico 5. O
número dele, 42,69% de receita e 21,92% de margem, tem no denominador
`Σ pₕᵢₛₜ·V̂(pₕᵢₛₜ)`, que é `multipartida._base`: o modelo avaliado no ponto
histórico. É modelo contra modelo. O objetivo 5 pede modelo contra **dado**.

Duas correções distintas separam um do outro, e confundi-las seria reportar um
número certo para a pergunta errada.

## Correção 1, a defasagem (D27, pendência 7.6)

`preco_lag1` e `preco_lag2` são atributos da rede. Numa avaliação em que o
otimizador reprecifica toda semana, o preço defasado da semana `t` é o preço que
o **próprio otimizador** pôs em `t−1`, e não o histórico. A avaliação com
defasagem histórica erra com **direção**: ela corta onde cortar parece bom e
nunca registra o empréstimo intertemporal que o corte gera. O brinquedo de D27
reporta 940 onde a verdade é 780.

Isso gera três objetos, e o módulo calcula os três sobre as MESMAS células:

| | política | defasagem na avaliação | o que é |
|---|---|---|---|
| **A** | resolvida com defasagem histórica | histórica | o que o item 8 reporta |
| **B** | a mesma de A | a que a própria A produz | o que A entregaria de fato |
| **C** | resolvida com a defasagem propagada | a que ela própria produz | a política sequencial |

`A/B − 1` é **quanto a avaliação ingênua superestima**, que é o número 1 de D27.
`C/B − 1` é o que resolver com a defasagem certa compra sobre simplesmente
aplicar a política ingênua, e é separado do anterior de propósito: um é erro de
**avaliação**, o outro é erro de **decisão**.

## Correção 2, o nível não cancela (decisão do Thiago, 18/09/2026)

Numa razão entre duas avaliações do mesmo modelo, um fator multiplicativo de
calibração some. Contra a receita real ele sobrevive. A pendência 9.4 já
registrou que o viés relevante **não** é o 0,9663 agregado de volume: 14 dos 20
SKUs são superprevistos no teste, três subprevistos respondem por 233% do
déficit, e o otimizador muda o mix.

A decisão adotada **não escolhe rota de correção**, porque toda rota assumiria
alguma coisa. Ela usa uma identidade exata:

    C / R_real  =  [C / M̂_hist]  ×  [M̂_hist / R_real]
                    └─ ganho ─┘      └─ calibração ─┘

O primeiro fator é da mesma família do item 8, modelo contra modelo, e é onde o
otimizador aparece. O segundo é medível hoje e nunca foi medido: é a calibração
do objetivo, ponderada por preço, nas células do escopo (o 0,9663 é de volume,
sem peso de preço, sobre 4.323 células e não sobre as do escopo de D38). O
produto é o número do objetivo 5, e nada nele é estipulado. O código confere a
identidade em vez de confiar nela.

**O que fica declarado e não medido:** se a calibração vale nos preços
contrafactuais. Não é medível, e por motivo estrutural: não existe volume
observado em preço que não foi praticado. Vai para a seção de limitações, não
para uma hipótese embutida no número.

## A regra da cadeia, e a exclusão (decisão do Thiago, 18/09/2026)

A cadeia sequencial vive dentro de uma **corrida**: semanas consecutivas da mesma
loja, todas no escopo de D38. Quando a semana `t−1` está fora do escopo, a
célula `t` não tem preço recomendado para herdar, e a decisão é **excluir**, não
cair no histórico. Cair no histórico misturaria as duas avaliações dentro da
mesma soma, que é exatamente o defeito que D27 existe para tirar.

As duas primeiras células de cada corrida são de **aquecimento**: a primeira não
tem nenhuma das duas defasagens em mãos do otimizador e a segunda só tem uma.
Elas são resolvidas e **propagam** o preço, mas ficam fora da amostra reportada.
Isso é inferência minha e não decisão do Thiago, e a alternativa seria começar a
corrida com defasagem histórica declarada, que é o que um piloto real faria.

Consequência que o texto tem de declarar: a amostra do item 9 é **menor** que as
4.022 células do item 8, e por isso o módulo reporta o ganho ingênuo nas duas
amostras. A diferença entre eles é efeito de **amostra**, não de método.

## O que NÃO muda, e é decisão de manter

A **caixa de ±15% e a hierarquia de D39 continuam ancoradas no preço histórico**
da própria célula, não no preço otimizado da semana anterior. Ancorar no anterior
deixaria o preço derivar sem limite ao longo da corrida e tiraria de D39 a
viabilidade por construção. O que a cadeia propaga é o **atributo** `preco_lagK`,
e nada mais.

## A miopia continua, e isto MEDE em vez de resolver

O otimizador decide uma célula por vez. Ele enxerga o ganho da semana `t` e não
enxerga o que aquele preço faz com a semana `t+1`. D27 declara isso como
limitação e não a contorna. O número 2 de D27, `C / M̂_hist − 1`, **pode sair
negativo** e continuar sendo resultado: significaria que reprecificar semana a
semana com otimizador míope não compensa.

Uso:
    python3 src/optimization/contrafactual_item9.py frj --sonda
    python3 src/optimization/contrafactual_item9.py frj -k 50
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

import hierarquia as HIER  # noqa: E402,F401
import mundos  # noqa: E402
import problema as P  # noqa: E402
import rede  # noqa: E402
from multipartida import sortear_partidas  # noqa: E402
from sonda_item8 import (OBJETIVOS, _percentis, carregar_modelo,  # noqa: E402
                         restricoes_do_escopo)

K_PADRAO = 50
# Células de aquecimento por corrida: a 1ª não tem nenhuma defasagem do
# otimizador, a 2ª só tem `lag1`. Da 3ª em diante a cadeia está cheia.
AQUECIMENTO = 2
# As quatro defasagens abrem CONTEXTO_SKU em toda especificação (rede.py).
LAGS_NA_ORDEM = ["preco_lag1", "preco_lag2", "falta_preco_lag1", "falta_preco_lag2"]


# ---------------------------------------------------------------------------
# Onde os lags moram dentro de `ctx`, e em que escala
# ---------------------------------------------------------------------------

def fatias_dos_lags(n: int) -> dict:
    """As colunas de `ctx` que guardam os quatro blocos de `CONTEXTO_SKU`.

    `escopo` monta `ctx = [ctx_celula | ctx_sku]`, e `pivotar` achata
    `CONTEXTO_SKU` na ordem `preco_lag1, preco_lag2, falta_preco_lag1,
    falta_preco_lag2`, cada um com `n` colunas. Errar esta conta é silencioso:
    o vetor tem o tamanho certo e a rede responde a outro atributo.

    A largura do contexto de célula é lida **na hora do uso**, e não na
    importação: desde D41 ela depende da especificação ativa (9 colunas na
    "adotada", 7 na "v1"), e uma constante de importação apontaria para a errada.
    """
    if rede.CONTEXTO_SKU[:4] != LAGS_NA_ORDEM:
        raise ValueError(f"as defasagens não abrem CONTEXTO_SKU: {rede.CONTEXTO_SKU}")
    b = len(rede.CONTEXTO_CELULA)
    return {"lag1": slice(b, b + n), "lag2": slice(b + n, b + 2 * n),
            "falta1": slice(b + 2 * n, b + 3 * n),
            "falta2": slice(b + 3 * n, b + 4 * n)}


def escalas_dos_lags(dados: dict, n: int) -> dict:
    """`(μ, σ)` do TREINO para cada bloco, que é o que D28 exige.

    O `ctx` do escopo já vem escalonado. Para escrever um preço novo ali é
    preciso aplicar a MESMA transformação, com as estatísticas do treino, e
    nunca reestimá-las na janela de teste.
    """
    mu = np.asarray(dados["escalonador"]["ctx_sku"]["media"], float)
    sd = np.asarray(dados["escalonador"]["ctx_sku"]["desvio"], float)
    esperado = len(rede.CONTEXTO_SKU) * n
    if len(mu) != esperado:
        raise ValueError(f"escalonador de ctx_sku com {len(mu)} colunas, "
                         f"esperado {esperado}")
    if rede.CONTEXTO_SKU[:4] != LAGS_NA_ORDEM:
        raise ValueError(f"as defasagens não abrem CONTEXTO_SKU: {rede.CONTEXTO_SKU}")
    return {"lag1": (mu[:n], sd[:n]), "lag2": (mu[n:2 * n], sd[n:2 * n]),
            "falta1": (mu[2 * n:3 * n], sd[2 * n:3 * n]),
            "falta2": (mu[3 * n:4 * n], sd[3 * n:4 * n])}


def ctx_com_lags(ctx0: np.ndarray, fat: dict, esc_lags: dict,
                 p1=None, p2=None) -> np.ndarray:
    """Cópia de `ctx0` com `preco_lag1` e/ou `preco_lag2` trocados por `p`.

    `None` mantém o valor histórico daquele bloco, que é o que as células de
    aquecimento usam.
    """
    ctx = np.array(ctx0, dtype=float, copy=True)
    for p, chave in ((p1, "lag1"), (p2, "lag2")):
        if p is None:
            continue
        mu, sd = esc_lags[chave]
        ctx[fat[chave]] = (np.asarray(p, float) - mu) / sd
    return ctx


# ---------------------------------------------------------------------------
# As corridas
# ---------------------------------------------------------------------------

def corridas(chaves) -> list:
    """Blocos de semanas CONSECUTIVAS da mesma loja, em ordem de semana.

    Devolve lista de arrays com as POSIÇÕES em `esc`. Uma corrida de tamanho
    `L` contribui `max(L − AQUECIMENTO, 0)` células para a amostra reportada.
    """
    loja = chaves["store"].to_numpy()
    semana = chaves["week"].to_numpy()
    fora = []
    for l in np.unique(loja):
        pos = np.flatnonzero(loja == l)
        pos = pos[np.argsort(semana[pos])]
        w = semana[pos]
        quebra = np.flatnonzero(np.diff(w) != 1) + 1
        fora.extend(np.split(pos, quebra))
    return [c for c in fora if len(c) > 0]


def mapa_das_corridas(chaves) -> dict:
    """Corridas, células elegíveis e as contagens que o texto tem de declarar."""
    cs = corridas(chaves)
    tam = np.array([len(c) for c in cs], dtype=float)
    elegiveis = np.concatenate([c[AQUECIMENTO:] for c in cs if len(c) > AQUECIMENTO]) \
        if any(len(c) > AQUECIMENTO for c in cs) else np.zeros(0, dtype=int)
    usadas = np.concatenate([c for c in cs if len(c) > AQUECIMENTO]) \
        if any(len(c) > AQUECIMENTO for c in cs) else np.zeros(0, dtype=int)
    total = len(chaves)
    return {
        "corridas": cs,
        "indices_elegiveis": np.sort(elegiveis).astype(int),
        "indices_resolvidos": np.sort(usadas).astype(int),
        "contagens": {
            "n_celulas_do_escopo": int(total),
            "n_corridas": int(len(cs)),
            "tamanho_das_corridas": _percentis(tam) if len(tam) else {},
            "n_corridas_curtas_demais": int((tam <= AQUECIMENTO).sum()),
            "n_celulas_resolvidas": int(len(usadas)),
            "n_celulas_de_aquecimento": int(len(usadas) - len(elegiveis)),
            "n_celulas_elegiveis": int(len(elegiveis)),
            "pct_do_escopo_elegivel": float(100.0 * len(elegiveis) / max(total, 1)),
            "n_excluidas_por_quebra_de_cadeia": int(total - len(elegiveis)),
            "semanas_de_aquecimento_por_corrida": AQUECIMENTO,
        },
    }


# ---------------------------------------------------------------------------
# Avaliação em lote
# ---------------------------------------------------------------------------

def valores_em_lote(modelo, CTX, U, loja_idx, centro, CUSTO) -> np.ndarray:
    """`M(p)` por célula, para `ctx` e `u` arbitrários. Sem gradiente."""
    import torch
    dtype = next(modelo.parameters()).dtype
    with torch.no_grad():
        valor, _ = P.valor_e_gradiente(
            modelo,
            torch.as_tensor(np.asarray(CTX), dtype=dtype),
            torch.as_tensor(np.asarray(U), dtype=dtype),
            torch.as_tensor(np.asarray(loja_idx).astype("int64"), dtype=torch.long),
            torch.as_tensor(np.asarray(centro), dtype=dtype),
            torch.as_tensor(np.asarray(CUSTO), dtype=dtype))
    return valor.numpy().astype(float)


def valor_observado(precos: np.ndarray, alvo: np.ndarray,
                    custos: np.ndarray) -> np.ndarray:
    """`Σᵢ (pᵢ − kᵢ)·Vᵢ` com o volume **observado**. É o lado do DADO."""
    return ((np.asarray(precos, float) - np.asarray(custos, float))
            * np.asarray(alvo, float)).sum(axis=1)


# ---------------------------------------------------------------------------
# Uma célula
# ---------------------------------------------------------------------------

def resolver_celula(modelo, centro, esc, c: int, ctx_c, custo_c, r_celula,
                    k: int, semente: int, metodo: str) -> dict:
    """K partidas numa célula, com `ctx` arbitrário. Devolve o MELHOR.

    A semente da partida é a mesma fórmula de `multipartida.rodar`, derivada do
    índice da célula, para que a passada ingênua e a sequencial partam dos
    MESMOS pontos. Sem isso a diferença entre elas misturaria efeito de
    defasagem com efeito de sorteio.
    """
    # D44: um mundo ou a política robusta são ancorados no preço HISTÓRICO da
    # célula, o mesmo ponto em que a caixa e a hierarquia se ancoram. A rede pura
    # passa sem mudança.
    prob = P.ProblemaDeCelula(mundos.ancorar(modelo, esc["u"][c]), ctx_c,
                              int(esc["loja_idx"][c]), centro,
                              custo_c, esc["u"][c], restricoes=r_celula)
    baixo, alto = P.caixa_em_u(esc["u"][c])
    U, ruim = sortear_partidas(esc["u"][c], baixo, alto, k,
                               semente * 1_000_003 + int(c),
                               *((r_celula[0], r_celula[1]) if r_celula
                                 else (None, None)))
    valores, us, reservas = [], [], 0
    for j in range(len(U)):
        r = P.resolver(prob, u_inicial=U[j], metodo=metodo)
        reservas += int(r["usou_reserva"])
        valores.append(r["valor"])
        us.append(r["u"])
    v = np.asarray(valores, float)
    jm = int(np.argmax(v))
    return {"u": np.asarray(us[jm], float), "valor": float(v[jm]),
            "descartes": int(ruim), "reservas": int(reservas)}


# ---------------------------------------------------------------------------
# As duas passadas
# ---------------------------------------------------------------------------

def _precos(u: np.ndarray, centro: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(u, float) + np.asarray(centro, float))


def rodar_passada(modelo, centro, esc, mapa, custos, restr, k: int,
                  semente: int, metodo: str, sequencial: bool,
                  progresso: int = 500) -> dict:
    """Uma passada completa sobre as corridas.

    `sequencial=False` resolve toda célula com a defasagem HISTÓRICA, que é o
    problema do item 8 restrito às corridas. `sequencial=True` alimenta
    `preco_lagK` da célula com o preço que esta mesma passada recomendou `K`
    posições antes na corrida.

    Devolve `u` e o `ctx` EFETIVAMENTE usado em cada célula resolvida, porque o
    `ctx` da sequencial não é reconstruível depois sem repetir o laço.
    """
    n = esc["n"]
    fat = fatias_dos_lags(n)
    esc_lags = esc["escalas_dos_lags"]
    total = len(esc["u"])
    U = np.array(esc["u"], dtype=float, copy=True)
    CTX = np.array(esc["ctx"], dtype=float, copy=True)
    resolvidas = np.zeros(total, dtype=bool)
    descartes = reservas = inviaveis = 0
    t0 = time.perf_counter()
    feito = 0

    for corrida in mapa["corridas"]:
        if len(corrida) <= AQUECIMENTO:
            continue
        for j, c in enumerate(corrida):
            c = int(c)
            if sequencial:
                p1 = _precos(U[int(corrida[j - 1])], centro) if j >= 1 else None
                p2 = _precos(U[int(corrida[j - 2])], centro) if j >= 2 else None
                ctx_c = ctx_com_lags(esc["ctx"][c], fat, esc_lags, p1, p2)
            else:
                ctx_c = np.array(esc["ctx"][c], dtype=float, copy=True)
            CTX[c] = ctx_c
            if restr is not None and restr[2][c]:
                inviaveis += 1
                resolvidas[c] = True
                feito += 1
                continue
            r_celula = None if restr is None else (restr[0], restr[1][c])
            r = resolver_celula(modelo, centro, esc, c, ctx_c, custos[c],
                                r_celula, k, semente, metodo)
            U[c] = r["u"]
            descartes += r["descartes"]
            reservas += r["reservas"]
            resolvidas[c] = True
            feito += 1
            if progresso and feito % progresso == 0:
                gasto = time.perf_counter() - t0
                alvo = len(mapa["indices_resolvidos"])
                print(f"  [{'seq' if sequencial else 'ing'}] {feito}/{alvo} "
                      f"células, {gasto / 60:.1f} min gastos, "
                      f"~{gasto * (alvo - feito) / max(feito, 1) / 60:.1f} min "
                      f"restantes", flush=True)

    n_res = int(resolvidas.sum())
    return {"u": U, "ctx": CTX, "resolvidas": resolvidas,
            "sequencial": bool(sequencial),
            "n_resolvidas": n_res,
            "n_inviaveis_na_caixa": inviaveis,
            "pct_partidas_descartadas":
                100.0 * descartes / max((n_res - inviaveis) * (k - 1), 1),
            "pct_solves_com_reserva":
                100.0 * reservas / max((n_res - inviaveis) * k, 1),
            "tempo_total_s": float(time.perf_counter() - t0)}


def ctx_honesto_da_politica(esc, mapa, U_pol, centro) -> np.ndarray:
    """O `ctx` que a política `U_pol` produziria se fosse de fato aplicada.

    Não resolve nada: só reescreve a defasagem com o preço que a própria
    política pôs na semana anterior. É o que transforma **A** em **B**, e é a
    diferença entre o que a avaliação ingênua REPORTA e o que a política
    ingênua ENTREGARIA.
    """
    fat = fatias_dos_lags(esc["n"])
    esc_lags = esc["escalas_dos_lags"]
    CTX = np.array(esc["ctx"], dtype=float, copy=True)
    for corrida in mapa["corridas"]:
        if len(corrida) <= AQUECIMENTO:
            continue
        for j, c in enumerate(corrida):
            if j < AQUECIMENTO:
                continue
            CTX[int(c)] = ctx_com_lags(
                esc["ctx"][int(c)], fat, esc_lags,
                _precos(U_pol[int(corrida[j - 1])], centro),
                _precos(U_pol[int(corrida[j - 2])], centro))
    return CTX


# ---------------------------------------------------------------------------
# Os números
# ---------------------------------------------------------------------------

def _razao(num: float, den: float) -> float:
    return 100.0 * (num / den - 1.0)


def agregar(modelo, centro, esc, mapa, custos, ing: dict, seq: dict,
            ctx_honesto: np.ndarray, objetivo: str) -> dict:
    """Os cinco agregados, sobre as MESMAS células elegíveis, e as razões.

    A identidade de dois fatores é CONFERIDA e não assumida: o produto do ganho
    modelo-contra-modelo pela calibração tem de dar o ganho contra o dado, a
    menos de erro de ponto flutuante.
    """
    S = mapa["indices_elegiveis"]
    if len(S) == 0:
        raise ValueError("nenhuma célula elegível; a cadeia não fechou em "
                         "corrida nenhuma")
    loja = esc["loja_idx"][S]
    kk = custos[S]
    # A avaliação pode ser num mundo DIFERENTE do que decidiu (D44): `ing` e `seq`
    # trazem as políticas, `modelo` é o mundo que as avalia. Âncora nas células S.
    modelo = mundos.ancorar(modelo, esc["u"][S])

    A = valores_em_lote(modelo, esc["ctx"][S], ing["u"][S], loja, centro, kk)
    B = valores_em_lote(modelo, ctx_honesto[S], ing["u"][S], loja, centro, kk)
    C = valores_em_lote(modelo, seq["ctx"][S], seq["u"][S], loja, centro, kk)
    M = valores_em_lote(modelo, esc["ctx"][S], esc["u"][S], loja, centro, kk)
    R = valor_observado(esc["precos"][S], esc["alvo"][S], kk)

    sA, sB, sC, sM, sR = (float(x.sum()) for x in (A, B, C, M, R))
    ganho_modelo = _razao(sC, sM)
    calibracao = sM / sR
    ganho_real = _razao(sC, sR)
    reconstruido = 100.0 * ((1.0 + ganho_modelo / 100.0) * calibracao - 1.0)
    erro_identidade = abs(reconstruido - ganho_real)
    if erro_identidade > 1e-6:
        raise ValueError(f"identidade de dois fatores não fecha: "
                         f"{reconstruido} contra {ganho_real}")

    return {
        "objetivo": objetivo,
        "n_celulas_elegiveis": int(len(S)),
        # Os cinco agregados, na unidade do objetivo.
        "agregados": {
            "A_ingenua_reportada": sA,
            "B_ingenua_honesta": sB,
            "C_sequencial": sC,
            "M_modelo_no_historico": sM,
            "R_observado_no_historico": sR,
        },
        # D27, número 1: erro de AVALIAÇÃO, mesma política, duas defasagens.
        "superestimacao_da_avaliacao_ingenua_pct": _razao(sA, sB),
        # O que resolver com a defasagem certa compra sobre aplicar a ingênua.
        "ganho_da_decisao_sequencial_sobre_a_ingenua_pct": _razao(sC, sB),
        # O número do item 8 restrito a esta amostra, para separar efeito de
        # amostra de efeito de método.
        "ganho_ingenuo_modelo_contra_modelo_pct": _razao(sA, sM),
        # D27, número 2, em unidade de modelo: reprecificar compensa?
        "ganho_sequencial_modelo_contra_modelo_pct": ganho_modelo,
        # A identidade de dois fatores (decisão do Thiago, 18/09/2026).
        "calibracao_do_objetivo_no_ponto_historico": calibracao,
        "ganho_sequencial_contra_o_OBSERVADO_pct": ganho_real,
        "erro_da_identidade_de_dois_fatores": erro_identidade,
        # Distribuições por célula.
        "por_celula": {
            "sequencial_sobre_modelo_pct": {
                k2: 100.0 * v for k2, v in _percentis(C / M - 1.0).items()},
            "sequencial_sobre_observado_pct": {
                k2: 100.0 * v for k2, v in _percentis(C / R - 1.0).items()},
            "superestimacao_ingenua_pct": {
                k2: 100.0 * v for k2, v in _percentis(A / B - 1.0).items()},
            "calibracao": _percentis(M / R),
        },
        "pct_celulas_em_que_a_sequencial_PERDE_para_o_historico_no_modelo":
            100.0 * float((C < M).mean()),
        "pct_celulas_em_que_a_sequencial_PERDE_para_o_observado":
            100.0 * float((C < R).mean()),
        "pct_celulas_em_que_a_ingenua_superestima":
            100.0 * float((A > B).mean()),
        # Por célula, BRUTO. Sai do JSON e vai para um .npz: é o que permite erro
        # padrão agrupado por loja (D24) sem repetir a otimização. A primeira
        # execução de 18/09 não gravou isto, e por isso não tem erro padrão.
        "_bruto": {"indice": np.asarray(S), "A": A, "B": B, "C": C, "M": M,
                   "R": R},
    }


# ---------------------------------------------------------------------------
# Erro padrão agrupado por loja (D24)
# ---------------------------------------------------------------------------

RAZOES = {
    "superestimacao_da_avaliacao_ingenua_pct": ("A", "B"),
    "ganho_da_decisao_sequencial_sobre_a_ingenua_pct": ("C", "B"),
    "ganho_ingenuo_modelo_contra_modelo_pct": ("A", "M"),
    "ganho_sequencial_modelo_contra_modelo_pct": ("C", "M"),
    "ganho_sequencial_contra_o_OBSERVADO_pct": ("C", "R"),
}


def bootstrap_por_loja(bruto: dict, loja: np.ndarray, n_boot: int = 2000,
                       semente: int = 0) -> dict:
    """Erro padrão e intervalo de 95% das razões de somas, reamostrando LOJAS.

    O grupo é a loja e não a célula, pela razão de D24: células da mesma loja
    compartilham embedding, clientela e a própria cadeia sequencial, e tratá-las
    como independentes encolheria o erro padrão por construção. Reamostrar a
    loja inteira leva junto toda essa dependência.

    As razões são de **somas**, como no ponto, e não médias de razões por
    célula: a média de razões dá peso igual a célula pequena e grande e cruza
    zero no denominador, que é o erro registrado em 9.4.
    """
    lojas = np.unique(loja)
    pos = {l: np.flatnonzero(loja == l) for l in lojas}
    somas = {k: np.array([bruto[k][pos[l]].sum() for l in lojas])
             for k in ("A", "B", "C", "M", "R")}
    rng = np.random.default_rng(semente)
    sorteio = rng.integers(0, len(lojas), size=(n_boot, len(lojas)))
    tot = {k: v[sorteio].sum(axis=1) for k, v in somas.items()}
    fora = {"n_lojas": int(len(lojas)), "n_boot": int(n_boot)}
    for nome, (num, den) in RAZOES.items():
        r = 100.0 * (tot[num] / tot[den] - 1.0)
        fora[nome] = {"ep": float(r.std(ddof=1)),
                      "ic95": [float(np.quantile(r, 0.025)),
                               float(np.quantile(r, 0.975))],
                      "pct_boot_abaixo_de_zero": 100.0 * float((r < 0).mean())}
    cal = tot["M"] / tot["R"]
    fora["calibracao_do_objetivo_no_ponto_historico"] = {
        "ep": float(cal.std(ddof=1)),
        "ic95": [float(np.quantile(cal, 0.025)), float(np.quantile(cal, 0.975))]}
    return fora


# ---------------------------------------------------------------------------
# A sonda: tudo o que se mede sem resolver nada
# ---------------------------------------------------------------------------

def checar_indicadores_de_falta(esc, mapa) -> dict:
    """`falta_preco_lagK` tem de ser 0 nas células elegíveis, e isto CONFERE.

    A cadeia só fecha em semanas consecutivas dentro do escopo, e semana
    anterior no escopo significa preço observado, logo defasagem definida. Se
    algum indicador vier 1 a premissa está errada e a célula teria defasagem
    preenchida pela mediana do treino, que não é preço de semana nenhuma.
    """
    n = esc["n"]
    fat = fatias_dos_lags(n)
    e = esc["escalas_dos_lags"]
    S = mapa["indices_elegiveis"]
    fora = {}
    for chave in ("falta1", "falta2"):
        mu, sd = e[chave]
        cru = esc["ctx"][S][:, fat[chave]] * sd + mu
        fora[chave] = {
            "pct_entradas_com_falta_1": float(100.0 * (cru > 0.5).mean()),
            "maximo": float(cru.max()) if cru.size else 0.0,
        }
    return fora


def sonda(modelo, centro, esc, mapa, objetivos) -> dict:
    """Calibração e estrutura das corridas, sem uma única otimização."""
    S = mapa["indices_elegiveis"]
    todos = np.arange(len(esc["u"]))
    blocos = {}
    for obj in objetivos:
        custos = (esc["custos"] if obj == "margem"
                  else np.zeros_like(esc["custos"]))
        linha = {}
        for nome, idx in (("escopo_inteiro", todos), ("elegiveis", S)):
            M = valores_em_lote(modelo, esc["ctx"][idx], esc["u"][idx],
                                esc["loja_idx"][idx], centro, custos[idx])
            R = valor_observado(esc["precos"][idx], esc["alvo"][idx], custos[idx])
            linha[nome] = {
                "n_celulas": int(len(idx)),
                "modelo_no_historico": float(M.sum()),
                "observado_no_historico": float(R.sum()),
                "calibracao_agregada": float(M.sum() / R.sum()),
                "calibracao_por_celula": _percentis(M / R),
                "pct_celulas_com_modelo_ABAIXO_do_observado":
                    100.0 * float((M < R).mean()),
            }
        blocos[obj] = linha
    return {"corridas": mapa["contagens"],
            "indicadores_de_falta": checar_indicadores_de_falta(esc, mapa),
            "calibracao": blocos}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

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
    ap.add_argument("--metodo", default="SLSQP")
    ap.add_argument("--saida", default="reports")
    ap.add_argument("--sufixo", default="")
    ap.add_argument("--objetivos", default=",".join(OBJETIVOS))
    ap.add_argument("--progresso", type=int, default=500)
    ap.add_argument("--sonda", action="store_true",
                    help="só a estrutura das corridas e a calibração; "
                         "nenhuma otimização")
    a = ap.parse_args()

    painel = Path(a.painel)
    dados = rede.preparar(a.categoria, painel)
    arq = Path(a.artefato or RAIZ / "models_artifacts" / f"rede_{a.categoria}.pt")
    modelo, cfg, centro = carregar_modelo(arq, dados)
    esc = P.escopo(dados, painel, a.categoria, "teste")
    esc["escalas_dos_lags"] = escalas_dos_lags(dados, esc["n"])
    mapa = mapa_das_corridas(esc["chaves"])

    objetivos = [o.strip() for o in a.objetivos.split(",") if o.strip()]
    desconhecidos = set(objetivos) - set(OBJETIVOS)
    if desconhecidos:
        raise SystemExit(f"objetivo desconhecido: {sorted(desconhecidos)}")

    if a.sonda:
        bloco = {"categoria": a.categoria, "artefato": arq.name,
                 "escopo": esc["contagens"], "sonda": sonda(modelo, centro, esc,
                                                            mapa, objetivos)}
        destino = Path(a.saida) / f"sonda_item9_{a.categoria}{a.sufixo}.json"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(bloco, indent=2, ensure_ascii=False))
        c = bloco["sonda"]["corridas"]
        print(f"escopo D38: {c['n_celulas_do_escopo']} células, "
              f"{c['n_corridas']} corridas\n"
              f"  corrida mediana {c['tamanho_das_corridas'].get('mediana', 0):.0f} "
              f"semanas, {c['n_corridas_curtas_demais']} curtas demais\n"
              f"  resolvidas {c['n_celulas_resolvidas']}  "
              f"aquecimento {c['n_celulas_de_aquecimento']}  "
              f"ELEGÍVEIS {c['n_celulas_elegiveis']} "
              f"({c['pct_do_escopo_elegivel']:.1f}% do escopo)")
        print("\nindicadores de falta nas elegíveis (têm de ser 0):")
        for k2, v in bloco["sonda"]["indicadores_de_falta"].items():
            print(f"  {k2}: {v['pct_entradas_com_falta_1']:.3f}% com 1")
        print("\ncalibração (modelo / observado) no ponto histórico:")
        for obj, linha in bloco["sonda"]["calibracao"].items():
            for nome, d in linha.items():
                print(f"  {obj:8s} {nome:15s} n={d['n_celulas']:5d}  "
                      f"agregada {d['calibracao_agregada']:.4f}  "
                      f"mediana por célula {d['calibracao_por_celula']['mediana']:.4f}")
        print(f"\n-> {destino}")
        return

    # A checagem barata ANTES da cara: uma execução completa leva dezenas de
    # minutos, e as duas condições abaixo a tornariam inútil.
    c0 = mapa["contagens"]
    if c0["n_celulas_elegiveis"] == 0:
        raise SystemExit("nenhuma célula elegível: a cadeia não fechou em "
                         "corrida nenhuma. Rode --sonda e veja §12.3.")
    falta = checar_indicadores_de_falta(esc, mapa)
    ruins = {k2: v for k2, v in falta.items() if v["pct_entradas_com_falta_1"] > 0}
    if ruins:
        print(f"AVISO: indicador de falta não nulo nas elegíveis: {ruins}.\n"
              "  D40 declara isto como critério de invalidação da construção "
              "da corrida.", flush=True)
    print(f"elegíveis {c0['n_celulas_elegiveis']} de "
          f"{c0['n_celulas_do_escopo']} células do escopo, em "
          f"{c0['n_corridas']} corridas\n", flush=True)

    restr = None
    pares = []
    if a.hierarquia != "nenhuma":
        A, B, pares, _, inviavel = restricoes_do_escopo(
            a.categoria, painel, a.bruto, centro, esc, a.hierarquia)
        restr = (A, B, inviavel)

    linhas, diagnosticos = [], []
    for obj in objetivos:
        custos = (esc["custos"] if obj == "margem"
                  else np.zeros_like(esc["custos"]))
        print(f"== {obj}: passada INGÊNUA (defasagem histórica)", flush=True)
        ing = rodar_passada(modelo, centro, esc, mapa, custos, restr, a.k,
                            a.semente_partidas, a.metodo, False, a.progresso)
        ctx_honesto = ctx_honesto_da_politica(esc, mapa, ing["u"], centro)
        print(f"== {obj}: passada SEQUENCIAL (D27)", flush=True)
        seq = rodar_passada(modelo, centro, esc, mapa, custos, restr, a.k,
                            a.semente_partidas, a.metodo, True, a.progresso)
        linhas.append(agregar(modelo, centro, esc, mapa, custos, ing, seq,
                              ctx_honesto, obj))
        diagnosticos.append({
            "objetivo": obj,
            "ingenua": {k2: v for k2, v in ing.items()
                        if k2 not in ("u", "ctx", "resolvidas")},
            "sequencial": {k2: v for k2, v in seq.items()
                           if k2 not in ("u", "ctx", "resolvidas")}})

    bloco = {"categoria": a.categoria, "artefato": arq.name,
             "configuracao": cfg, "hierarquia": a.hierarquia,
             "n_pares": len(pares), "escopo": esc["contagens"],
             "corridas": mapa["contagens"],
             "indicadores_de_falta": falta,
             "k": a.k, "semente_partidas": a.semente_partidas,
             "objetivos": objetivos, "resultados": linhas,
             "diagnosticos": diagnosticos}
    destino = (Path(a.saida) / f"contrafactual_item9_{a.categoria}_k{a.k}"
               f"_{a.hierarquia}{a.sufixo}.json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    brutos = {}
    for r in linhas:
        b = r.pop("_bruto")
        for chave, v in b.items():
            brutos[f"{r['objetivo']}_{chave}"] = v
    idx = brutos[f"{linhas[0]['objetivo']}_indice"]
    brutos["loja"] = esc["chaves"]["store"].to_numpy()[idx]
    for r in linhas:
        b = {k: brutos[f"{r['objetivo']}_{k}"] for k in ("A", "B", "C", "M", "R")}
        r["bootstrap_por_loja"] = bootstrap_por_loja(b, brutos["loja"])
    brutos["semana"] = esc["chaves"]["week"].to_numpy()[idx]
    np.savez_compressed(destino.with_suffix(".npz"), **brutos)
    destino.write_text(json.dumps(bloco, indent=2, ensure_ascii=False))

    c = mapa["contagens"]
    print(f"\nelegíveis {c['n_celulas_elegiveis']} de "
          f"{c['n_celulas_do_escopo']} células do escopo\n")
    for r in linhas:
        print(f"== {r['objetivo']}")
        print(f"  D27 nº1  a avaliação ingênua superestima em "
              f"{r['superestimacao_da_avaliacao_ingenua_pct']:8.2f}%")
        print(f"  D27 nº2  ganho da sequencial contra não fazer nada, no modelo "
              f"{r['ganho_sequencial_modelo_contra_modelo_pct']:8.2f}%")
        print(f"           (o ingênuo, na mesma amostra, dizia "
              f"{r['ganho_ingenuo_modelo_contra_modelo_pct']:8.2f}%)")
        print(f"  resolver com a defasagem certa compra "
              f"{r['ganho_da_decisao_sequencial_sobre_a_ingenua_pct']:8.2f}%")
        print(f"  calibração no ponto histórico "
              f"{r['calibracao_do_objetivo_no_ponto_historico']:8.4f}")
        print(f"  OBJETIVO 5: ganho contra o OBSERVADO "
              f"{r['ganho_sequencial_contra_o_OBSERVADO_pct']:8.2f}%")
        print(f"  a sequencial perde para o histórico em "
              f"{r['pct_celulas_em_que_a_sequencial_PERDE_para_o_historico_no_modelo']:5.2f}% "
              f"das células (modelo) e "
              f"{r['pct_celulas_em_que_a_sequencial_PERDE_para_o_observado']:5.2f}% "
              f"(observado)")
        bt = r["bootstrap_por_loja"]
        print(f"  erro padrão agrupado por loja ({bt['n_lojas']} lojas, "
              f"{bt['n_boot']} reamostras):")
        for nome in RAZOES:
            e = bt[nome]
            print(f"    {nome[:52]:52s} ep {e['ep']:6.2f}  "
                  f"IC95 [{e['ic95'][0]:7.2f}, {e['ic95'][1]:7.2f}]  "
                  f"<0 em {e['pct_boot_abaixo_de_zero']:5.1f}%")
        print()
    print(f"-> {destino}\n-> {destino.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
