"""O problema de PNL do item 8: objetivo, gradiente analítico e caixa.

Implementa **D38** (escopo: janela de teste, células com preço e custo completos,
os dois objetivos na mesma amostra) e a formulação da seção 3.3 do projeto de
pesquisa sobre a rede treinada do item 6, com os pesos congelados.

## A condição que este módulo existe para exercitar

Com `Rᵢ = pᵢV̂ᵢ`, a derivada da receita total em relação ao preço próprio é

    ∂Z/∂uᵢ  =  Rᵢ·(1 + εᵢᵢ)  +  Σ_{j≠i} γⱼᵢ·Rⱼ
               └── o próprio ──┘   └── o cruzado ──┘

O item 7 mediu `|εᵢᵢ| > 1` em toda a caixa de ±15% para os 20 SKUs, de modo que o
primeiro termo é **negativo em todo ponto viável**. Segue que:

| o cruzado contra o próprio | derivada | solução |
|---|---|---|
| sempre menor em módulo | sempre < 0 | borda inferior, −15% |
| sempre maior | sempre > 0 | borda superior, +15% |
| igual em algum ponto da caixa | cruza zero | **ótimo interior** |

Ou seja: o item 8 só produz otimização se o saldo **mudar de sinal** dentro da
caixa. Se não mudar, quem decidiu o preço foi a restrição arbitrada, não a rede.
O diagnóstico 8.13 (`rede.diagnosticos`) já mede isso **coordenada a coordenada**,
com os outros N−1 preços no histórico, e deu 9,49%. O problema conjunto não é a
soma de N problemas de uma variável, porque mover todos os preços muda todo `Rⱼ`
que aparece no termo cruzado, e é só isso que o solver responde.

## Um objetivo, dois problemas

Receita é margem com custo zero:

    M(p) = Σᵢ (pᵢ − kᵢ)·V̂ᵢ(p)        Z(p) = M(p) com k ≡ 0

de modo que uma implementação, com `k` como parâmetro, entrega os dois objetivos
de D21. O gradiente único, em forma compacta que vale para os dois:

    ∂M/∂uᵢ = pᵢV̂ᵢ + Σⱼ (pⱼ − kⱼ)·V̂ⱼ·εⱼᵢ          (a soma inclui j = i)

Com `k = 0` isso é exatamente `Rᵢ + Σⱼ Rⱼεⱼᵢ`, que é a forma que `rede.diagnosticos`
já usa em 8.13. Não é coincidência e não deve virar duas fórmulas: aqui está a
versão geral, e 8.13 é o caso `k = 0` avaliado nas bordas.

**Atenção ao índice.** `εⱼᵢ = ∂ln V̂ⱼ/∂ln pᵢ` é a **coluna** i da matriz, não a
linha. Trocar linha por coluna é silencioso: a matriz é quadrada, o código roda,
e o gradiente aponta para outro lugar. É o que
`test_gradiente_fechado_bate_com_autograd` existe para pegar.

## A variável de decisão é `u`, e isto é uma RETRATAÇÃO

A primeira versão desta formulação decidia em `p`, com o argumento de que a
hierarquia de marca de D22 é linear em `p` e deixaria de ser em `u`. **Errado.**
A hierarquia é uma comparação de **razões**, e tomar log de razão preserva
linearidade:

    p_own/oz_own ≤ p_nat/oz_nat
    ⟺ u_own − u_nat ≤ (c_nat − c_own) + ln(oz_own/oz_nat)

As duas famílias de restrição do projeto sobrevivem em `u`, e a caixa também,
porque `ln` é monótona. Decidir em `u` elimina a multiplicação por `1/pᵢ` da regra
da cadeia, que é um ponto a mais para errar sinal ou índice sem que nada acuse.

**O que faria a decisão virar de novo:** qualquer restrição que seja **soma** de
preços (preço médio da cesta limitado, orçamento promocional fixo). Soma de preços
é linear em `p` e vira soma de exponenciais em `u`. Nenhuma existe no projeto hoje.

## Precisão, e ela não é detalhe aqui

A rede é `float32`. A conferência do gradiente por **diferença finita** em
`float32` tem erro relativo da ordem de `1e-7/h`, isto é, cerca de `1e-2` com
`h = 1e-5`, o que é grande demais para distinguir um bug de ruído. Por isso a
conferência de diferença finita roda em `float64` (`modelo.double()`), e a de
autograd, que é exata, roda na precisão nativa. Medido: `float64` dá concordância
de `8e-12`, `float32` dá `8e-3`, com a mesma álgebra.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src" / "models"))

from rede import LIMITES_CAIXA  # noqa: E402


def _torch():
    import torch
    return torch


# ---------------------------------------------------------------------------
# A caixa
# ---------------------------------------------------------------------------

def caixa_em_u(u0: np.ndarray, limites=LIMITES_CAIXA) -> tuple:
    """Os limites da caixa de ±15% escritos em `u`, não em `p`.

    `pᵢ ∈ [ℓ·pᵢ⁰, h·pᵢ⁰]` equivale a `uᵢ ∈ [uᵢ⁰ + ln ℓ, uᵢ⁰ + ln h]`, porque `ln`
    é monótona crescente e `uᵢ = ln pᵢ − cᵢ` desloca por constante.

    A caixa é **simétrica em `p`** e **assimétrica em `u`**: `ln 0,85 = −0,1625`
    contra `ln 1,15 = +0,1398`. Isso não é defeito, é o que a restrição de negócio
    de fato diz, e quem escrever `u0 ± 0,15` estará impondo outra restrição.
    """
    baixo, alto = float(limites[0]), float(limites[1])
    if not 0 < baixo <= 1 <= alto:
        raise ValueError(f"limites incoerentes: {limites}")
    return u0 + np.log(baixo), u0 + np.log(alto)


# ---------------------------------------------------------------------------
# Objetivo e gradiente
# ---------------------------------------------------------------------------

def valor_e_gradiente(modelo, ctx, u, loja, centro, custo):
    """`(M, ∂M/∂u)` por célula, com a matriz de elasticidades em forma fechada.

    Argumentos são tensores do torch: `ctx` (B, dim), `u` (B, n), `loja` (B,),
    `centro` (n,) o `cᵢ` de D28, `custo` (B, n) o `kᵢ` de D38. Passe `custo`
    nulo para obter receita.

    Devolve `(valor, gradiente)` com formas (B,) e (B, n). O gradiente é
    **analítico**, não por autograd: a diagonal vem de `mᵢ′(uᵢ)` em forma fechada
    e a fora-diagonal é `γ`, que D36 fixou constante.
    """
    torch = _torch()
    # D44: um conjunto (a política robusta) tem objetivo igual à MÉDIA do
    # objetivo dos membros, e gradiente igual à média dos gradientes. A média é
    # linear, de modo que a álgebra de cada membro fica intocada.
    if hasattr(modelo, "membros"):
        pares = [valor_e_gradiente(m, ctx, u, loja, centro, custo)
                 for m in modelo.membros]
        k = float(len(pares))
        return (sum(v for v, _ in pares) / k, sum(g for _, g in pares) / k)
    with torch.no_grad():
        p = torch.exp(u + centro)
        v = torch.exp(modelo.log_demanda(ctx, u, loja))
        eps = modelo.elasticidades_fechadas(u)
        m = (p - custo) * v                                   # (B, n)
        valor = m.sum(dim=1)                                  # (B,)
        # g_i = p_i·v_i + Σ_j m_j·ε_ji   → a COLUNA i de ε, via "bj,bji->bi"
        grad = p * v + torch.einsum("bj,bji->bi", m, eps)
    return valor, grad


def gradiente_por_autograd(modelo, ctx, u, loja, centro, custo):
    """O mesmo gradiente por diferenciação automática, e ele existe para CONFERIR.

    Mesma razão de `rede.elasticidades_por_autograd`: as duas vias concordarem é
    o teste de que a álgebra escrita à mão bate com o que o grafo calcula. Esta
    via também é a única disponível se `γ` passar a ser contextual.
    """
    torch = _torch()
    if hasattr(modelo, "membros"):
        gs = [gradiente_por_autograd(m, ctx, u, loja, centro, custo)
              for m in modelo.membros]
        return sum(gs) / float(len(gs))
    u = u.clone().requires_grad_(True)
    p = torch.exp(u + centro)
    v = torch.exp(modelo.log_demanda(ctx, u, loja))
    (g,) = torch.autograd.grad(((p - custo) * v).sum(), u)
    return g


# ---------------------------------------------------------------------------
# Escopo (D38)
# ---------------------------------------------------------------------------

def custos_por_celula(chaves: pd.DataFrame, painel, categoria: str,
                      n: int) -> np.ndarray:
    """`kᵢ` alinhado às células de `chaves`, na ordem de posição do painel.

    O custo vive no painel de células e **não** no painel de modelagem, que o
    exclui de propósito: ele afeta a demanda só através do preço e é o
    instrumento de D18. Aqui ele entra como coeficiente do objetivo, não como
    atributo da rede, de modo que a exclusão de D18 não é violada.

    A coluna `custo_ii` corresponde à posição `i` do vetor, a mesma que
    `painel.pivotar` usa para `preco_ii`, que é a ordem de `<cat>_upcs.json`.
    Confiar nessa correspondência é o que torna obrigatório o teste
    `test_custo_alinha_com_a_posicao_do_vetor`.
    """
    colunas = [f"custo_{i:02d}" for i in range(n)]
    cel = pd.read_parquet(Path(painel) / f"{categoria}_celulas.parquet",
                          columns=["store", "week"] + colunas)
    cel = cel.set_index(["store", "week"])
    idx = pd.MultiIndex.from_arrays(
        [chaves["store"].to_numpy(), chaves["week"].to_numpy()],
        names=["store", "week"])
    faltando = (~idx.isin(cel.index)).sum()
    if faltando:
        raise ValueError(f"{faltando} células da partição não estão no painel "
                         "de células; ordem ou partição divergiram")
    return cel.reindex(idx)[colunas].to_numpy(float)


def escopo(dados: dict, painel, categoria: str, particao: str = "teste") -> dict:
    """As células que o item 8 otimiza, e a contagem que D38 devia ao registro.

    Aplica as três partes de D38: partição de teste, custo ausente excluído e
    nunca imputado, e a mesma amostra para os dois objetivos.

    Devolve os tensores da partição já recortados, mais `contagens`, que é o
    bloco que fecha as pendências 10.1 e 10.2: os números de escopo passam a ter
    produtor em vez de sair de comando ad hoc em conversa.
    """
    n = dados["n"]
    parte = dados[particao]
    k = custos_por_celula(parte["chaves"], painel, categoria, n)
    completo = ~np.isnan(k).any(axis=1)

    p = parte["precos"]
    dentro = k[completo]
    if (dentro <= 0).any():
        raise ValueError("custo não positivo nas células selecionadas")
    if (dentro >= p[completo]).any():
        raise ValueError("custo maior ou igual ao preço nas células selecionadas")

    recorte = {
        "n": n,
        "chaves": parte["chaves"].loc[completo].reset_index(drop=True),
        "u": parte["u"][completo],
        "precos": p[completo],
        "custos": dentro,
        "ctx": np.concatenate([parte["ctx_celula"], parte["ctx_sku"]], axis=1)[completo],
        "loja_idx": parte["loja_idx"][completo],
        "alvo": parte["alvo"][completo],
    }
    recorte["contagens"] = {
        "particao": particao,
        "n_celulas_da_particao": int(len(completo)),
        "n_celulas_com_custo_completo": int(completo.sum()),
        "n_celulas_excluidas_por_custo": int((~completo).sum()),
        "pct_excluidas_por_custo": float(100.0 * (~completo).mean()),
        "pct_entradas_de_custo_ausentes": float(100.0 * np.isnan(k).mean()),
        "margem_relativa_mediana": float(
            np.median(1.0 - dentro / p[completo])),
        "semana_min": int(recorte["chaves"]["week"].min()),
        "semana_max": int(recorte["chaves"]["week"].max()),
        "n_lojas": int(recorte["chaves"]["store"].nunique()),
    }
    return recorte


# ---------------------------------------------------------------------------
# Uma célula, pronta para o scipy
# ---------------------------------------------------------------------------

class ProblemaDeCelula:
    """O problema de uma célula: N variáveis, caixa, objetivo e gradiente.

    Cada célula é um problema **independente**: a rede não liga células, de modo
    que não existe acoplamento entre lojas nem entre semanas. O acoplamento que
    importa é entre os N preços **dentro** da célula, e é ele que o solver resolve.

    O scipy **minimiza**, então `fun` devolve `−M`. Quem ler o valor tem de
    trocar o sinal de volta, e por isso `resolver` já devolve o valor com o sinal
    econômico.
    """

    def __init__(self, modelo, ctx, loja_idx, centro, custo, u0,
                 limites=LIMITES_CAIXA, restricoes=None):
        torch = _torch()
        self.modelo = modelo
        dtype = next(modelo.parameters()).dtype
        self.ctx = torch.as_tensor(np.asarray(ctx)[None, :], dtype=dtype)
        self.loja = torch.as_tensor(np.asarray([loja_idx]), dtype=torch.long)
        self.centro = torch.as_tensor(np.asarray(centro), dtype=dtype)
        self.custo = torch.as_tensor(np.asarray(custo)[None, :], dtype=dtype)
        self.u0 = np.asarray(u0, dtype=float)
        self.dtype = dtype
        self.baixo, self.alto = caixa_em_u(self.u0, limites)
        # Restricoes lineares `A·u <= b` desta celula (hierarquia de marca, D39).
        # `None` significa so a caixa, que e o problema antes da emenda a D22.
        if restricoes is None:
            self.A = self.b = None
        else:
            self.A = np.asarray(restricoes[0], dtype=float)
            self.b = np.asarray(restricoes[1], dtype=float).ravel()
            if self.A.ndim != 2 or self.A.shape[1] != len(self.u0):
                raise ValueError(f"A com forma {self.A.shape} nao casa com n="
                                 f"{len(self.u0)}")
            if self.b.shape[0] != self.A.shape[0]:
                raise ValueError("A e b com numero de linhas diferente")

    @property
    def limites_scipy(self) -> list:
        return list(zip(self.baixo.tolist(), self.alto.tolist()))

    @property
    def restricoes_scipy(self) -> tuple:
        """As lineares no formato do scipy, que espera `fun(u) >= 0`.

        Por isso a forma e `b - A·u` e a Jacobiana e `-A`, constante. Passar a
        Jacobiana importa: sem ela o SLSQP diferencia a restricao por diferenca
        finita, o que custa uma avaliacao por variavel por iteracao e introduz
        erro numerico numa restricao que e exata.
        """
        if self.A is None:
            return ()
        A, b = self.A, self.b
        return ({"type": "ineq", "fun": lambda u: b - A @ u,
                 "jac": lambda u: -A},)

    def folga_das_restricoes(self, u_np) -> np.ndarray:
        """`b - A·u`. Negativo e violacao."""
        if self.A is None:
            return np.zeros(0)
        return self.b - self.A @ np.asarray(u_np, dtype=float)

    def fun_e_jac(self, u_np: np.ndarray):
        torch = _torch()
        u = torch.as_tensor(np.asarray(u_np)[None, :], dtype=self.dtype)
        valor, grad = valor_e_gradiente(self.modelo, self.ctx, u, self.loja,
                                        self.centro, self.custo)
        return -float(valor[0]), -grad[0].numpy().astype(float)

    def valor(self, u_np: np.ndarray) -> float:
        return -self.fun_e_jac(u_np)[0]


# Tolerancia de viabilidade, DECLARADA na unidade em que ela se julga.
#
# `u` e log de preco, de modo que uma violacao `v` em `u` e `exp(v) - 1` de preco.
# O criterio e meio centavo no preco tipico da categoria (US$ 1,15 ponderado por
# volume, D21): 0,005 / 1,15 = 4,3e-3. Abaixo disso a violacao e menor que a
# RESOLUCAO do dado, porque os precos do DFF vem em centavos, e exigir mais seria
# exigir do solver precisao que o preco nao tem.
#
# 1e-7, que era o valor anterior, nao e um criterio, e um numero pequeno: ele
# aciona a reserva por residuo numerico do trust-constr e depois nao consegue ser
# cumprido nem por ela, o que deixa o relatorio dizendo que verifica algo que nao
# verifica. Medido em 17/09/2026 no ambiente oficial: a violacao residual maxima
# com a reserva ligada e 2,75e-04 em `u`, ou 0,0275% do preco, que num preco de
# US$ 1,22 sao 0,0034 centavos.
TOL_VIABILIDADE = 4.3e-3

# Gatilho da reserva, que e OUTRO papel e por isso outro numero.
#
# Colapsar os dois num so foi erro meu, e ele mudou resultado: com o gatilho em
# 4,3e-3 a reserva deixa de ser acionada em cerca de 15% das celulas da receita, e
# ali fica valendo a resposta do SLSQP, que prega coordenadas na borda. Medido em
# 18/09/2026, 4.022 celulas: o interior da receita cai de 25,36% para 13,87%
# enquanto o ganho agregado nao se move (41,14% contra 41,17%).
#
# A leitura disso e que perto do otimo o objetivo e PLANO em algumas direcoes: o
# valor e robusto a escolha do solver e a classificacao borda/interior nao e. Logo
# o gatilho deve ser generoso (tentar a reserva sempre que houver duvida) e a
# escolha entre os dois candidatos deve ser por VALOR entre os viaveis, nao por
# quem chegou primeiro.
TOL_GATILHO_RESERVA = 1e-7


def _uma_tentativa(problema, u_ini, metodo, opcoes):
    from scipy.optimize import minimize
    return minimize(problema.fun_e_jac, u_ini, jac=True, method=metodo,
                    bounds=problema.limites_scipy,
                    constraints=problema.restricoes_scipy,
                    options=opcoes or {})


def resolver(problema: ProblemaDeCelula, u_inicial=None, metodo: str = "SLSQP",
             opcoes: dict | None = None, metodo_reserva: str | None = "trust-constr",
             opcoes_reserva: dict | None = None,
             tol_viabilidade: float = TOL_VIABILIDADE,
             tol_gatilho: float = TOL_GATILHO_RESERVA) -> dict:
    """Uma otimizacao, de um ponto de partida, com gradiente exato e RESERVA.

    `jac=True` diz ao scipy que `fun` devolve `(valor, gradiente)` junto, o que
    evita recalcular a passada direta da rede para obter o gradiente. Sem isso o
    numero de avaliacoes da rede dobra.

    ## Por que existe metodo de reserva, e ele foi medido antes de ser escrito

    Com a hierarquia de marca ligada (D39) o SLSQP falha, e falha DEVOLVENDO
    ponto inviavel. Medido em 400 celulas da janela de teste, objetivo receita:

    | metodo | falhas | violacao maxima | ganho mediano | tempo |
    |---|---:|---:|---:|---:|
    | SLSQP | 51/400 | **1,11e-01** | 35,68% | 2,6 s |
    | SLSQP, maxiter 500 | 51/400 | 1,11e-01 | 35,68% | 2,4 s |
    | trust-constr | 17/400 | **1,93e-07** | 37,14% | 56,6 s |

    Violacao de 1,1e-01 em `u` e cerca de 11% em preco: nao e residuo numerico, e
    solucao que nao respeita a restricao. Mais iteracao nao conserta, o que aponta
    para conjunto ativo degenerado: na receita muitas coordenadas vao para a borda
    da caixa e varias linhas da hierarquia amarram ao mesmo tempo, e o SLSQP
    tropeca nessa combinacao. O trust-constr resolve, com ganho um pouco MAIOR (o
    que confirma que as falhas do SLSQP eram falhas e nao otimos), mas e 22 vezes
    mais lento.

    Dai a reserva: tenta o rapido, e troca pelo robusto **so** onde o rapido falha
    ou devolve ponto inviavel. Com 12,75% de reserva o custo fica em cerca de
    0,025 s por celula, contra 0,14 s se tudo rodasse no trust-constr.

    `usou_reserva` vai no resultado de proposito: a taxa de reserva e diagnostico,
    e se ela subir muito e sinal de que a formulacao mudou de dificuldade.
    """
    u_ini = problema.u0 if u_inicial is None else np.asarray(u_inicial, float)
    u_ini = np.clip(u_ini, problema.baixo, problema.alto)

    def _falta(r):
        return -problema.folga_das_restricoes(np.asarray(r.x, float)).min(initial=0.0)

    res = _uma_tentativa(problema, u_ini, metodo, opcoes)
    usada = metodo
    # GATILHO: generoso de proposito. Qualquer duvida manda tentar a reserva.
    if (not res.success or _falta(res) > tol_gatilho) and metodo_reserva:
        alt = _uma_tentativa(problema, u_ini, metodo_reserva,
                             opcoes_reserva or {"maxiter": 300})
        # ESCOLHA entre os dois: viabilidade na tolerancia DECLARADA primeiro,
        # valor depois. Nunca "quem chegou primeiro".
        viavel, viavel_alt = _falta(res) <= tol_viabilidade, _falta(alt) <= tol_viabilidade
        if viavel_alt and (not viavel or alt.fun <= res.fun):
            res, usada = alt, metodo_reserva
        elif (not viavel) and (not viavel_alt) and _falta(alt) < _falta(res):
            # Nenhum dos dois cumpre: fica o MENOS inviavel, e o relatorio acusa.
            res, usada = alt, metodo_reserva

    u = np.asarray(res.x, dtype=float)
    folga_baixo = u - problema.baixo
    folga_alto = problema.alto - u
    tol = 1e-8
    folga_lin = problema.folga_das_restricoes(u)
    return {
        "u": u,
        "valor": -float(res.fun),
        "valor_inicial": problema.valor(u_ini),
        "sucesso": bool(res.success),
        "mensagem": str(res.message),
        "metodo_usado": usada,
        "usou_reserva": usada != metodo,
        "viavel": bool(float(max(0.0, -folga_lin.min(initial=0.0))) <= tol_viabilidade),
        "n_avaliacoes": int(res.nfev),
        "n_iteracoes": int(getattr(res, "nit", -1)),
        "na_borda_inferior": (folga_baixo <= tol),
        "na_borda_superior": (folga_alto <= tol),
        "n_interior": int(((folga_baixo > tol) & (folga_alto > tol)).sum()),
        "n_restricoes": 0 if problema.A is None else int(problema.A.shape[0]),
        "n_restricoes_ativas": int((folga_lin <= 1e-8).sum()),
        "violacao_maxima_das_restricoes": float(max(0.0, -folga_lin.min(initial=0.0))),
        # A MESMA violacao lida em PRECO, que e a unidade em que ela se julga.
        # `u` e log de preco, entao `exp(v) - 1` e a fracao do preco.
        "violacao_maxima_em_fracao_de_preco": float(
            np.expm1(max(0.0, -folga_lin.min(initial=0.0)))),
    }
