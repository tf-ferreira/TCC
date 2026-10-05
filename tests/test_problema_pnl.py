"""Testes da formulação do PNL (item 8), antes de tocar dado real.

A convenção da seção 8 do CLAUDE.md pede teste unitário para toda formulação de
restrição do PNL **antes** do uso com dado real, e o motivo é o mesmo dos lags:
um erro de índice aqui não produz exceção, produz um gradiente que aponta para
outro lugar, e o sintoma só aparece no resultado final.

Este arquivo roda no processo do `torch` (ver `testes.sh`), junto de
`test_rede.py`, e nunca no processo do lightgbm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src" / "models"))
sys.path.insert(0, str(RAIZ / "src" / "optimization"))
sys.path.insert(0, str(RAIZ / "src" / "data"))

import problema as P  # noqa: E402
from rede import construir_modulos  # noqa: E402

torch = pytest.importorskip("torch")

N, DIM, N_LOJAS, B = 5, 4, 3, 4


def _modelo(semente: int = 0, dupla: bool = False):
    """Rede restrita com `γ` ASSIMÉTRICO, que é o que dá poder aos testes.

    `RedeDemanda` inicializa `γ` em zeros, e com `γ = 0` a fora-diagonal não
    existe: trocar linha por coluna passaria despercebido. Preencher `γ` com uma
    matriz assimétrica é o que torna o teste de transposição capaz de falhar.
    """
    mods = construir_modulos()
    m = mods["RedeDemanda"](n=N, dim_ctx=DIM, n_lojas=N_LOJAS, largura=8,
                            profundidade=2, dim_emb=3, h_monotona=4,
                            restrita=True, semente=semente)
    g = torch.Generator().manual_seed(semente + 7)
    with torch.no_grad():
        m.gama.copy_(torch.randn(N, N, generator=g) * 0.4)
    m.eval()
    return m.double() if dupla else m


def _dados(dupla: bool = False):
    dtype = torch.float64 if dupla else torch.float32
    g = torch.Generator().manual_seed(3)
    ctx = torch.randn(B, DIM, generator=g).to(dtype)
    u = (torch.randn(B, N, generator=g) * 0.1).to(dtype)
    loja = torch.randint(0, N_LOJAS, (B,), generator=g)
    centro = (torch.randn(N, generator=g) * 0.05).to(dtype)
    custo = (torch.rand(B, N, generator=g) * 0.4).to(dtype)
    return ctx, u, loja, centro, custo


# ---------------------------------------------------------------------------
# A caixa
# ---------------------------------------------------------------------------

def test_caixa_em_u_equivale_a_caixa_em_p():
    p0 = np.array([1.0, 2.5, 0.79, 1.33])
    c = np.array([0.1, -0.2, 0.05, 0.0])
    u0 = np.log(p0) - c
    baixo, alto = P.caixa_em_u(u0)
    assert np.allclose(np.exp(baixo + c), 0.85 * p0)
    assert np.allclose(np.exp(alto + c), 1.15 * p0)


def test_caixa_em_u_nao_e_simetrica():
    """Simétrica em `p` não é simétrica em `u`, e quem escrever `u0 ± 0,15`
    estará impondo outra restrição."""
    u0 = np.zeros(3)
    baixo, alto = P.caixa_em_u(u0)
    assert np.allclose(baixo, np.log(0.85))
    assert np.allclose(alto, np.log(1.15))
    assert abs(baixo[0]) > abs(alto[0])
    assert not np.allclose(-baixo, alto)


def test_caixa_rejeita_limites_incoerentes():
    with pytest.raises(ValueError):
        P.caixa_em_u(np.zeros(2), limites=(1.15, 0.85))


# ---------------------------------------------------------------------------
# Objetivo e gradiente
# ---------------------------------------------------------------------------

def test_valor_bate_com_a_soma_direta():
    m = _modelo()
    ctx, u, loja, centro, custo = _dados()
    valor, _ = P.valor_e_gradiente(m, ctx, u, loja, centro, custo)
    with torch.no_grad():
        p = torch.exp(u + centro)
        v = torch.exp(m.log_demanda(ctx, u, loja))
        direto = ((p - custo) * v).sum(dim=1)
    assert torch.allclose(valor, direto)


@pytest.mark.parametrize("com_custo", [True, False])
def test_gradiente_fechado_bate_com_autograd(com_custo):
    """A álgebra escrita à mão contra o que o grafo de fato calcula."""
    m = _modelo()
    ctx, u, loja, centro, custo = _dados()
    if not com_custo:
        custo = torch.zeros_like(custo)
    _, fechado = P.valor_e_gradiente(m, ctx, u, loja, centro, custo)
    auto = P.gradiente_por_autograd(m, ctx, u, loja, centro, custo)
    escala = float(fechado.abs().max())
    assert float((fechado - auto).abs().max()) < 1e-5 * max(escala, 1.0)


def test_gradiente_fechado_bate_com_diferenca_finita():
    """Em `float64`. Em `float32` o erro da diferença finita é `~1e-2`, grande
    demais para separar bug de ruído, e foi medido antes de escrever isto."""
    m = _modelo(dupla=True)
    ctx, u, loja, centro, custo = _dados(dupla=True)
    _, fechado = P.valor_e_gradiente(m, ctx, u, loja, centro, custo)

    h = 1e-6
    fd = torch.zeros_like(u)
    for i in range(N):
        up, um = u.clone(), u.clone()
        up[:, i] += h
        um[:, i] -= h
        vp, _ = P.valor_e_gradiente(m, ctx, up, loja, centro, custo)
        vm, _ = P.valor_e_gradiente(m, ctx, um, loja, centro, custo)
        fd[:, i] = (vp - vm) / (2 * h)
    assert float((fd - fechado).abs().max()) < 1e-6


def test_gradiente_usa_a_COLUNA_da_matriz_e_nao_a_linha():
    """O teste anterior só tem poder se a versão transposta de fato falhar.

    `εⱼᵢ = ∂ln V̂ⱼ/∂ln pᵢ` é a coluna `i`. Somar pela linha roda sem erro, porque
    a matriz é quadrada, e devolve outro gradiente. Aqui isso fica afirmado.
    """
    m = _modelo()
    ctx, u, loja, centro, custo = _dados()
    _, fechado = P.valor_e_gradiente(m, ctx, u, loja, centro, custo)
    with torch.no_grad():
        p = torch.exp(u + centro)
        v = torch.exp(m.log_demanda(ctx, u, loja))
        eps = m.elasticidades_fechadas(u)
        errado = p * v + torch.einsum("bj,bij->bi", (p - custo) * v, eps)
    assert float((fechado - errado).abs().max()) > 1e-3


def test_custo_zero_da_a_forma_da_receita():
    """Com `k = 0`, o gradiente é `Rᵢ(1 + εᵢᵢ) + Σ_{j≠i} εⱼᵢRⱼ`, que é a forma
    que `rede.diagnosticos` usa em 8.13."""
    m = _modelo()
    ctx, u, loja, centro, _ = _dados()
    zero = torch.zeros(B, N)
    _, g = P.valor_e_gradiente(m, ctx, u, loja, centro, zero)
    with torch.no_grad():
        p = torch.exp(u + centro)
        r = p * torch.exp(m.log_demanda(ctx, u, loja))
        eps = m.elasticidades_fechadas(u)
        diag = torch.diagonal(eps, dim1=1, dim2=2)
        escada = (r * (1.0 + diag)
                  + torch.einsum("bj,bji->bi", r, eps) - r * diag)
    assert torch.allclose(g, escada, atol=1e-5)


def test_derivada_propria_da_receita_e_negativa_quando_a_elasticidade_passa_de_um():
    """O achado do item 7 como invariante executável: olhando SÓ o termo próprio,
    `|εᵢᵢ| > 1` implica termo negativo, que é o que manda o preço para a borda."""
    m = _modelo()
    ctx, u, loja, centro, _ = _dados()
    with torch.no_grad():
        p = torch.exp(u + centro)
        r = p * torch.exp(m.log_demanda(ctx, u, loja))
        diag = torch.diagonal(m.elasticidades_fechadas(u), dim1=1, dim2=2)
    proprio = r * (1.0 + diag)
    assert torch.all(proprio[diag < -1.0] < 0)
    assert torch.all(proprio[diag > -1.0] >= 0)


# ---------------------------------------------------------------------------
# Custo alinhado à posição do vetor
# ---------------------------------------------------------------------------

def _painel_falso(tmp_path: Path, n: int = 3) -> Path:
    d = pd.DataFrame({
        "store": [2, 2, 5],
        "week": [10, 11, 10],
        **{f"custo_{i:02d}": [i + 0.1, i + 0.2, i + 0.3] for i in range(n)},
    })
    d.to_parquet(tmp_path / "xxx_celulas.parquet")
    return tmp_path


def test_custo_alinha_com_a_posicao_do_vetor(tmp_path):
    painel = _painel_falso(tmp_path)
    chaves = pd.DataFrame({"store": [5, 2], "week": [10, 11]})
    k = P.custos_por_celula(chaves, painel, "xxx", 3)
    assert k.shape == (2, 3)
    assert np.allclose(k[0], [0.3, 1.3, 2.3])   # loja 5, semana 10
    assert np.allclose(k[1], [0.2, 1.2, 2.2])   # loja 2, semana 11


def test_custo_falha_alto_se_a_celula_nao_existe(tmp_path):
    painel = _painel_falso(tmp_path)
    chaves = pd.DataFrame({"store": [99], "week": [10]})
    with pytest.raises(ValueError):
        P.custos_por_celula(chaves, painel, "xxx", 3)


# ---------------------------------------------------------------------------
# O solver, na caixa
# ---------------------------------------------------------------------------

def test_resolver_respeita_a_caixa_e_nao_piora_o_ponto_inicial():
    pytest.importorskip("scipy")
    m = _modelo()
    ctx, u, loja, centro, custo = _dados()
    prob = P.ProblemaDeCelula(m, ctx[0].numpy(), int(loja[0]), centro.numpy(),
                              custo[0].numpy(), u[0].numpy())
    r = P.resolver(prob)
    assert np.all(r["u"] >= prob.baixo - 1e-9)
    assert np.all(r["u"] <= prob.alto + 1e-9)
    assert r["valor"] >= r["valor_inicial"] - 1e-9


def test_resolver_conta_borda_e_interior_sem_sobreposicao():
    pytest.importorskip("scipy")
    m = _modelo()
    ctx, u, loja, centro, custo = _dados()
    prob = P.ProblemaDeCelula(m, ctx[1].numpy(), int(loja[1]), centro.numpy(),
                              custo[1].numpy(), u[1].numpy())
    r = P.resolver(prob)
    assert not np.any(r["na_borda_inferior"] & r["na_borda_superior"])
    assert (int(r["na_borda_inferior"].sum()) + int(r["na_borda_superior"].sum())
            + r["n_interior"]) == N


# ---------------------------------------------------------------------------
# Restrições lineares (hierarquia de marca, D39)
# ---------------------------------------------------------------------------

def _prob_com_restricao(forma, u0=None):
    import hierarquia as H
    m = _modelo()
    ctx, u, loja, centro, custo = _dados()
    u0 = u[0].numpy() if u0 is None else np.asarray(u0, float)
    pares = [(1, 0), (3, 2)]
    A, b = H.restricoes(pares, centro.numpy(), N, forma=forma,
                        u0=(u0[None, :] if forma == "nao_piorar" else None))
    b = b[0] if forma == "nao_piorar" else b
    return P.ProblemaDeCelula(m, ctx[0].numpy(), int(loja[0]), centro.numpy(),
                              custo[0].numpy(), u0, restricoes=(A, b))


def test_solucao_respeita_as_restricoes_lineares():
    pytest.importorskip("scipy")
    prob = _prob_com_restricao("nao_piorar")
    r = P.resolver(prob)
    assert r["sucesso"]
    assert r["violacao_maxima_das_restricoes"] < 1e-7
    assert np.all(prob.folga_das_restricoes(r["u"]) > -1e-7)


def test_nao_piorar_deixa_o_ponto_historico_viavel_no_solver():
    """A propriedade que motivou a escolha da forma: sem ela o ponto de partida
    já estaria fora da região viável e o ganho apurado misturaria otimizar com
    mudar as regras."""
    pytest.importorskip("scipy")
    prob = _prob_com_restricao("nao_piorar")
    assert np.all(prob.folga_das_restricoes(prob.u0) > -1e-12)


def test_restricao_nunca_melhora_o_valor():
    """O conjunto viável com restrição está CONTIDO no sem restrição, então o
    ótimo restrito não pode valer mais. Se valer, o solver mentiu."""
    pytest.importorskip("scipy")
    m = _modelo()
    ctx, u, loja, centro, custo = _dados()
    livre = P.ProblemaDeCelula(m, ctx[0].numpy(), int(loja[0]), centro.numpy(),
                               custo[0].numpy(), u[0].numpy())
    restrito = _prob_com_restricao("nao_piorar")
    assert P.resolver(restrito)["valor"] <= P.resolver(livre)["valor"] + 1e-9


def test_jacobiana_da_restricao_e_menos_A():
    prob = _prob_com_restricao("nao_piorar")
    (c,) = prob.restricoes_scipy
    u = prob.u0
    assert np.allclose(c["fun"](u), prob.b - prob.A @ u)
    assert np.allclose(c["jac"](u), -prob.A)


def test_sem_restricao_a_lista_do_scipy_fica_vazia():
    m = _modelo()
    ctx, u, loja, centro, custo = _dados()
    prob = P.ProblemaDeCelula(m, ctx[0].numpy(), int(loja[0]), centro.numpy(),
                              custo[0].numpy(), u[0].numpy())
    assert prob.restricoes_scipy == ()
    assert prob.folga_das_restricoes(u[0].numpy()).size == 0


def test_A_com_forma_errada_falha_alto():
    m = _modelo()
    ctx, u, loja, centro, custo = _dados()
    with pytest.raises(ValueError):
        P.ProblemaDeCelula(m, ctx[0].numpy(), int(loja[0]), centro.numpy(),
                           custo[0].numpy(), u[0].numpy(),
                           restricoes=(np.zeros((2, N + 1)), np.zeros(2)))
