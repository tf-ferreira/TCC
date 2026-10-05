"""Testes do reescalamento de elasticidades (sensibilidade de D21).

O risco é o invólucro mexer no que não devia: se ele mudar a previsão no ponto
histórico, o ganho passa a ser razão entre coisas diferentes e a sensibilidade mede
nível em vez de elasticidade.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
for sub in ("models", "data", "experiments", "optimization"):
    sys.path.insert(0, str(RAIZ / "src" / sub))

torch = pytest.importorskip("torch")
import problema as P  # noqa: E402
import sensibilidade_item8 as S  # noqa: E402
from rede import construir_modulos  # noqa: E402

N, DIM, N_LOJAS = 5, 4, 3


def _modelo():
    m = construir_modulos()["RedeDemanda"](
        n=N, dim_ctx=DIM, n_lojas=N_LOJAS, largura=8, profundidade=2, dim_emb=3,
        h_monotona=4, restrita=True, semente=0)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        m.gama.copy_(torch.randn(N, N, generator=g) * 0.4)
    m.eval()
    return m


def _dados(b=3):
    g = torch.Generator().manual_seed(3)
    return (torch.randn(b, DIM, generator=g),
            (torch.randn(b, N, generator=g) * 0.1),
            torch.randint(0, N_LOJAS, (b,), generator=g),
            (torch.randn(N, generator=g) * 0.05),
            (torch.rand(b, N, generator=g) * 0.4))


@pytest.mark.parametrize("lam", [0.5, 1.0, 1.5])
def test_a_previsao_NO_PONTO_HISTORICO_nao_muda(lam):
    """A ancoragem é o ponto todo do invólucro: em u⁰ o modelo escalado tem de
    prever exatamente o mesmo, senão a sensibilidade mede nível."""
    m = _modelo()
    ctx, u, loja, centro, _ = _dados()
    u0 = u[0].numpy()
    esc = S.ModeloEscalado(m, lam, u0)
    uu = torch.as_tensor(u0[None, :], dtype=torch.float32)
    with torch.no_grad():
        a = m.log_demanda(ctx[:1], uu, loja[:1])
        b = esc.log_demanda(ctx[:1], uu, loja[:1])
    assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("lam", [0.5, 1.0, 2.0])
def test_a_matriz_de_elasticidades_e_multiplicada_por_lambda(lam):
    m = _modelo()
    _, u, _, _, _ = _dados()
    esc = S.ModeloEscalado(m, lam, u[0].numpy())
    with torch.no_grad():
        assert torch.allclose(esc.elasticidades_fechadas(u),
                              lam * m.elasticidades_fechadas(u), atol=1e-6)


def test_lambda_um_reproduz_o_modelo_original_em_qualquer_ponto():
    """λ = 1 é a conferência de que o invólucro não mudou nada por acidente."""
    m = _modelo()
    ctx, u, loja, centro, custo = _dados()
    esc = S.ModeloEscalado(m, 1.0, u[0].numpy())
    with torch.no_grad():
        assert torch.allclose(m.log_demanda(ctx, u, loja),
                              esc.log_demanda(ctx, u, loja), atol=1e-6)
    v1, g1 = P.valor_e_gradiente(m, ctx, u, loja, centro, custo)
    v2, g2 = P.valor_e_gradiente(esc, ctx, u, loja, centro, custo)
    assert torch.allclose(v1, v2, atol=1e-5)
    assert torch.allclose(g1, g2, atol=1e-5)


def test_lambda_maior_da_derivada_maior_em_modulo():
    """Consequência esperada e testável: elasticidade maior em módulo produz
    gradiente de receita maior em módulo no canal próprio."""
    m = _modelo()
    ctx, u, loja, centro, _ = _dados()
    zero = torch.zeros(len(u), N)
    _, g1 = P.valor_e_gradiente(S.ModeloEscalado(m, 1.0, u[0].numpy()),
                                ctx, u, loja, centro, zero)
    _, g2 = P.valor_e_gradiente(S.ModeloEscalado(m, 2.0, u[0].numpy()),
                                ctx, u, loja, centro, zero)
    assert float(g2.abs().sum()) > float(g1.abs().sum())


def test_u_base_de_mais_de_uma_celula_falha_alto():
    m = _modelo()
    _, u, _, _, _ = _dados()
    with pytest.raises(ValueError):
        S.ModeloEscalado(m, 1.0, u.numpy())
