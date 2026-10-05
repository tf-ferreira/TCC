"""Testes dos mundos de resposta cruzada e da política robusta (D44).

Três propriedades seguram a leitura dos resultados da v1, e nenhuma aparece no
número final se falhar:

- μ = 1 é a rede, exatamente; senão "a política da rede" não é a rede;
- no ponto histórico todos os mundos preveem igual; senão a calibração de D40
  muda entre mundos e o contraste entre eles mistura nível com resposta;
- a garantia de D33 (diagonal não positiva) sobrevive em todo mundo.

E o objetivo do conjunto tem de ser a média dos membros, com o gradiente
conferido contra autograd, que é o mesmo padrão de `test_problema_pnl`.
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
import mundos as MU  # noqa: E402
import problema as P  # noqa: E402
import rede  # noqa: E402

N, DIM, LOJAS, B = 4, 5, 3, 6


def _rede(semente=0, escala=1.0):
    mods = rede.construir_modulos()
    m = mods["RedeDemanda"](n=N, dim_ctx=DIM, n_lojas=LOJAS, largura=8,
                            profundidade=1, dim_emb=2, h_monotona=4,
                            restrita=True, semente=semente).double()
    g = torch.Generator().manual_seed(semente + 50)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64) * escala)
    return m.eval()


def _entradas(semente=1):
    g = torch.Generator().manual_seed(semente)
    ctx = torch.randn(B, DIM, generator=g, dtype=torch.float64)
    u0 = torch.randn(B, N, generator=g, dtype=torch.float64) * 0.1
    u = u0 + torch.randn(B, N, generator=g, dtype=torch.float64) * 0.1
    loja = torch.randint(0, LOJAS, (B,), generator=g)
    return ctx, u0, u, loja


def test_mu_1_e_a_rede_exatamente():
    m = _rede()
    ctx, u0, u, loja = _entradas()
    w = MU.Mundo(m, escala=1.0).ancorado(u0.numpy())
    assert torch.allclose(w.log_demanda(ctx, u, loja), m.log_demanda(ctx, u, loja),
                          atol=1e-12)
    assert torch.allclose(w.elasticidades_fechadas(u), m.elasticidades_fechadas(u),
                          atol=1e-12)


def test_todos_os_mundos_preveem_igual_na_ancora():
    m = _rede()
    ctx, u0, _, loja = _entradas()
    base = m.log_demanda(ctx, u0, loja)
    for nome, mundo in MU.mundos_v1(m).items():
        w = mundo.ancorado(u0.numpy())
        assert torch.allclose(w.log_demanda(ctx, u0, loja), base, atol=1e-12), nome


def test_mu_0_zera_a_resposta_cruzada_e_preserva_a_propria():
    m = _rede()
    _, u0, u, _ = _entradas()
    w = MU.Mundo(m, escala=0.0).ancorado(u0.numpy())
    e = w.elasticidades_fechadas(u)
    fora = e * (1 - torch.eye(N, dtype=e.dtype))
    assert torch.all(fora == 0)
    assert torch.allclose(torch.diagonal(e, dim1=1, dim2=2),
                          torch.diagonal(m.elasticidades_fechadas(u), dim1=1, dim2=2))


def test_so_substitutos_zera_so_as_complementaridades():
    m = _rede()
    _, u0, u, _ = _entradas()
    G = (m.gama * m.fora_da_diagonal).detach()
    w = MU.Mundo(m, so_substitutos=True).ancorado(u0.numpy())
    Gf = w.elasticidades_fechadas(u)[0] * (1 - torch.eye(N, dtype=G.dtype))
    assert torch.all(Gf >= 0)
    assert torch.allclose(Gf[G > 0], G[G > 0])


@pytest.mark.parametrize("semente", [0, 1, 2])
def test_a_diagonal_continua_nao_positiva_em_todo_mundo(semente):
    """D33 com pesos arbitrários: `f` só mexe fora da diagonal."""
    m = _rede(semente, escala=3.0)
    _, u0, u, _ = _entradas(semente + 10)
    for mundo in MU.mundos_v1(m).values():
        e = mundo.ancorado(u0.numpy()).elasticidades_fechadas(u)
        assert torch.all(torch.diagonal(e, dim1=1, dim2=2) <= 1e-12)


def test_a_forma_fechada_do_mundo_bate_com_autograd():
    m = _rede()
    ctx, u0, u, loja = _entradas()
    for mundo in MU.mundos_v1(m).values():
        w = mundo.ancorado(u0.numpy())
        e_auto = rede.elasticidades_por_autograd(w, ctx, u, loja)
        assert torch.allclose(e_auto, w.elasticidades_fechadas(u), atol=1e-10)


def test_o_conjunto_e_a_media_dos_membros_e_o_gradiente_bate_com_autograd():
    m = _rede()
    ctx, u0, u, loja = _entradas()
    centro = torch.zeros(N, dtype=torch.float64)
    custo = torch.full((B, N), 0.3, dtype=torch.float64)
    conj = MU.politica_robusta_v1(m).ancorado(u0.numpy())
    v, g = P.valor_e_gradiente(conj, ctx, u, loja, centro, custo)
    vs = [P.valor_e_gradiente(w, ctx, u, loja, centro, custo)[0] for w in conj.membros]
    assert torch.allclose(v, sum(vs) / len(vs), atol=1e-12)
    g_auto = P.gradiente_por_autograd(conj, ctx, u, loja, centro, custo)
    assert torch.allclose(g, g_auto, atol=1e-9)


def test_ancorar_deixa_a_rede_pura_intocada():
    m = _rede()
    assert MU.ancorar(m, np.zeros(N)) is m


def test_ancora_com_numero_de_linhas_errado_falha_alto():
    m = _rede()
    ctx, u0, u, loja = _entradas()
    w = MU.Mundo(m, escala=0.5).ancorado(u0.numpy()[:2])
    with pytest.raises(ValueError):
        w.log_demanda(ctx, u, loja)


def test_problema_de_celula_aceita_o_conjunto_ancorado():
    """O adaptador do scipy lê dtype e parâmetros do conjunto sem tropeçar."""
    m = _rede()
    ctx, u0, _, loja = _entradas()
    conj = MU.politica_robusta_v1(m).ancorado(u0.numpy()[:1])
    prob = P.ProblemaDeCelula(conj, ctx[0].numpy(), int(loja[0]), np.zeros(N),
                              np.full(N, 0.3), u0[0].numpy())
    r = P.resolver(prob)
    assert r["valor"] >= r["valor_inicial"] - 1e-9
