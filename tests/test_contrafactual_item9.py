"""Testes do contrafactual do item 9, antes de rodar com dado real.

Quase tudo aqui é propriedade estrutural: onde os lags moram em `ctx`, como a
corrida é cortada, e o que a cadeia escreve. São justamente os erros que NÃO
aparecem no resultado, porque o vetor continua com o tamanho certo e a rede
responde alguma coisa.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
for sub in ("models", "data", "experiments", "optimization"):
    sys.path.insert(0, str(RAIZ / "src" / sub))

pytest.importorskip("torch")
import contrafactual_item9 as C9  # noqa: E402
import rede  # noqa: E402

N = 3


def _chaves(pares):
    return pd.DataFrame(pares, columns=["store", "week"])


def _esc(n=N, ctx=None, escalas=None):
    largura = len(rede.CONTEXTO_CELULA) + 4 * n
    return {
        "n": n,
        "ctx": np.zeros((0, largura)) if ctx is None else ctx,
        "escalas_dos_lags": escalas or {
            "lag1": (np.zeros(n), np.ones(n)),
            "lag2": (np.zeros(n), np.ones(n)),
            "falta1": (np.zeros(n), np.ones(n)),
            "falta2": (np.zeros(n), np.ones(n)),
        },
    }


# ---------------------------------------------------------------------------
# Onde os lags moram
# ---------------------------------------------------------------------------

def test_as_fatias_batem_com_a_ordem_de_CONTEXTO_SKU():
    """O teste que impede o erro silencioso: `pivotar` achata `CONTEXTO_SKU` numa
    ordem, e ler outra devolve o atributo errado sem quebrar nada."""
    assert rede.CONTEXTO_SKU == ["preco_lag1", "preco_lag2",
                                 "falta_preco_lag1", "falta_preco_lag2"]
    fat = C9.fatias_dos_lags(N)
    base = len(rede.CONTEXTO_CELULA)
    assert fat["lag1"] == slice(base, base + N)
    assert fat["lag2"] == slice(base + N, base + 2 * N)
    assert fat["falta1"] == slice(base + 2 * N, base + 3 * N)
    assert fat["falta2"] == slice(base + 3 * N, base + 4 * N)


def test_as_fatias_recuperam_o_bloco_que_pivotar_escreveu():
    """Monta `ctx` do mesmo jeito que `pivotar` e `escopo` montam, e confere que
    a fatia devolve o bloco certo."""
    blocos = {nome: np.full((5, N), i + 1.0)
              for i, nome in enumerate(rede.CONTEXTO_SKU)}
    ctx_sku = np.concatenate([blocos[nome] for nome in rede.CONTEXTO_SKU], axis=1)
    ctx = np.concatenate([np.zeros((5, len(rede.CONTEXTO_CELULA))), ctx_sku], axis=1)
    fat = C9.fatias_dos_lags(N)
    assert np.allclose(ctx[:, fat["lag1"]], 1.0)
    assert np.allclose(ctx[:, fat["lag2"]], 2.0)
    assert np.allclose(ctx[:, fat["falta1"]], 3.0)
    assert np.allclose(ctx[:, fat["falta2"]], 4.0)


def test_escalas_recusam_escalonador_de_tamanho_errado():
    with pytest.raises(ValueError, match="ctx_sku"):
        C9.escalas_dos_lags(
            {"escalonador": {"ctx_sku": {"media": [0.0] * 5,
                                         "desvio": [1.0] * 5}}}, N)


# ---------------------------------------------------------------------------
# Escrever um preço novo na defasagem
# ---------------------------------------------------------------------------

def test_ctx_com_lags_escreve_na_escala_do_TREINO_e_volta():
    mu1, sd1 = np.array([2.0, 3.0, 4.0]), np.array([0.5, 0.5, 2.0])
    esc = _esc(escalas={"lag1": (mu1, sd1), "lag2": (np.zeros(N), np.ones(N)),
                        "falta1": (np.zeros(N), np.ones(N)),
                        "falta2": (np.zeros(N), np.ones(N))})
    fat = C9.fatias_dos_lags(N)
    ctx0 = np.zeros(len(rede.CONTEXTO_CELULA) + 4 * N)
    p = np.array([1.0, 5.0, 9.0])
    ctx = C9.ctx_com_lags(ctx0, fat, esc["escalas_dos_lags"], p1=p)
    assert np.allclose(ctx[fat["lag1"]] * sd1 + mu1, p)


def test_ctx_com_lags_nao_toca_no_que_nao_foi_pedido():
    esc = _esc()
    fat = C9.fatias_dos_lags(N)
    ctx0 = np.arange(len(rede.CONTEXTO_CELULA) + 4 * N, dtype=float)
    ctx = C9.ctx_com_lags(ctx0, fat, esc["escalas_dos_lags"],
                          p1=np.ones(N), p2=None)
    assert np.allclose(ctx[:len(rede.CONTEXTO_CELULA)],
                       ctx0[:len(rede.CONTEXTO_CELULA)])
    assert np.allclose(ctx[fat["lag2"]], ctx0[fat["lag2"]])
    assert np.allclose(ctx[fat["falta1"]], ctx0[fat["falta1"]])
    assert not np.allclose(ctx[fat["lag1"]], ctx0[fat["lag1"]])


def test_ctx_com_lags_nao_altera_a_entrada():
    esc = _esc()
    fat = C9.fatias_dos_lags(N)
    ctx0 = np.zeros(len(rede.CONTEXTO_CELULA) + 4 * N)
    C9.ctx_com_lags(ctx0, fat, esc["escalas_dos_lags"], p1=np.ones(N))
    assert np.allclose(ctx0, 0.0)


# ---------------------------------------------------------------------------
# As corridas
# ---------------------------------------------------------------------------

def test_corrida_quebra_em_semana_faltante_e_nunca_atravessa_loja():
    ch = _chaves([(1, 10), (1, 11), (1, 13), (1, 14), (2, 10), (2, 11)])
    cs = C9.corridas(ch)
    tamanhos = sorted(len(c) for c in cs)
    assert tamanhos == [2, 2, 2]
    for c in cs:
        assert len(set(ch["store"].to_numpy()[c])) == 1


def test_corrida_e_ordenada_por_semana_mesmo_com_entrada_fora_de_ordem():
    ch = _chaves([(1, 12), (1, 10), (1, 11)])
    (c,) = C9.corridas(ch)
    assert list(ch["week"].to_numpy()[c]) == [10, 11, 12]


def test_o_aquecimento_sai_da_amostra_e_as_contagens_FECHAM():
    # loja 1: 5 semanas seguidas -> 3 elegiveis; loja 2: 2 semanas -> nenhuma.
    ch = _chaves([(1, w) for w in range(10, 15)] + [(2, 10), (2, 11)])
    m = C9.mapa_das_corridas(ch)
    c = m["contagens"]
    assert c["n_celulas_do_escopo"] == 7
    assert c["n_corridas"] == 2
    assert c["n_corridas_curtas_demais"] == 1
    assert c["n_celulas_resolvidas"] == 5
    assert c["n_celulas_de_aquecimento"] == C9.AQUECIMENTO
    assert c["n_celulas_elegiveis"] == 3
    assert (c["n_celulas_elegiveis"] + c["n_excluidas_por_quebra_de_cadeia"]
            == c["n_celulas_do_escopo"])
    assert list(m["indices_elegiveis"]) == [2, 3, 4]


def test_corrida_do_tamanho_exato_do_aquecimento_nao_entra():
    ch = _chaves([(1, 10), (1, 11)])
    m = C9.mapa_das_corridas(ch)
    assert m["contagens"]["n_celulas_elegiveis"] == 0
    assert m["contagens"]["n_celulas_resolvidas"] == 0


# ---------------------------------------------------------------------------
# A cadeia
# ---------------------------------------------------------------------------

def test_a_cadeia_honesta_escreve_o_preco_da_POLITICA_nas_duas_defasagens():
    ch = _chaves([(1, w) for w in range(10, 15)])
    m = C9.mapa_das_corridas(ch)
    largura = len(rede.CONTEXTO_CELULA) + 4 * N
    esc = _esc(ctx=np.zeros((5, largura)))
    esc["chaves"] = ch
    centro = np.zeros(N)
    U = np.arange(5 * N, dtype=float).reshape(5, N) / 10.0
    CTX = C9.ctx_honesto_da_politica(esc, m, U, centro)
    fat = C9.fatias_dos_lags(N)
    for j in (2, 3, 4):
        assert np.allclose(CTX[j, fat["lag1"]], np.exp(U[j - 1]))
        assert np.allclose(CTX[j, fat["lag2"]], np.exp(U[j - 2]))
    # aquecimento intocado
    for j in (0, 1):
        assert np.allclose(CTX[j], esc["ctx"][j])


def test_a_cadeia_honesta_nao_atravessa_a_quebra_da_corrida():
    ch = _chaves([(1, 10), (1, 11), (1, 12), (1, 20), (1, 21), (1, 22)])
    m = C9.mapa_das_corridas(ch)
    largura = len(rede.CONTEXTO_CELULA) + 4 * N
    esc = _esc(ctx=np.zeros((6, largura)))
    centro = np.zeros(N)
    U = np.ones((6, N))
    CTX = C9.ctx_honesto_da_politica(esc, m, U, centro)
    # As duas primeiras de CADA corrida ficam intocadas: 0, 1, 3 e 4.
    for j in (0, 1, 3, 4):
        assert np.allclose(CTX[j], esc["ctx"][j])
    for j in (2, 5):
        assert not np.allclose(CTX[j], esc["ctx"][j])


# ---------------------------------------------------------------------------
# O lado do dado
# ---------------------------------------------------------------------------

def test_valor_observado_com_custo_zero_e_a_receita_observada():
    p = np.array([[2.0, 3.0]])
    v = np.array([[10.0, 5.0]])
    assert np.allclose(C9.valor_observado(p, v, np.zeros_like(p)), 35.0)


def test_valor_observado_com_custo_e_a_margem_observada():
    p = np.array([[2.0, 3.0]])
    v = np.array([[10.0, 5.0]])
    k = np.array([[1.0, 1.0]])
    assert np.allclose(C9.valor_observado(p, v, k), 10.0 + 10.0)


def test_a_identidade_de_dois_fatores_e_aritmetica_e_nao_aproximacao():
    """`C/R = (C/M)·(M/R)`. O `agregar` recusa a execução se isto não fechar."""
    sC, sM, sR = 137.0, 100.0, 104.0
    ganho = C9._razao(sC, sM)
    calib = sM / sR
    assert np.isclose(100.0 * ((1 + ganho / 100.0) * calib - 1.0),
                      C9._razao(sC, sR))


# ---------------------------------------------------------------------------
# Fim a fim, com rede de verdade e pesos aleatórios
# ---------------------------------------------------------------------------

def _modelo_e_escopo(semente: int = 0):
    """Rede restrita com `γ` assimétrico e um escopo sintético de duas lojas.

    `γ` em zeros é o padrão de `RedeDemanda`, e com ele a fora-diagonal some;
    preencher com matriz assimétrica é o que dá poder a qualquer teste que
    dependa de cruzada, aqui inclusive.
    """
    import torch
    from rede import construir_modulos
    dim = len(rede.CONTEXTO_CELULA) + 4 * N
    mods = construir_modulos()
    m = mods["RedeDemanda"](n=N, dim_ctx=dim, n_lojas=2, largura=8,
                            profundidade=2, dim_emb=3, h_monotona=4,
                            restrita=True, semente=semente)
    g = torch.Generator().manual_seed(semente + 7)
    with torch.no_grad():
        m.gama.copy_(torch.randn(N, N, generator=g) * 0.4)
    m.eval()

    # loja 1: semanas 10..14 (3 elegíveis); loja 2: semanas 10..13 (2).
    ch = _chaves([(1, w) for w in range(10, 15)] + [(2, w) for w in range(10, 14)])
    c = len(ch)
    rng = np.random.default_rng(semente)
    precos = np.exp(rng.normal(0.0, 0.2, size=(c, N)))
    centro = np.log(precos).mean(axis=0)
    esc = {
        "n": N,
        "chaves": ch,
        "ctx": rng.normal(0.0, 1.0, size=(c, dim)),
        "u": np.log(precos) - centro,
        "precos": precos,
        "custos": precos * 0.6,
        "alvo": rng.uniform(5.0, 50.0, size=(c, N)),
        "loja_idx": ch["store"].to_numpy().astype("int64") - 1,
        "escalas_dos_lags": {
            "lag1": (np.zeros(N), np.ones(N)), "lag2": (np.zeros(N), np.ones(N)),
            "falta1": (np.zeros(N), np.ones(N)),
            "falta2": (np.zeros(N), np.ones(N))},
    }
    return m, esc, centro


def test_fim_a_fim_a_cadeia_escreve_lag1_da_posicao_1_e_lag2_da_posicao_2():
    """A posição 1 da corrida JÁ herda `lag1` do otimizador, embora seja
    aquecimento e fique fora da amostra. Quem confundir "aquecimento" com "sem
    cadeia" escreve defasagem histórica onde já há preço recomendado."""
    m, esc, centro = _modelo_e_escopo()
    mapa = C9.mapa_das_corridas(esc["chaves"])
    custos = np.zeros_like(esc["custos"])
    seq = C9.rodar_passada(m, centro, esc, mapa, custos, None, 3, 0, "SLSQP",
                           True, progresso=0)
    fat = C9.fatias_dos_lags(N)

    def _igual(c, bloco):
        return np.allclose(seq["ctx"][c, fat[bloco]], esc["ctx"][c, fat[bloco]])

    resolvidas = set(mapa["indices_resolvidos"].tolist())
    for corrida in mapa["corridas"]:
        if len(corrida) <= C9.AQUECIMENTO:
            continue
        for j, c in enumerate(corrida):
            c = int(c)
            assert _igual(c, "lag1") == (j == 0)
            assert _igual(c, "lag2") == (j <= 1)
            assert _igual(c, "falta1") and _igual(c, "falta2")
    for c in range(len(esc["u"])):
        if c not in resolvidas:
            assert np.allclose(seq["ctx"][c], esc["ctx"][c])


def test_fim_a_fim_a_PRIMEIRA_celula_da_corrida_e_a_mesma_nas_duas_passadas():
    """Invariante da cadeia: na posição 0 não há preço do otimizador para herdar,
    então o problema sequencial é IDÊNTICO ao ingênuo. Se divergir, a cadeia está
    escrevendo defasagem onde não devia."""
    m, esc, centro = _modelo_e_escopo()
    mapa = C9.mapa_das_corridas(esc["chaves"])
    custos = np.zeros_like(esc["custos"])
    ing = C9.rodar_passada(m, centro, esc, mapa, custos, None, 3, 0, "SLSQP",
                           False, progresso=0)
    seq = C9.rodar_passada(m, centro, esc, mapa, custos, None, 3, 0, "SLSQP",
                           True, progresso=0)
    for corrida in mapa["corridas"]:
        if len(corrida) <= C9.AQUECIMENTO:
            continue
        c0 = int(corrida[0])
        assert np.allclose(ing["u"][c0], seq["u"][c0])


def test_fim_a_fim_agregar_fecha_a_identidade_e_separa_avaliacao_de_decisao():
    m, esc, centro = _modelo_e_escopo()
    mapa = C9.mapa_das_corridas(esc["chaves"])
    custos = np.zeros_like(esc["custos"])
    ing = C9.rodar_passada(m, centro, esc, mapa, custos, None, 3, 0, "SLSQP",
                           False, progresso=0)
    honesto = C9.ctx_honesto_da_politica(esc, mapa, ing["u"], centro)
    seq = C9.rodar_passada(m, centro, esc, mapa, custos, None, 3, 0, "SLSQP",
                           True, progresso=0)
    r = C9.agregar(m, centro, esc, mapa, custos, ing, seq, honesto, "receita")

    assert r["n_celulas_elegiveis"] == mapa["contagens"]["n_celulas_elegiveis"]
    assert r["erro_da_identidade_de_dois_fatores"] < 1e-6
    a = r["agregados"]
    # A e B são a MESMA política em defasagens diferentes: têm de diferir, senão
    # a cadeia não está entrando na rede.
    assert not np.isclose(a["A_ingenua_reportada"], a["B_ingenua_honesta"])
    # O otimizador nunca pode ficar abaixo do ponto histórico no PRÓPRIO
    # problema que resolveu: a partida 0 é a histórica.
    assert a["A_ingenua_reportada"] >= a["M_modelo_no_historico"] - 1e-9


def test_agregar_RECUSA_quando_nenhuma_corrida_fecha():
    m, esc, centro = _modelo_e_escopo()
    esc["chaves"] = _chaves([(1, 10), (2, 10)] + [(3, 10), (4, 10)]
                            + [(5, 10), (6, 10), (7, 10), (8, 10), (9, 10)])
    mapa = C9.mapa_das_corridas(esc["chaves"])
    custos = np.zeros_like(esc["custos"])
    vazio = {"u": esc["u"], "ctx": esc["ctx"]}
    with pytest.raises(ValueError, match="nenhuma célula elegível"):
        C9.agregar(m, centro, esc, mapa, custos, vazio, vazio, esc["ctx"],
                   "receita")


# ---------------------------------------------------------------------------
# Erro padrão agrupado por loja
# ---------------------------------------------------------------------------

def test_bootstrap_com_razao_identica_em_toda_loja_tem_ep_zero():
    """Se toda loja tem a mesma razão, reamostrar lojas não move a razão das
    somas. É o que separa 'razão de somas' de 'média de razões'."""
    loja = np.repeat(np.arange(6), 4)
    base = np.linspace(1.0, 3.0, len(loja))
    b = {"A": 1.2 * base, "B": base, "C": 1.1 * base, "M": base, "R": 0.9 * base}
    r = C9.bootstrap_por_loja(b, loja, n_boot=300)
    for nome in C9.RAZOES:
        assert r[nome]["ep"] < 1e-9
    assert r["n_lojas"] == 6


def test_bootstrap_reamostra_LOJA_e_nao_celula():
    """Uma loja com razão muito diferente das outras tem de gerar dispersão; se o
    sorteio fosse por célula dentro da loja, a razão da loja não mudaria de peso."""
    loja = np.repeat(np.arange(8), 5)
    B = np.ones(len(loja))
    A = np.where(loja == 0, 3.0, 1.0)
    b = {"A": A, "B": B, "C": B, "M": B, "R": B}
    r = C9.bootstrap_por_loja(b, loja, n_boot=500)
    e = r["superestimacao_da_avaliacao_ingenua_pct"]
    assert e["ep"] > 1.0
    assert e["ic95"][0] <= 100.0 * (A.sum() / B.sum() - 1.0) <= e["ic95"][1]


# ---------------------------------------------------------------------------
# A Jacobiana da defasagem, conferida por diferença finita
# ---------------------------------------------------------------------------

def test_jacobiana_da_defasagem_bate_com_diferenca_finita_em_ln_p():
    """A regra da cadeia do escalonamento, `∂/∂ln p = (p/σ)·∂/∂x`, é o ponto em
    que um erro passa despercebido: o sinal sai certo e a magnitude sai errada.
    Confere-se em float64 pelo motivo registrado em `problema.py`."""
    import torch
    import diagnostico_item9 as D9
    m, esc, centro = _modelo_e_escopo()
    m = m.double()
    mu = np.array([1.0, 2.0, 0.5])
    sd = np.array([0.3, 0.8, 0.2])
    esc["escalas_dos_lags"]["lag1"] = (mu, sd)
    fat = C9.fatias_dos_lags(N)
    # defasagem com preço positivo, que é o domínio do ln
    esc["ctx"][:, fat["lag1"]] = (np.array([1.2, 2.5, 0.6]) - mu) / sd
    idx = np.array([0, 3])
    J, _ = D9.jacobiana_da_defasagem(m, esc, idx, "lag1")

    h = 1e-6
    for c_pos, c in enumerate(idx):
        p = esc["ctx"][c, fat["lag1"]] * sd + mu
        for j in range(N):
            def lnv(delta):
                pp = p.copy()
                pp[j] = p[j] * np.exp(delta)
                ctx = esc["ctx"][c].copy()
                ctx[fat["lag1"]] = (pp - mu) / sd
                with torch.no_grad():
                    return m.log_demanda(
                        torch.as_tensor(ctx[None, :], dtype=torch.float64),
                        torch.as_tensor(esc["u"][c][None, :], dtype=torch.float64),
                        torch.as_tensor([int(esc["loja_idx"][c])])).numpy()[0]
            fd = (lnv(h) - lnv(-h)) / (2 * h)
            assert np.allclose(J[c_pos, :, j], fd, atol=1e-7)


# ---------------------------------------------------------------------------
# 50 sementes (sementes_item9)
# ---------------------------------------------------------------------------

def test_com_K_1_a_unica_partida_e_a_historica_e_a_passada_roda():
    """`sementes_item9` usa K = 1. Com uma partida só, ela tem de ser a histórica,
    senão a execução por sementes mediria outra coisa que a canônica."""
    import multipartida as M
    u0 = np.linspace(-0.1, 0.1, N)
    baixo, alto = u0 + np.log(0.85), u0 + np.log(1.15)
    U, ruim = M.sortear_partidas(u0, baixo, alto, 1, semente=5)
    assert U.shape == (1, N) and ruim == 0
    assert np.allclose(U[0], u0)
    m, esc, centro = _modelo_e_escopo()
    mapa = C9.mapa_das_corridas(esc["chaves"])
    r = C9.rodar_passada(m, centro, esc, mapa, np.zeros_like(esc["custos"]),
                         None, 1, 0, "SLSQP", True, progresso=0)
    assert r["n_resolvidas"] == mapa["contagens"]["n_celulas_resolvidas"]


def test_agregar_sementes_conta_sinais_e_resume_cada_campo():
    import sementes_item9 as S9
    linhas = []
    for i, dec in enumerate([-1.0, -0.5, 0.3]):
        obj = {k: float(i) for k in S9.CAMPOS}
        obj["ganho_da_decisao_sequencial_sobre_a_ingenua_pct"] = dec
        obj["superestimacao_da_avaliacao_ingenua_pct"] = 1.0
        linhas.append({"margem": dict(obj), "receita": dict(obj),
                       "wmape_teste": 48.0 + i, "vies_de_nivel_teste": 1.0})
    ag = S9.agregar_sementes(linhas)
    for obj in ("margem", "receita"):
        assert np.isclose(ag[obj]["pct_sementes_decisao_negativa"], 200.0 / 3)
        assert ag[obj]["pct_sementes_superestima"] == 100.0
        assert set(S9.CAMPOS) <= set(ag[obj])
        assert ag[obj]["ganho_ingenuo_modelo_contra_modelo_pct"]["n"] == 3
    assert np.isclose(ag["wmape_teste"]["media"], 49.0)


def test_concordancia_de_sinal_segue_a_previsao_de_12_10():
    """Sustentada negativa prevê receita SUBESTIMANDO e margem SUPERESTIMANDO."""
    import sementes_item9 as S9
    casos = [(-0.2, -3.0, +5.0), (-0.1, -1.0, +2.0), (+0.3, +4.0, -1.0),
             (+0.1, -2.0, +3.0)]          # o último contraria a previsão
    linhas = []
    for sus, vr, vm in casos:
        base = {k: 0.0 for k in S9.CAMPOS}
        m, r = dict(base), dict(base)
        m["superestimacao_da_avaliacao_ingenua_pct"] = vm
        r["superestimacao_da_avaliacao_ingenua_pct"] = vr
        linhas.append({"margem": m, "receita": r, "wmape_teste": 48.0,
                       "vies_de_nivel_teste": 1.0,
                       "defasagem_lag1_agregada": sus / 2,
                       "defasagem_lag2_agregada": sus / 2,
                       "defasagem_sustentada": sus})
    ag = S9.agregar_sementes(linhas)
    assert ag["receita"]["pct_sementes_sinal_previsto_pela_defasagem"] == 75.0
    assert ag["margem"]["pct_sementes_sinal_previsto_pela_defasagem"] == 75.0
    assert ag["pct_sementes_defasagem_sustentada_negativa"] == 50.0
    assert ag["receita"]["correlacao_vies_ingenuo_defasagem_sustentada"] > 0


# ---------------------------------------------------------------------------
# Consolidação de sementes
# ---------------------------------------------------------------------------

def _bloco_sementes(sementes, cb, config="x"):
    import sementes_item9 as S9
    linhas = []
    for s_, v in zip(sementes, cb):
        base = {k: 1.0 for k in S9.CAMPOS}
        base[S9.CB] = v
        linhas.append({"semente": s_, "margem": dict(base),
                       "receita": dict(base), "wmape_teste": 48.0,
                       "vies_de_nivel_teste": 1.0})
    return {"categoria": "frj", "configuracao": {"c": config},
            "hierarquia": "nao_piorar", "n_pares": 12, "escopo": {"n": 1},
            "corridas": {"n": 1}, "partidas_por_celula": 1,
            "por_semente": linhas, "tempo_total_min": 1.0}


def test_juntar_consolida_e_testa_cb_com_o_limiar_declarado(tmp_path):
    import json
    import sementes_item9 as S9
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(_bloco_sementes([0, 1, 2], [-1.0, -1.2, -0.8])))
    b.write_text(json.dumps(_bloco_sementes([50, 51], [-1.1, -0.9])))
    r = S9.juntar([a, b])
    assert r["n_sementes"] == 5 and r["sementes"] == [0, 1, 2, 50, 51]
    t = r["teste_cb_contra_zero"]["margem"]
    assert t["limiar"] == S9.LIMIAR_CB == 2.241
    assert t["passa"] and t["z"] < -S9.LIMIAR_CB


def test_juntar_RECUSA_semente_repetida_e_configuracao_diferente(tmp_path):
    import json
    import sementes_item9 as S9
    a, b, c = (tmp_path / f"{x}.json" for x in "abc")
    a.write_text(json.dumps(_bloco_sementes([0, 1], [0.1, -0.1])))
    b.write_text(json.dumps(_bloco_sementes([1, 2], [0.1, -0.1])))
    c.write_text(json.dumps(_bloco_sementes([5, 6], [0.1, -0.1], config="y")))
    with pytest.raises(ValueError, match="repetida"):
        S9.juntar([a, b])
    with pytest.raises(ValueError, match="configuracao"):
        S9.juntar([a, c])


# ---------------------------------------------------------------------------
# D41 e D42: a especificação v1
# ---------------------------------------------------------------------------

@pytest.fixture
def especificacao_v1():
    """Ativa a v1 e devolve a adotada no fim, mesmo se o teste falhar."""
    rede.usar_especificacao("v1")
    try:
        yield
    finally:
        rede.usar_especificacao("adotada")


def test_as_fatias_acompanham_a_especificacao_ativa(especificacao_v1):
    """Largura do contexto de célula lida na hora: 7 colunas na v1, não 9."""
    assert len(rede.CONTEXTO_CELULA) == 7
    fat = C9.fatias_dos_lags(N)
    assert fat["lag1"] == slice(7, 7 + N)
    assert fat["falta2"] == slice(7 + 3 * N, 7 + 4 * N)


def test_o_contrafactual_nunca_reescreve_a_funcao_de_controle(especificacao_v1):
    """D42: `v̂` é o choque da célula e fica congelado no histórico.

    Reescrever as defasagens com um preço novo não pode tocar nos blocos de
    `v_cf` e `falta_v_cf`, que vêm depois das quatro defasagens em CONTEXTO_SKU.
    Se tocasse, `v̂` viraria função do preço decidido, que é o que D25 proíbe.
    """
    largura = len(rede.CONTEXTO_CELULA) + len(rede.CONTEXTO_SKU) * N
    ctx0 = np.arange(largura, dtype=float)
    esc = _esc()
    fat = C9.fatias_dos_lags(N)
    ctx = C9.ctx_com_lags(ctx0, fat, esc["escalas_dos_lags"],
                          p1=np.full(N, 9.0), p2=np.full(N, 7.0))
    inicio_cf = len(rede.CONTEXTO_CELULA) + 4 * N
    assert rede.CONTEXTO_SKU[4:] == ["v_cf", "falta_v_cf"]
    assert np.array_equal(ctx[inicio_cf:], ctx0[inicio_cf:])
    assert not np.array_equal(ctx[fat["lag1"]], ctx0[fat["lag1"]])


def test_escalas_aceitam_os_blocos_da_funcao_de_controle(especificacao_v1):
    n = 2
    k = len(rede.CONTEXTO_SKU) * n
    esc = C9.escalas_dos_lags({"escalonador": {"ctx_sku": {
        "media": list(range(k)), "desvio": [1.0] * k}}}, n)
    assert np.allclose(esc["lag1"][0], [0, 1])
    assert np.allclose(esc["falta2"][0], [6, 7])
