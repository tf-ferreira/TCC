"""Testes da hierarquia de marca, antes de a restrição tocar dado real.

Não precisam de torch, então rodam no processo do lightgbm junto do resto.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src" / "optimization"))

import hierarquia as H  # noqa: E402


# ---------------------------------------------------------------------------
# Tamanho
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("texto,esperado", [
    ("12 OZ", 12.0), (" 6 OZ", 6.0), ("16 OZ ", 16.0), ("6oz", 6.0),
    ("11.5 OZ", 11.5),
])
def test_oncas_reconhece_os_formatos_do_painel(texto, esperado):
    assert H.oncas_de(texto) == esperado


@pytest.mark.parametrize("texto", ["24/12O", "12 CT", "", "PACK"])
def test_oncas_falha_alto_em_formato_desconhecido(texto):
    """Falhar alto é a decisão: tamanho lido errado emparelharia embalagens
    diferentes e a restrição compararia coisas incomparáveis em silêncio."""
    with pytest.raises(ValueError):
        H.oncas_de(texto)


# ---------------------------------------------------------------------------
# Um painel de brinquedo, com a mesma estrutura do frj
# ---------------------------------------------------------------------------

def _info_falsa() -> pd.DataFrame:
    """Quatro posições: própria 12oz e 16oz, nacional 12oz e 16oz."""
    return pd.DataFrame({
        "pos": [0, 1, 2, 3],
        "upc": [111, 3828190001, 222, 3828190002],
        "descrip": ["NAT A 12", "OWN A 12", "NAT A 16", "OWN A 16"],
        "size": ["12 OZ", "12 OZ", "16 OZ", "16 OZ"],
        "oncas": [12.0, 12.0, 16.0, 16.0],
        "tipo": ["a", "a", "a", "a"],
        "marca_propria": [False, True, False, True],
        "descontinuado": [False, False, False, False],
    })


def test_pares_saem_de_tipo_e_tamanho_e_nunca_cruzam_tamanho():
    ps = H.pares(_info_falsa())
    assert ps == [(1, 0), (3, 2)]


def test_pares_nao_existem_sem_contraparte():
    info = _info_falsa()
    info.loc[info["pos"] == 0, "tipo"] = "b"     # a nacional 12oz muda de tipo
    assert H.pares(info) == [(3, 2)]


def test_pares_de_uma_propria_contra_varias_nacionais():
    """`own ≤ cada nacional comparável` é o mesmo que `own ≤ a mais barata`, e por
    isso uma linha por par é a forma certa, não uma linha por própria."""
    info = _info_falsa()
    info.loc[info["pos"] == 2, "oncas"] = 12.0
    info.loc[info["pos"] == 2, "size"] = "12 OZ"
    assert H.pares(info) == [(1, 0), (1, 2)]


# ---------------------------------------------------------------------------
# Auditoria
# ---------------------------------------------------------------------------

def test_auditoria_conta_celula_e_nao_mediana():
    """O ponto da pendência: mediana confortável convive com violação em parte
    das células, e é a contagem por célula que decide a viabilidade."""
    # própria (col 1) mais barata em 3 de 4 células, e mais cara numa só
    precos = np.array([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [2.0, 3.0]])
    r = H.auditar(precos, [(1, 0)])
    assert r["por_par"][0]["pct_violacao"] == 25.0
    assert r["pct_celulas_com_alguma_violacao"] == 25.0
    # a mediana da própria continua ABAIXO da nacional
    assert r["por_par"][0]["preco_propria_mediana"] < r["por_par"][0]["preco_nacional_mediana"]


def test_auditoria_uniao_entre_pares_nao_soma_duas_vezes():
    precos = np.array([[1.0, 2.0, 1.0, 2.0],    # nenhuma violação
                       [1.0, 2.0, 1.0, 2.0],
                       [3.0, 2.0, 3.0, 2.0]])   # as DUAS violadas na mesma célula
    r = H.auditar(precos, [(0, 1), (2, 3)])
    assert r["pct_violacao_media_entre_pares"] == pytest.approx(100 / 3)
    assert r["pct_celulas_com_alguma_violacao"] == pytest.approx(100 / 3)


# ---------------------------------------------------------------------------
# A restrição
# ---------------------------------------------------------------------------

def test_matriz_tem_um_mais_um_e_um_menos_um_por_linha():
    A, b = H.restricoes([(1, 0), (3, 2)], np.zeros(4), 4)
    assert A.shape == (2, 4) and b.shape == (2,)
    for linha in A:
        assert linha.sum() == 0.0
        assert np.count_nonzero(linha) == 2
    assert A[0, 1] == 1.0 and A[0, 0] == -1.0


def test_absoluta_equivale_a_comparar_precos():
    """`A·u ≤ b` com `b = c_nat − c_own` é `p_own ≤ p_nat`, e o teste confere pela
    definição de `u`, não pela álgebra repetida."""
    centro = np.array([0.3, -0.2])
    A, b = H.restricoes([(1, 0)], centro, 2)
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.uniform(0.5, 3.0, size=2)
        u = np.log(p) - centro
        assert (float((A @ u)[0]) <= float(b[0])) == bool(p[1] <= p[0] + 1e-15)


def test_nao_piorar_mantem_o_historico_viavel_sempre():
    centro = np.array([0.3, -0.2, 0.0, 0.1])
    pares_ = [(1, 0), (3, 2)]
    rng = np.random.default_rng(1)
    U = rng.normal(0.0, 0.5, size=(500, 4))
    A, B = H.restricoes(pares_, centro, 4, forma="nao_piorar", u0=U)
    assert B.shape == (500, 2)
    assert np.all(U @ A.T <= B + 1e-12)


def test_nao_piorar_e_a_absoluta_onde_a_hierarquia_JA_vale():
    centro = np.array([0.0, 0.0])
    A, b_abs = H.restricoes([(1, 0)], centro, 2)
    u0 = np.array([[0.5, -0.5]])        # própria mais barata: folga
    _, B = H.restricoes([(1, 0)], centro, 2, forma="nao_piorar", u0=u0)
    assert B[0, 0] == pytest.approx(b_abs[0])


def test_nao_piorar_congela_a_violacao_onde_ela_JA_existe():
    centro = np.array([0.0, 0.0])
    u0 = np.array([[0.1, 0.4]])         # própria MAIS CARA: viola
    A, B = H.restricoes([(1, 0)], centro, 2, forma="nao_piorar", u0=u0)
    assert B[0, 0] == pytest.approx(0.3)      # a folga histórica, congelada
    assert float((A @ u0[0])[0]) == pytest.approx(0.3)


def test_forma_desconhecida_falha():
    with pytest.raises(ValueError):
        H.restricoes([(1, 0)], np.zeros(2), 2, forma="qualquer")


def test_nao_piorar_exige_u0():
    with pytest.raises(ValueError):
        H.restricoes([(1, 0)], np.zeros(2), 2, forma="nao_piorar")


# ---------------------------------------------------------------------------
# Identificação, com arquivos de brinquedo
# ---------------------------------------------------------------------------

def test_identificar_usa_a_ordem_do_vetor_e_o_prefixo(tmp_path, monkeypatch):
    painel = tmp_path / "painel"
    painel.mkdir()
    ordem = [222, 3828190001, 111]
    (painel / "xxx_upcs.json").write_text(json.dumps({"n": 3, "ordem": ordem}))
    bruto = tmp_path / "raw" / "xxx"
    bruto.mkdir(parents=True)
    pd.DataFrame({"com_code": [1, 1, 1, 1],
                  "upc": [111, 222, 3828190001, 999],
                  "descrip": ["~NAT B", "NAT A", "OWN A", "FORA"],
                  "size": ["12 OZ", "12 OZ", "12 OZ", "12 OZ"],
                  "case": [1, 1, 1, 1], "nitem": [1, 2, 3, 4],
                  }).to_csv(bruto / "upcxxx.csv", index=False)
    monkeypatch.setitem(H.TIPO, 0, "a")
    monkeypatch.setitem(H.TIPO, 1, "a")
    monkeypatch.setitem(H.TIPO, 2, "a")
    info = H.identificar("xxx", painel, bruto=tmp_path / "raw")
    assert info["upc"].tolist() == ordem              # ordem do vetor, não do csv
    assert info["marca_propria"].tolist() == [False, True, False]
    assert info["descontinuado"].tolist() == [False, False, True]
    assert H.pares(info) == [(1, 0), (1, 2)]


def test_identificar_falha_se_falta_upc(tmp_path, monkeypatch):
    painel = tmp_path / "painel"
    painel.mkdir()
    (painel / "xxx_upcs.json").write_text(json.dumps({"n": 2, "ordem": [111, 555]}))
    bruto = tmp_path / "raw" / "xxx"
    bruto.mkdir(parents=True)
    pd.DataFrame({"upc": [111], "descrip": ["A"], "size": ["12 OZ"]}).to_csv(
        bruto / "upcxxx.csv", index=False)
    monkeypatch.setitem(H.TIPO, 0, "a")
    with pytest.raises(ValueError):
        H.identificar("xxx", painel, bruto=tmp_path / "raw")


# ---------------------------------------------------------------------------
# Viabilidade dentro da caixa
# ---------------------------------------------------------------------------

def test_minimo_na_caixa_usa_limite_inferior_no_positivo_e_superior_no_negativo():
    A = np.array([[1.0, -1.0], [-1.0, 1.0]])
    baixo, alto = np.array([0.0, 10.0]), np.array([1.0, 20.0])
    m = H.minimo_na_caixa(A, baixo, alto)
    assert m[0] == pytest.approx(0.0 - 20.0)
    assert m[1] == pytest.approx(10.0 - 1.0)


def test_nao_piorar_nunca_e_inviavel_na_caixa():
    """A propriedade que sustenta D39: o ponto histórico está na caixa e satisfaz a
    restrição, então existe ponto viável em toda célula, sempre."""
    centro = np.array([0.1, -0.1, 0.0, 0.2])
    rng = np.random.default_rng(7)
    U = rng.normal(0.0, 0.6, size=(300, 4))
    A, B = H.restricoes([(1, 0), (3, 2)], centro, 4, forma="nao_piorar", u0=U)
    baixo, alto = U + np.log(0.85), U + np.log(1.15)
    assert not H.inviaveis_na_caixa(A, B, baixo, alto).any()


def test_absoluta_fica_inviavel_quando_a_caixa_nao_alcanca():
    """Com a caixa de ±15%, a razão entre dois preços só pode melhorar por um fator
    de 0,85/1,15 = 0,739, isto é, 0,3023 em log. Uma violação maior que isso
    não tem ponto viável, e é o que acontece em 4,60% das células do escopo."""
    centro = np.zeros(2)
    folga_max = np.log(1.15) - np.log(0.85)          # ~0,3016
    u_ok = np.array([[0.0, 0.2]])                    # viola 0,2 < 0,3023: alcançável
    u_nao = np.array([[0.0, 0.5]])                   # viola 0,5 > 0,3023: inviável
    A, b = H.restricoes([(1, 0)], centro, 2)
    for u0, esperado in ((u_ok, False), (u_nao, True)):
        baixo, alto = u0 + np.log(0.85), u0 + np.log(1.15)
        B = np.tile(b, (1, 1))
        assert bool(H.inviaveis_na_caixa(A, B, baixo, alto)[0]) is esperado
    assert folga_max == pytest.approx(0.302281, abs=1e-5)
