"""Testes dos erros padrão agrupados de `elasticidades.py` (F1).

O erro que estes testes existem para impedir é de uma classe diferente da de
`test_estimacao.py`. Lá a pergunta era se o estimador devolve o parâmetro que
foi posto no dado. Aqui o **coeficiente já estava certo** e o que estava errado
era a precisão declarada: um erro padrão que supunha independência entre
células da mesma loja, e que por isso derrubava hipóteses que o dado não
derruba. Um teste de coeficiente jamais teria pego isso.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
for pasta in ("data", "experiments"):
    sys.path.insert(0, str(RAIZ / "src" / pasta))

from elasticidades import bootstrap_wild, ols, projetar, sanduiche  # noqa: E402


def test_um_grupo_por_linha_reproduz_hc1_exatamente():
    """Com cada linha em seu próprio grupo não há correlação intragrupo para
    capturar, e o agrupado tem que degenerar no robusto a heterocedasticidade.

    A conta fechada é `(X'X)⁻¹ (Σ u_i² x_i x_i') (X'X)⁻¹` vezes n/(n-k), que é
    o HC1. Se a correção de amostra finita estivesse escrita errado, este é o
    teste que quebra, porque aqui ela vale exatamente n/(n-k).
    """
    rng = np.random.default_rng(3)
    n, k = 200, 2
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    u = rng.normal(size=n) * (1 + np.abs(X[:, 1]))
    y = X @ np.array([1.0, 2.0]) + u
    b, _, r = ols(y, X, k)

    pao = np.linalg.inv(X.T @ X)
    hc1 = np.sqrt(np.diag(pao @ ((X * r[:, None]).T @ (X * r[:, None])) @ pao)
                  * (n / (n - k)))
    obtido = sanduiche(X, r, {"linha": np.arange(n)}, k)["linha"]["ep"]
    assert np.allclose(obtido, hc1)
    assert obtido.shape == (2,)


def test_agrupar_infla_quando_o_choque_e_do_grupo():
    """O caso que motivou F1: choque comum dentro do grupo.

    Com um choque por grupo somado ao ruído de linha, as observações de um
    mesmo grupo repetem informação. O erro padrão iid conta cada linha como
    nova, e por isso sai pequeno demais. O agrupado tem que sair maior, e a
    razão entre os dois é o fator de inflação.
    """
    rng = np.random.default_rng(11)
    G, por_grupo = 40, 50
    g = np.repeat(np.arange(G), por_grupo)
    n = len(g)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    u = rng.normal(size=G)[g] * 3.0 + rng.normal(size=n) * 0.3
    y = X @ np.array([0.5, 1.0]) + u
    b, se_iid, r = ols(y, X, 2)

    ep = sanduiche(X, r, {"grupo": g}, 2)["grupo"]["ep"]
    assert ep[0] > 3.0 * se_iid[0], "inflação menor que o choque construído"

    # E o contrário: sem choque de grupo, agrupar não deve inflar muito.
    u2 = rng.normal(size=n)
    y2 = X @ np.array([0.5, 1.0]) + u2
    _, se_iid2, r2 = ols(y2, X, 2)
    ep2 = sanduiche(X, r2, {"grupo": g}, 2)["grupo"]["ep"]
    assert ep2[0] < 1.5 * se_iid2[0]


def test_o_residuo_do_2sls_usa_X_e_nao_X_projetado():
    """Erro clássico de implementação de 2SLS agrupado.

    O pão é `(X̂'X̂)⁻¹` mas o resíduo é `y - X b`, com o X original. Usar
    `y - X̂ b` dá outro número, igualmente plausível à inspeção, e ninguém
    notaria: `y - X̂ b` é `u` mais o resíduo do primeiro estágio vezes `b`.

    A direção do erro **não** é fixa, e a primeira versão deste teste afirmava
    que era. Ela depende do sinal da correlação entre `u` e o resíduo do
    primeiro estágio, que é a própria endogeneidade. Por isso o teste compara a
    versão correta contra a conta fechada, e apenas exige que a errada difira.
    """
    rng = np.random.default_rng(5)
    n = 600
    g = np.repeat(np.arange(30), 20)
    Z = rng.normal(size=(n, 1))
    v = rng.normal(size=n)
    X = 0.8 * Z + np.column_stack([v])
    u = v * 1.5 + rng.normal(size=n) * 0.5
    y = X @ np.array([1.0]) + u

    Xh = projetar(Z, X)
    b = np.linalg.solve(Xh.T @ Xh, Xh.T @ y)
    u_certo = y - X @ b
    certo = sanduiche(Xh, u_certo, {"g": g}, 2)["g"]["ep"]
    errado = sanduiche(Xh, (y - Xh @ b), {"g": g}, 2)["g"]["ep"]

    # A conta fechada, escrita à mão, sem passar pela função testada.
    pao = np.linalg.inv(Xh.T @ Xh)
    G, k = 30, 2
    meat = np.zeros((1, 1))
    for gg in np.unique(g):
        m = g == gg
        s = Xh[m].T @ u_certo[m]
        meat += np.outer(s, s)
    c = (G / (G - 1)) * ((n - 1) / (n - k))
    esperado = np.sqrt(np.diag(pao @ (c * meat) @ pao))

    assert np.allclose(certo, esperado)
    assert not np.allclose(certo, errado)


def test_bootstrap_wild_devolve_corte_na_ordem_de_grandeza_da_normal():
    """Sem correlação intragrupo e com muitos grupos, o corte construído no
    dado tem que ficar perto de 1,96. Serve de sanidade: um corte de 0,3 ou de
    12 indicaria erro de montagem do laço, não propriedade do dado.
    """
    rng = np.random.default_rng(17)
    G, por_grupo = 80, 10
    g = np.repeat(np.arange(G), por_grupo)
    n = len(g)
    Z = rng.normal(size=(n, 1))
    X = 0.9 * Z + rng.normal(size=(n, 1)) * 0.4
    y = X @ np.array([1.0]) + rng.normal(size=n)

    r = bootstrap_wild(y, X, Z, g, k=2, B=199, semente=1)
    assert r["G_bootstrap"] == G and r["B"] == 199
    corte = r["critico_95_bicaudal"][0]
    assert 1.2 < corte < 3.5, corte
    assert r["critico_90_bicaudal"][0] < corte


def test_bootstrap_wild_e_reprodutivel():
    """Mesma semente, mesmo corte. Um bootstrap que muda de resposta a cada
    execução não pode ancorar número citado em texto."""
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(25), 8)
    n = len(g)
    Z = rng.normal(size=(n, 1))
    X = Z + rng.normal(size=(n, 1)) * 0.5
    y = X @ np.array([2.0]) + rng.normal(size=n)
    a = bootstrap_wild(y, X, Z, g, k=2, B=99, semente=42)
    b = bootstrap_wild(y, X, Z, g, k=2, B=99, semente=42)
    assert a == b
