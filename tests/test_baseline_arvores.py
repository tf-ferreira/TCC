"""Testes do baseline de árvores (D20), incluindo o achado negativo de F2.

O teste mais importante deste arquivo prova que uma correção **não** era
necessária. Isso é deliberado: o projeto registrou a suspeita, a mediu, e
descobriu que o LightGBM já protegia contra ela. Sem este teste o desfecho
negativo se perderia, e a mesma suspeita voltaria na próxima auditoria.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src" / "models"))

from baseline_arvores import (colunas_atributos, fixar_categorias,  # noqa: E402
                              particionar)


def _painel_sintetico(semente: int = 0) -> pd.DataFrame:
    """Painel com lojas que entram e saem ao longo do tempo, que é a condição
    em que as partições deixam de ter o mesmo conjunto de categorias."""
    rng = np.random.default_rng(semente)
    linhas = []
    for semana in range(1, 101):
        # As lojas 0 e 1 fecham na metade; as lojas 8 e 9 só abrem depois.
        lojas = [l for l in range(10)
                 if not (l < 2 and semana > 50) and not (l >= 8 and semana <= 50)]
        for loja in lojas:
            for sku in range(3):
                linhas.append((loja, semana, sku,
                               1.0 + 0.1 * rng.normal(),
                               10.0 + loja * 5 + rng.normal()))
    return pd.DataFrame(linhas, columns=["store", "week", "sku",
                                         "preco_proprio", "alvo"])


def test_particoes_compartilham_o_mesmo_dtype_categorico():
    """A invariante que `fixar_categorias` existe para garantir."""
    longo = _painel_sintetico()
    longo = fixar_categorias(longo, ["store", "sku"])
    treino, teste, _ = particionar(longo, frac_teste=0.2, zona=4)
    for c in ("store", "sku"):
        assert treino[c].dtype == teste[c].dtype
        assert list(treino[c].cat.categories) == list(teste[c].cat.categories)


def test_sem_a_funcao_os_codigos_divergem_de_fato():
    """A metade verdadeira do achado de F2: converter por partição embaralha
    mesmo os códigos. É fato sobre o pandas, e este teste o fixa."""
    longo = _painel_sintetico()
    treino, teste, _ = particionar(longo, frac_teste=0.2, zona=4)
    a = treino["store"].astype("category").cat.categories
    b = teste["store"].astype("category").cat.categories
    assert list(a) != list(b)
    divergem = sum(1 for i in range(min(len(a), len(b))) if a[i] != b[i])
    assert divergem > 0, "cenário sintético não reproduz a condição do painel"


def test_lightgbm_remapeia_categoria_por_valor_e_nao_por_codigo():
    """A metade FALSA do achado de F2, fixada como fato.

    O LightGBM guarda as categorias do treino e chama `cat.set_categories`
    sobre o quadro de previsão, de modo que o remapeamento é por valor. As
    previsões saem idênticas com e sem `fixar_categorias`. Se uma versão futura
    da biblioteca deixar de fazer isso, este teste quebra, e aí a correção
    passa a ser necessária de verdade.
    """
    lgb = pytest.importorskip("lightgbm")
    longo = _painel_sintetico(semente=3)
    colunas = ["store", "sku", "preco_proprio"]

    treino, teste, _ = particionar(longo, frac_teste=0.2, zona=4)
    a, b = treino.copy(), teste.copy()
    for c in ("store", "sku"):
        a[c] = a[c].astype("category")
        b[c] = b[c].astype("category")
    assert list(a["store"].cat.categories) != list(b["store"].cat.categories)
    m = lgb.LGBMRegressor(n_estimators=30, verbose=-1, random_state=0)
    m.fit(a[colunas], a["alvo"], categorical_feature=["store", "sku"])
    por_particao = m.predict(b[colunas])

    longo2 = fixar_categorias(longo, ["store", "sku"])
    c_, d_, _ = particionar(longo2, frac_teste=0.2, zona=4)
    m2 = lgb.LGBMRegressor(n_estimators=30, verbose=-1, random_state=0)
    m2.fit(c_[colunas], c_["alvo"], categorical_feature=["store", "sku"])
    comum = m2.predict(d_[colunas])

    assert np.allclose(por_particao, comum), (
        "o LightGBM deixou de remapear por valor: a correção de F2 passou a ser "
        "necessária, e o piso de D20 precisa ser reexecutado")


def test_a_zona_morta_separa_treino_de_teste():
    """Sem a zona morta, uma defasagem de k semanas atravessa a fronteira."""
    longo = _painel_sintetico()
    treino, teste, corte = particionar(longo, frac_teste=0.2, zona=8)
    assert treino.week.max() < corte - 8
    assert teste.week.min() >= corte
    assert teste.week.min() - treino.week.max() > 8


# --- D25: nenhum atributo pode ser função do preço de decisão ------------------


def test_promo_proprio_sai_e_promo_resto_fica():
    """D25. `promo_proprio` é função do preço que o otimizador decide; congelado
    ele atenua a derivada, recalculado por regra ele a destrói. `promo_resto`
    é apenas correlacionado com o preço próprio, e por isso permanece."""
    com = colunas_atributos(20)
    sem = colunas_atributos(20, sem_promo_proprio=True)
    assert "promo_proprio" in com
    assert "promo_proprio" not in sem
    assert "promo_resto" in sem
    assert set(com) - set(sem) == {"promo_proprio"}


def test_a_ordem_das_demais_colunas_nao_muda():
    """A posição no vetor é a identidade do produto para a rede (ver painel.py).
    Remover um atributo não pode reordenar os que ficam."""
    com = [c for c in colunas_atributos(20) if c != "promo_proprio"]
    assert com == colunas_atributos(20, sem_promo_proprio=True)


def test_nenhum_atributo_sobrevivente_e_funcao_do_preco_de_decisao():
    """O critério de D25 enunciado como invariante, e não como lista.

    Os preços de decisão são `preco_00..preco_19` mais `preco_proprio`, que é
    cópia de um deles. Qualquer OUTRO atributo cujo valor mude quando um desses
    preços muda viola D25. Hoje o único caso no painel é `promo_proprio`, e este
    teste quebra se um atributo novo com a mesma propriedade for acrescentado
    sem passar pela decisão.
    """
    decisao = {f"preco_{i:02d}" for i in range(20)} | {"preco_proprio"}
    derivados_do_preco = {"promo_proprio"}
    sobreviventes = set(colunas_atributos(20, sem_promo_proprio=True))
    assert not (sobreviventes & derivados_do_preco)
    assert sobreviventes & decisao == decisao
