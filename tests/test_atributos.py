"""Testes do painel de modelagem (o produto do item 5).

Os dois primeiros escrevem as decisões da fase como **invariantes executáveis**,
não como lista: eles quebram se alguém acrescentar um atributo proibido, mesmo um
que ninguém previu ao escrever a especificação.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src" / "models"))
sys.path.insert(0, str(RAIZ / "src" / "data"))

from atributos import (ALVO, BEM_EXTERNO, CATEGORICAS, LAGS,  # noqa: E402
                       NAO_ATRIBUTOS, SAZONAIS, atributos, atributos_piso,
                       preparar_treino_teste)

N = 4


def _painel(semanas: int = 60, lojas: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    linhas = []
    for loja in range(lojas):
        for semana in range(1, semanas + 1):
            precos = np.round(2.0 + rng.random(N), 2)
            for sku in range(N):
                linhas.append({
                    "store": loja, "week": semana, "sku": sku,
                    **{f"preco_{i:02d}": precos[i] for i in range(N)},
                    "preco_proprio": precos[sku],
                    "promo_proprio": float(rng.random() < 0.2),
                    "semana_do_ano": semana % 52,
                    "sen_1": np.sin(semana), "cos_1": np.cos(semana),
                    "sen_2": np.sin(2 * semana), "cos_2": np.cos(2 * semana),
                    "idx_preco_resto": 1.0 + rng.random() * 0.1,
                    "vol_resto": 100.0 + rng.random() * 10,
                    "promo_resto": rng.random(), "n_resto": 39.0,
                    "preco_lag1": precos[sku] if semana > 1 else np.nan,
                    "preco_lag2": precos[sku] if semana > 2 else np.nan,
                    ALVO: 50.0 + rng.random() * 10,
                })
    return pd.DataFrame(linhas)


def test_nenhum_atributo_proibido_entra():
    """D25, D27, D29 e D30 escritas como invariante. A lista de proibidos é o
    que as decisões excluíram; se um atributo novo com o mesmo defeito for
    acrescentado a `atributos`, este teste quebra."""
    cols = set(atributos(N))
    proibidos = {"promo_proprio", "semana_do_ano", "volume_lag1", "volume_lag2",
                 "clientes", "ln_clientes", "custo_00", "zone", "scluster",
                 "income", "weekvol", "selas1"}
    proibidos |= {f"promo_{i:02d}" for i in range(N)}
    assert not (cols & proibidos), sorted(cols & proibidos)


def test_os_precos_de_decisao_estao_todos_presentes_e_em_ordem():
    """A posição no vetor é a identidade do produto para a rede.

    O filtro é por padrão numérico e **não** por prefixo, e a diferença é uma
    armadilha real: `preco_lag1` começa com `preco_` e não é preço de decisão.
    Qualquer código que selecione o vetor com `startswith("preco_")` vai pegar
    as defasagens junto e perturbar a coluna errada na validação de derivadas.
    """
    cols = atributos(N)
    esperado = [f"preco_{i:02d}" for i in range(N)]
    assert [c for c in cols if c in esperado] == esperado
    assert "preco_proprio" in cols

    por_prefixo = [c for c in cols if c.startswith("preco_")]
    assert set(por_prefixo) - set(esperado) - {"preco_proprio"} == set(LAGS), (
        "o filtro por prefixo pega as defasagens; use o padrão numérico")


def test_o_piso_e_o_final_diferem_so_no_que_a_fase_decidiu():
    """A comparação com o piso de D20 tem de medir atributo, não amostra nem
    layout. O que os separa é exatamente o conteúdo das seções 2, 3 e 5."""
    final, piso = set(atributos(N)), set(atributos_piso(N))
    assert final - piso == set(SAZONAIS) | {"tempo"} | set(LAGS) | {
        f"falta_{c}" for c in LAGS}
    assert piso - final == {"promo_proprio", "semana_do_ano"}


def test_as_colunas_que_nao_sao_atributos_ficam_fora_da_lista():
    for c in NAO_ATRIBUTOS:
        assert c not in atributos(N)


def test_o_preparo_nao_deixa_ausencia_nos_atributos():
    treino, teste, escala, vocab, _ = preparar_treino_teste(_painel())
    for parte in (treino, teste):
        presentes = [c for c in atributos(N) if c in parte.columns]
        assert not parte[presentes].isna().to_numpy().any()


def test_o_escalonador_e_ajustado_no_treino_e_nao_muda_com_o_teste():
    """D28, segunda regra, agora sobre o painel inteiro e não sobre um vetor."""
    painel = _painel()
    _, _, escala_a, _, _ = preparar_treino_teste(painel)

    mexido = painel.copy()
    fim = mexido.week.max()
    alvo = mexido.week > fim - 5           # só o final, que cai no teste
    mexido.loc[alvo, "preco_proprio"] *= 100.0
    _, _, escala_b, _, _ = preparar_treino_teste(mexido)
    assert escala_a["preco_proprio"] == escala_b["preco_proprio"]


def test_o_tempo_congela_fora_da_janela_de_treino():
    treino, teste, _, _, meta = preparar_treino_teste(_painel())
    assert treino["tempo"].max() <= 1.0 and treino["tempo"].min() >= 0.0
    # O teste está inteiro além do fim do treino, então congela em 1.
    assert np.allclose(teste["tempo"].to_numpy(), 1.0)


def test_o_indicador_de_ausencia_marca_as_primeiras_semanas():
    treino, _, _, _, _ = preparar_treino_teste(_painel())
    primeira = treino[treino.week == treino.week.min()]
    assert (primeira["falta_preco_lag1"] == 1.0).all()
