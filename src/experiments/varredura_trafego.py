"""Varredura da contagem de clientes (seção 7 da especificação de atributos).

A contagem de clientes é **pós-tratamento**: o preço mexe no tráfego e o tráfego
mexe no volume. O argumento de princípio diz para não condicionar nela. A
medição de `src/experiments/trafego.py` mostrou que, **neste dado**, o canal é
vazio: controlar por tráfego move a elasticidade própria em −0,1% e a agregada
em −2,2%, nenhuma das duas distinguível de zero.

Com o argumento forte derrubado, a decisão passa a depender do mesmo critério
que decidiu D25 e D27: o que ela faz com o **ajuste** e o que ela faz com a
**derivada**, medidos lado a lado. É isso que esta varredura entrega.

A base já está sob todas as seções fechadas: sem `promo_proprio` (D25), duas
harmônicas sobre a data real (seção 2) e defasagens de preço de 1 e 2 semanas
(D27). A única coisa que muda entre as duas configurações é o tráfego.

Cobertura: a contagem existe em cerca de 83% das células completas. As linhas
sem ela recebem a mediana da loja na janela de treino mais indicador de
ausência, que é a regra de D28.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src" / "experiments"))
sys.path.insert(0, str(RAIZ / "src" / "models"))
sys.path.insert(0, str(RAIZ / "src" / "data"))

from baseline_arvores import fixar_categorias, montar_longo, particionar  # noqa: E402
from calendario import data_da_semana, harmonicos  # noqa: E402
from defasagens import adicionar_defasagens  # noqa: E402
from trafego import clientes_por_celula  # noqa: E402

CATEGORICAS = ["sku", "store"]
SAZONAIS = ["sen_1", "cos_1", "sen_2", "cos_2"]
LAGS = ["preco_lag1", "preco_lag2"]
CAMINHO_CLIENTES = RAIZ / "data/raw/customer_count/ccount_stata.zip"


def preparar(cel: pd.DataFrame, fora: pd.DataFrame, n: int) -> pd.DataFrame:
    longo = montar_longo(cel, fora, n).reset_index(drop=True)
    datas = data_da_semana(longo["week"].to_numpy())
    longo = pd.concat([longo, harmonicos(datas, 2)], axis=1)
    longo = adicionar_defasagens(longo, ks_preco=(1, 2))

    cli = clientes_por_celula(CAMINHO_CLIENTES)
    longo = longo.merge(cli, on=["store", "week"], how="left")
    longo["falta_clientes"] = longo.clientes.isna().astype("float64")
    # Preenchimento pela mediana da loja na janela de TREINO (D28). A partição
    # é recalculada aqui só para saber onde o treino termina; o preenchimento
    # não pode olhar o teste.
    _, _, corte = particionar(longo.assign(alvo=longo["alvo"]), 0.20, 8)
    treino = longo[longo.week < corte - 8]
    mediana = treino.groupby("store", observed=True)["clientes"].median()
    referencia = longo.store.map(mediana)
    longo["clientes"] = longo.clientes.fillna(referencia).fillna(
        float(treino.clientes.median()))
    longo["ln_clientes"] = np.log(longo.clientes.clip(lower=1.0))
    return fixar_categorias(longo, CATEGORICAS)


def configuracoes(_=None) -> dict[str, list[str]]:
    base = SAZONAIS + LAGS
    return {
        "sem_trafego": base,
        "com_trafego": base + ["ln_clientes", "falta_clientes"],
    }
