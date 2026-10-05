"""O valor do item 5, medido: o piso de D20 contra o painel final.

D20 prometeu esta medida ao declarar a ressalva da sua primeira execução:

> Esta execução é um **piso**, não a comparação final. (...) reexecutada depois
> do item 5, a diferença entre as duas execuções **mede o valor da engenharia de
> atributos**, que de outro modo ficaria sem medida.

Aqui está a execução prometida. As duas configurações rodam sobre **as mesmas
linhas**, saídas do mesmo painel de `src/data/atributos.py`, e diferem apenas no
conjunto de atributos:

    piso    `promo_proprio`, `semana_do_ano`, sem defasagem
    final   as seis decisões da especificação (D25, seção 2, D27, D28, D29, D30)

É por isso que o painel de modelagem grava `promo_proprio` e `semana_do_ano` sem
listá-los como atributos: sem eles a comparação teria de reconstruir o piso a
partir de outro painel, e a diferença misturaria efeito de atributo com efeito de
amostra.

**O escalonamento e a codificação de embedding não são aplicados aqui**, de
propósito. Árvore é invariante a transformação monótona das contínuas, e
transformar categoria em índice quebraria o tratamento categórico do LightGBM. As
duas operações são da fase da rede; o que esta varredura mede é o **conjunto de
atributos**.

Uso: via `src/experiments/ruido_semente.py --familia item5`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src" / "models"))
sys.path.insert(0, str(RAIZ / "src" / "data"))

from atributos import (CATEGORICAS, atributos, atributos_piso,  # noqa: E402
                       construir, preparar_treino_teste)

# As configurações trazem a lista COMPLETA de atributos, e não um acréscimo
# sobre uma base comum: é o conjunto inteiro que está em disputa.
COMPLETO = True

_N = {"n": None}


def preparar(cel: pd.DataFrame, fora: pd.DataFrame, n: int) -> pd.DataFrame:
    """Painel final, com as regras de D28 já aplicadas a partir do treino.

    O preenchimento e o truncamento do tempo dependem da partição, então são
    aplicados aqui, com a mediana e a janela vindas **do treino**. As linhas da
    zona morta somem, como devem: elas não entram nem no treino nem no teste.
    """
    _N["n"] = n
    painel = construir(cel, fora, n)
    treino, teste, _, _, _ = preparar_treino_teste(painel)
    return pd.concat([treino, teste], ignore_index=False)


def configuracoes(_=None) -> dict[str, list[str]]:
    n = _N["n"]
    if n is None:
        raise RuntimeError("chame preparar antes de configuracoes")
    final = atributos(n)
    return {
        "piso": atributos_piso(n),
        "final": final,
        # Isola o `tempo` truncado de D28. Ele é a única coluna do conjunto
        # final que não foi medida sozinha em nenhuma varredura anterior, e a
        # pendência 7.9 registrava exatamente essa lacuna.
        "final_sem_tempo": [c for c in final if c != "tempo"],
        # Isola o efeito de recência de outra forma: só o piso mais o tempo.
        "piso_com_tempo": atributos_piso(n) + ["tempo"],
    }
