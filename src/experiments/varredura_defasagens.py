"""Varredura de defasagens (seção 3 da especificação de atributos).

Mede três coisas de uma vez, com todo o resto igual e já sob as seções fechadas
(D25: sem `promo_proprio`; seção 2: duas harmônicas sobre a data real):

    sem_lag        nenhuma defasagem, que é o estado do piso de D20
    lagp1..lagp1a8 defasagem do PREÇO próprio, 1, 2, 4 e 8 semanas
    lagp1_volume   defasagem de preço mais defasagem do PRÓPRIO VOLUME

**A grandeza decisiva aqui não é o WMAPE, é a elasticidade implícita.** A seção 3
rejeita a defasagem de volume por dois motivos, e o segundo é que ela absorve
variância e **encolhe a derivada em relação ao preço**, que é a mesma família de
atenuação de D25 e é o que este trabalho menos pode pagar. Essa afirmação tem de
ser medida, não repetida.

E ela é barata de medir: D26 exige 50 sementes para afirmar diferença de WMAPE,
porque a amplitude do WMAPE entre sementes chega a 4,8 pontos. A elasticidade
implícita, medida nas mesmas execuções, tem amplitude de 0,013 a 0,039. São
ordens de grandeza diferentes, e por isso a conclusão sobre a derivada se
sustenta com poucas sementes enquanto a conclusão sobre o ajuste não. O artefato
reporta as duas amplitudes lado a lado justamente para que a assimetria fique
verificável, em vez de ser alegada.

Cobertura: a defasagem não está definida em toda linha, porque o painel de
células completas tem buracos. `src/data/defasagens.py` faz a junção por
calendário e conta a ausência; a cobertura vai no artefato, porque comparar uma
configuração definida em 87% com outra definida em 100% sem declarar isso
contamina a comparação.

Uso: via `src/experiments/ruido_semente.py --familia defasagens`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src" / "models"))
sys.path.insert(0, str(RAIZ / "src" / "data"))

from baseline_arvores import montar_longo  # noqa: E402
from calendario import data_da_semana, harmonicos  # noqa: E402
from defasagens import adicionar_defasagens, cobertura  # noqa: E402

CATEGORICAS = ["sku", "store"]
K_MAX = 8
# Seção 2, fechada: duas harmônicas sobre a data real.
SAZONAIS = ["sen_1", "cos_1", "sen_2", "cos_2"]


def preparar(cel: pd.DataFrame, fora: pd.DataFrame, n: int) -> pd.DataFrame:
    """Painel longo com o calendário da seção 2 e todas as defasagens candidatas."""
    from baseline_arvores import fixar_categorias
    longo = montar_longo(cel, fora, n).reset_index(drop=True)
    datas = data_da_semana(longo["week"].to_numpy())
    longo = pd.concat([longo, harmonicos(datas, 2)], axis=1)
    longo = adicionar_defasagens(longo, ks_preco=tuple(range(1, K_MAX + 1)),
                                 ks_volume=(1,))
    return fixar_categorias(longo, CATEGORICAS)


def configuracoes(_=None) -> dict[str, list[str]]:
    """Atributos ACRESCENTADOS à base. A base já inclui os sazonais da seção 2."""
    p = lambda ks: [f"preco_lag{k}" for k in ks]  # noqa: E731
    return {
        "sem_lag": list(SAZONAIS),
        "lagp1": SAZONAIS + p([1]),
        "lagp12": SAZONAIS + p([1, 2]),
        "lagp1a4": SAZONAIS + p(range(1, 5)),
        "lagp1a8": SAZONAIS + p(range(1, 9)),
        "lagp1_volume": SAZONAIS + p([1]) + ["volume_lag1"],
    }


def cobertura_das_defasagens(longo: pd.DataFrame) -> dict:
    cols = [f"preco_lag{k}" for k in range(1, K_MAX + 1)] + ["volume_lag1"]
    return cobertura(longo, cols)


def main() -> None:
    """Grava a cobertura das defasagens como artefato próprio.

    A cobertura é citada em D27 e na seção 3, então precisa de produtor
    declarado, como manda `src/reports/numeros_oficiais.py`. A varredura em si
    sai por `ruido_semente.py --familia defasagens`.
    """
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--saida", default="reports")
    a = ap.parse_args()

    pa = Path(a.painel)
    n = json.loads((pa / f"{a.categoria}_upcs.json").read_text())["n"]
    longo = preparar(pd.read_parquet(pa / f"{a.categoria}_celulas.parquet"),
                     pd.read_parquet(pa / f"{a.categoria}_bem_externo.parquet"), n)
    cob = cobertura_das_defasagens(longo)
    destino = Path(a.saida) / f"defasagens_{a.categoria}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(
        {"categoria": a.categoria, "n_linhas": int(len(longo)),
         "cobertura": cob}, indent=2, ensure_ascii=False))
    for c, v in cob.items():
        print(f"  {c:14s} {v:6.1%}")
    print(f"gravado em {destino}")


if __name__ == "__main__":
    main()
