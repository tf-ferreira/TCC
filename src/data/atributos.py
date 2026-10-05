"""Painel de modelagem: o produto do item 5 do cronograma.

Monta, num lugar só, o painel que a rede e o modelo de referência consomem,
aplicando as seis decisões da especificação de atributos. Nenhum atributo entra
aqui sem uma seção correspondente em `docs/especificacao_atributos.md`, e nenhum
atributo daquela especificação fica de fora.

## O que entra, e de onde

| bloco | colunas | decisão |
|---|---|---|
| preços de decisão | `preco_00..preco_19`, `preco_proprio` | painel, D12 |
| identidade | `store`, `sku` (por embedding) | D29 |
| bem externo | `idx_preco_resto`, `vol_resto`, `promo_resto`, `n_resto` | D23 |
| sazonalidade | `sen_1`, `cos_1`, `sen_2`, `cos_2` | seção 2 |
| tendência | `tempo` (normalizado e truncado na janela de treino) | D28 |
| defasagens | `preco_lag1`, `preco_lag2` e seus indicadores de ausência | D27, D28 |

## O que NÃO entra, e por quê em uma linha cada

`promo_proprio` e a promoção dos outros dezenove são função do preço decidido
(D25). `volume_lagK` atenua a derivada (D27). Custo afeta a demanda só através do
preço e é o instrumento de D18 (D20). Demografia, `zone`, `scluster` e faixa de
preço são redundantes com o embedding e faltam justo nas lojas onde importariam
(D29). Contagem de clientes e elasticidades pré-calculadas, D30.

## Por que o parquet guarda colunas que não são atributos

`promo_proprio` e `semana_do_ano` ficam gravados e **fora** de `ATRIBUTOS`. Eles
existem para que a comparação com o piso de D20 rode sobre **exatamente as mesmas
linhas** do painel final: sem isso, a diferença entre o piso e o resultado do
item 5 misturaria efeito de atributo com efeito de amostra. Quem treina usa
`ATRIBUTOS`; quem compara com o piso usa `ATRIBUTOS_PISO`.

## Por que o preenchimento e o escalonamento NÃO são feitos aqui

Os dois são ajustados na janela de treino (D28), então dependem da partição.
Gravá-los no parquet assaria a partição dentro do dado e tornaria impossível
mudar a fração de teste sem reconstruir tudo. `preparar_treino_teste` aplica-os
no momento em que a partição existe.

Uso:
    python3 src/data/atributos.py frj
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calendario import data_da_semana, harmonicos
from defasagens import adicionar_defasagens, cobertura
from preparo import (aplicar_escalonador, ajustar_escalonador, codificar,
                     medianas_de_treino, preencher_defasagens, tempo_truncado,
                     vocabulario)

CHAVES = ["store", "week", "sku"]
ALVO = "alvo"
LAGS = ["preco_lag1", "preco_lag2"]
SAZONAIS = ["sen_1", "cos_1", "sen_2", "cos_2"]
BEM_EXTERNO = ["idx_preco_resto", "vol_resto", "promo_resto", "n_resto"]
CATEGORICAS = ["sku", "store"]

# Versão do pipeline de preparo. Entra na impressão digital de todo fragmento de
# treino (varredura_rede, espelho_d16), porque uma mudança de pipeline muda o
# modelo sem mudar a configuração, e o fragmento antigo seria reaproveitado com o
# significado errado. Trocada em 17/09/2026 pela pendência 9.6: o vocabulário de
# loja passou a ser do treino.
VERSAO_PIPELINE = "2026-09-17_vocabulario_do_treino"
# Colunas gravadas mas que NÃO são atributos: existem só para a comparação com
# o piso de D20 rodar sobre as mesmas linhas.
NAO_ATRIBUTOS = ["promo_proprio", "semana_do_ano"]


def atributos(n: int) -> list[str]:
    """A lista canônica, na ordem fixa. A posição no vetor é a identidade do
    produto para a rede (ver `painel.py`), então a ordem não pode variar."""
    return ([f"preco_{i:02d}" for i in range(n)]
            + ["preco_proprio"] + CATEGORICAS + SAZONAIS + ["tempo"]
            + LAGS + [f"falta_{c}" for c in LAGS] + BEM_EXTERNO)


def atributos_piso(n: int) -> list[str]:
    """O conjunto do piso de D20: sem defasagem, com `promo_proprio` e com
    `semana_do_ano` no lugar das harmônicas."""
    return ([f"preco_{i:02d}" for i in range(n)]
            + ["preco_proprio", "promo_proprio"] + CATEGORICAS
            + ["semana_do_ano"] + BEM_EXTERNO)


def construir(cel: pd.DataFrame, fora: pd.DataFrame, n: int) -> pd.DataFrame:
    """Painel longo, uma linha por (célula, SKU), só de células completas."""
    from baseline_arvores import montar_longo
    longo = montar_longo(cel, fora, n).reset_index(drop=True)
    datas = data_da_semana(longo["week"].to_numpy())
    longo = pd.concat([longo, harmonicos(datas, 2)], axis=1)
    longo["data"] = datas.to_numpy()
    longo = adicionar_defasagens(longo, ks_preco=(1, 2))
    return longo


def preparar_treino_teste(painel: pd.DataFrame, frac_teste: float = 0.20,
                          zona: int = 8) -> tuple:
    """Aplica as regras de D28, que dependem da partição, e devolve os dois lados.

    A ordem importa e não é arbitrária: particiona, **depois** ajusta tudo no
    treino, **depois** aplica nos dois. Ajustar antes de particionar seria o
    vazamento que D28 proíbe, e o teste `test_o_escalonador_nao_ve_o_teste` fixa
    a metade disso que é verificável sem dado real.
    """
    from baseline_arvores import fixar_categorias, particionar
    painel = fixar_categorias(painel.copy(), CATEGORICAS)
    treino, teste, corte = particionar(painel, frac_teste, zona)

    mediana = medianas_de_treino(treino)
    treino = preencher_defasagens(treino, mediana, LAGS)
    teste = preencher_defasagens(teste, mediana, LAGS)

    w0, w1 = int(treino.week.min()), int(treino.week.max())
    for parte in (treino, teste):
        parte["tempo"] = tempo_truncado(parte.week.to_numpy(), w0, w1)

    contínuas = ([c for c in painel.columns if c.startswith("preco_")
                  and c != "preco_proprio"]
                 + ["preco_proprio"] + SAZONAIS + LAGS + BEM_EXTERNO)
    escala = ajustar_escalonador(treino, contínuas)

    # Do TREINO (pendência 9.6, 17/09/2026). Do painel inteiro, as 7 lojas que
    # só aparecem no teste recebiam código próprio, nunca caíam no índice de
    # reserva, e liam linhas de embedding que não recebiam gradiente.
    vocab = {c: vocabulario(treino, c) for c in CATEGORICAS}
    meta = {"semana_de_corte": int(corte), "zona_morta_semanas": int(zona),
            "versao_pipeline": VERSAO_PIPELINE,
            "janela_treino": [w0, w1], "escalonador": escala,
            "vocabulario_tamanho": {c: len(v) for c, v in vocab.items()},
            "n_treino": int(len(treino)), "n_teste": int(len(teste))}
    return treino, teste, escala, vocab, meta


def aplicar(quadro: pd.DataFrame, escala: dict, vocab: dict) -> pd.DataFrame:
    """Escalona as contínuas e codifica as categóricas para índice de embedding."""
    saida = aplicar_escalonador(quadro, escala)
    for c, v in vocab.items():
        saida[c] = codificar(saida[c].to_numpy(), v)
    return saida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--saida", default="data/processed")
    ap.add_argument("--frac-teste", type=float, default=0.20)
    ap.add_argument("--zona", type=int, default=8)
    a = ap.parse_args()

    pa = Path(a.painel)
    n = json.loads((pa / f"{a.categoria}_upcs.json").read_text())["n"]
    cel = pd.read_parquet(pa / f"{a.categoria}_celulas.parquet")
    fora = pd.read_parquet(pa / f"{a.categoria}_bem_externo.parquet")
    painel = construir(cel, fora, n)

    _, _, _, _, meta = preparar_treino_teste(painel, a.frac_teste, a.zona)
    colunas = CHAVES + [ALVO, "data"] + [c for c in painel.columns
                                         if c not in CHAVES + [ALVO, "data"]]
    destino = Path(a.saida) / f"{a.categoria}_modelagem.parquet"
    destino.parent.mkdir(parents=True, exist_ok=True)
    painel[colunas].to_parquet(destino, index=False)

    diag = {
        "categoria": a.categoria,
        "n_sortimento": n,
        "n_linhas": int(len(painel)),
        "n_celulas": int(painel.groupby(["store", "week"]).ngroups),
        "n_atributos": len(atributos(n)),
        "atributos": atributos(n),
        "nao_atributos_gravados": NAO_ATRIBUTOS,
        "cobertura_defasagem": cobertura(painel, LAGS),
        "particao": {k: meta[k] for k in ("semana_de_corte", "zona_morta_semanas",
                                          "janela_treino", "n_treino", "n_teste")},
        "vocabulario_tamanho": meta["vocabulario_tamanho"],
    }
    (Path(a.saida) / f"{a.categoria}_modelagem.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False))

    print(f"painel de modelagem: {diag['n_linhas']} linhas, "
          f"{diag['n_celulas']} células, {diag['n_atributos']} atributos")
    print(f"  cobertura: " + ", ".join(
        f"{k} {v:.1%}" for k, v in diag["cobertura_defasagem"].items()))
    print(f"  partição: treino até {meta['janela_treino'][1]}, "
          f"teste a partir de {meta['semana_de_corte']}, "
          f"{meta['n_treino']} e {meta['n_teste']} linhas")
    print(f"gravado em {destino}")


if __name__ == "__main__":
    import sys
    RAIZ = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(RAIZ / "src" / "models"))
    sys.path.insert(0, str(RAIZ / "src" / "data"))
    main()
