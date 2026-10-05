"""Duas medições que sustentam D12, e o único lugar onde a regra superada vive.

1. **Robustez da busca gulosa.** Repete a seleção a partir de cada SKU
   elegível como semente, em vez de começar sempre pelo mais presente, e
   compara o melhor resultado com o adotado. A busca não tem certificado
   teórico (maximizar interseção equivale a minimizar união de complementos,
   e para isso não há garantia de fator constante), então a defesa é empírica.

2. **Desconfundir a tabela de sinal.** Mede variação de preço e frequência
   promocional das quatro configurações relevantes, com N fixo, para separar
   o efeito do **critério** do efeito de **N**. Foi essa separação que
   derrubou a afirmação de que a completude conjunta melhorava o sinal.

> **A regra superada de D12 vive aqui e só aqui.** `selecionar_por_regularidade_individual`
> é objeto de comparação, não caminho de produção. Ela ficava exposta em
> `nucleo.py` como API pública e foi chamada por engano pelo diagnóstico de
> endogeneidade, produzindo estimativas sobre um sortimento que o projeto já
> tinha aposentado. Mantê-la aqui preserva a reprodutibilidade da evidência
> sem que ninguém possa usá-la para gerar resultado novo.

Cada medida de sinal sai em **duas agregações**, porque as duas circulam nos
documentos e divergem:

    dois_estagios   mediana por SKU sobre as séries, depois mediana entre SKUs
    direto          mediana única sobre o conjunto indistinto de séries

A oficial é a de dois estágios, que é a de D3: a pergunta é sobre o SKU
típico do sortimento, não sobre a série típica. A direta pondera cada SKU
pelo número de lojas em que ele aparece.

Uso:
    python3 src/experiments/robustez_selecao.py frj
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from sortimento import (elegiveis, extrair_longo, guloso, matriz_presenca,   # noqa: E402
                        agregar_direto, agregar_dois_estagios, semente_padrao)


def selecionar_por_regularidade_individual(cobertura: pd.DataFrame,
                                           n: int) -> list[int]:
    """A regra superada de D12. **Não use fora desta comparação.**

    Ordena os elegíveis por regularidade individual e pega os n maiores. Mede
    assiduidade ("enquanto existiu, esteve na prateleira?") quando a pergunta
    da pesquisa é coexistência ("esteve na prateleira ao mesmo tempo que os
    outros?"). Pelos limites de Fréchet, N produtos com presença p admitem
    interseção entre max(0, 1 - N(1-p)) e p, e com p = 0,95 o limite inferior
    chega a zero exatamente em N = 20: a métrica individual não restringe o
    que importa.
    """
    apto = cobertura.dropna(subset=["regularidade"])
    apto = apto[(apto["longevidade"] >= 0.50) & (apto["amplitude"] >= 0.50)]
    return sorted(apto.sort_values("regularidade", ascending=False)
                  .head(n)["upc"].tolist())


def sinal(series: pd.DataFrame, upcs) -> dict:
    s = series[series.upc.isin(set(upcs))]
    return {
        "n_series": int(len(s)),
        "n_skus": int(s.upc.nunique()),
        "cv_dois_estagios": agregar_dois_estagios(series, upcs, "cv"),
        "cv_direto": agregar_direto(series, upcs, "cv"),
        "promo_preco_dois_estagios":
            agregar_dois_estagios(series, upcs, "frac_promo_preco"),
        "promo_preco_direto": agregar_direto(series, upcs, "frac_promo_preco"),
        "sale_declarado_dois_estagios":
            agregar_dois_estagios(series, upcs, "frac_sale_declarado"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--raiz", default="data/raw")
    ap.add_argument("--screening", default="data/interim/screening")
    ap.add_argument("--saida", default="data/interim/painel")
    a = ap.parse_args()

    cob = pd.read_csv(Path(a.screening) / f"{a.categoria}_cobertura.csv")
    series = pd.read_csv(Path(a.screening) / f"{a.categoria}_series.csv")
    cand = elegiveis(cob)
    zips = sorted((Path(a.raiz) / a.categoria).glob("w*.zip"))
    longo = extrair_longo(zips[0], set(cand))
    P = matriz_presenca(longo, cand)
    print(f"celulas={P.shape[0]}  elegiveis={P.shape[1]}", flush=True)

    out = {"categoria": a.categoria, "n_celulas": int(P.shape[0]),
           "n_elegiveis": int(P.shape[1])}

    # ---- 1. robustez da busca gulosa -------------------------------------
    padrao = semente_padrao(P)
    for n in (20, 30):
        res = []
        for s in range(P.shape[1]):
            esc, cel = guloso(P, n, s)
            res.append((cel, s, sorted(esc)))
        res.sort(key=lambda r: -r[0])
        adotado = next(r for r in res if r[1] == padrao)
        melhor = res[0]
        out[f"guloso_n{n}"] = {
            "semente_padrao": padrao,
            "celulas_adotado": adotado[0],
            "celulas_melhor": melhor[0],
            "semente_melhor": melhor[1],
            "ganho_absoluto": melhor[0] - adotado[0],
            "ganho_relativo_pct": 100.0 * (melhor[0] - adotado[0]) / adotado[0],
            "celulas_pior": res[-1][0],
            "celulas_mediana": float(np.median([r[0] for r in res])),
            "celulas_p05": float(np.percentile([r[0] for r in res], 5)),
            "skus_trocados_no_melhor": len(set(melhor[2]) - set(adotado[2])),
        }
        print(f"N={n}", json.dumps(out[f"guloso_n{n}"]), flush=True)

    # ---- 2. desconfundir a tabela de sinal --------------------------------
    conj = {}
    for n in (20, 30):
        esc, cel = guloso(P, n, padrao)
        upcs = sorted(cand[j] for j in esc)
        conj[n] = {"celulas": cel, "upcs": upcs, **sinal(series, upcs)}

    ind30 = selecionar_por_regularidade_individual(cob, 30)
    tabela = {
        "categoria_inteira": sinal(series, series.upc.unique()),
        "regularidade_individual_n30": {"upcs": ind30, **sinal(series, ind30)},
        "completude_conjunta_n30": conj[30],
        "completude_conjunta_n20": conj[20],
    }
    out["sinal"] = tabela
    print(f"\n{'configuracao':32s} {'cv(2est)':>9} {'cv(dir)':>9} "
          f"{'promo(2est)':>12} {'promo(dir)':>11}")
    for k, v in tabela.items():
        print(f"{k:32s} {v['cv_dois_estagios']:9.4f} {v['cv_direto']:9.4f} "
              f"{v['promo_preco_dois_estagios']:12.4f} "
              f"{v['promo_preco_direto']:11.4f}", flush=True)

    dest = Path(a.saida) / f"{a.categoria}_robustez.json"
    dest.write_text(json.dumps(out, indent=2))
    print("\ngravado em", dest)


if __name__ == "__main__":
    main()
