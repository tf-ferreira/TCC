"""Instrumentos alternativos para o preço: relevância e deslocamento do estimando.

Motivação registrada em 13/09/2026. O instrumento de D18 é o custo unitário
derivado de `price × (1 - profit/100) / qty`. Ele tem relevância alta no nível
de série (F de 88.593), mas duas fragilidades declaradas:

1. toda a variação que o torna instrumento vem da **margem reportada**, de modo
   que a exclusão exige que a margem não responda ao choque de demanda;
2. a separação entre a variação do custo e a do preço dentro da série é menor
   do que os documentos afirmavam: CV mediano de 0,1144 contra 0,1382.

Este experimento não substitui o instrumento principal. Ele mede três
alternativas na **mesma especificação** (elasticidade própria de SKU, efeitos
fixos de série e de semana) e reporta, para cada uma, relevância e quanto o
estimando se move. Um instrumento alternativo que produza elasticidade próxima
é evidência de robustez; um que produza elasticidade muito diferente é evidência
de que pelo menos um dos dois viola exclusão, e o texto tem de dizer qual e por
quê.

## Os três candidatos

**A. Hausman, preço do mesmo SKU em outras lojas.** Média do preço do mesmo UPC
na mesma semana, nas demais lojas da rede (deixa-uma-de-fora). Explora o
componente comum de custo entre lojas e descarta o choque local. **Exclusão**
exige que o choque de demanda não seja comum às lojas, hipótese discutível numa
mesma região metropolitana: feriado, clima e campanha nacional atingem todas.

**B. Custo do mesmo SKU em outras lojas.** Mesma construção, sobre o custo. Mais
defensável que A, porque não usa o preço de ninguém, e mais fraca, porque o
custo já é derivado.

**C. Custo defasado uma semana na própria série.** Quebra a simultaneidade
contemporânea. Não quebra a persistência: se o choque de demanda for
autocorrelacionado, o custo de t-1 continua correlacionado com o erro de t.

Uso:
    python3 -m src.experiments.instrumentos frj
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sortimento import carregar_sortimento, extrair_longo            # noqa: E402
from elasticidades import bloco, centralizar, zip_unico              # noqa: E402


def deixa_uma_de_fora(d: pd.DataFrame, coluna: str) -> pd.Series:
    """Média do valor nas DEMAIS lojas, para o mesmo UPC e semana.

    Com `n` lojas reportando o UPC na semana, a média deixa-uma-de-fora é
    `(soma - proprio) / (n - 1)`. Onde só uma loja reporta, é indefinida.
    """
    g = d.groupby(["upc", "week"], observed=True)[coluna]
    soma = g.transform("sum")
    n = g.transform("count")
    return (soma - d[coluna]) / (n - 1)


def custo_defasado(d: pd.DataFrame) -> pd.Series:
    """Custo da semana anterior na mesma série (loja, SKU), só se contígua."""
    d = d.sort_values(["store", "upc", "week"])
    g = d.groupby(["store", "upc"], observed=True)
    anterior = g["custo"].shift(1)
    semana_anterior = g["week"].shift(1)
    return anterior.where(semana_anterior == d["week"] - 1).reindex(d.index)


def estimar(d: pd.DataFrame, instrumento: str, nome: str) -> dict:
    """A especificação de `estimar_serie`, trocando apenas o instrumento."""
    u = d[d[instrumento].notna() & (d[instrumento] > 0)].copy()
    u["serie"] = u.store.astype(np.int64) * 10**12 + u.upc
    M = centralizar(np.column_stack([
        np.log(u.volume.to_numpy()),
        np.log(u.preco.to_numpy()),
        np.log(u[instrumento].to_numpy()),
    ]), [u.serie.to_numpy(), u.week.to_numpy()])
    nfe = u.serie.nunique() + u.week.nunique()
    r = bloco(nome, f"serie, instrumento = {instrumento}",
              M[:, 0], M[:, 1:2], M[:, 2:3], nfe, ["ln_preco_proprio"],
              grupos={"serie": u.serie.to_numpy(),
                      "loja": u.store.to_numpy(),
                      "semana": u.week.to_numpy()})
    r["n_series"] = int(u.serie.nunique())
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--raiz", default="data/raw")
    ap.add_argument("--painel", default="data/interim/painel")
    a = ap.parse_args()

    upcs = carregar_sortimento(a.categoria, Path(a.painel))
    d = extrair_longo(zip_unico(Path(a.raiz) / a.categoria), set(upcs))
    d = d[(d.volume > 0) & d.custo.notna() & (d.custo > 0)].copy()

    d["preco_outras_lojas"] = deixa_uma_de_fora(d, "preco")
    d["custo_outras_lojas"] = deixa_uma_de_fora(d, "custo")
    d["custo_defasado"] = custo_defasado(d)

    especs = [
        ("custo", "D18_custo_proprio"),
        ("preco_outras_lojas", "A_hausman_preco_outras_lojas"),
        ("custo_outras_lojas", "B_custo_outras_lojas"),
        ("custo_defasado", "C_custo_defasado"),
    ]
    est = {nome: estimar(d, col, nome) for col, nome in especs}

    base = est["D18_custo_proprio"]["iv"]["ln_preco_proprio"]["coef"]
    for nome, r in est.items():
        c = r["iv"]["ln_preco_proprio"]["coef"]
        r["deslocamento_pct_vs_D18"] = 100.0 * (c - base) / abs(base)

    destino = Path(a.painel) / f"{a.categoria}_instrumentos.json"
    destino.write_text(json.dumps({"categoria": a.categoria,
                                   "estimativas": est}, indent=2,
                                  ensure_ascii=False))

    print(f"{'instrumento':32s} {'n':>9s} {'MQO':>9s} {'IV':>9s} {'ep':>7s} "
          f"{'F1':>10s} {'r2parc':>8s} {'desloc%':>8s} {'corr':>7s}")
    for nome, r in est.items():
        rot = "ln_preco_proprio"
        print(f"{nome:32s} {r['n']:9d} {r['ols'][rot]['coef']:+9.4f} "
              f"{r['iv'][rot]['coef']:+9.4f} {r['iv'][rot]['ep']:7.4f} "
              f"{r['primeiro_estagio'][rot]['F_instrumentos']:10.0f} "
              f"{100*r['primeiro_estagio'][rot]['r2_parcial']:7.2f}% "
              f"{r['deslocamento_pct_vs_D18']:+7.1f}% "
              f"{r['corr_endogeno_instrumento'][rot]:+7.3f}")
    print(f"\ngravado em {destino}")


if __name__ == "__main__":
    main()
