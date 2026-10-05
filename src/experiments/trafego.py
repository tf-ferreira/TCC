"""Quanto da resposta ao preço passa pelo tráfego de clientes (seção 7).

A contagem de clientes por loja e semana existe no arquivo `ccount` e é
tentadora: mede tráfego, varia no tempo, e não é função direta do preço de
nenhum SKU. Ela é **pós-tratamento**, e este script mede o custo disso em vez
de afirmá-lo.

## O mecanismo, no menor caso em que ele aparece

Uma loja, uma semana. A 4,00 a loja tem 100 clientes que levam 2 unidades cada,
200 no total. A 3,00 ela tem 120 clientes que levam 3 cada, 360. O corte de
preço vale **+160**.

Mantendo os clientes em 100, como faria um modelo que recebe a contagem e a
congela durante a otimização, o mesmo corte vale 100 × 3 = 300, ou **+100**. As
60 unidades que vieram dos 20 clientes a mais somem, 37,5% da resposta.

O modelo não está errado: ele mede o efeito **dentro do mesmo tráfego**. O
problema é que o objeto que o trabalho precisa é o efeito **total**, porque o
gerente que baixa o preço recebe as duas parcelas.

## O que este script mede

A mesma especificação de série de D18 (efeitos fixos de série e de semana, custo
como instrumento), com e sem `ln(clientes)` como regressor exógeno. A diferença
entre os dois coeficientes de preço é a parcela que o controle pós-tratamento
bloqueia, agora no dado e não no brinquedo.

Uso:
    python3 src/experiments/trafego.py frj
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src" / "experiments"))
sys.path.insert(0, str(RAIZ / "src" / "data"))

from elasticidades import (carregar_sortimento, centralizar,  # noqa: E402
                           extrair_longo, montar_celula, ols,
                           preparar_celulas, sanduiche, tsls, zip_unico)

SEMANA_MIN, SEMANA_MAX = 1, 399
DIAS_NA_SEMANA = 7


def clientes_por_celula(caminho: Path) -> pd.DataFrame:
    """Soma o `custcoun` diário por (loja, semana), só de semanas completas.

    O arquivo é **diário**, sete linhas por loja e semana, e traz semanas fora
    de faixa (valores negativos) que são registro corrompido. Somar semana
    incompleta criaria variação que é de dias faltando e não de tráfego, e ela
    entraria no coeficiente como se fosse.
    """
    z = zipfile.ZipFile(caminho)
    nome = [n for n in z.namelist() if n.lower().endswith(".dta")][0]
    d = pd.read_stata(io.BytesIO(z.read(nome)), columns=["store", "week", "custcoun"])
    d = d.dropna(subset=["week"])
    d = d[(d.week >= SEMANA_MIN) & (d.week <= SEMANA_MAX) & (d.custcoun > 0)]
    g = d.groupby(["store", "week"]).agg(clientes=("custcoun", "sum"),
                                         dias=("custcoun", "size")).reset_index()
    return g[g.dias == DIAS_NA_SEMANA].drop(columns="dias")


def estimar(d: pd.DataFrame, com_clientes: bool) -> dict:
    """IV de série, com ou sem `ln(clientes)` entre os exógenos."""
    colunas = [np.log(d.volume.to_numpy()), np.log(d.preco.to_numpy()),
               np.log(d.custo.to_numpy())]
    if com_clientes:
        colunas.append(np.log(d.clientes.to_numpy()))
    M = centralizar(np.column_stack(colunas),
                    [d.serie.to_numpy(), d.week.to_numpy()])
    y = M[:, 0]
    endogeno = M[:, 1:2]
    instrumento = M[:, 2:3]
    exogenos = M[:, 3:] if com_clientes else np.empty((len(y), 0))

    X = np.hstack([endogeno, exogenos])
    Z = np.hstack([instrumento, exogenos])
    nfe = d.serie.nunique() + d.week.nunique()
    k = nfe + X.shape[1]

    b_iv, _, u_iv = tsls(y, X, Z, k)
    # Para 2SLS o "R" do sanduíche é o X projetado, não o X bruto (ver o
    # docstring de `sanduiche` em elasticidades.py).
    Xh = Z @ np.linalg.solve(Z.T @ Z, Z.T @ X)
    ep_iv = sanduiche(Xh, u_iv, {"loja": d.store.to_numpy()}, k)
    b_ols, _, u_ols = ols(y, X, k)
    ep_ols = sanduiche(X, u_ols, {"loja": d.store.to_numpy()}, k)
    return {
        "n": int(len(y)),
        "iv_preco": float(b_iv[0]),
        "iv_ep_agrupado_loja": float(ep_iv["loja"]["ep"][0]),
        "ols_preco": float(b_ols[0]),
        "ols_ep_agrupado_loja": float(ep_ols["loja"]["ep"][0]),
        "coef_clientes_iv": float(b_iv[1]) if com_clientes else None,
    }


def estimar_celula_com_trafego(d, P, V, C, com_clientes: bool) -> dict:
    """A mesma pergunta no estimando que o otimizador de fato move.

    A especificação de série mede o efeito do preço de **um** SKU, e um SKU de
    suco congelado não muda o tráfego da loja. O otimizador move **vinte preços
    ao mesmo tempo**, que é um movimento de categoria, e é aí que o canal de
    tráfego teria chance de existir. Medir só a de série responderia a pergunta
    errada.
    """
    g = montar_celula(d, P, V, C)
    colunas = [g["y"], g["lnP"], g["lnI"]]
    if com_clientes:
        colunas.append(np.log(d.clientes.to_numpy()))
    colunas += [g["lnZ"], g["lnW"]]
    M = centralizar(np.column_stack(colunas),
                    [d.store.to_numpy(), d.week.to_numpy()])
    y = M[:, 0]
    if com_clientes:
        X = np.column_stack([M[:, 1], M[:, 2], M[:, 3]])
        Z = np.column_stack([M[:, 4], M[:, 5], M[:, 3]])
    else:
        X = M[:, 1:3]
        Z = M[:, 3:5]
    nfe = d.store.nunique() + d.week.nunique()
    k = nfe + X.shape[1]
    b_iv, _, u_iv = tsls(y, X, Z, k)
    Xh = Z @ np.linalg.solve(Z.T @ Z, Z.T @ X)
    ep_iv = sanduiche(Xh, u_iv, {"loja": d.store.to_numpy()}, k)
    b_ols, _, u_ols = ols(y, X, k)
    ep_ols = sanduiche(X, u_ols, {"loja": d.store.to_numpy()}, k)
    return {
        "n": int(len(y)),
        "iv_preco": float(b_iv[0]),
        "iv_ep_agrupado_loja": float(ep_iv["loja"]["ep"][0]),
        "ols_preco": float(b_ols[0]),
        "ols_ep_agrupado_loja": float(ep_ols["loja"]["ep"][0]),
        "coef_clientes_iv": float(b_iv[2]) if com_clientes else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--raiz", default="data/raw")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--clientes", default="data/raw/customer_count/ccount_stata.zip")
    ap.add_argument("--saida", default="reports")
    a = ap.parse_args()

    pa = Path(a.painel)
    upcs = carregar_sortimento(a.categoria, pa)
    longo = extrair_longo(zip_unico(Path(a.raiz) / a.categoria), set(upcs))
    longo = longo[(longo.volume > 0) & longo.custo.notna() & (longo.custo > 0)].copy()
    longo["serie"] = longo.store.astype(np.int64) * 10**12 + longo.upc

    cli = clientes_por_celula(Path(a.clientes))
    antes = len(longo)
    d = longo.merge(cli, on=["store", "week"], how="inner")
    d["serie"] = d.store.astype(np.int64) * 10**12 + d.upc

    # A MESMA amostra nas duas estimativas, senão a diferença misturaria efeito
    # do controle com efeito de amostra.
    saida = {
        "categoria": a.categoria,
        "linhas_serie": antes,
        "linhas_com_clientes": int(len(d)),
        "pct_serie_com_clientes": float(100.0 * len(d) / antes),
        "sem_controle_de_trafego": estimar(d, False),
        "com_controle_de_trafego": estimar(d, True),
    }
    s, c = saida["sem_controle_de_trafego"], saida["com_controle_de_trafego"]
    saida["atenuacao_iv_pct"] = float(
        100.0 * (1.0 - abs(c["iv_preco"]) / abs(s["iv_preco"])))
    saida["atenuacao_ols_pct"] = float(
        100.0 * (1.0 - abs(c["ols_preco"]) / abs(s["ols_preco"])))

    # --- o mesmo teste no estimando agregado, que é o que o otimizador move ---
    n = len(upcs)
    cel = pd.read_parquet(pa / f"{a.categoria}_celulas.parquet")
    fora = pd.read_parquet(pa / f"{a.categoria}_bem_externo.parquet")
    dc, P, V, C = preparar_celulas(cel, fora, n)
    completas = dc[(dc[P] > 0).all(axis=1) & (dc[C] > 0).all(axis=1)]
    completas = completas.merge(cli, on=["store", "week"], how="inner")
    sc = estimar_celula_com_trafego(completas, P, V, C, False)
    cc = estimar_celula_com_trafego(completas, P, V, C, True)
    saida["celula_sem_controle"] = sc
    saida["celula_com_controle"] = cc
    saida["atenuacao_celula_iv_pct"] = float(
        100.0 * (1.0 - abs(cc["iv_preco"]) / abs(sc["iv_preco"])))

    destino = Path(a.saida) / f"trafego_{a.categoria}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))

    print(f"amostra: {saida['linhas_com_clientes']} de {antes} linhas de série "
          f"({saida['pct_serie_com_clientes']:.1f}%)")
    print(f"{'':28s} {'IV':>9} {'ep(loja)':>9} {'MQO':>9} {'ep(loja)':>9}")
    for rot, r in (("sem controle de tráfego", s), ("com controle de tráfego", c)):
        print(f"{rot:28s} {r['iv_preco']:9.3f} {r['iv_ep_agrupado_loja']:9.4f} "
              f"{r['ols_preco']:9.3f} {r['ols_ep_agrupado_loja']:9.4f}")
    print(f"\natenuação da elasticidade: IV {saida['atenuacao_iv_pct']:.1f}%, "
          f"MQO {saida['atenuacao_ols_pct']:.1f}%")
    print(f"coeficiente de ln(clientes), IV: {c['coef_clientes_iv']:.3f}")
    print(f"\ncélula (os vinte preços juntos), {sc['n']} células:")
    for rot, r in (("sem controle de tráfego", sc), ("com controle de tráfego", cc)):
        print(f"{rot:28s} {r['iv_preco']:9.3f} {r['iv_ep_agrupado_loja']:9.4f} "
              f"{r['ols_preco']:9.3f} {r['ols_ep_agrupado_loja']:9.4f}")
    print(f"atenuação da agregada: IV {saida['atenuacao_celula_iv_pct']:.1f}%")
    print(f"coeficiente de ln(clientes) na célula, IV: {cc['coef_clientes_iv']:.3f}")
    print(f"gravado em {destino}")


if __name__ == "__main__":
    main()
