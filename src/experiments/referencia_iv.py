"""A referência IV da própria, como faixa (rodada R2 do plano de 23/09/2026).

D18 usa o custo (AAC) como instrumento do preço próprio, com efeitos fixos de
série e de semana, e dá −1,757 na amostra do sortimento. O diagnóstico de 23/09
(F2) mostrou três coisas que o número sozinho esconde, e este script as mede com
o código do projeto (`elasticidades.centralizar`, `tsls`, `sanduiche`):

1. **O aparato promocional.** O custo cai nas semanas de promoção, e o acordo
   comercial que o derruba costuma vir com encarte e exposição: efeito direto do
   instrumento sobre a demanda. Controlar os códigos de promoção declarados
   (B, C e S, variáveis exógenas nos dois estágios) move o coeficiente.
2. **A exclusão plausível** (Conley, Hansen e Rossi, 2012, versão de ponto): se o
   custo tem efeito direto ρ sobre `ln V`, o IV devolve β + ρ/π, e a curva
   β(ρ) = β_IV − ρ/π diz quanto de violação muda a conclusão.
3. **Onde agrupar.** 97,5% da variação intra-série do custo é SKU × semana, comum
   às lojas. O erro agrupado por loja (D24) trata como independentes semanas em
   que o instrumento é o mesmo em toda a rede. Aqui vão lado a lado: série, loja,
   SKU × semana, SKU, e duas vias (loja e SKU × semana).

A faixa sem e com os códigos é **referência de ordem de grandeza** para a
elasticidade própria da v1, e não critério (D42): D16 mostrou que o IV sem preços
cruzados estima a própria mais a cruzada vezes o arrasto do instrumento.

Uso:
    python3 src/experiments/referencia_iv.py frj
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src" / "experiments"))

from elasticidades import Z95, centralizar, ols, projetar, sanduiche, tsls  # noqa: E402

RHOS = (0.0, -0.02, -0.05, -0.10, -0.15, -0.20)
CODIGOS = ("B", "C", "S")


def carregar(bruto: Path, painel: Path, categoria: str) -> pd.DataFrame:
    """Linhas do sortimento com preço, venda e custo válidos (as de D18)."""
    upcs = json.loads((painel / f"{categoria}_upcs.json").read_text())["ordem"]
    z = zipfile.ZipFile(bruto / categoria / f"w{categoria}.zip")
    d = pd.read_csv(z.open(f"w{categoria}.csv"),
                    usecols=["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE",
                             "SALE", "PROFIT", "OK"])
    d.columns = [c.lower() for c in d.columns]
    d = d[(d.ok == 1) & (d.price > 0) & (d.move > 0) & d.upc.isin(upcs)].copy()
    d["p"] = d.price / d.qty
    d["c"] = d.price * (1 - d.profit / 100) / d.qty
    d = d[(d.c > 0) & (d.c < d.p)].copy()
    d["sale"] = d["sale"].fillna("").astype(str).str.strip().str.upper()
    for cod in CODIGOS:
        d[f"cod_{cod}"] = (d["sale"] == cod).astype(float)
    d["serie"] = d.store.astype(np.int64) * 10**12 + d.upc.astype(np.int64)
    d["sku_semana"] = d.upc.astype(np.int64) * 10**4 + d.week.astype(np.int64)
    d["loja_sku_semana"] = d.serie * 10**4 + d.week.astype(np.int64)
    return d.reset_index(drop=True)


def estimar(d: pd.DataFrame, com_codigos: bool) -> dict:
    """IV com efeitos fixos de série e semana; códigos como exógenas, se pedido."""
    extras = [f"cod_{c}" for c in CODIGOS] if com_codigos else []
    base = np.column_stack([np.log(d.move.to_numpy(float)), np.log(d.p.to_numpy(float)),
                            np.log(d.c.to_numpy(float))]
                           + [d[c].to_numpy(float) for c in extras])
    diag: dict = {}
    M = centralizar(base, [d.serie.to_numpy(), d.week.to_numpy()], diagnostico=diag)
    y = M[:, 0]
    X = np.column_stack([M[:, 1]] + [M[:, 3 + j] for j in range(len(extras))])
    Z = np.column_stack([M[:, 2]] + [M[:, 3 + j] for j in range(len(extras))])
    nfe = d.serie.nunique() + d.week.nunique()
    k = nfe + X.shape[1]
    b_ols, _, _ = ols(y, X, k)
    b_iv, ep_iid, r_iv = tsls(y, X, Z, k)
    # Primeiro estágio: π é o coeficiente do custo no preço, dadas as exógenas.
    b_fs, ep_fs, _ = ols(X[:, 0], Z, k)
    pi = float(b_fs[0])
    grupos = {"serie": d.serie.to_numpy(), "loja": d.store.to_numpy(),
              "sku_semana": d.sku_semana.to_numpy(), "sku": d.upc.to_numpy(),
              "loja_sku_semana": d.loja_sku_semana.to_numpy()}
    ag = sanduiche(projetar(Z, X), r_iv, grupos, k)
    ep = {g: float(ag[g]["ep"][0]) for g in grupos}
    # Duas vias (Cameron, Gelbach e Miller, 2011): V_loja + V_skusemana − V_interseção,
    # na diagonal do coeficiente de interesse.
    v2 = ep["loja"] ** 2 + ep["sku_semana"] ** 2 - ep["loja_sku_semana"] ** 2
    ep["duas_vias_loja_e_sku_semana"] = float(np.sqrt(max(v2, 0.0)))
    beta = float(b_iv[0])
    return {"com_codigos": com_codigos, "n": int(len(d)),
            "mqo": float(b_ols[0]), "iv": beta, "ep_iid": float(ep_iid[0]),
            "ep_agrupado": ep, "G": {g: int(ag[g]["G"]) for g in grupos},
            "primeiro_estagio_pi": pi, "primeiro_estagio_t": float(pi / ep_fs[0]),
            "ic95_duas_vias": [beta - Z95 * ep["duas_vias_loja_e_sku_semana"],
                               beta + Z95 * ep["duas_vias_loja_e_sku_semana"]],
            "exclusao_plausivel": {f"{r:+.2f}": beta - r / pi for r in RHOS},
            "rho_que_leva_modulo_a_1": float((beta + 1.0) * pi),
            "absorcao_efeitos_fixos": diag}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--bruto", default="data/raw")
    ap.add_argument("--saida", default="reports")
    a = ap.parse_args()
    d = carregar(Path(a.bruto), Path(a.painel), a.categoria)
    blocos = {"sem_codigos": estimar(d, False), "com_codigos": estimar(d, True)}
    frac_cod = {c: float(d[f"cod_{c}"].mean()) for c in CODIGOS}
    saida = {"categoria": a.categoria, "fracao_de_linhas_por_codigo": frac_cod, **blocos}
    destino = Path(a.saida) / f"referencia_iv_{a.categoria}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))
    for nome, b in blocos.items():
        print(f"== {nome}: n {b['n']}, MQO {b['mqo']:+.3f}, IV {b['iv']:+.3f}, π {b['primeiro_estagio_pi']:.3f}")
        print("   ep: " + ", ".join(f"{g} {v:.3f}" for g, v in b["ep_agrupado"].items()))
        print("   β(ρ): " + ", ".join(f"ρ {r}: {v:+.3f}" for r, v in b["exclusao_plausivel"].items()))
        print(f"   ρ que leva |β| a 1: {b['rho_que_leva_modulo_a_1']:+.3f}")
    print(f"\n-> {destino}")


if __name__ == "__main__":
    main()
