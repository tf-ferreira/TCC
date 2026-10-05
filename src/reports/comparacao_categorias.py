"""Tabela comparativa das categorias sob o critério vigente, versionada.

`data/` é ignorado pelo git (decisão de não versionar binário grande), então
as saídas de `nucleo.py` vivem só no disco. Este script destila o que vai para
o texto e grava em `reports/`, que **é** versionado. Sem isso a comparação
entre categorias volta a ser um número sem artefato rastreável, que é a classe
de erro que a fase 0 existe para impedir.

Uso:
    python3 src/reports/comparacao_categorias.py --n 20
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

NOMES = {
    "sdr": "Refrigerantes", "fre": "Refeições congeladas", "cso": "Sopas enlatadas",
    "frj": "Sucos congelados", "frd": "Pratos congelados", "che": "Queijos",
    "bjc": "Sucos engarrafados", "tti": "Papel higiênico",
}

CAMPOS = [
    ("n_upcs_categoria", "SKUs", 0),
    ("n_elegiveis", "elegíveis", 0),
    ("regularidade_minima", "reg. mín.", 3),
    ("celulas_completas", "células", 0),
    ("observacoes_completas", "observações", 0),
    ("cv_dois_estagios", "CV preço", 3),
    ("promo_preco_dois_estagios", "promoção", 3),
    ("cobertura_volume_categoria_pct", "cobertura vol. %", 1),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--screening", default="data/interim/screening")
    ap.add_argument("--saida", default="reports")
    a = ap.parse_args()

    linhas = []
    for caminho in sorted(Path(a.screening).glob(f"*_nucleo{a.n}.json")):
        d = json.loads(caminho.read_text())
        d["nome"] = NOMES.get(d["categoria"], d["categoria"])
        linhas.append({k: d.get(k) for k, _, _ in CAMPOS}
                      | {"categoria": d["categoria"], "nome": d["nome"],
                         "criterio": d.get("criterio")})
    linhas.sort(key=lambda r: -(r["cobertura_volume_categoria_pct"] or 0))

    saida = {"gerado_em": date.today().isoformat(), "n": a.n,
             "criterio": "completude_conjunta",
             "produtor": "src/data/nucleo.py",
             "observacao": ("Comparação refeita sob o critério vigente de D12. "
                            "A comparação original do texto usa tau_c(30) sob "
                            "regularidade individual, que é a regra superada."),
             "categorias": linhas}
    destino = Path(a.saida) / f"comparacao_categorias_n{a.n}.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))

    cab = ["categoria"] + [r for _, r, _ in CAMPOS]
    print("| " + " | ".join(cab) + " |")
    print("|" + "|".join(["---"] * len(cab)) + "|")
    for r in linhas:
        celulas = [r["nome"]]
        for chave, _, casas in CAMPOS:
            v = r[chave]
            celulas.append("—" if v is None else
                           (f"{int(v):,}".replace(",", ".") if casas == 0
                            else f"{v:.{casas}f}".replace(".", ",")))
        print("| " + " | ".join(celulas) + " |")
    print(f"\ngravado em {destino}")


if __name__ == "__main__":
    main()
