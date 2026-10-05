"""D16, suavidade: a derivada própria da rede ao longo da faixa de preço.

## O que D16 pede

"Inspeção da derivada ao longo da faixa de preço: oscilação de alta frequência
indica superfície irregular ainda que o nível pareça bom."

## O que a arquitetura já decide, antes de medir

Na rede restrita a elasticidade própria é `εᵢᵢ(u) = −mᵢ′(u)`, com

    mᵢ′(u)  = Σ_h w2[i,h]·w1[i,h]·σ(w1[i,h]·u + b[i,h])
    mᵢ″(u)  = Σ_h w2[i,h]·w1[i,h]²·σ′(w1[i,h]·u + b[i,h])

com `w1, w2 = softplus(·) > 0` e `σ′ > 0`. Todo termo de `mᵢ″` é positivo, então
**`mᵢ′` é crescente em u para qualquer valor dos pesos**: a elasticidade própria
fica monotonamente mais negativa conforme o preço sobe. Uma função monótona não
oscila. O risco que D16 descreve (derivada ondulada com nível bom) é **impossível
por construção** no termo próprio, e o termo cruzado é constante (D36).

Então este script não mede *se* há oscilação: ele confere a propriedade nos pesos
treinados (tem de dar zero trocas de sinal) e mede o que sobra, que é **quanto** a
elasticidade varia dentro da caixa de ±15%. Isso é o que importa para o PNL: uma
elasticidade quase constante na caixa quer dizer que o otimizador vê a mesma
inclinação em toda a região em que pode andar.

## O limiar que conecta ao item 8

Sem cruzadas, a receita própria `pᵢVᵢ` cresce com o preço onde `|εᵢᵢ| < 1` e cai
onde `|εᵢᵢ| > 1`. O ponto `u*` com `εᵢᵢ = −1` é o estacionário da receita própria.
Ele é reportado por SKU, com a informação de cair dentro ou fora da faixa
observada. **Não é o ótimo do PNL**, que tem cruzadas e restrições (item 8).

Uso:
    python3 src/experiments/suavidade_d16.py frj
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CAIXA = (np.log(0.85), np.log(1.15))
PONTOS = 2001


def _softplus(x):
    return np.logaddexp(0.0, x)


def _sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


def derivadas(w1_bruto, w2_bruto, b, u):
    """`mᵢ′` e `mᵢ″` em forma fechada. `u` com forma (P,) ou (P, n); saída (P, n)."""
    w1, w2 = _softplus(w1_bruto), _softplus(w2_bruto)            # (n, h)
    u = np.asarray(u, float)
    if u.ndim == 1:
        u = np.repeat(u[:, None], w1.shape[0], axis=1)
    z = u[..., None] * w1 + b                                     # (P, n, h)
    s = _sigmoide(z)
    m1 = (s * w1 * w2).sum(-1)
    m2 = (s * (1 - s) * w1 ** 2 * w2).sum(-1)
    return m1, m2


def trocas_de_sinal(v, tol=0.0):
    """Número de trocas de sinal de uma série, ignorando |v| <= tol."""
    s = np.sign(np.where(np.abs(v) <= tol, 0.0, v))
    s = s[s != 0]
    return int((s[1:] != s[:-1]).sum())


def u_de_elasticidade_um(w1_bruto, w2_bruto, b, lo, hi, it=200):
    """`u` com `mᵢ′(u) = 1`, por bisseção (mᵢ′ é crescente). nan fora de [lo, hi]."""
    n = w1_bruto.shape[0]
    a = np.full(n, lo, float)
    c = np.full(n, hi, float)
    fa = derivadas(w1_bruto, w2_bruto, b, a[None, :])[0][0] - 1
    fc = derivadas(w1_bruto, w2_bruto, b, c[None, :])[0][0] - 1
    dentro = (fa <= 0) & (fc >= 0)
    for _ in range(it):
        m = (a + c) / 2
        fm = derivadas(w1_bruto, w2_bruto, b, m[None, :])[0][0] - 1
        a = np.where(fm < 0, m, a)
        c = np.where(fm < 0, c, m)
    raiz = (a + c) / 2
    return np.where(dentro, raiz, np.nan), fa + 1, fc + 1


def analisar(w1_bruto, w2_bruto, b, U: np.ndarray) -> dict:
    """`U` (células, n): u observados. Devolve o registro por SKU."""
    n = U.shape[1]
    q = np.quantile(U, [0.01, 0.5, 0.99], axis=0)                 # (3, n)
    lo = q[0] + CAIXA[0]
    hi = q[2] + CAIXA[1]
    grade = lo[None, :] + (hi - lo)[None, :] * np.linspace(0, 1, PONTOS)[:, None]
    m1, m2 = derivadas(w1_bruto, w2_bruto, b, grade)
    eps = -m1
    # a caixa de ±15% em torno do preço mediano, que é a região do PNL típico
    caixa = np.stack([q[1] + CAIXA[0], q[1], q[1] + CAIXA[1]])
    ec = -derivadas(w1_bruto, w2_bruto, b, caixa)[0]              # (3, n)
    raiz, m1_lo, m1_hi = u_de_elasticidade_um(w1_bruto, w2_bruto, b, lo, hi)
    # por célula: amplitude na caixa de CADA célula observada
    e_baixo = -derivadas(w1_bruto, w2_bruto, b, U + CAIXA[0])[0]
    e_alto = -derivadas(w1_bruto, w2_bruto, b, U + CAIXA[1])[0]
    e_obs = -derivadas(w1_bruto, w2_bruto, b, U)[0]

    por_sku = []
    for i in range(n):
        # Oscilação é a INCLINAÇÃO da elasticidade trocar de sinal: primeira
        # diferença de ε, que aproxima −mᵢ″. A segunda diferença aproxima −mᵢ‴,
        # que troca de sinal legitimamente no ponto de inflexão de cada sigmoide.
        d1 = np.diff(eps[:, i])
        por_sku.append({
            "sku": i,
            "min_m2": float(m2[:, i].min()),
            "trocas_de_sinal_da_inclinacao": trocas_de_sinal(d1, tol=1e-14),
            "elast_p01_menos_15": float(eps[0, i]),
            "elast_p99_mais_15": float(eps[-1, i]),
            "elast_mediana_menos_15": float(ec[0, i]),
            "elast_mediana": float(ec[1, i]),
            "elast_mediana_mais_15": float(ec[2, i]),
            "amplitude_caixa_mediana": float(ec[0, i] - ec[2, i]),
            "amplitude_relativa_caixa_mediana": float((ec[0, i] - ec[2, i]) / abs(ec[1, i])),
            "u_elasticidade_menos_um": None if np.isnan(raiz[i]) else float(raiz[i]),
            "elasticidade_um_dentro_da_faixa": bool(np.isfinite(raiz[i])),
            "pct_celulas_inelasticas": float(100 * (e_obs[:, i] > -1).mean()),
        })
    amp_cel = e_baixo - e_alto
    return {
        "n_celulas": int(U.shape[0]), "pontos_na_grade": PONTOS,
        "min_m2_global": float(m2.min()),
        "total_trocas_de_sinal": int(sum(s["trocas_de_sinal_da_inclinacao"]
                                         for s in por_sku)),
        "amplitude_por_celula": {"mediana": float(np.median(amp_cel)),
                                 "p90": float(np.quantile(amp_cel, 0.9)),
                                 "max": float(amp_cel.max())},
        "amplitude_relativa_por_celula_mediana": float(np.median(amp_cel / np.abs(e_obs))),
        "pct_celulas_inelasticas": float(100 * (e_obs > -1).mean()),
        "skus_com_elasticidade_um_na_faixa": int(sum(s["elasticidade_um_dentro_da_faixa"]
                                                     for s in por_sku)),
        "por_sku": por_sku,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--artefato", default=None)
    a = ap.parse_args()

    import torch
    sys.path.insert(0, str(RAIZ / "src" / "experiments"))
    sys.path.insert(0, str(RAIZ / "src" / "models"))
    sys.path.insert(0, str(RAIZ / "src" / "data"))
    from varredura_rede import MELHOR

    arq = Path(a.artefato or RAIZ / "models_artifacts" / f"rede_{a.categoria}.pt")
    pt = torch.load(arq, map_location="cpu", weights_only=False)
    cfg = pt["cfg"]
    divergentes = {k: (cfg.get(k), v) for k, v in MELHOR.items() if cfg.get(k) != v}
    if divergentes or not cfg.get("restrita", False):
        raise SystemExit(f"o artefato {arq} não é o ponto adotado: {divergentes}. "
                         "Refaça com `python3 src/models/rede.py frj --adotada`.")
    est = pt["estado"]
    w1 = est["monotona.w1_bruto"].numpy().astype("float64")
    w2 = est["monotona.w2_bruto"].numpy().astype("float64")
    b = est["monotona.b"].numpy().astype("float64")
    n = w1.shape[0]

    cel = pd.read_parquet(Path(a.painel) / f"{a.categoria}_celulas.parquet")
    P = cel[[f"preco_{i:02d}" for i in range(n)]].to_numpy(float)
    P = P[np.all(P > 0, axis=1)]
    U = np.log(P) - np.asarray(pt["centro_de_preco"], float)

    r = analisar(w1, w2, b, U)
    saida = {"categoria": a.categoria, "artefato": str(arq.name),
             "semente": cfg.get("semente"), "configuracao": cfg,
             "caixa": [0.85, 1.15], **r}
    destino = RAIZ / "reports" / f"suavidade_d16_{a.categoria}.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))

    print("min m'' global (tem de ser > 0):", f'{r["min_m2_global"]:.3e}')
    print("trocas de sinal da inclinação de ε (tem de ser 0):", r["total_trocas_de_sinal"])
    print("amplitude da elasticidade na caixa ±15%, por célula:",
          json.dumps({k: round(v, 4) for k, v in r["amplitude_por_celula"].items()}),
          "relativa mediana", round(r["amplitude_relativa_por_celula_mediana"], 4))
    print("% células inelásticas no preço observado:", round(r["pct_celulas_inelasticas"], 2))
    print("SKUs com ε = −1 dentro da faixa:", r["skus_com_elasticidade_um_na_faixa"], "de", n)
    for s in r["por_sku"]:
        print(f'  {s["sku"]:2d}  ε mediana {s["elast_mediana"]:7.3f}  '
              f'caixa [{s["elast_mediana_mais_15"]:7.3f}, {s["elast_mediana_menos_15"]:7.3f}]  '
              f'amp rel {s["amplitude_relativa_caixa_mediana"]:.3f}  '
              f'inelást {s["pct_celulas_inelasticas"]:5.1f}%')
    print("->", destino)


if __name__ == "__main__":
    main()
