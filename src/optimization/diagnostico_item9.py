"""O sinal da resposta da rede à DEFASAGEM de preço, que decide a leitura de D27.

## Por que este script existe

A execução de 18/09/2026 do item 9 contradisse D27 na receita. D27 afirmava que a
avaliação com defasagem histórica "superestima sempre no sentido favorável", e o
argumento era o brinquedo de estocagem: corte hoje, o consumidor antecipa a compra,
a demanda de amanhã cai. Medido:

| objetivo | A/B − 1 | o que o otimizador faz com o preço |
|---|---:|---|
| margem | **+5,79%** | sobe em boa parte das coordenadas (49,95% na borda superior, 10.4) |
| receita | **−3,23%** | corta (64,92% na borda inferior, 10.4) |

Um só sinal da resposta à defasagem explica as duas linhas: se **preço passado mais
alto reduz a demanda de hoje** (`∂ln V̂/∂ln p_lag < 0`), a margem, que sobe preço,
herda defasagem alta e perde volume na avaliação honesta; a receita, que corta,
herda defasagem baixa e **ganha** volume. É o **oposto** da estocagem, que prevê
`∂ln V̂/∂ln p_lag > 0`.

Isso é hipótese até ser medido, e este script mede. Ele não otimiza nada.

## O que mede

Sobre as células ELEGÍVEIS do item 9, no ponto histórico, a Jacobiana
`Jᵢⱼ = ∂ln V̂ᵢ/∂ln p_lagⱼ` por autograd em `ctx`, com a regra da cadeia do
escalonamento: `x = (p − μ)/σ`, logo `∂/∂ln p = (p/σ)·∂/∂x`.

- **diagonal** (`Jᵢᵢ`): defasagem própria;
- **resposta agregada**: variação percentual do volume total da célula a uma alta
  UNIFORME de 1% em todas as vinte defasagens, `Σᵢ V̂ᵢ Σⱼ Jᵢⱼ / Σᵢ V̂ᵢ`. É a forma
  imune a composição, pelo mesmo argumento da identidade de 2.8.

E, sem rede, a seleção da amostra: nível observado e calibração nas elegíveis
contra as excluídas, que é o critério de invalidação de D38 transposto para D40.

Uso:
    python3 src/optimization/diagnostico_item9.py frj
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
for sub in ("models", "data", "experiments", "optimization"):
    sys.path.insert(0, str(RAIZ / "src" / sub))

import contrafactual_item9 as C9  # noqa: E402
import problema as P  # noqa: E402
import rede  # noqa: E402
from sonda_item8 import _percentis, carregar_modelo  # noqa: E402


def jacobiana_da_defasagem(modelo, esc, idx, bloco: str) -> tuple:
    """`(J, V̂)`, com `J` de forma (B, n, n) em unidade de elasticidade."""
    import torch
    n = esc["n"]
    fat = C9.fatias_dos_lags(n)[bloco]
    mu, sd = esc["escalas_dos_lags"][bloco]
    dtype = next(modelo.parameters()).dtype
    ctx = torch.as_tensor(esc["ctx"][idx], dtype=dtype).requires_grad_(True)
    u = torch.as_tensor(esc["u"][idx], dtype=dtype)
    loja = torch.as_tensor(esc["loja_idx"][idx].astype("int64"), dtype=torch.long)
    lnv = modelo.log_demanda(ctx, u, loja)                       # (B, n)
    J = np.empty((len(idx), n, n))
    for i in range(n):
        (g,) = torch.autograd.grad(lnv[:, i].sum(), ctx, retain_graph=True)
        J[:, i, :] = g[:, fat].detach().numpy()                   # ∂lnVᵢ/∂xⱼ
    x = esc["ctx"][idx][:, fat]
    p_lag = x * sd + mu
    J = J * (p_lag / sd)[:, None, :]                              # ∂lnVᵢ/∂ln pⱼ
    return J, np.exp(lnv.detach().numpy())


def resumo_da_jacobiana(J, V) -> dict:
    diag = np.einsum("bii->bi", J)
    agregada = (V * J.sum(axis=2)).sum(axis=1) / V.sum(axis=1)
    fora = J.sum(axis=2) - diag
    return {
        "propria_por_sku_mediana": np.median(diag, axis=0).tolist(),
        "propria": _percentis(diag.ravel()),
        "pct_propria_negativa": 100.0 * float((diag < 0).mean()),
        "cruzada_somada": _percentis(fora.ravel()),
        "agregada_uniforme_por_celula": _percentis(agregada),
        "pct_celulas_agregada_negativa": 100.0 * float((agregada < 0).mean()),
        "agregada_uniforme_ponderada":
            float((V * J.sum(axis=2)).sum() / V.sum()),
    }


def selecao(esc, idx_eleg, modelo, centro) -> dict:
    """Elegíveis contra excluídas, no lado do dado e na calibração."""
    todos = np.arange(len(esc["u"]))
    excl = np.setdiff1d(todos, idx_eleg)
    fora = {}
    for nome, idx in (("elegiveis", idx_eleg), ("excluidas", excl)):
        k0 = np.zeros_like(esc["custos"][idx])
        R = C9.valor_observado(esc["precos"][idx], esc["alvo"][idx], k0)
        M = C9.valores_em_lote(modelo, esc["ctx"][idx], esc["u"][idx],
                               esc["loja_idx"][idx], centro, k0)
        fora[nome] = {
            "n_celulas": int(len(idx)),
            "n_lojas": int(esc["chaves"]["store"].iloc[idx].nunique()),
            "volume_total_por_celula": _percentis(esc["alvo"][idx].sum(axis=1)),
            "receita_observada_por_celula": _percentis(R),
            "preco_medio_da_celula": _percentis(esc["precos"][idx].mean(axis=1)),
            "calibracao_receita_agregada": float(M.sum() / R.sum()),
        }
    return fora


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--artefato", default=None)
    ap.add_argument("--saida", default="reports")
    a = ap.parse_args()

    painel = Path(a.painel)
    dados = rede.preparar(a.categoria, painel)
    arq = Path(a.artefato or RAIZ / "models_artifacts" / f"rede_{a.categoria}.pt")
    modelo, cfg, centro = carregar_modelo(arq, dados)
    esc = P.escopo(dados, painel, a.categoria, "teste")
    esc["escalas_dos_lags"] = C9.escalas_dos_lags(dados, esc["n"])
    mapa = C9.mapa_das_corridas(esc["chaves"])
    S = mapa["indices_elegiveis"]

    bloco = {"categoria": a.categoria, "artefato": arq.name,
             "n_celulas_elegiveis": int(len(S)), "defasagem": {}}
    for chave in ("lag1", "lag2"):
        J, V = jacobiana_da_defasagem(modelo, esc, S, chave)
        bloco["defasagem"][chave] = resumo_da_jacobiana(J, V)
    bloco["selecao"] = selecao(esc, S, modelo, centro)

    destino = Path(a.saida) / f"diagnostico_item9_{a.categoria}.json"
    destino.write_text(json.dumps(bloco, indent=2, ensure_ascii=False))

    print(f"{len(S)} células elegíveis, ponto histórico\n")
    print("elasticidade do volume à DEFASAGEM de preço")
    print("  (estocagem prevê > 0; a hipótese de 18/09 prevê < 0)")
    for chave, r in bloco["defasagem"].items():
        print(f"  {chave}: própria mediana {r['propria']['mediana']:+.4f} "
              f"(p10 {r['propria']['p10']:+.4f}, p90 {r['propria']['p90']:+.4f}), "
              f"negativa em {r['pct_propria_negativa']:.1f}%")
        print(f"        agregada uniforme ponderada "
              f"{r['agregada_uniforme_ponderada']:+.4f}, "
              f"negativa em {r['pct_celulas_agregada_negativa']:.1f}% das células")
    print("\nseleção da amostra")
    for nome, s in bloco["selecao"].items():
        print(f"  {nome:10s} n={s['n_celulas']:5d} lojas={s['n_lojas']:3d}  "
              f"volume/célula mediana {s['volume_total_por_celula']['mediana']:8.1f}  "
              f"preço médio {s['preco_medio_da_celula']['mediana']:.3f}  "
              f"calibração receita {s['calibracao_receita_agregada']:.4f}")
    print(f"\n-> {destino}")


if __name__ == "__main__":
    main()
