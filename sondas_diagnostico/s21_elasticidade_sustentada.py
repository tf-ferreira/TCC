"""Sonda 21: a elasticidade própria SUSTENTADA, por SKU, na adotada e na v1.

D19 e D33 garantem que, dentro da semana, a demanda de um produto cai quando o
preço dele sobe: `∂ ln V̂ᵢ / ∂ ln pᵢ = −mᵢ′ ≤ 0`. A garantia não diz nada sobre o
que acontece quando o preço fica mais alto por várias semanas, porque os preços
das semanas anteriores entram em `g(x)` pelas defasagens, sem restrição.

A elasticidade sustentada de `i` é a resposta de `ln V̂ᵢ` a subir 10% o preço de
`i` na semana corrente E nas duas anteriores, com todo o resto no histórico:

    εᵢ(sust) = [ln V̂ᵢ(p·1,1; defasagens·1,1) − ln V̂ᵢ(p; defasagens)] / ln 1,1

decomposta na parte contemporânea (só o preço corrente, que é −mᵢ′ integrado) e na
parte das defasagens (só as duas defasagens de `i`). Medida nas células elegíveis
do teste, mediana por SKU, algumas sementes.

Se `εᵢ(sust) > 0`, a rede diz que manter o preço de `i` mais alto AUMENTA a
demanda de `i`, e uma avaliação sequencial credita isso à política.

Uso: `cd sondas_diagnostico && python3 s21_elasticidade_sustentada.py [n_sementes]`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
for sub in ("models", "data", "experiments", "optimization"):
    sys.path.insert(0, str(RAIZ / "src" / sub))

import contrafactual_item9 as C9  # noqa: E402
import problema as P  # noqa: E402
import rede  # noqa: E402
import torch  # noqa: E402
from varredura_rede import MELHOR  # noqa: E402

K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
PAINEL = RAIZ / "data" / "interim" / "painel"
H = np.log(1.1)


def logv(modelo, ctx, u, loja):
    with torch.no_grad():
        return modelo.log_demanda(torch.tensor(ctx, dtype=torch.float32),
                                  torch.tensor(u, dtype=torch.float32),
                                  torch.tensor(loja, dtype=torch.long)).numpy()


def main():
    saida = {}
    for espec in ("adotada", "v1"):
        rede.usar_especificacao(espec)
        dados = rede.preparar("frj", PAINEL)
        esc = P.escopo(dados, PAINEL, "frj", "teste")
        n = esc["n"]
        el = C9.escalas_dos_lags(dados, n)
        fat = C9.fatias_dos_lags(n)
        S = C9.mapa_das_corridas(esc["chaves"])["indices_elegiveis"]
        ctx0, u0, loja = esc["ctx"][S], esc["u"][S], esc["loja_idx"][S]
        # os preços históricos das duas semanas anteriores, lidos do próprio contexto
        p1 = ctx0[:, fat["lag1"]] * el["lag1"][1] + el["lag1"][0]
        p2 = ctx0[:, fat["lag2"]] * el["lag2"][1] + el["lag2"][0]
        linhas = []
        for s in range(K):
            cfg = dict(MELHOR, semente=s, restrita=True, especificacao=espec, gama_contextual=False)
            modelo = rede.treinar(dados, cfg)["modelo"].eval()
            base = logv(modelo, ctx0, u0, loja)
            cont, defa, sust = np.zeros(n), np.zeros(n), np.zeros(n)
            for i in range(n):
                u1 = u0.copy(); u1[:, i] += H
                c1 = ctx0.copy()
                c1[:, fat["lag1"].start + i] = (p1[:, i] * 1.1 - el["lag1"][0][i]) / el["lag1"][1][i]
                c1[:, fat["lag2"].start + i] = (p2[:, i] * 1.1 - el["lag2"][0][i]) / el["lag2"][1][i]
                cont[i] = np.median((logv(modelo, ctx0, u1, loja) - base)[:, i]) / H
                defa[i] = np.median((logv(modelo, c1, u0, loja) - base)[:, i]) / H
                sust[i] = np.median((logv(modelo, c1, u1, loja) - base)[:, i]) / H
            linhas.append({"semente": s, "contemporanea": cont.tolist(),
                           "defasagens": defa.tolist(), "sustentada": sust.tolist()})
            print(f"{espec} semente {s}: sustentada > 0 em {int((sust > 0).sum())} de {n} SKUs; "
                  f"defasagens mediana {np.median(defa):+.2f} (máx {defa.max():+.2f}); "
                  f"sustentada mediana {np.median(sust):+.2f} (máx {sust.max():+.2f})", flush=True)
        S_ = np.array([l["sustentada"] for l in linhas]); D_ = np.array([l["defasagens"] for l in linhas])
        saida[espec] = {"por_semente": linhas,
                        "skus_com_sustentada_positiva_por_semente": (S_ > 0).sum(1).tolist(),
                        "sustentada_media_por_sku": S_.mean(0).tolist(),
                        "defasagens_media_por_sku": D_.mean(0).tolist(),
                        # para o registro: o pior caso entre as sementes
                        "max_skus_com_sustentada_positiva": int((S_ > 0).sum(1).max()),
                        "sustentada_maxima": float(S_.max()),
                        "n_sementes": int(len(linhas))}
        print(f"  {espec}: sustentada média por SKU {np.round(S_.mean(0), 2)}")
        print(f"  {espec}: defasagens média por SKU {np.round(D_.mean(0), 2)}")
    rede.usar_especificacao("adotada")
    Path(__file__).with_name("s21_elasticidade_sustentada.json").write_text(json.dumps(saida, indent=1))


if __name__ == "__main__":
    main()
