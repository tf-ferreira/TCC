"""A régua com que a rede aprende: Poisson contra MSLE, medida antes de decidir.

A seção 4 do `CLAUDE.md` permite duas perdas, MSLE ou Poisson, e a escolha não é
de conveniência numérica: as duas **ponderam SKUs de forma oposta**. No caso de
brinquedo de dois SKUs que abriu esta decisão, um vendendo 500 e outro 5, ambos
previstos com erro relativo de 10% e 100%:

    MSLE      contribuição 0,0090 (o grande) contra 0,3674 (o pequeno)  ~41x
    Poisson   contribuição 4,689  (o grande) contra 3,069  (o pequeno)  ~1,5x
    WMAPE     o grande pesa 10x   (é o erro absoluto sobre volume somado)
    receita   o grande pesa 100x  (é preço vezes volume)

Isto é, a MSLE aponta para o lado oposto das duas coisas que o trabalho de fato
entrega. O argumento fecha a decisão a favor de Poisson, e é por isso mesmo que
ele não basta: a fase do item 5 mudou de conclusão cinco vezes ao medir o que
parecia decidido, e sempre porque faltava a **magnitude**, não o mecanismo.

A pergunta que esta varredura responde é a de D25 e D27 outra vez: **a régua
muda o ajuste, a derivada, ou as duas?** Se mudar só o ajuste, a decisão fecha
sem ressalva. Se mudar a derivada, ela vira compromisso declarado, como
`promo_proprio` e `volume_lag1` foram.

## As três configurações, e por que são três

Todas rodam sobre **as mesmas linhas** e o **mesmo conjunto de atributos**, o
final do item 5. Só a régua muda.

    poisson         `objective="poisson"` sobre o alvo cru. A referência de D20.
    msle            `objective="regression"` sobre ln(1+y), destransformada por
                    `expm1`. É a MSLE por definição.
    msle_smearing   a mesma, com a correção de Duan aplicada.

**A terceira existe para que a comparação seja honesta**, e a razão é um viés
conhecido e não um detalhe de implementação. Minimizar erro quadrático em
ln(1+y) estima a média condicional **do logaritmo**, e `expm1` dela devolve algo
próximo da mediana condicional, que é sistematicamente **menor** que a média. O
WMAPE compara contra o volume realizado, cuja expectativa é a média. Sem a
correção, parte da derrota da MSLE no WMAPE seria esse deslocamento de nível e
não a ponderação entre SKUs, e a leitura ficaria ambígua.

A correção de Duan é um fator multiplicativo estimado **no treino**, o que D28
exige: `S = média(exp(r))` sobre os resíduos de treino na escala log, e
`ŷ = exp(t̂)·S − 1`. Se a MSLE perder mesmo com a correção, a conclusão é da
ponderação, que é o que a escada previu.

Uso: via `src/experiments/ruido_semente.py --familia perda`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src" / "models"))
sys.path.insert(0, str(RAIZ / "src" / "data"))

from atributos import atributos, construir, preparar_treino_teste  # noqa: E402

COMPLETO = True

_N = {"n": None}


def preparar(cel: pd.DataFrame, fora: pd.DataFrame, n: int) -> pd.DataFrame:
    """O painel final do item 5, idêntico ao de `varredura_item5.preparar`."""
    _N["n"] = n
    painel = construir(cel, fora, n)
    treino, teste, _, _, _ = preparar_treino_teste(painel)
    return pd.concat([treino, teste], ignore_index=False)


def configuracoes(_=None) -> dict[str, list[str]]:
    n = _N["n"]
    if n is None:
        raise RuntimeError("chame preparar antes de configuracoes")
    final = atributos(n)
    # O conjunto de atributos é O MESMO nas três. O que está em disputa é a
    # perda, e misturar as duas coisas na mesma tabela mediria a soma.
    return {nome: final for nome in ("poisson", "msle", "msle_smearing")}


def _inverso_simples(_modelo, _treino, _colunas, _alvo):
    return np.expm1


def _inverso_smearing(modelo, treino, colunas, alvo):
    """Fator de Duan estimado NO TREINO (D28), nunca no teste."""
    t = alvo(treino["alvo"].to_numpy(float))
    residuo = t - modelo.predict(treino[colunas])
    fator = float(np.mean(np.exp(residuo)))
    return lambda z: np.exp(z) * fator - 1.0


def especificacao(nome: str) -> dict:
    """Régua de cada configuração, lida por `ruido_semente.py`.

    `alvo` é a transformação aplicada ao y de treino; `inverso` é uma FÁBRICA e
    não uma função, porque a correção de Duan só pode ser estimada depois do
    ajuste, com o resíduo de treino na mão.
    """
    if nome == "poisson":
        return {"objective": "poisson"}
    if nome in ("msle", "msle_smearing"):
        return {
            "objective": "regression",
            "alvo": np.log1p,
            "inverso": (_inverso_smearing if nome == "msle_smearing"
                        else _inverso_simples),
        }
    raise KeyError(nome)

# ---------------------------------------------------------------------------
# A decomposição dos dois eixos, que é o resultado desta varredura
# ---------------------------------------------------------------------------

# O par (msle, msle_smearing) é a MESMA perda com a MESMA ponderação entre
# SKUs, diferindo só por um fator multiplicativo que empurra a previsão na
# direção da média. Ele é, portanto, um experimento de nível PURO, e dele sai
# a taxa de câmbio entre nível e cada eixo.
PAR_NIVEL_PURO = ("msle", "msle_smearing")
PAR_TOTAL = ("msle", "poisson")


def _dif(comparacao: dict, eixo: str, par: tuple[str, str]) -> float:
    a, b = par
    pares = comparacao["eixos"][eixo]["pares"]
    if f"{a}__menos__{b}" in pares:
        return float(pares[f"{a}__menos__{b}"]["diferenca_media"])
    return -float(pares[f"{b}__menos__{a}"]["diferenca_media"])


def decompor(comparacao: dict) -> dict:
    """Quanto de cada diferença é nível, e quanto é ponderação entre SKUs.

    O mecanismo, e ele depende de a terceira configuração existir. Mover o
    nível sem mexer na ponderação é o que `msle_smearing` faz, então o par
    (`msle`, `msle_smearing`) mede a **taxa de câmbio**: quanto de WMAPE e de
    derivada custa uma unidade de nível, com tudo o mais parado. Aplicada ao par
    (`msle`, `poisson`), essa taxa diz quanto da diferença total seria esperada
    **só** pelo deslocamento de nível; o resto é a ponderação.

    **Aproximação declarada:** a taxa é medida num ponto e extrapolada
    linearmente para um deslocamento de nível 1,6 vez maior. Ela não é
    identidade, é atribuição aproximada, e a leitura que ela sustenta é de
    proporção grosseira, não de terceira casa.

    O que torna a conta informativa é que a taxa da derivada é quase nula, o
    que a teoria prevê: um fator multiplicativo constante não muda
    ∂ln V / ∂ln p. Se ela saísse grande, a decomposição estaria medindo outra
    coisa e o número deveria ser descartado em vez de interpretado.
    """
    d_nivel_puro = _dif(comparacao, "vies_de_nivel", PAR_NIVEL_PURO)
    d_nivel_total = _dif(comparacao, "vies_de_nivel", PAR_TOTAL)
    if d_nivel_puro == 0.0:
        # Sem deslocamento de nivel no par puro nao existe taxa de cambio, e
        # devolver zero ou infinito aqui seria inventar. Falha explicita.
        raise ValueError("o par de nivel puro nao deslocou o nivel; "
                         "a taxa de cambio e indefinida")
    saida = {"par_de_nivel_puro": list(PAR_NIVEL_PURO),
             "par_total": list(PAR_TOTAL),
             "deslocamento_de_nivel_puro": d_nivel_puro,
             "deslocamento_de_nivel_total": d_nivel_total,
             "aviso": ("taxa medida num ponto e extrapolada linearmente; a "
                       "leitura e de proporcao grosseira, nao de terceira casa"),
             "eixos": {}}
    for eixo in ("wmape", "elast_h10"):
        puro = _dif(comparacao, eixo, PAR_NIVEL_PURO)
        total = _dif(comparacao, eixo, PAR_TOTAL)
        taxa = puro / d_nivel_puro
        atribuido = taxa * d_nivel_total
        saida["eixos"][eixo] = {
            "diferenca_total": total,
            "taxa_por_unidade_de_nivel": taxa,
            "atribuido_ao_nivel": atribuido,
            "residuo_de_ponderacao": total - atribuido,
            # Sem diferenca total nao existe proporcao a explicar. None e a
            # resposta honesta; 0 ou 100 seriam duas invencoes diferentes, e
            # foi um teste que pegou a versao sem esta guarda.
            "pct_explicado_pelo_nivel":
                (float(100.0 * abs(atribuido) / abs(total)) if total != 0.0
                 else None),
        }
    return saida


def main() -> None:
    ap = argparse.ArgumentParser(description=decompor.__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--saida", default="reports")
    a = ap.parse_args()
    origem = Path(a.saida) / f"comparacao_{a.categoria}_perda.json"
    saida = decompor(json.loads(origem.read_text()))
    destino = Path(a.saida) / f"decomposicao_perda_{a.categoria}.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))

    print(f"nivel puro ({' -> '.join(PAR_NIVEL_PURO)}): "
          f"{saida['deslocamento_de_nivel_puro']:+.4f}")
    print(f"nivel total ({' -> '.join(PAR_TOTAL)}): "
          f"{saida['deslocamento_de_nivel_total']:+.4f}\n")
    print(f"  {'eixo':12s} {'dif total':>10} {'taxa/nivel':>11} "
          f"{'do nivel':>10} {'resto':>10} {'% nivel':>8}")
    for eixo, d in saida["eixos"].items():
        print(f"  {eixo:12s} {d['diferenca_total']:+10.4f} "
              f"{d['taxa_por_unidade_de_nivel']:+11.3f} "
              f"{d['atribuido_ao_nivel']:+10.4f} "
              f"{d['residuo_de_ponderacao']:+10.4f} "
              + (f"{d['pct_explicado_pelo_nivel']:7.1f}%"
                 if d["pct_explicado_pelo_nivel"] is not None else "      —"))
    print(f"\ngravado em {destino}")


if __name__ == "__main__":
    main()
