"""A hierarquia de marca do PNL: identificação, pares, auditoria e restrição.

D22 suprimiu a hierarquia de **embalagem** por evidência, e manteve a de **marca**
com a justificativa de que ela "é respeitada no preço praticado": medianas de 9,08
centavos por onça nas marcas da rede contra 11,81 nas nacionais. Este módulo existe
porque aquela justificativa tem dois problemas, e o segundo é fatal:

1. Os dois números **não têm produtor** em código. Ver pendência 10.1 e o caso de
   2.1, que foi o que motivou `numeros_oficiais.py`.
2. Medianas de categoria **não respondem a pergunta de viabilidade**. A restrição
   entra célula por célula no problema de otimização, e o que importa é se o ponto
   histórico de **cada célula** a satisfaz. Uma mediana confortável conviver com
   violação em parte das células é perfeitamente possível, e é o que se mede aqui.

## A identificação da marca própria, e ela é mecânica

O manual da Kilts Center (ver `references/schema_notes.md`) diz que os dígitos
iniciais do UPC identificam o **fabricante**. Os sete UPCs do sortimento com prefixo
`3828190` são Heritage House (`HH`) e Dominick's (`DOM`), os rótulos da própria rede,
e sete é exatamente o número que D22 afirma sem produtor. A identificação deixa de
ser lista digitada e passa a ser consequência do prefixo.

## Os pares comparáveis, e onde mora o julgamento

"Marca nacional comparável" exige dizer o que é comparável. O julgamento é
**exclusivamente** o mapa `TIPO` abaixo, vinte entradas lidas da descrição do
produto. Os pares saem mecanicamente de (mesmo tipo, mesmo tamanho), e não são
digitados: digitar os pares esconderia o julgamento dentro de uma lista de doze
linhas, onde ninguém o audita.

Como todo par tem o **mesmo tamanho**, a normalização por onça cancela e a restrição
é simplesmente `p_própria ≤ p_nacional`. Em `u`, com `uᵢ = ln pᵢ − cᵢ`:

    u_own − u_nat  ≤  c_nat − c_own

linear, como a seção 3.3 do projeto promete. A linearidade em `u` é o que a
retratação registrada em `problema.py` estabeleceu: log de razão preserva
linearidade.

## Duas formas da restrição, e a diferença é quem fica viável

**Absoluta**, que é o que D22 enuncia: `p_own ≤ p_nat` em toda célula. Se o
histórico violar, o ponto histórico fica **fora da região viável**, e é exatamente o
defeito que D22 usou para suprimir a hierarquia de embalagem: o ganho apurado passa
a misturar o efeito de otimizar com o efeito de ter mudado as regras do jogo.

**Não piorar**, que D22 não considerou:

    u_own − u_nat  ≤  max( c_nat − c_own ,  u_own⁰ − u_nat⁰ )

O lado direito é o maior entre o limite absoluto e a razão histórica. Onde a
hierarquia já vale, a restrição é a absoluta; onde ela já é violada, a restrição
proíbe **aprofundar** a violação sem exigir consertá-la. O ponto histórico é viável
**por construção**, em toda célula, o que é o que preserva a interpretação do
contrafactual.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PREFIXO_MARCA_PROPRIA = "3828190"

# O ÚNICO julgamento deste módulo: o tipo de produto de cada posição do vetor, lido
# da descrição do arquivo de UPC. Os pares comparáveis saem daqui mais o tamanho.
# `blend` e `cranberry` não têm contraparte de marca própria no sortimento, e por
# isso não geram par nenhum, o que é resultado do mapa e não exceção codificada.
TIPO = {
    0: "laranja",    # ~MM ORANGE JUICE 6 OZ  (o "~" marca item descontinuado)
    1: "blend",      # 5 ALIVE FRUIT BEVERAGE
    2: "laranja",    # MM COUNTRY STYLE ORANGE
    3: "laranja",    # MM ORANGE JUICE
    4: "laranja",    # MM ORANGE JUICE 16 OZ
    5: "laranja",    # MM ORANGE JUICE W/CALCIUM
    6: "punch",      # MM FRUIT PUNCH
    7: "limonada",   # MM LEMONADE
    8: "maca",       # TREE TOP APPLE JUICE
    9: "limonada",   # HH LEMONADE
    10: "limonada",  # HH PINK LEMONADE
    11: "maca",      # DOM APPLE JUICE
    12: "laranja",   # HH ORANGE JUICE CONC 12 OZ
    13: "laranja",   # HH ORANGE JUICE CONC 16 OZ
    14: "laranja",   # HH ORANGE JUICE CONC 6 OZ
    15: "punch",     # DOM FRUIT PUNCH
    16: "cranberry",  # WELCH'S CRANBERRY JUICE
    17: "laranja",   # TROP SB ORANGE JUICE
    18: "laranja",   # TROP ORANGE JUICE 16 OZ
    19: "laranja",   # TROP SB HOME STYLE ORANGE
}

FORMAS = ("absoluta", "nao_piorar")


def oncas_de(size: str) -> float:
    """`12 OZ` para 12,0. Falha alto em formato que não reconhece.

    O campo `size` é texto livre, e o manual avisa que ele não é confiável. Falhar
    alto é de propósito: um tamanho lido errado emparelharia produtos de embalagens
    diferentes e a restrição passaria a comparar coisas incomparáveis em silêncio.
    """
    m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*OZ\s*", str(size).upper())
    if not m:
        raise ValueError(f"tamanho não reconhecido: {size!r}")
    return float(m.group(1))


def identificar(categoria: str, painel, bruto="data/raw") -> pd.DataFrame:
    """Uma linha por posição do vetor: UPC, descrição, tamanho, marca e tipo.

    É o produtor que D22 não tem. A ordem das linhas **é** a ordem do vetor da rede,
    que é a de `<cat>_upcs.json`, e trocá-la invalidaria os pesos treinados.
    """
    painel = Path(painel)
    ordem = json.loads((painel / f"{categoria}_upcs.json").read_text())["ordem"]
    arq = Path(bruto) / categoria / f"upc{categoria}.csv"
    d = pd.read_csv(arq)
    d.columns = [c.lower() for c in d.columns]
    pos = {u: i for i, u in enumerate(ordem)}
    sub = d[d["upc"].isin(ordem)].copy()
    if len(sub) != len(ordem):
        raise ValueError(f"{len(sub)} UPCs encontrados para {len(ordem)} do sortimento")
    sub["pos"] = sub["upc"].map(pos)
    sub = sub.sort_values("pos").reset_index(drop=True)
    sub["oncas"] = sub["size"].map(oncas_de)
    sub["marca_propria"] = sub["upc"].astype(str).str.startswith(PREFIXO_MARCA_PROPRIA)
    sub["descontinuado"] = sub["descrip"].astype(str).str.startswith("~")
    sub["tipo"] = sub["pos"].map(TIPO)
    if sub["tipo"].isna().any():
        faltando = sub.loc[sub["tipo"].isna(), "pos"].tolist()
        raise ValueError(f"TIPO não cobre as posições {faltando}")
    return sub[["pos", "upc", "descrip", "size", "oncas", "tipo",
                "marca_propria", "descontinuado"]]


def pares(info: pd.DataFrame) -> list:
    """Pares `(propria, nacional)` de mesmo tipo e mesmo tamanho, em ordem fixa.

    Derivados, nunca digitados: o julgamento está em `TIPO` e em nenhum outro lugar.
    """
    propria = info[info["marca_propria"]]
    nacional = info[~info["marca_propria"]]
    saida = []
    for _, o in propria.iterrows():
        iguais = nacional[(nacional["tipo"] == o["tipo"])
                          & (nacional["oncas"] == o["oncas"])]
        for _, nn in iguais.iterrows():
            saida.append((int(o["pos"]), int(nn["pos"])))
    return sorted(saida)


def auditar(precos: np.ndarray, pares_: list) -> dict:
    """O ponto histórico satisfaz a hierarquia, célula por célula?

    `precos` é (B, n). A pergunta que D22 respondeu com mediana de categoria, e que
    só a contagem por célula responde de fato.
    """
    por_par, qualquer = [], np.zeros(len(precos), dtype=bool)
    for o, n in pares_:
        v = precos[:, o] > precos[:, n]
        qualquer |= v
        por_par.append({
            "propria": o, "nacional": n,
            "pct_violacao": float(100.0 * v.mean()),
            "preco_propria_mediana": float(np.median(precos[:, o])),
            "preco_nacional_mediana": float(np.median(precos[:, n])),
        })
    return {
        "n_celulas": int(len(precos)),
        "n_pares": len(pares_),
        "pct_celulas_com_alguma_violacao": float(100.0 * qualquer.mean()),
        "n_celulas_com_alguma_violacao": int(qualquer.sum()),
        "pct_violacao_media_entre_pares": float(
            np.mean([x["pct_violacao"] for x in por_par])) if por_par else 0.0,
        "pct_violacao_maxima_entre_pares": float(
            np.max([x["pct_violacao"] for x in por_par])) if por_par else 0.0,
        "por_par": por_par,
    }


def centavos_por_onca(precos: np.ndarray, info: pd.DataFrame) -> dict:
    """Os dois números de D22, agora com produtor."""
    oz = info.sort_values("pos")["oncas"].to_numpy(float)
    propria = info.sort_values("pos")["marca_propria"].to_numpy(bool)
    cpo = 100.0 * precos / oz
    return {"centavos_por_onca_propria_mediana": float(np.median(cpo[:, propria])),
            "centavos_por_onca_nacional_mediana": float(np.median(cpo[:, ~propria])),
            "n_skus_marca_propria": int(propria.sum())}


def restricoes(pares_: list, centro: np.ndarray, n: int, forma: str = "absoluta",
               u0: np.ndarray | None = None):
    """`(A, b)` com `A·u ≤ b`, uma linha por par.

    Cada linha tem exatamente um `+1` na propria e um `−1` na nacional, de modo que
    `A·u = u_own − u_nat`. O lado direito depende da forma:

    - `absoluta`:   `b = c_nat − c_own`, igual em toda célula, forma (m,).
    - `nao_piorar`: `b = max(c_nat − c_own, u_own⁰ − u_nat⁰)`, por célula, forma
      (B, m). O ponto histórico é viável por construção, e o teste
      `test_nao_piorar_mantem_o_historico_viavel` é o que fixa isso.
    """
    if forma not in FORMAS:
        raise ValueError(f"forma desconhecida: {forma!r}; use {FORMAS}")
    m = len(pares_)
    A = np.zeros((m, n), dtype=float)
    limite = np.zeros(m, dtype=float)
    centro = np.asarray(centro, dtype=float)
    for k, (o, nn) in enumerate(pares_):
        A[k, o] = 1.0
        A[k, nn] = -1.0
        limite[k] = centro[nn] - centro[o]
    if forma == "absoluta":
        return A, limite
    if u0 is None:
        raise ValueError("a forma `nao_piorar` exige u0")
    U = np.atleast_2d(np.asarray(u0, dtype=float))
    return A, np.maximum(limite[None, :], U @ A.T)


def main() -> None:
    """A auditoria, com produtor, nos dois recortes que a decisão precisa.

    `todas` são as células completas em preço, que é o recorte de que D22 fala
    quando cita medianas de categoria. `escopo` são as 4.022 células da janela de
    teste com custo completo (D38), que é onde a restrição de fato entraria.
    """
    import argparse
    import sys as _sys

    raiz = Path(__file__).resolve().parents[2]
    for sub in ("models", "data", "experiments", "optimization"):
        _sys.path.insert(0, str(raiz / "src" / sub))
    import rede  # noqa: E402

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--bruto", default="data/raw")
    ap.add_argument("--saida", default="reports")
    a = ap.parse_args()

    painel = Path(a.painel)
    info = identificar(a.categoria, painel, a.bruto)
    pares_ = pares(info)
    n = len(info)

    cel = pd.read_parquet(painel / f"{a.categoria}_celulas.parquet")
    colunas = [f"preco_{i:02d}" for i in range(n)]
    completas = cel[colunas].notna().all(axis=1)
    P_todas = cel.loc[completas, colunas].to_numpy(float)

    dados = rede.preparar(a.categoria, painel)
    import problema as Prob  # noqa: E402
    esc = Prob.escopo(dados, painel, a.categoria, "teste")
    P_escopo = esc["precos"]

    bloco = {
        "categoria": a.categoria,
        "prefixo_marca_propria": PREFIXO_MARCA_PROPRIA,
        "n_skus": n,
        "produtos": info.to_dict(orient="records"),
        "pares": [list(p) for p in pares_],
        "n_pares": len(pares_),
        "skus_sem_par": sorted(set(range(n))
                               - {p for par in pares_ for p in par}),
        "centavos_por_onca_todas": centavos_por_onca(P_todas, info),
        "auditoria_todas": auditar(P_todas, pares_),
        "auditoria_escopo_d38": auditar(P_escopo, pares_),
    }
    destino = Path(a.saida) / f"auditoria_hierarquia_{a.categoria}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(bloco, indent=2, ensure_ascii=False))

    cpo = bloco["centavos_por_onca_todas"]
    print(f"marca própria: {cpo['n_skus_marca_propria']} de {n} SKUs, "
          f"prefixo {PREFIXO_MARCA_PROPRIA}")
    print(f"centavos por onça, mediana: própria {cpo['centavos_por_onca_propria_mediana']:.2f}"
          f"  nacional {cpo['centavos_por_onca_nacional_mediana']:.2f}"
          "   (D22 afirma 9,08 e 11,81, sem produtor)")
    descont = info.loc[info["descontinuado"], "pos"].tolist()
    print(f"descontinuados (prefixo ~ na descrição): posições {descont}")
    print(f"sem par comparável: posições {bloco['skus_sem_par']}\n")

    nomes = {int(r["pos"]): f"{r['descrip'].strip()} {r['size'].strip()}"
             for r in bloco["produtos"]}
    for rotulo, chave in (("TODAS as completas em preço", "auditoria_todas"),
                          ("ESCOPO de D38 (teste, custo completo)", "auditoria_escopo_d38")):
        r = bloco[chave]
        print(f"== {rotulo}: {r['n_celulas']} células")
        print(f"{'própria':26s} {'nacional':26s} {'viola':>7}")
        for x in r["por_par"]:
            print(f"{nomes[x['propria']]:26s} {nomes[x['nacional']]:26s} "
                  f"{x['pct_violacao']:6.2f}%")
        print(f"células com ALGUMA violação: {r['pct_celulas_com_alguma_violacao']:.2f}% "
              f"({r['n_celulas_com_alguma_violacao']} de {r['n_celulas']})\n")
    print(f"-> {destino}")


if __name__ == "__main__":
    main()


def minimo_na_caixa(A: np.ndarray, baixo: np.ndarray, alto: np.ndarray) -> np.ndarray:
    """`min_{u na caixa} A·u`, linha por linha.

    Para uma linha `a`, o mínimo é atingido pondo cada coordenada no seu limite
    INFERIOR onde `a_j > 0` e no SUPERIOR onde `a_j < 0`. Serve para decidir
    viabilidade antes de chamar o solver: se o mínimo já excede `b`, aquela célula
    não tem ponto viável nenhum na caixa, e mandar o solver nela produz um
    "resultado" que é só o ponto menos inviável que ele achou.
    """
    A = np.asarray(A, dtype=float)
    lo = np.asarray(baixo, dtype=float)
    hi = np.asarray(alto, dtype=float)
    return np.clip(A, 0, None) @ lo + np.clip(A, None, 0) @ hi


def inviaveis_na_caixa(A: np.ndarray, B: np.ndarray, baixo: np.ndarray,
                       alto: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Máscara (B,) das células sem ponto viável na caixa.

    `B` é (B, m) e `baixo`/`alto` são (B, n): os limites variam por célula porque a
    caixa de ±15% é ancorada no preço histórico de cada uma.
    """
    B = np.atleast_2d(np.asarray(B, dtype=float))
    baixo = np.atleast_2d(np.asarray(baixo, dtype=float))
    alto = np.atleast_2d(np.asarray(alto, dtype=float))
    minimos = np.stack([minimo_na_caixa(A, baixo[k], alto[k])
                        for k in range(len(B))])
    return (minimos > B + tol).any(axis=1)
