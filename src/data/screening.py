"""Triagem quantitativa das categorias do painel Dominick's Finer Foods (DFF).

Implementa os critérios de seleção de categoria declarados na seção 3.1 do
projeto de pesquisa, sob as regras de operacionalização registradas em
`docs/decisoes_operacionais.md`.

Notação do projeto de pesquisa, para rastreabilidade entre código e texto:

    p_{s,i,t}   preço praticado do SKU i na loja s na semana t
    V_{s,i,t}   volume de vendas unitário

Neste módulo, "série" designa o par (i, s): um SKU numa loja específica,
observado ao longo de t. É a unidade em que a variância de preço é medida
(decisão D3), porque só a variação temporal dentro de uma mesma série
identifica elasticidade-preço.

Estratégia de memória: em vez de reter as observações, o acumulador guarda
a contagem de cada preço unitário distinto por série (um histograma). Média,
desvio, coeficiente de variação, moda e fração promocional são todos
deriváveis de contagens ponderadas, então o histograma responde tudo sem
revisitar o dado bruto e cabe em memória mesmo nas categorias grandes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Colunas necessárias. PROFIT, PRICE_HEX e PROFIT_HEX são ignorados de
# propósito: não entram em nenhuma métrica da triagem e as colunas hex são
# largas, então lê-las custaria tempo de parse sem contrapartida.
# A ORDEM das colunas varia entre categorias, por isso a seleção é por nome.
COLUNAS = ["STORE", "UPC", "WEEK", "MOVE", "PRICE", "QTY", "SALE", "OK"]

# Todas as colunas numéricas são lidas como float, inclusive as que
# conceitualmente são inteiras (STORE, UPC, WEEK, OK). Motivo: algumas
# categorias trazem valores ausentes nessas colunas, e o parser do pandas
# aborta a leitura ao encontrar NA numa coluna declarada como inteira. Lendo
# como float, as linhas com chave ausente são contadas e descartadas de forma
# explícita, virando métrica de qualidade em vez de erro de execução.
DTYPES = {
    "STORE": "float64",
    "UPC": "float64",
    "WEEK": "float64",
    "MOVE": "float64",
    "PRICE": "float64",
    "QTY": "float64",
    "SALE": "object",
    "OK": "float64",
}

COLUNAS_CHAVE = ["STORE", "UPC", "WEEK", "OK"]

CODIGOS_PROMOCAO = ("B", "C", "S")  # Bonus Buy, Coupon, Sale (redução simples)

# Fração do preço regular abaixo da qual a semana é classificada como
# promoção detectada por preço (decisão D4). Exposto como parâmetro para que
# a sensibilidade a esta escolha possa ser reportada.
LIMIAR_PROMOCAO_PRECO = 0.90


def preco_unitario(price: np.ndarray, qty: np.ndarray) -> np.ndarray:
    """Preço pago por unidade individual do produto.

    Decisão D1: em promoções de pacote (ex.: 3 latas por $2), `price` guarda
    o preço do pacote inteiro e `qty` o tamanho do pacote. O preço relevante
    para o consumidor, e portanto para elasticidade, é price/qty. Usar
    `price` cru faz uma promoção de pacote aparecer como aumento de preço,
    invertendo o sinal do fenômeno.

    Valores de `qty` ausentes ou não positivos são tratados como 1, o caso
    de "sem empacotamento". Isso é defensivo: o esperado é que a esmagadora
    maioria das linhas tenha qty == 1.
    """
    qty_seguro = np.where(np.isfinite(qty) & (qty > 0), qty, 1.0)
    return price / qty_seguro


class AcumuladorCategoria:
    """Acumula, em passadas sucessivas sobre blocos, tudo que a triagem exige.

    Nenhuma observação individual é retida. O que sobrevive entre blocos são
    contadores e um histograma de preços por série.
    """

    def __init__(self) -> None:
        self.n_linhas = 0
        self.n_chave_ausente = 0  # NA em STORE, UPC, WEEK ou OK
        self.n_ok_zero = 0
        self.n_sem_atividade = 0  # price == 0, ver decisão D2
        self.n_validas = 0  # ok == 1 e price > 0
        self.contagem_sale = {codigo: 0 for codigo in CODIGOS_PROMOCAO}
        self.n_sale_preenchido = 0
        self.lojas: set[int] = set()
        self.semanas: set[int] = set()
        self.upcs: set[int] = set()
        # Series com MultiIndex (upc, store, preco_centavos) -> contagem
        self._histograma: pd.Series | None = None
        # Series upc -> nº de pares (loja, semana) com atividade
        self._cobertura: pd.Series | None = None
        # Series upc -> primeira e última semana com atividade
        self._semana_min_upc: pd.Series | None = None
        self._semana_max_upc: pd.Series | None = None
        # Series store -> primeira e última semana com atividade na categoria.
        # Necessário porque as lojas entram e saem do painel (decisão D10):
        # sem isso, todo SKU vendido numa loja que abriu tarde é penalizado
        # pelas semanas anteriores à abertura dela.
        self._semana_min_loja: pd.Series | None = None
        self._semana_max_loja: pd.Series | None = None
        # Series (upc, store) -> semanas válidas com código de promoção.
        # Acumulado por série, e não globalmente, para que a promoção
        # declarada seja comparável à detectada por preço: mesmo
        # denominador (semanas válidas da série) e mesma agregação
        # (mediana entre séries). Ver decisão D4.
        self._sale_por_serie: pd.Series | None = None

    def atualizar(self, bloco: pd.DataFrame) -> None:
        self.n_linhas += len(bloco)

        faltantes = bloco[COLUNAS_CHAVE].isna().any(axis=1)
        self.n_chave_ausente += int(faltantes.sum())
        bloco = bloco.loc[~faltantes]
        if bloco.empty:
            return
        bloco = bloco.astype({"STORE": "int32", "UPC": "int64",
                              "WEEK": "int32", "OK": "int8"})

        self.n_ok_zero += int((bloco["OK"] == 0).sum())
        self.lojas.update(bloco["STORE"].unique().tolist())
        self.semanas.update(bloco["WEEK"].unique().tolist())
        self.upcs.update(bloco["UPC"].unique().tolist())

        sale = bloco["SALE"]
        preenchido = sale.notna() & (sale.astype(str).str.strip() != "")
        self.n_sale_preenchido += int(preenchido.sum())
        for codigo in CODIGOS_PROMOCAO:
            self.contagem_sale[codigo] += int(
                (sale.astype(str).str.strip().str.upper() == codigo).sum()
            )

        self.n_sem_atividade += int((bloco["PRICE"] <= 0).sum())

        # Decisão D5: só linhas marcadas como válidas entram nas estatísticas.
        # Decisão D2: presença exige price > 0, não mera existência de linha.
        validas = bloco[(bloco["OK"] == 1) & (bloco["PRICE"] > 0)]
        if validas.empty:
            return
        self.n_validas += len(validas)

        preco = preco_unitario(
            validas["PRICE"].to_numpy(), validas["QTY"].to_numpy()
        )
        # Arredondar em centavos: preço de varejo é discreto, e o
        # arredondamento é o que faz o histograma colapsar de fato.
        centavos = np.rint(preco * 100).astype("int64")

        chave = pd.DataFrame(
            {
                "upc": validas["UPC"].to_numpy(),
                "store": validas["STORE"].to_numpy(),
                "centavos": centavos,
            }
        )
        parcial = chave.groupby(["upc", "store", "centavos"], sort=False).size()
        self._histograma = _somar_series(self._histograma, parcial)

        sale_validas = validas["SALE"]
        marcadas = sale_validas.notna() & (sale_validas.astype(str).str.strip() != "")
        if marcadas.any():
            self._sale_por_serie = _somar_series(
                self._sale_por_serie,
                chave.loc[marcadas.to_numpy()]
                .groupby(["upc", "store"], sort=False)
                .size(),
            )

        por_upc = validas.groupby("UPC")["WEEK"]
        self._cobertura = _somar_series(self._cobertura, por_upc.size())
        self._semana_min_upc = _reduzir_series(
            self._semana_min_upc, por_upc.min(), "min"
        )
        self._semana_max_upc = _reduzir_series(
            self._semana_max_upc, por_upc.max(), "max"
        )

        por_loja = validas.groupby("STORE")["WEEK"]
        self._semana_min_loja = _reduzir_series(
            self._semana_min_loja, por_loja.min(), "min"
        )
        self._semana_max_loja = _reduzir_series(
            self._semana_max_loja, por_loja.max(), "max"
        )

    # ---------------------------------------------------------------- saída

    def estatisticas_por_serie(self) -> pd.DataFrame:
        """Deriva média, CV, preço regular e fração promocional por série.

        Tudo sai do histograma por contagem ponderada, sem revisitar o dado:
            média  = Σ(preço · contagem) / Σcontagem
            E[X²]  = Σ(preço² · contagem) / Σcontagem
            var    = E[X²] − média²
            regular = preço de maior contagem (moda)
        """
        if self._histograma is None or self._histograma.empty:
            return pd.DataFrame(
                columns=["upc", "store", "n_semanas", "media", "cv",
                         "preco_regular", "frac_promo_preco"]
            )

        hist = self._histograma.rename("contagem").reset_index()
        hist["preco"] = hist["centavos"] / 100.0
        hist["peso_preco"] = hist["preco"] * hist["contagem"]
        hist["peso_preco2"] = hist["preco"] ** 2 * hist["contagem"]

        grupos = hist.groupby(["upc", "store"], sort=False)
        agregado = grupos.agg(
            n_semanas=("contagem", "sum"),
            soma=("peso_preco", "sum"),
            soma2=("peso_preco2", "sum"),
        )
        agregado["media"] = agregado["soma"] / agregado["n_semanas"]
        variancia = agregado["soma2"] / agregado["n_semanas"] - agregado["media"] ** 2
        # Erro de ponto flutuante pode gerar variância levemente negativa
        # numa série de preço constante.
        variancia = variancia.clip(lower=0.0)
        agregado["cv"] = np.sqrt(variancia) / agregado["media"]

        # Preço regular = moda. Em empate, o maior preço, porque o valor de
        # tabela é o teto e a promoção é a exceção.
        ordenado = hist.sort_values(["contagem", "preco"], ascending=[True, True])
        moda = (
            ordenado.groupby(["upc", "store"], sort=False)
            .tail(1)
            .set_index(["upc", "store"])["preco"]
            .rename("preco_regular")
        )
        agregado = agregado.join(moda)

        hist_indexado = hist.set_index(["upc", "store"])
        limite = agregado["preco_regular"] * LIMIAR_PROMOCAO_PRECO
        abaixo = hist_indexado.join(limite.rename("limite"))
        promo = (
            abaixo[abaixo["preco"] < abaixo["limite"]]
            .groupby(level=["upc", "store"])["contagem"]
            .sum()
            .rename("n_promo")
        )
        agregado = agregado.join(promo)
        agregado["n_promo"] = agregado["n_promo"].fillna(0)
        agregado["frac_promo_preco"] = agregado["n_promo"] / agregado["n_semanas"]

        # Promoção declarada, com o mesmo denominador da detectada por preço.
        if self._sale_por_serie is not None:
            agregado = agregado.join(self._sale_por_serie.rename("n_sale"))
        else:
            agregado["n_sale"] = 0
        agregado["n_sale"] = agregado["n_sale"].fillna(0)
        agregado["frac_sale_declarado"] = agregado["n_sale"] / agregado["n_semanas"]

        # Divergência entre as duas vias. Positiva significa que o preço caiu
        # em semanas sem código de promoção, ou seja, o campo `sale`
        # subdeclara. Negativa significa o oposto: código marcado sem queda de
        # preço correspondente (típico de cupom, que não altera o preço de
        # gôndola).
        agregado["divergencia_promo"] = (
            agregado["frac_promo_preco"] - agregado["frac_sale_declarado"]
        )

        return agregado.reset_index()[
            ["upc", "store", "n_semanas", "media", "cv", "preco_regular",
             "frac_promo_preco", "frac_sale_declarado", "divergencia_promo"]
        ]

    def cobertura_por_upc(self) -> pd.DataFrame:
        """Decompoe a presenca de cada SKU em quatro fatores independentes.

        Decisoes D8 e D10. A cobertura bruta (celulas ativas sobre o retangulo
        inteiro da categoria) mistura fenomenos distintos: trata como
        "instavel" tanto um produto que ainda nao existia quanto um produto
        vendido numa loja que ainda nao havia aberto. A decomposicao separa
        cada causa:

            cobertura_total = amplitude * longevidade * disponibilidade * regularidade

            amplitude       = lojas que alguma vez carregaram o SKU / lojas
                              totais ("distribuicao ampla ou de nicho?")
            longevidade     = semanas entre a primeira e a ultima atividade do
                              SKU / semanas da categoria ("por quanto tempo o
                              produto existiu?")
            disponibilidade = semanas em que as lojas que carregam o SKU de
                              fato operavam, dentro da vida dele, sobre o
                              retangulo (lojas x vida) ("houve oportunidade
                              de venda?")
            regularidade    = celulas ativas / semanas efetivamente
                              disponiveis ("dentro da janela em que era
                              possivel estar la, esteve?")

        A identidade e exata por construcao e e verificada em teste.

        Sem o fator `disponibilidade`, abertura e fechamento de loja
        contaminam a regularidade: em sucos congelados, 14 das 93 lojas
        comecam depois da semana 10 (uma so a partir da 373) e 10 encerram
        antes da 390, o que impede qualquer SKU de se aproximar de
        regularidade 1 por um motivo que nada tem a ver com o produto.

        Para o criterio de selecao de categoria interessa sobretudo
        `regularidade`, porque elasticidade cruzada exige que os SKUs
        coexistam de forma consistente. Os outros tres fatores dimensionam o
        recorte depois e diagnosticam por que a cobertura bruta e baixa.

        Decisao D6: devolve a distribuicao completa, sem nenhum limiar.
        """
        colunas = ["upc", "n_ativo", "n_lojas_ativas", "primeira_semana",
                   "ultima_semana", "n_semanas_vida", "n_semanas_disponiveis",
                   "amplitude", "longevidade", "disponibilidade",
                   "regularidade", "cobertura_total"]
        celulas = len(self.lojas) * len(self.semanas)
        if self._cobertura is None or celulas == 0 or self._histograma is None:
            return pd.DataFrame(columns=colunas)

        semanas_ord = np.array(sorted(self.semanas))

        def janela(inicio: np.ndarray, fim: np.ndarray) -> np.ndarray:
            """Quantas semanas observadas na categoria caem em [inicio, fim].

            Conta sobre as semanas que a categoria de fato observa, nao sobre
            a diferenca de numeracao: o painel pode ter buracos de calendario,
            e isso nao e responsabilidade do SKU nem da loja.
            """
            i0 = np.searchsorted(semanas_ord, inicio, "left")
            i1 = np.searchsorted(semanas_ord, fim, "right")
            return np.maximum(0, i1 - i0)

        df = self._cobertura.rename("n_ativo").rename_axis("upc").reset_index()
        df = df.merge(
            self._semana_min_upc.rename("primeira_semana").rename_axis("upc").reset_index(),
            on="upc", how="left",
        ).merge(
            self._semana_max_upc.rename("ultima_semana").rename_axis("upc").reset_index(),
            on="upc", how="left",
        )
        df["n_semanas_vida"] = janela(
            df["primeira_semana"].to_numpy(), df["ultima_semana"].to_numpy()
        )

        # Pares (upc, store) ativos saem do proprio indice do histograma: cada
        # par presente ali e, por construcao, uma serie com atividade.
        pares = (
            self._histograma.index.to_frame(index=False)[["upc", "store"]]
            .drop_duplicates()
            .merge(df[["upc", "primeira_semana", "ultima_semana"]], on="upc")
            .merge(
                self._semana_min_loja.rename("loja_inicio").rename_axis("store").reset_index(),
                on="store", how="left",
            )
            .merge(
                self._semana_max_loja.rename("loja_fim").rename_axis("store").reset_index(),
                on="store", how="left",
            )
        )
        # Oportunidade real de venda: intersecao entre a vida do produto e a
        # janela de operacao daquela loja.
        pares["disponivel"] = janela(
            np.maximum(pares["primeira_semana"].to_numpy(), pares["loja_inicio"].to_numpy()),
            np.minimum(pares["ultima_semana"].to_numpy(), pares["loja_fim"].to_numpy()),
        )
        por_upc = pares.groupby("upc").agg(
            n_lojas_ativas=("store", "size"),
            n_semanas_disponiveis=("disponivel", "sum"),
        )
        df = df.merge(por_upc.reset_index(), on="upc", how="left")

        retangulo = df["n_lojas_ativas"] * df["n_semanas_vida"]
        df["amplitude"] = df["n_lojas_ativas"] / len(self.lojas)
        df["longevidade"] = df["n_semanas_vida"] / len(self.semanas)
        df["disponibilidade"] = np.where(
            retangulo > 0, df["n_semanas_disponiveis"] / retangulo, np.nan
        )
        df["regularidade"] = np.where(
            df["n_semanas_disponiveis"] > 0,
            df["n_ativo"] / df["n_semanas_disponiveis"],
            np.nan,
        )
        df["cobertura_total"] = df["n_ativo"] / celulas

        return df[colunas].sort_values("regularidade", ascending=False)

    def resumo(self, categoria: str) -> dict:
        series = self.estatisticas_por_serie()
        cobertura = self.cobertura_por_upc()
        quantis = [0.10, 0.25, 0.50, 0.75, 0.90]

        resumo = {
            "categoria": categoria,
            # Bloco 0: dimensões e qualidade
            "n_linhas": self.n_linhas,
            "n_upcs": len(self.upcs),
            "n_lojas": len(self.lojas),
            "n_semanas": len(self.semanas),
            "semana_min": min(self.semanas) if self.semanas else None,
            "semana_max": max(self.semanas) if self.semanas else None,
            "pct_ok_zero": _pct(self.n_ok_zero, self.n_linhas),
            "pct_chave_ausente": _pct(self.n_chave_ausente, self.n_linhas),
            "pct_sem_atividade": _pct(self.n_sem_atividade, self.n_linhas),
            "n_linhas_validas": self.n_validas,
            "n_series": len(series),
            # Composição dos códigos de promoção, sobre o arquivo inteiro.
            # Serve só para saber quais códigos a categoria usa; a taxa
            # comparável é a por série, mais abaixo.
            "pct_linhas_com_sale": _pct(self.n_sale_preenchido, self.n_linhas),
            **{
                f"pct_linhas_sale_{codigo}": _pct(
                    self.contagem_sale[codigo], self.n_linhas
                )
                for codigo in CODIGOS_PROMOCAO
            },
        }

        # Bloco 1 e 2: distribuições contínuas, sem limiar (decisão D6).
        # As duas medidas de promoção compartilham denominador e agregação.
        for coluna, prefixo in (
            ("cv", "cv_preco"),
            ("frac_promo_preco", "promo_preco"),
            ("frac_sale_declarado", "promo_sale"),
            ("divergencia_promo", "divergencia"),
        ):
            if len(series):
                valores = series[coluna].dropna()
                for q in quantis:
                    resumo[f"{prefixo}_p{int(q * 100)}"] = float(valores.quantile(q))
                resumo[f"{prefixo}_media"] = float(valores.mean())
            else:
                for q in quantis:
                    resumo[f"{prefixo}_p{int(q * 100)}"] = None
                resumo[f"{prefixo}_media"] = None

        # Bloco 3: estabilidade do sortimento decomposta, também sem limiar
        for coluna in ("amplitude", "longevidade", "disponibilidade",
                       "regularidade", "cobertura_total"):
            valores = cobertura[coluna].dropna() if len(cobertura) else []
            for q in quantis:
                chave = f"{coluna}_p{int(q * 100)}"
                resumo[chave] = float(valores.quantile(q)) if len(valores) else None

        return resumo


def _somar_series(acumulada: pd.Series | None, parcial: pd.Series) -> pd.Series:
    """Soma duas Series alinhando pelo índice, tratando o primeiro bloco."""
    if acumulada is None:
        return parcial
    return acumulada.add(parcial, fill_value=0).astype("int64")


def _reduzir_series(acumulada: pd.Series | None, parcial: pd.Series,
                    operacao: str) -> pd.Series:
    """Combina duas Series por mínimo ou máximo, alinhando pelo índice.

    Usado para primeira e última semana de atividade por SKU. Diferente da
    soma, min e max precisam ignorar o lado ausente em vez de tratá-lo como
    zero, senão um SKU que só aparece no segundo bloco teria primeira semana
    igual a zero.
    """
    if acumulada is None:
        return parcial
    juntas = pd.concat([acumulada, parcial], axis=1)
    return juntas.min(axis=1) if operacao == "min" else juntas.max(axis=1)


def _pct(parte: int, total: int) -> float | None:
    return None if total == 0 else 100.0 * parte / total


def processar_categoria(caminho_zip: Path, tamanho_bloco: int = 2_000_000):
    """Percorre o arquivo de movimento de uma categoria em blocos."""
    acumulador = AcumuladorCategoria()
    leitor = pd.read_csv(
        caminho_zip,
        compression="zip",
        usecols=COLUNAS,
        dtype=DTYPES,
        chunksize=tamanho_bloco,
    )
    for bloco in leitor:
        acumulador.atualizar(bloco)
    return acumulador


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("categoria", help="sigla de três letras, ex.: sdr")
    parser.add_argument("--raiz", default="data/raw")
    parser.add_argument("--saida", default="data/interim/screening")
    parser.add_argument("--bloco", type=int, default=2_000_000)
    args = parser.parse_args()

    raiz = Path(args.raiz) / args.categoria
    candidatos = sorted(raiz.glob("w*.zip"))
    if not candidatos:
        raise SystemExit(f"arquivo de movimento não encontrado em {raiz}")

    acumulador = processar_categoria(candidatos[0], args.bloco)

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    acumulador.estatisticas_por_serie().to_csv(
        saida / f"{args.categoria}_series.csv", index=False
    )
    acumulador.cobertura_por_upc().to_csv(
        saida / f"{args.categoria}_cobertura.csv", index=False
    )
    with open(saida / f"{args.categoria}_resumo.json", "w") as arquivo:
        json.dump(acumulador.resumo(args.categoria), arquivo, indent=2)

    print(f"{args.categoria}: {acumulador.n_linhas} linhas, "
          f"{len(acumulador.upcs)} upcs, {len(acumulador.lojas)} lojas")


if __name__ == "__main__":
    main()
