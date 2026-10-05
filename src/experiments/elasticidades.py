"""Todas as elasticidades do projeto, num script só, com as amostras explícitas.

Absorve `diagnostico_endogeneidade.py`, `piloto_bem_externo.py` e
`iv_bem_externo.py`. O motivo de juntar não é economia de arquivos, é impedir
uma classe de erro: com três scripts separados, três estimativas circulavam
nos documentos como se fossem comparáveis, e ninguém tinha em mãos as
amostras lado a lado para ver que **não são o mesmo objeto**.

## Os três estimandos, e por que são diferentes

**A. Própria de SKU.** `d ln V_i / d ln p_i` com os concorrentes parados. É a
diagonal da matriz de elasticidades, o objeto que D16 usa para validar a
derivada da rede.

**B e C. Agregada da cesta.** `d ln (soma V_i) / d delta` quando **todos** os N
preços sobem `delta` por cento juntos. É o objeto que D21 e a figura 6 usam,
porque a pergunta ali é sobre mover a cesta inteira.

Elas se relacionam por uma identidade de agregação. Com `s_i` a participação
de volume de cada SKU:

    eps_agregada = soma_i s_i eps_ii + soma_i s_i soma_{j!=i} eps_ij

O segundo termo é positivo quando os produtos são substitutos, o que implica

    |eps_agregada| <= |media ponderada das proprias|

Essa desigualdade é **restrição testável**, não expectativa. Nas estimativas
atuais os pontos a violam (2,507 contra 1,757) e o intervalo de confiança da
agregada não, porque ela é imprecisa demais para refutar coisa alguma. Duas
ressalvas impedem afirmar violação: a de série pondera por variância intra do
preço e a identidade pede ponderação por volume; e o agregador do lado direito
é média simples de logs, não índice ponderado por participação.

## Os erros padrão são agrupados, e a versão iid fica ao lado

Correção de 14/09/2026. Até aqui todo erro padrão do projeto supunha linhas
independentes. Em painel de varejo isso é falso por construção, e o efeito foi
medido: no nível de célula o erro padrão da elasticidade agregada é 2,66 vezes
maior quando agrupado por loja. **A conclusão de D21 muda com a correção**, e
por isso as três versões saem lado a lado, como já saíam MQO e IV. A elasticidade
própria de série sobrevive bem, com inflação de 3,25 vezes e intervalo que ainda
exclui −1 com folga.

## As amostras, que são a metade do resultado

    serie                 loja x SKU x semana, move > 0, custo válido
    celula_completas      células com os N preços observados
    celula_com_custo      idem, exigindo custo válido nos N (subamostra)
    celula_todas          qualquer célula com atividade e volume positivo

A última existe como **diagnóstico**, não como estimativa a citar: nela a
composição do agregado varia entre células, e o IV troca de sinal. Um
instrumento que muda o sinal do estimando conforme a amostra não está
identificando, está capturando composição, e isso é evidência a favor de D12.

Uso:
    python3 src/experiments/elasticidades.py frj
"""
from __future__ import annotations

import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from sortimento import carregar_sortimento, extrair_longo            # noqa: E402

Z95 = 1.959963984540054
warnings.filterwarnings("ignore", message="Mean of empty slice")


# ---------------------------------------------------------------- álgebra --
def centralizar(M: np.ndarray, chaves: list[np.ndarray], it: int = 200,
                tol: float = 1e-10, diagnostico: dict | None = None) -> np.ndarray:
    """Absorve efeitos fixos por centralização alternada (Frisch-Waugh-Lovell).

    ## O mecanismo, e por que precisa iterar

    Com **uma** dimensão de efeito fixo, subtrair a média do grupo resolve em
    um passo: o resíduo já é ortogonal ao espaço dos efeitos daquela dimensão.
    Com **duas**, os dois espaços não são ortogonais entre si, e centralizar
    pela segunda estraga parte do que a primeira tinha feito. Alternar as duas
    projeções converge para a interseção, mas geometricamente, não de uma vez:
    a velocidade é governada pelo ângulo entre os dois espaços, que é pequeno
    quando o painel é mal conectado (lojas que compartilham poucas semanas com
    as outras, séries curtas).

    ## A correção de 14/09/2026 (F4)

    A versão anterior fazia **25 passadas fixas e não verificava nada**.
    Convergência incompleta atenua o coeficiente na direção do MQO, em
    silêncio, e o único teste que existia usava painel balanceado de 12 por 9
    sem ruído, onde uma passada já resolve. Ele provava que a fórmula está
    certa, não que 25 passadas bastam nos 631.365 registros reais.

    Agora o laço para por **critério**, não por contagem, e o número de
    passadas usadas vai para `diagnostico`, de modo que fica auditável no JSON
    de saída em vez de ser suposto.

    `tol` é medido na escala do próprio dado: a maior mudança absoluta de uma
    passada, dividida por `1 + max|M|`.
    """
    M = M.astype(float).copy()
    usadas, ultimo = 0, float("inf")
    for passada in range(it):
        anterior = M
        for k in chaves:
            df = pd.DataFrame(M)
            M = (df - df.groupby(k).transform("mean")).to_numpy()
        usadas = passada + 1
        ultimo = float(np.max(np.abs(M - anterior)) / (1.0 + np.max(np.abs(M))))
        if ultimo < tol:
            break
    if diagnostico is not None:
        diagnostico.update({"passadas": usadas, "variacao_final": ultimo,
                            "tolerancia": tol, "convergiu": ultimo < tol})
    return M


def ols(y, X, k):
    X = np.asarray(X, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    s2 = float(r @ r) / (len(y) - k)
    return b, np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X))), r


def projetar(Z, X):
    return Z @ np.linalg.solve(Z.T @ Z, Z.T @ X)


def tsls(y, X, Z, k):
    Xh = projetar(Z, X)
    A = Xh.T @ Xh
    b = np.linalg.solve(A, Xh.T @ y)
    r = y - X @ b
    s2 = float(r @ r) / (len(y) - k)
    return b, np.sqrt(np.diag(s2 * np.linalg.inv(A))), r


def cragg_donald(X, Z, k_fe):
    """Autovalor mínimo da matriz de concentração.

    Com mais de um endógeno o F de cada primeiro estágio não basta: os
    instrumentos podem identificar bem uma combinação linear e mal outra.
    Cragg-Donald mede a pior direção.
    """
    n, kz = Z.shape
    Xh = projetar(Z, X)
    Vhat = X - Xh
    Sig = (Vhat.T @ Vhat) / (n - k_fe - kz)
    L = np.linalg.cholesky(np.linalg.inv(Sig))
    G = L.T @ (Xh.T @ Xh) @ L / kz
    return float(np.min(np.linalg.eigvalsh(G)))


def sanduiche(R: np.ndarray, u: np.ndarray, grupos: dict, k: int) -> dict:
    """Erro padrão agrupado, para MQO e para 2SLS.

    ## O que estava errado antes

    `ols` e `tsls` devolvem `s² (R'R)⁻¹`, que supõe as n linhas independentes.
    Num painel de varejo isso é falso por construção: a demanda de uma loja é
    serialmente correlacionada ao longo de quase quatrocentas semanas, e um
    choque de semana atinge as 93 lojas ao mesmo tempo. O erro padrão iid
    conta 23.374 observações independentes onde existem 93 lojas, e promete
    uma precisão que o dado não tem.

    ## O mecanismo, em uma frase

    A variância de um estimador linear é `(R'R)⁻¹ Var(R'u) (R'R)⁻¹`, e a
    única coisa que muda entre as versões é o miolo. O iid supõe
    `Var(R'u) = σ² R'R`, soma de n termos independentes. O agrupado **soma o
    escore dentro de cada grupo antes de elevar ao quadrado**,
    `Σ_g (R_g'u_g)(R_g'u_g)'`, e com isso conta G termos independentes em vez
    de n. A razão entre os dois erros padrão é o fator de inflação, e ele mede
    exatamente quanta observação repetida a suposição de independência estava
    tratando como informação nova.

    `R` é `X` no MQO e `X̂` no 2SLS, porque em ambos `b = (R'R)⁻¹R'y`. O
    resíduo `u` usa sempre o `X` original, nunca o projetado: `X̂` entra no pão,
    `X` entra no resíduo.

    ## Limitação declarada

    A correção é assintótica em **G**, não em n: ela conserta o miolo, não a
    distribuição de referência. Com G = 93 lojas a aproximação normal ainda é
    aproximação, e por isso `bootstrap_wild` reporta o valor crítico construído
    no próprio dado em vez de importar 1,96.
    """
    n = len(u)
    pao = np.linalg.inv(R.T @ R)
    saida = {}
    for nome, chave in grupos.items():
        escore = (pd.DataFrame(R * u[:, None])
                  .groupby(np.asarray(chave)).sum().to_numpy())
        G = escore.shape[0]
        # Correção de amostra finita usual (Cameron, Gelbach e Miller, 2011).
        c = (G / (G - 1)) * ((n - 1) / (n - k))
        V = pao @ (c * (escore.T @ escore)) @ pao
        saida[nome] = {"G": int(G), "ep": np.sqrt(np.diag(V))}
    return saida


def bootstrap_wild(y, X, Z, chave, k, B: int = 999, semente: int = 0) -> dict:
    """Bootstrap wild agrupado do t, para o regime de G pequeno.

    ## Por que existe

    O sanduíche agrupado é consistente quando o número de GRUPOS cresce, não
    quando o número de linhas cresce. Com G = 93 o t agrupado não é
    necessariamente normal, e o corte de 1,96 é importado de uma distribuição
    que pode não ser a certa. Este procedimento constrói a distribuição do t no
    próprio dado em vez de importá-la.

    ## O que a medição de 14/09/2026 devolveu, e como lê-la

    Corte de 1,70 contra os 1,96 da normal, na especificação agregada com bem
    externo. **Isto não autoriza dizer que a normal é conservadora aqui.** A
    versão implementada é a não restrita, e a literatura mostra que ela erra
    justamente nessa direção, produzindo corte pequeno demais. A leitura
    defensável é a fraca: nenhuma das duas réguas salva a conclusão de D21, já
    que mesmo com o corte de 1,70 os z agrupados de 1,04 e 1,13 não rejeitam.
    Trocar WCU por WCR é a pendência registrada.

    ## O mecanismo

    Mantém o desenho fixo, `X`, `Z` e a estrutura de grupos, e reamostra apenas
    o **sinal** do resíduo, um sinal por grupo, sorteado em {-1, +1} com peso
    igual (pesos de Rademacher). O sinal ser constante dentro do grupo é o
    ponto inteiro: ele preserva a correlação intragrupo que um bootstrap por
    linha destruiria, e é por isso que o resultado difere do bootstrap ingênuo.

    Devolve o quantil 95% de |t*|, que é o valor crítico a comparar com o t
    observado de cada hipótese, no lugar de 1,96.

    ## Limitações declaradas

    Versão **não restrita** (WCU): o processo gerador das réplicas usa a
    estimativa livre, e não a estimativa sob a hipótese nula. Davidson e
    MacKinnon (2010) mostram que a versão restrita (WCR) tem erro de tamanho
    menor. E os efeitos fixos são absorvidos uma vez, antes do laço, e tratados
    como fixos nas réplicas.
    """
    chave = np.asarray(chave)
    ordem = np.argsort(chave, kind="stable")
    y, X, Z, chave = y[ordem], X[ordem], Z[ordem], chave[ordem]
    inicios = np.flatnonzero(np.r_[True, chave[1:] != chave[:-1]])
    G, n, m = len(inicios), len(y), X.shape[1]
    tamanhos = np.diff(np.r_[inicios, n])

    Xh = projetar(Z, X)
    pao = np.linalg.inv(Xh.T @ Xh)
    b = pao @ (Xh.T @ y)
    u = y - X @ b
    c = (G / (G - 1)) * ((n - 1) / (n - k))

    rng = np.random.default_rng(semente)
    ts = np.empty((B, m))
    for r in range(B):
        v = np.repeat(rng.choice([-1.0, 1.0], size=G), tamanhos)
        ys = X @ b + u * v
        bs = pao @ (Xh.T @ ys)
        us = ys - X @ bs
        escore = np.add.reduceat(Xh * us[:, None], inicios, axis=0)
        V = pao @ (c * (escore.T @ escore)) @ pao
        ts[r] = (bs - b) / np.sqrt(np.diag(V))
    return {"B": B, "G_bootstrap": int(G),
            "critico_95_bicaudal": [float(q) for q in
                                    np.quantile(np.abs(ts), 0.95, axis=0)],
            "critico_90_bicaudal": [float(q) for q in
                                    np.quantile(np.abs(ts), 0.90, axis=0)]}


def bloco(nome: str, amostra: str, y, X, Z, nfe: int, rotulos: list[str],
          grupos: dict | None = None) -> dict:
    """MQO, IV, primeiros estágios, Hausman-Wu, IC e erros padrão agrupados."""
    k = nfe + X.shape[1]
    b_ols, se_ols, r_ols = ols(y, X, k)
    b_iv, se_iv, r_iv = tsls(y, X, Z, k)

    primeiro = {}
    for i, rot in enumerate(rotulos):
        bb, _, rr = ols(X[:, i], Z, k)
        sst = float(((X[:, i] - X[:, i].mean()) ** 2).sum())
        r2 = 1.0 - float(rr @ rr) / sst
        kz = Z.shape[1]
        primeiro[rot] = {
            "r2_parcial": r2,
            "F_instrumentos": float((r2 / kz) / ((1 - r2) / (len(y) - nfe - kz))),
        }

    Vhat = X - projetar(Z, X)
    b_aug, se_aug, _ = ols(y, np.column_stack([X, Vhat]), k + X.shape[1])
    m = X.shape[1]

    saida = {
        "especificacao": nome,
        "amostra": amostra,
        "n": int(len(y)),
        "regressores": rotulos,
        "ols": {r: {"coef": float(b_ols[i]), "ep": float(se_ols[i])}
                for i, r in enumerate(rotulos)},
        "iv": {r: {"coef": float(b_iv[i]), "ep": float(se_iv[i]),
                   "ic95_inf": float(b_iv[i] - Z95 * se_iv[i]),
                   "ic95_sup": float(b_iv[i] + Z95 * se_iv[i])}
               for i, r in enumerate(rotulos)},
        "primeiro_estagio": primeiro,
        "hausman_wu_t": {r: float(b_aug[m + i] / se_aug[m + i])
                         for i, r in enumerate(rotulos)},
        "corr_endogeno_instrumento": {
            r: float(np.corrcoef(X[:, i], Z[:, i])[0, 1]) for i, r in enumerate(rotulos)},
    }
    if X.shape[1] > 1:
        saida["cragg_donald_F"] = cragg_donald(X, Z, nfe)

    # Erros padrão agrupados. O iid acima fica no arquivo de propósito: é o
    # contraste entre os dois que é o resultado, do mesmo modo que MQO contra
    # IV. Citar o iid sozinho como precisão é que deixou de ser permitido.
    if grupos:
        ag_ols = sanduiche(X, r_ols, grupos, k)
        ag_iv = sanduiche(projetar(Z, X), r_iv, grupos, k)
        saida["ep_agrupado"] = {
            g: {
                "G": ag_iv[g]["G"],
                "ols": {r: {"ep": float(ag_ols[g]["ep"][i]),
                            "inflacao": float(ag_ols[g]["ep"][i] / se_ols[i])}
                        for i, r in enumerate(rotulos)},
                "iv": {r: {"ep": float(ag_iv[g]["ep"][i]),
                           "inflacao": float(ag_iv[g]["ep"][i] / se_iv[i]),
                           "ic95_inf": float(b_iv[i] - Z95 * ag_iv[g]["ep"][i]),
                           "ic95_sup": float(b_iv[i] + Z95 * ag_iv[g]["ep"][i])}
                       for i, r in enumerate(rotulos)},
            }
            for g in grupos
        }
    return saida


# ------------------------------------------------------------- estimandos --
def estimar_serie(longo: pd.DataFrame) -> dict:
    """A. Elasticidade própria de SKU, efeitos fixos de série e de semana."""
    d = longo[(longo.volume > 0) & longo.custo.notna() & (longo.custo > 0)].copy()
    d["serie"] = d.store.astype(np.int64) * 10**12 + d.upc
    diag_fe: dict = {}
    M = centralizar(np.column_stack([
        np.log(d.volume.to_numpy()),
        np.log(d.preco.to_numpy()),
        np.log(d.custo.to_numpy()),
    ]), [d.serie.to_numpy(), d.week.to_numpy()], diagnostico=diag_fe)
    nfe = d.serie.nunique() + d.week.nunique()
    r = bloco("propria_de_sku", "serie", M[:, 0], M[:, 1:2], M[:, 2:3], nfe,
              ["ln_preco_proprio"],
              grupos={"serie": d.serie.to_numpy(),
                      "loja": d.store.to_numpy(),
                      "semana": d.week.to_numpy()})
    r["bootstrap_wild_loja"] = bootstrap_wild(
        M[:, 0], M[:, 1:2], M[:, 2:3], d.store.to_numpy(), nfe + 1)
    r["n_series"] = int(d.serie.nunique())
    r["n_semanas"] = int(d.week.nunique())
    r["absorcao_efeitos_fixos"] = diag_fe
    return r


def preparar_celulas(cel: pd.DataFrame, fora: pd.DataFrame, n: int):
    P = [f"preco_{i:02d}" for i in range(n)]
    V = [f"volume_{i:02d}" for i in range(n)]
    C = [f"custo_{i:02d}" for i in range(n)]
    d = cel.merge(fora, on=["store", "week"])
    d = d[(d.idx_preco_resto > 0) & (d.idx_custo_resto > 0)]
    return d, P, V, C


def montar_celula(d: pd.DataFrame, P, V, C) -> dict:
    """Agregados da célula sobre os SKUs **presentes**.

    Média de logs (média geométrica) e não índice ponderado por participação.
    A escolha importa para comparar com a identidade de agregação, e está
    declarada como limitação no topo deste arquivo.
    """
    return {
        "y": np.log(np.nansum(d[V].to_numpy(), axis=1)),
        "lnP": np.nanmean(np.log(d[P].to_numpy()), axis=1),
        "lnZ": np.nanmean(np.log(d[C].to_numpy()), axis=1),
        "lnI": np.log(d.idx_preco_resto.to_numpy()),
        "lnW": np.log(d.idx_custo_resto.to_numpy()),
    }


def estimar_celula(d: pd.DataFrame, P, V, C, nome: str, amostra: str,
                   com_bem_externo: bool) -> dict:
    g = montar_celula(d, P, V, C)
    colunas = [g["y"], g["lnP"]] + ([g["lnI"]] if com_bem_externo else [])
    colunas += [g["lnZ"]] + ([g["lnW"]] if com_bem_externo else [])
    diag_fe: dict = {}
    M = centralizar(np.column_stack(colunas),
                    [d.store.to_numpy(), d.week.to_numpy()],
                    diagnostico=diag_fe)
    m = 2 if com_bem_externo else 1
    nfe = d.store.nunique() + d.week.nunique()
    rot = ["ln_preco_sortimento"] + (["ln_preco_resto"] if com_bem_externo else [])
    r = bloco(nome, amostra, M[:, 0], M[:, 1:1 + m], M[:, 1 + m:1 + 2 * m],
              nfe, rot, grupos={"loja": d.store.to_numpy(),
                                "semana": d.week.to_numpy()})
    r["absorcao_efeitos_fixos"] = diag_fe
    if com_bem_externo:
        r["bootstrap_wild_loja"] = bootstrap_wild(
            M[:, 0], M[:, 1:1 + m], M[:, 1 + m:1 + 2 * m],
            d.store.to_numpy(), nfe + m)
    return r


# ----------------------------------------------------------- alavancagem ---
def alavancagem(longo: pd.DataFrame, elasticidades: dict,
                cortes=(-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20)) -> dict:
    """O fator de alavancagem de D21, medido, e os dois limiares.

    ## A correção de agregação

    Os documentos usam preço unitário **médio simples** e custo médio simples,
    fixados à mão em `figura_receita_margem.py` a partir de uma amostra de
    266.524 linhas. A agregação está errada, independentemente da amostra.

    A pergunta de D21 é sobre a margem **da cesta**, não sobre a margem do SKU
    típico. Com elasticidade comum eps e variação comum delta,

        M(delta) = (1+delta)^eps [ (1+delta) R - K ]
        R = soma p_i V_i (receita)      K = soma c_i V_i (custo da mercadoria)

    derivando em delta = 0, a margem cresce ao subir o preço se e somente se

        |eps| < R / (R - K)

    isto é, **receita dividida pela margem total**, que é o preço médio
    ponderado por volume dividido pela margem média ponderada por volume. A
    média simples pondera cada linha igualmente e portanto dá peso demais aos
    SKUs de baixo giro, que aqui são os mais caros e de maior margem
    percentual. Medido nesta categoria a diferença é grande, e vai na direção
    que amplia a faixa de conflito de D21.

    ## Local contra discreto

    O limiar acima é **local**, válido para delta infinitesimal. Para um corte
    discreto o limiar é maior, porque a receita responde proporcionalmente a
    delta enquanto o volume responde exponencialmente, e a assimetria cresce
    com |delta|. O limiar discreto é o |eps| que zera M(delta)/M(0) - 1.
    """
    d = longo[longo.custo.notna() & (longo.custo > 0)]
    v = d.volume.to_numpy(float)
    R = float((d.preco.to_numpy() * v).sum())
    K = float((d.custo.to_numpy() * v).sum())
    Q = float(v.sum())

    def resposta(eps: float, delta: float) -> dict:
        r = 1.0 + delta
        vol = r ** eps
        return {"volume": vol - 1.0,
                "receita": r * vol - 1.0,
                "margem": vol * (r * R - K) / (R - K) - 1.0}

    def limiar_discreto(delta: float) -> float:
        lo, hi = 1.0, 30.0
        for _ in range(200):
            meio = (lo + hi) / 2
            if resposta(-meio, delta)["margem"] < 0:
                lo = meio
            else:
                hi = meio
        return (lo + hi) / 2

    # Variação intra-série: mediana, entre séries (loja, SKU), do coeficiente
    # de variação ao longo das semanas. É o que sustenta a afirmação de que o
    # custo se move menos que o preço, usada como argumento de relevância do
    # instrumento em D18. Nos documentos os valores citados eram 0,0789 contra
    # 0,131, vindos da
    # amostra de 266.524 linhas sem produtor declarado.
    s = d.copy()
    s["serie"] = s.store.astype(np.int64) * 10**12 + s.upc
    g = s.groupby("serie", observed=True)

    def cv_mediano(coluna: str) -> float:
        m = g[coluna].mean()
        dp = g[coluna].std(ddof=1)
        r = (dp / m).replace([np.inf, -np.inf], np.nan).dropna()
        return float(r.median())

    return {
        "amostra": "linhas (loja, SKU, semana) do sortimento com custo válido",
        "cv_intra_preco": cv_mediano("preco"),
        "cv_intra_custo": cv_mediano("custo"),
        "n_linhas": int(len(d)),
        "receita_total": R,
        "custo_mercadoria_total": K,
        "margem_total": R - K,
        "volume_total": Q,
        "preco_medio_ponderado": R / Q,
        "custo_medio_ponderado": K / Q,
        "margem_percentual": 100.0 * (R - K) / R,
        "fator_alavancagem": R / (R - K),
        "limiar_local": R / (R - K),
        "limiar_discreto": {f"{dl:+.0%}": limiar_discreto(dl)
                            for dl in cortes if dl < 0},
        # Reportado apenas para explicar a divergência com os documentos, que
        # usam a média simples. NÃO é o fator que a álgebra pede.
        "comparacao_media_simples": {
            "preco_medio_simples": float(d.preco.mean()),
            "custo_medio_simples": float(d.custo.mean()),
            "fator_se_usasse_media_simples":
                float(d.preco.mean() / (d.preco.mean() - d.custo.mean())),
        },
        "cenarios": {chave: {"elasticidade": eps,
                             "resposta": {f"{dl:+.0%}": resposta(eps, dl)
                                          for dl in cortes}}
                     for chave, eps in elasticidades.items()},
    }


def checagem_agregacao(eps_agregada: float, ep_agregada: float,
                       eps_propria: float, ep_propria: float) -> dict:
    """A identidade de agregação como restrição testável, e não como ressalva.

    ## A restrição

    Com `s_i` a participação de volume de cada SKU,

        eps_agregada = soma_i s_i eps_ii + soma_i s_i soma_{j!=i} eps_ij

    O segundo termo é **positivo** quando os produtos são substitutos: subir o
    preço do SKU i empurra volume para os outros, o que amortece a queda do
    volume da cesta. Logo

        |eps_agregada| <= |media ponderada das proprias|

    Isto não é expectativa nem regra de bolso, é consequência algébrica da
    definição. Se a estimação a violar, alguma das duas estimativas está errada
    ou os produtos não são substitutos.

    ## Por que isto vira número

    Os pontos a violavam, 2,507 contra 1,757, e o arquivo registrava a violação
    como ressalva em prosa. Com os erros padrão agrupados de D24 a diferença
    está a meio erro padrão e a violação some. Passando a sair como número, a
    checagem serve a dois usos: documenta que a coerência interna foi verificada
    e não apenas afirmada, e reaparece na fase da rede, onde a **matriz** de
    elasticidades tem de satisfazer a mesma desigualdade, aí com as
    participações de verdade em vez de um único coeficiente comum.

    ## Ressalvas declaradas, e são três

    1. A variância da diferença é calculada como `ep1² + ep2²`, isto é,
       supondo as duas estimativas **independentes**. Elas não são: as amostras
       se sobrepõem e o instrumento é o mesmo. A aproximação serve para dar
       ordem de grandeza, não para um teste formal.
    2. A elasticidade de série é uma média implícita ponderada por variância
       intra do preço, e a identidade pede ponderação por **volume**.
    3. O agregador do lado esquerdo é média simples de logs, e não índice
       ponderado por participação.
    """
    folga = abs(eps_propria) - abs(eps_agregada)
    ep_dif = float(np.hypot(ep_agregada, ep_propria))
    return {
        "modulo_agregada": abs(eps_agregada),
        "modulo_propria": abs(eps_propria),
        "folga": folga,
        "ep_da_diferenca_supondo_independencia": ep_dif,
        "z_da_violacao": -folga / ep_dif,
        "viola_no_ponto": bool(folga < 0),
        "viola_em_sentido_testavel": bool(-folga / ep_dif > 1.96),
        "ressalvas": ["independencia entre as duas estimativas e aproximacao",
                      "serie pondera por variancia intra, a identidade pede volume",
                      "agregador e media simples de logs, nao indice ponderado"],
    }


def teste_faixa_conflito(eps: float, eps_por_versao: dict,
                         limiares: dict, critico_wild: float | None = None) -> dict:
    """Distância, em erros padrão, entre a elasticidade e cada borda da faixa.

    Estar na faixa de conflito de D21 exige duas coisas, e cada uma é uma
    hipótese a derrubar: `|eps|` abaixo do limiar e `|eps|` acima de 1. O
    intervalo de confiança "atravessar" o limiar é observação visual; o que o
    texto pode afirmar vem destes números, comparados aos cortes de uma cauda,
    1,645 para 5% e 2,326 para 1%.

    **Correção de 14/09/2026.** A versão anterior desta função recebia um erro
    padrão só, o iid, e com ele as duas hipóteses eram derrubadas a 1%. Isso
    era artefato da suposição de independência entre células da mesma loja. A
    função passa a receber as três versões e a reportar as três lado a lado,
    porque a conclusão de D21 **depende de qual delas se usa**, e esconder isso
    seria o mesmo erro que o registro de números oficiais existe para impedir.

    O limiar continua tratado como **conhecido**. Ele vem de R e K medidos
    sobre a mesma amostra, com erro desprezível ao lado de `ep`, e essa escolha
    fica ainda mais confortável com o `ep` agrupado, que é maior. Mas continua
    sendo escolha declarada e não fato automático.
    """
    m = abs(eps)

    def contra(ep: float) -> dict:
        return {
            "erro_padrao": ep,
            "z_borda_inferior_1": (m - 1.0) / ep,
            "z_borda_superior": {nome: (float(v) - m) / ep
                                 for nome, v in limiares.items()},
        }

    saida = {
        "elasticidade_modulo": m,
        "cortes_uma_cauda": {"5pct": 1.645, "1pct": 2.326},
        "versoes": {nome: contra(ep) for nome, ep in eps_por_versao.items()},
        "versao_citavel": "agrupado_loja",
    }
    if critico_wild is not None:
        # Com G = 93 o corte da normal é otimista. Este é o corte construído no
        # próprio dado, e é ele que decide, não o 1,96.
        saida["critico_wild_bicaudal_95"] = critico_wild
    return saida


def zip_unico(pasta: Path) -> Path:
    """O arquivo de movimento da categoria, exigindo que haja exatamente um.

    A versão anterior fazia `sorted(pasta.glob("w*.zip"))[0]` e descartava o
    resto em silêncio. Hoje sucos congelados tem um zip só, então não havia
    erro. Se a categoria mudar, ou o download for refeito em partes, o erro
    passa a existir e é do pior tipo: silencioso, e desloca todos os números
    de uma vez.
    """
    zips = sorted(pasta.glob("w*.zip"))
    if len(zips) != 1:
        raise SystemExit(
            f"esperava exatamente um w*.zip em {pasta}, encontrei {len(zips)}: "
            f"{[z.name for z in zips]}. Junte-os antes, ou ajuste o leitor.")
    return zips[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--raiz", default="data/raw")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--sufixo", default="",
                    help="sufixo do bem externo a usar, ex.: _p95 (checagem de D23)")
    a = ap.parse_args()
    pa = Path(a.painel)

    upcs = carregar_sortimento(a.categoria, pa)
    n = len(upcs)
    longo = extrair_longo(zip_unico(Path(a.raiz) / a.categoria), set(upcs))

    cel = pd.read_parquet(pa / f"{a.categoria}_celulas.parquet")
    fora = pd.read_parquet(pa / f"{a.categoria}_bem_externo{a.sufixo}.parquet")
    d, P, V, C = preparar_celulas(cel, fora, n)

    completas = d[(d[P] > 0).all(axis=1)]
    com_custo = completas[(completas[C] > 0).all(axis=1)]
    todas = d[np.nansum(d[V].to_numpy(), axis=1) > 0]
    todas = todas[np.isfinite(np.nanmean(np.log(todas[C].to_numpy()), axis=1))]

    est = {
        "serie": estimar_serie(longo),
        "celula_sem_bem_externo": estimar_celula(
            com_custo, P, V, C, "agregada_sem_bem_externo",
            "celulas completas com custo válido nos N", False),
        "celula_com_bem_externo": estimar_celula(
            com_custo, P, V, C, "agregada_com_bem_externo",
            "celulas completas com custo válido nos N", True),
        # Só o MQO desta linha é citável: o instrumento seria a média dos
        # custos sobre um subconjunto variável de SKUs, o que não é o mesmo
        # objeto entre células. Serve para reproduzir o piloto de D23.
        # Par de MQO na amostra de 25.234, que é a do piloto de D23: é dele
        # que sai o viés de omissão de 71%, hoje citado sem artefato.
        "celula_completas_sem_bem_externo": estimar_celula(
            completas, P, V, C, "agregada_sem_bem_externo",
            "celulas completas, SO O MQO e citavel", False),
        "celula_completas_sem_exigir_custo": estimar_celula(
            completas, P, V, C, "agregada_com_bem_externo",
            "celulas completas, SO O MQO e citavel", True),
        "diagnostico_celulas_incompletas": estimar_celula(
            todas, P, V, C, "agregada_com_bem_externo",
            "TODAS as celulas com atividade (composicao variavel, NAO citar)", True),
    }

    # As bordas do intervalo vêm da versão AGRUPADA por loja. A versão iid
    # continua no arquivo, mas deixou de ser citável como precisão (F1), e
    # portanto não pode mais alimentar os cenários do contrafactual.
    _ivcl = est["celula_com_bem_externo"]["ep_agrupado"]["loja"]["iv"]["ln_preco_sortimento"]
    escolhidas = {
        "propria_de_sku_iv": est["serie"]["iv"]["ln_preco_proprio"]["coef"],
        "agregada_iv": est["celula_com_bem_externo"]["iv"]["ln_preco_sortimento"]["coef"],
        "agregada_ic95cl_inf": _ivcl["ic95_sup"],
        "agregada_ic95cl_sup": _ivcl["ic95_inf"],
    }

    # Viés de omissão do bem externo, em MQO, na amostra do piloto de D23.
    # É deslocamento do conjunto de condicionamento, não magnitude do canal de
    # substituição: sob IV ele quase desaparece, e D23 já registra isso.
    sem = est["celula_completas_sem_bem_externo"]["ols"]["ln_preco_sortimento"]["coef"]
    com = est["celula_completas_sem_exigir_custo"]["ols"]["ln_preco_sortimento"]["coef"]
    sem_iv = est["celula_sem_bem_externo"]["iv"]["ln_preco_sortimento"]["coef"]
    com_iv = est["celula_com_bem_externo"]["iv"]["ln_preco_sortimento"]["coef"]

    saida = {"categoria": a.categoria, "n_sortimento": n,
             "bem_externo": a.sufixo or "principal", "estimativas": est,
             "vies_omissao": {
                 "ols_sem": sem, "ols_com": com,
                 "deslocamento_pct_ols": 100.0 * (com - sem) / abs(sem),
                 "iv_sem": sem_iv, "iv_com": com_iv,
                 "deslocamento_pct_iv": 100.0 * (com_iv - sem_iv) / abs(sem_iv)},
             "alavancagem": None}
    saida["alavancagem"] = alavancagem(longo, escolhidas)
    _al = saida["alavancagem"]
    _cbe = est["celula_com_bem_externo"]
    _eps = {
        "iid": _cbe["iv"]["ln_preco_sortimento"]["ep"],
        "agrupado_loja": _cbe["ep_agrupado"]["loja"]["iv"]["ln_preco_sortimento"]["ep"],
        "agrupado_semana": _cbe["ep_agrupado"]["semana"]["iv"]["ln_preco_sortimento"]["ep"],
    }
    saida["teste_faixa_conflito"] = teste_faixa_conflito(
        escolhidas["agregada_iv"], _eps,
        {"local": _al["limiar_local"], **_al["limiar_discreto"]},
        critico_wild=_cbe["bootstrap_wild_loja"]["critico_95_bicaudal"][0])
    saida["checagem_agregacao"] = checagem_agregacao(
        escolhidas["agregada_iv"], _eps["agrupado_loja"],
        est["serie"]["iv"]["ln_preco_proprio"]["coef"],
        est["serie"]["ep_agrupado"]["serie"]["iv"]["ln_preco_proprio"]["ep"])
    destino = pa / f"{a.categoria}_elasticidades{a.sufixo}.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))

    print(f"{'especificacao':38s} {'amostra':>9s} {'MQO':>9s} {'IV':>9s} "
          f"{'ep iid':>8s} {'ep loja':>8s} {'infl':>6s} {'F1':>10s} {'r2parc':>8s}")
    for chave, r in est.items():
        rot = r["regressores"][0]
        cl = r.get("ep_agrupado", {}).get("loja", {}).get("iv", {}).get(rot)
        ep_cl = format(cl["ep"], ".4f") if cl else "-"
        infl = format(cl["inflacao"], ".2f") + "x" if cl else "-"
        print(f"{chave:38s} {r['n']:9d} {r['ols'][rot]['coef']:+9.4f} "
              f"{r['iv'][rot]['coef']:+9.4f} {r['iv'][rot]['ep']:8.4f} "
              f"{ep_cl:>8s} {infl:>6s} "
              f"{r['primeiro_estagio'][rot]['F_instrumentos']:10.0f} "
              f"{r['primeiro_estagio'][rot]['r2_parcial']*100:7.2f}%")

    tf = saida["teste_faixa_conflito"]
    print(f"\nteste da faixa de conflito de D21, |eps| = {tf['elasticidade_modulo']:.3f}")
    print(f"  {'versao':18s} {'ep':>8s} {'z ate 1':>9s} {'z ate local':>12s} "
          f"{'z ate -15%':>11s}")
    for nome, v in tf["versoes"].items():
        print(f"  {nome:18s} {v['erro_padrao']:8.4f} "
              f"{v['z_borda_inferior_1']:9.2f} "
              f"{v['z_borda_superior']['local']:12.2f} "
              f"{v['z_borda_superior']['-15%']:11.2f}")
    print(f"  corte da normal 1,96  |  corte wild agrupado por loja "
          f"{tf['critico_wild_bicaudal_95']:.2f}  (a versao citavel e "
          f"{tf['versao_citavel']})")

    ca = saida["checagem_agregacao"]
    veredito = ("VIOLA de forma testavel" if ca["viola_em_sentido_testavel"]
                else "viola no ponto, nao em sentido testavel"
                if ca["viola_no_ponto"] else "nao viola")
    print(f"\nidentidade de agregacao |eps_agregada| <= |media das proprias|: "
          f"{veredito}")
    print(f"  {ca['modulo_agregada']:.3f} contra {ca['modulo_propria']:.3f}, "
          f"folga {ca['folga']:+.3f}, z da violacao {ca['z_da_violacao']:.2f}")

    for nome, r in est.items():
        af = r.get("absorcao_efeitos_fixos")
        if af:
            print(f"  absorcao FE {nome:34s} {af['passadas']:3d} passadas, "
                  f"variacao final {af['variacao_final']:.2e}, "
                  f"convergiu={af['convergiu']}")
    al = saida["alavancagem"]
    print(f"\npreco medio ponderado {al['preco_medio_ponderado']:.4f}  "
          f"custo {al['custo_medio_ponderado']:.4f}  "
          f"margem {al['margem_percentual']:.1f}%")
    print(f"fator de alavancagem (R/(R-K)) = {al['fator_alavancagem']:.4f}"
          f"   contra {al['comparacao_media_simples']['fator_se_usasse_media_simples']:.4f}"
          f" se usasse media simples, e 2,889 nos documentos")
    print("limiar discreto:", {k: round(v, 3) for k, v in al["limiar_discreto"].items()})
    print("\nresposta a um corte de 15%:")
    print(f"  {'cenario':26s} {'eps':>8s} {'volume':>9s} {'receita':>9s} {'margem':>9s}")
    for chave, c in al["cenarios"].items():
        r = c["resposta"]["-15%"]
        print(f"  {chave:26s} {c['elasticidade']:+8.3f} {r['volume']:+8.1%} "
              f"{r['receita']:+9.1%} {r['margem']:+9.1%}")
    print("\ngravado em", destino)


if __name__ == "__main__":
    main()
