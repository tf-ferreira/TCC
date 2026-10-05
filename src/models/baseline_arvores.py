"""D20: ensemble de árvores como referência obrigatória de previsão.

D20 exige um modelo de *gradient boosting* sobre árvores treinado com **as
mesmas variáveis e a mesma partição temporal** da rede neural, reportado ao
lado dela. Ele tem duas funções, e a segunda é a que decide o cronograma.

**Comparação honesta.** A literatura mostra ensembles de árvores competindo com
ou superando redes neurais em dados tabulares de varejo. Um trabalho que
alegasse superioridade preditiva da rede sem testar contra esta referência
estaria afirmando o que a literatura não sustenta.

**Controle de sanidade.** Se a rede não alcançar o ensemble com as mesmas
variáveis, o problema é de **treinamento** e não de arquitetura. Detectar isso
antes da etapa prescritiva evita construir a otimização inteira sobre uma rede
mal ajustada.

## Uma investigação registrada, 14/09/2026, com desfecho negativo

Uma auditoria apontou que o tipo categórico era fixado em cada partição
separadamente, e que por isso a identidade da loja estaria embaralhada entre
treino e teste. A primeira metade é verdade e está medida em
`fixar_categorias`. **A segunda não é**: o LightGBM remapeia categorias por
valor na previsão, e as métricas são idênticas com e sem a correção. O piso de
D20 não estava corrompido.

Fica registrado porque desfecho negativo também é resultado, e porque a mesma
armadilha **é** real um passo adiante, no embedding de loja da rede, onde nada
remapeia por valor.

## Uma ressalva de ordem, declarada e não escondida

O item 5 do cronograma, engenharia de atributos, **não foi feito**. Não existem
defasagens, sazonalidade fina nem variáveis de loja a partir da demografia.
Este script roda sobre o que existe hoje, e portanto é um **piso**, não a
comparação final.

Isso não é desperdício, por três motivos. Ele mede se o painel tem sinal antes
de se investir em atributos. Ele fixa a partição temporal e o formato de dados
que a rede vai ter de usar. E, rodado de novo depois do item 5, a **diferença
entre as duas execuções mede o valor da engenharia de atributos**, que de outro
modo ficaria sem medida.

## O formato, e por que ele é justo com as árvores

A rede mapeia contexto mais vetor de preços em **vetor** de demandas, uma saída
por SKU. Árvore não tem saída vetorial. A comparação se faz em formato longo,
uma linha por (célula, SKU), com a identidade do SKU como variável categórica.

Para que a comparação meça **modelo** e não **preparação de dado**, o preço
próprio entra como coluna explícita além de estar no vetor. Sem isso, a árvore
teria de aprender "quando o SKU é 7, olhe a coluna `preco_07`", que é um
handicap de layout e não de classe de modelo. A rede tem essa correspondência
por construção.

O custo **não** entra como atributo. Custo afeta demanda apenas através do
preço, e ele é o instrumento de D18: usá-lo como preditor de demanda
contaminaria a identificação.

## O que a árvore não pode fazer, medido e não afirmado

Árvore produz função **constante por partes**: o espaço é particionado e dentro
de cada região a previsão é um número fixo. A derivada é zero no interior de
cada região e indefinida nas fronteiras. Logo não há elasticidade extraível nem
gradiente para o otimizador, e **esta é a razão técnica de a rede neural ser
necessária neste trabalho**.

D20 e a seção 2.2 do texto afirmam isso. Aqui ele é medido, e o teste decisivo
não é a fração de zeros num passo qualquer: é a **varredura do passo**.

Com passo `h` grande, a diferença centrada quase nunca dá zero, porque entre
centenas de árvores alguma troca de folha. Isso parece derivada e não é: o valor
obtido é uma razão entre saltos, e depende de `h`. Uma derivada de verdade
**estabiliza** quando `h` diminui. A de uma função constante por partes faz o
oposto: quanto menor o passo, mais pontos caem estritamente no interior de uma
folha, então a fração de zeros exatos **cresce**, e onde não é zero o valor
**explode**, porque um salto fixo é dividido por um `h` que encolhe.

A varredura produz essa evidência, e o mesmo procedimento será aplicado à rede
na validação de D16, onde a expectativa é a inversa: estabilizar e coincidir com
a retropropagação.

Uso:
    python3 src/models/baseline_arvores.py frj
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd


def montar_longo(cel: pd.DataFrame, fora: pd.DataFrame, n: int) -> pd.DataFrame:
    """Formato longo: uma linha por (célula, SKU), só de células completas."""
    P = [f"preco_{i:02d}" for i in range(n)]
    V = [f"volume_{i:02d}" for i in range(n)]
    R = [f"promo_{i:02d}" for i in range(n)]
    d = cel.merge(fora, on=["store", "week"])
    d = d[(d[P] > 0).all(axis=1)]

    partes = []
    for i in range(n):
        parte = d[["store", "week"] + P + ["idx_preco_resto", "vol_resto",
                                          "promo_resto", "n_resto"]].copy()
        parte["sku"] = i
        parte["preco_proprio"] = d[P[i]].to_numpy()
        parte["promo_proprio"] = d[R[i]].to_numpy()
        parte["alvo"] = d[V[i]].to_numpy()
        partes.append(parte)
    longo = pd.concat(partes, ignore_index=True)
    # Sazonalidade como semana do ano. A semana bruta fica de fora de propósito:
    # com partição temporal ela só assume valores novos no teste, e a árvore não
    # extrapola. Escolha provisória, que pertence ao item 5.
    #
    # A deriva está medida e declarada (F9): o ano tem 52,18 semanas, não 52, e
    # ao longo dos 7,6 anos do painel o módulo acumula cerca de 1,4 semana de
    # defasagem. Ou seja, "semana 3 do ano" no fim do painel é meados de
    # janeiro, não o começo. É pequeno ao lado do resto do erro, mas o item 5
    # deve trocar isto por uma data de calendário derivada da semana 1 do DFF,
    # que é 14/09/1989, e por seno e cosseno do ângulo do ano, para que a
    # variável seja contínua na virada e não dê um salto entre 51 e 0.
    longo["semana_do_ano"] = longo["week"] % 52
    return longo.dropna(subset=["alvo"])


def fixar_categorias(longo: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """Fixa o tipo categórico sobre o painel INTEIRO, antes de particionar.

    ## O que a versão anterior fazia, e o que se descobriu ao investigar

    A versão anterior fazia `for df in (treino, teste): df[c] =
    df[c].astype("category")`, convertendo cada partição **separadamente**. O
    pandas constrói as categorias a partir dos valores observados em cada
    quadro, e as partições não têm o mesmo conjunto de lojas: seis lojas só
    aparecem no treino e sete só no teste, porque lojas abrem e fecham ao longo
    das 396 semanas. Medido: dos 86 códigos de loja do treino, **73 apontavam
    para uma loja diferente no teste**. O código 13 era a loja 45 no treino e a
    loja 47 no teste.

    **E não houve consequência sobre os resultados, o que também foi medido.**
    O LightGBM guarda as categorias do treino em `pandas_categorical` e, na
    previsão, chama `cat.set_categories` sobre o quadro recebido, remapeando
    **por valor** e não por código. As previsões com e sem esta função são
    idênticas bit a bit, e `tests/test_baseline_arvores.py` fixa esse fato. O
    WMAPE de teste de 52,4% contra 23,7% de treino é real, e mede regime
    temporal mais sobreajuste, não embaralhamento de loja.

    ## Por que a função fica, então

    Por três motivos, e nenhum deles é o susto.

    Primeiro, a invariante passa a ser **do projeto** e não de uma biblioteca:
    hoje o LightGBM protege, e o próximo consumidor do painel pode não proteger.
    Segundo, é exatamente isso que acontece na fase da rede: o embedding de loja
    é uma tabela indexada por código, e **nada** vai remapear por valor. Se a
    tabela de códigos for construída na partição em vez de no painel, o
    embedding lê a linha errada, e aí o erro é real e silencioso. Terceiro, a
    asserção em `main` documenta a exigência para quem ler o código depois.

    ## A regra geral

    A codificação de uma variável categórica não é propriedade da variável, é
    propriedade do ajuste que a criou. O mesmo erro aparece com escalonador
    ajustado no teste e com vocabulário de embedding construído por partição, e
    vai reaparecer na fase da rede, onde o embedding de loja é exatamente esse
    objeto.
    """
    longo = longo.copy()
    for c in colunas:
        longo[c] = longo[c].astype(
            pd.CategoricalDtype(categories=np.sort(longo[c].unique())))
    return longo


def importancia_completa(colunas: list[str], valores: list[float]) -> dict:
    """Importância de todos os atributos, com posição no ranking e participação.

    Existe porque D25 cita a posição de `promo_proprio`, que fica no fim da
    lista e por isso não aparecia no recorte das doze maiores. Número citado em
    documento precisa sair de um artefato.
    """
    imp = dict(zip(colunas, valores))
    ordem = sorted(imp.items(), key=lambda kv: -kv[1])
    total = float(sum(imp.values())) or 1.0
    return {
        "n_atributos": len(colunas),
        "importancia_completa": {k: float(v) for k, v in ordem},
        "posicao_importancia": {k: i + 1 for i, (k, _) in enumerate(ordem)},
        "share_importancia_pct": {k: 100.0 * v / total for k, v in ordem},
    }


def colunas_atributos(n: int, sem_promo_proprio: bool = False) -> list[str]:
    """Lista de atributos do modelo de referência, na ordem fixa.

    O preço próprio entra como coluna explícita **além** de estar no vetor, para
    que a comparação meça modelo e não layout de dado. O custo não entra: ele
    afeta a demanda só através do preço e é o instrumento de D18.

    `sem_promo_proprio` implementa D25. `promo_proprio` é o único atributo do
    painel que é função do preço que o otimizador decide, e por isso não pode
    entrar: congelado no valor histórico ele rouba efeito do preço e **atenua**
    a derivada; recalculado por regra do preço ele reintroduz descontinuidade e
    **destrói** a derivada, que é o que a seção 4 do projeto proíbe.

    `promo_resto` permanece, e a distinção é a que D25 registra: ele é
    *correlacionado* com o preço próprio (calendário promocional comum da
    categoria) mas não é *função* dele. Congelá-lo durante a otimização não tira
    nada do gradiente e é exatamente o cenário de piloto já declarado em D23.
    """
    colunas = ([f"preco_{i:02d}" for i in range(n)]
               + ["preco_proprio", "promo_proprio", "sku", "store",
                  "semana_do_ano", "idx_preco_resto", "vol_resto",
                  "promo_resto", "n_resto"])
    if sem_promo_proprio:
        colunas = [c for c in colunas if c != "promo_proprio"]
    return colunas


def particionar(longo: pd.DataFrame, frac_teste: float, zona: int):
    """Partição temporal com zona morta.

    A zona morta existe para que nenhuma defasagem atravesse a fronteira quando
    elas entrarem (item 5). Sem ela, uma defasagem de k semanas faz o conjunto
    de treino conter informação do teste, e a métrica fora da amostra deixa de
    medir o que promete.
    """
    semanas = np.sort(longo.week.unique())
    corte = semanas[int(len(semanas) * (1 - frac_teste))]
    treino = longo[longo.week < corte - zona]
    teste = longo[longo.week >= corte]
    return treino, teste, int(corte)


def metricas(y, yhat) -> dict:
    y = np.asarray(y, float); yhat = np.maximum(np.asarray(yhat, float), 0.0)
    return {
        "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
        "mae": float(np.mean(np.abs(y - yhat))),
        "rmse_log": float(np.sqrt(np.mean((np.log1p(y) - np.log1p(yhat)) ** 2))),
        "wmape_pct": float(100.0 * np.abs(y - yhat).sum() / y.sum()),
    }


def derivada_por_diferencas(modelo, X: pd.DataFrame, colunas: list[str],
                            n: int, h_rel: float = 0.02,
                            amostra: int = 20000, semente: int = 0) -> dict:
    """Derivada da previsão em relação ao preço próprio, por diferença centrada.

    Perturba o preço próprio **e** a posição correspondente no vetor de preços,
    porque as duas colunas representam a mesma grandeza. Perturbar só uma
    mediria a sensibilidade a uma incoerência, não a preço.
    """
    rng = np.random.default_rng(semente)
    idx = rng.choice(len(X), size=min(amostra, len(X)), replace=False)
    base = X.iloc[idx].copy().reset_index(drop=True)
    h = h_rel * base["preco_proprio"].to_numpy()

    derivadas = np.empty(len(base))
    for sinal, guarda in ((+1, "mais"), (-1, "menos")):
        pert = base.copy()
        pert["preco_proprio"] = base["preco_proprio"].to_numpy() + sinal * h
        posicoes = base["sku"].to_numpy().astype(int)
        coluna_preco = pert[[f"preco_{i:02d}" for i in range(n)]].to_numpy().copy()
        coluna_preco[np.arange(len(base)), posicoes] += sinal * h
        pert[[f"preco_{i:02d}" for i in range(n)]] = coluna_preco
        pred = modelo.predict(pert[colunas])
        derivadas = pred if sinal == +1 else (derivadas - pred)
    derivadas = derivadas / (2 * h)

    p = base["preco_proprio"].to_numpy()
    q = np.maximum(modelo.predict(base[colunas]), 1e-9)
    elast = derivadas * p / q
    nulas = derivadas == 0.0
    n_nao_nulas = int((~nulas).sum())
    return {
        "n_avaliadas": int(len(base)),
        "h_relativo": h_rel,
        "fracao_derivada_exatamente_zero_pct": float(100.0 * nulas.mean()),
        "fracao_derivada_positiva_pct": float(100.0 * (derivadas > 0).mean()),
        # A fração sobre o total NÃO é comparável entre linhas da varredura,
        # porque o denominador muda: quanto menor o passo, mais pontos viram
        # zero exato, e zero não é positivo nem negativo. A queda aparente da
        # coluna "positiva" mede o encolhimento do conjunto de não nulos, não o
        # comportamento do modelo. A fração CONDICIONAL abaixo é a comparável.
        "fracao_positiva_entre_as_nao_nulas_pct":
            float(100.0 * (derivadas[~nulas] > 0).mean()) if n_nao_nulas else None,
        "n_nao_nulas": n_nao_nulas,
        "elasticidade_implicita_mediana": float(np.median(elast)),
        "elasticidade_implicita_p10": float(np.percentile(elast, 10)),
        "elasticidade_implicita_p90": float(np.percentile(elast, 90)),
        "elasticidade_mediana_entre_as_nao_nulas":
            float(np.median(elast[~nulas])) if (~nulas).any() else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--saida", default="reports")
    ap.add_argument("--frac-teste", type=float, default=0.20)
    ap.add_argument("--zona", type=int, default=8,
                    help="zona morta em semanas entre treino e teste")
    ap.add_argument("--arvores", type=int, default=800)
    ap.add_argument("--sem-promo-proprio", action="store_true",
                    help="remove promo_proprio dos atributos (D25). Serve para "
                         "medir o custo preditivo de tirar do modelo o unico "
                         "atributo que e funcao do preco de decisao.")
    ap.add_argument("--sufixo", default="",
                    help="sufixo do arquivo de saida, para nao sobrescrever a "
                         "execucao de referencia")
    a = ap.parse_args()
    import lightgbm as lgb

    pa = Path(a.painel)
    n = json.loads((pa / f"{a.categoria}_upcs.json").read_text())["n"]
    cel = pd.read_parquet(pa / f"{a.categoria}_celulas.parquet")
    fora = pd.read_parquet(pa / f"{a.categoria}_bem_externo.parquet")
    longo = montar_longo(cel, fora, n)
    categoricas = ["sku", "store"]
    # Antes de particionar, nunca depois: ver fixar_categorias.
    longo = fixar_categorias(longo, categoricas)
    treino, teste, corte = particionar(longo, a.frac_teste, a.zona)

    colunas = colunas_atributos(n, a.sem_promo_proprio)

    for c in categoricas:
        assert treino[c].dtype == teste[c].dtype, (
            f"{c}: as particoes tem categorias diferentes, ver fixar_categorias")

    modelo = lgb.LGBMRegressor(
        objective="poisson", n_estimators=a.arvores, learning_rate=0.05,
        num_leaves=127, min_child_samples=40, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    modelo.fit(treino[colunas], treino["alvo"],
               categorical_feature=categoricas)

    pred_tr = modelo.predict(treino[colunas])
    pred_te = modelo.predict(teste[colunas])

    # Referência trivial: a mediana histórica do SKU naquela loja. Serve para
    # saber se o ensemble está aprendendo algo além de "cada SKU vende o que
    # costuma vender", que é o piso que qualquer modelo precisa superar.
    med = treino.groupby(["store", "sku"], observed=True)["alvo"].median()
    ingenuo = teste.set_index(["store", "sku"]).index.map(med).to_numpy(float)
    ingenuo = np.where(np.isnan(ingenuo), treino["alvo"].median(), ingenuo)

    saida = {
        "categoria": a.categoria,
        "n_sortimento": n,
        "sem_promo_proprio": bool(a.sem_promo_proprio),
        "aviso": ("PISO, não a comparação final: o item 5 do cronograma "
                  "(engenharia de atributos) não foi feito, então não há "
                  "defasagens, sazonalidade fina nem variáveis de loja. "
                  "Reexecutar com a especificação final; a diferença entre as "
                  "duas execuções mede o valor da engenharia de atributos."),
        "particao": {"semana_de_corte": corte, "zona_morta_semanas": a.zona,
                     "n_treino": int(len(treino)), "n_teste": int(len(teste)),
                     "semanas_treino": int(treino.week.nunique()),
                     "semanas_teste": int(teste.week.nunique())},
        "atributos": colunas,
        "modelo": {"tipo": "LightGBM", "objetivo": "poisson",
                   "n_estimators": a.arvores, "num_leaves": 127},
        "desempenho": {
            "treino": metricas(treino["alvo"], pred_tr),
            "teste": metricas(teste["alvo"], pred_te),
            "teste_referencia_ingenua": metricas(teste["alvo"], ingenuo),
        },
        # Varredura do passo: é ela que distingue "derivada" de "razão entre
        # saltos". Ver a seção do docstring sobre o que a árvore não pode fazer.
        "derivadas": [derivada_por_diferencas(modelo, teste, colunas, n, h)
                      for h in (0.10, 0.05, 0.02, 0.01, 0.005, 0.001)],
        "importancia": dict(sorted(
            zip(colunas, modelo.feature_importances_.tolist()),
            key=lambda kv: -kv[1])[:12]),
        # Importância completa, e não só as doze maiores. D25 precisa citar a
        # posição de um atributo que está no fim da lista, e um número citável
        # tem de ter produtor declarado (ver src/reports/numeros_oficiais.py).
        **importancia_completa(colunas, modelo.feature_importances_.tolist()),
    }
    destino = Path(a.saida) / f"baseline_arvores_{a.categoria}{a.sufixo}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))

    d = saida["desempenho"]
    print(f"partição: treino até a semana {corte - a.zona}, teste a partir da {corte}, "
          f"zona morta de {a.zona} semanas")
    print(f"{'':28s} {'RMSE':>9} {'MAE':>9} {'RMSE log':>9} {'WMAPE':>8}")
    for rot, m in d.items():
        print(f"{rot:28s} {m['rmse']:9.2f} {m['mae']:9.2f} "
              f"{m['rmse_log']:9.4f} {m['wmape_pct']:7.1f}%")
    print(f"\nvarredura do passo, {saida['derivadas'][0]['n_avaliadas']} pontos de teste:")
    print(f"  {'h relativo':>11} {'zeros exatos':>13} {'positiva/total':>15} "
          f"{'positiva/nao nulas':>19} {'elast. mediana':>15} {'p10':>9} {'p90':>9}")
    for g in saida["derivadas"]:
        cond = g["fracao_positiva_entre_as_nao_nulas_pct"]
        print(f"  {g['h_relativo']:11.3%} "
              f"{g['fracao_derivada_exatamente_zero_pct']:12.1f}% "
              f"{g['fracao_derivada_positiva_pct']:14.1f}% "
              f"{(f'{cond:.1f}%' if cond is not None else '-'):>19} "
              f"{g['elasticidade_implicita_mediana']:15.3f} "
              f"{g['elasticidade_implicita_p10']:9.2f} "
              f"{g['elasticidade_implicita_p90']:9.2f}")
    print(f"\ngravado em {destino}")


if __name__ == "__main__":
    main()
