"""Piso de ruído de uma comparação de WMAPE neste desenho experimental.

Existe por uma razão específica, e ela vale registrar. A varredura de
sazonalidade separou as configurações por cerca de um ponto de WMAPE, que é a
mesma ordem de grandeza do efeito medido em D25. Antes de ler essa separação
como resultado, é preciso saber quanto o **mesmo modelo, com a mesma
configuração**, se move só por trocar a semente: `colsample_bytree` e
`subsample` sorteiam colunas e linhas a cada árvore, e o número de colunas muda
entre as configurações, o que mexe no sorteio.

Sem esta medida, qualquer diferença abaixo do piso seria reportada como
descoberta. Com ela, a leitura fica objetiva: a comparação entre configurações
passa a ser entre **médias com erro padrão**, e não entre dois pontos.

Vale para as comparações seguintes também, defasagens inclusive, e é por isso
que é script próprio e não um trecho de `sazonalidade.py`.

## Por que a execução é fatiada

Cada ajuste leva cerca de 8 segundos com quatro linhas de execução, e 50
sementes por configuração são centenas de ajustes. O script é **retomável**:
cada par (configuração, semente) vira um arquivo próprio no diretório de
fragmentos, e uma semente já calculada não é refeita. Fatiar e retomar é o que
torna a medida executável em ambiente com limite de tempo por chamada, sem
nunca recomeçar do zero.

O diretório de fragmentos fica **fora do repositório** por padrão, porque são
centenas de arquivos intermediários; o artefato versionado é só o consolidado.

Uso:
    # calcula (pode ser chamado várias vezes, retoma de onde parou)
    python3 src/experiments/ruido_semente.py frj --config harm3 --sementes 0-49
    # consolida os fragmentos no artefato final
    python3 src/experiments/ruido_semente.py frj --consolidar
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src" / "experiments"))
sys.path.insert(0, str(RAIZ / "src" / "models"))
sys.path.insert(0, str(RAIZ / "src" / "data"))

from baseline_arvores import (colunas_atributos,  # noqa: E402
                              derivada_por_diferencas, metricas, particionar)
import sazonalidade  # noqa: E402
import varredura_defasagens  # noqa: E402
import varredura_item5  # noqa: E402
import varredura_perda  # noqa: E402
import varredura_trafego  # noqa: E402

# Uma família é um conjunto de configurações comparáveis entre si: mesmo painel
# preparado, mesma base de atributos, só o bloco em teste mudando. Misturar
# famílias na mesma tabela compararia coisas que não diferem só no que se quer
# medir.
FAMILIAS = {"sazonalidade": sazonalidade, "defasagens": varredura_defasagens,
            "trafego": varredura_trafego, "item5": varredura_item5,
            "perda": varredura_perda}
CATEGORICAS = sazonalidade.CATEGORICAS

FRAGMENTOS_PADRAO = Path(os.environ.get("TCC_FRAGMENTOS",
                                        Path.home() / ".tcc_fragmentos_ruido"))


def faixa(texto: str) -> list[int]:
    """Aceita "0-49", "3", ou "0-9,20,30-32"."""
    saida: list[int] = []
    for pedaco in texto.split(","):
        if "-" in pedaco:
            a, b = pedaco.split("-")
            saida.extend(range(int(a), int(b) + 1))
        else:
            saida.append(int(pedaco))
    return sorted(set(saida))


def colunas_da_config(nome: str, n: int, base: list[str], cfg: dict,
                      completo: bool = False) -> list[str]:
    """O par de D25 não é configuração de sazonalidade: ele troca a presença de
    `promo_proprio` e mantém `semana_do_ano`. Fica aqui porque a pergunta é a
    mesma, quanto do efeito medido é semente."""
    if nome.startswith("d25_"):
        return colunas_atributos(n, sem_promo_proprio=(nome == "d25_sem"))
    return cfg[nome] if completo else base + cfg[nome]


class _Destransformado:
    """Modelo cuja previsão já volta na escala do alvo.

    Existe para que `metricas` e `derivada_por_diferencas` não precisem saber em
    que escala a perda treinou: as duas só chamam `.predict`. Destransformar nos
    dois lugares separadamente seria o modo de falhar em que se esquece um deles
    e a elasticidade sai medida na escala errada, sem erro visível.
    """

    def __init__(self, modelo, inverso):
        self._modelo, self._inverso = modelo, inverso

    def predict(self, X):
        return self._inverso(self._modelo.predict(X))


EIXOS = {"wmape": "wmape_por_semente",
         "vies_de_nivel": "vies_de_nivel_por_semente",
         "elast_h10": "elast_h10_por_semente"}

# Qual é o "melhor" em cada eixo, e o `None` é o ponto importante. No WMAPE,
# menor é melhor. No viés de nível, melhor é **perto de 1**, e ordenar por menor
# elegeria a configuração que mais subestima. Na elasticidade própria **não
# existe melhor**: ela é diagnóstico, confrontado com o estimando independente
# de D18, e inventar uma ordem aqui seria transformar diagnóstico em critério de
# seleção pelas costas.
ORDENACAO = {"wmape": lambda v: v,
             "vies_de_nivel": lambda v: abs(v - 1.0),
             "elast_h10": None}


def comparar(consolidado: dict, alfa: float = 0.05) -> dict:
    """O teste que D26 pede, como script em vez de conta feita à mão.

    Três coisas que a conta manual errava com facilidade, e é por isso que ela
    virou código:

    1. **Pareado.** As configurações correm sobre as MESMAS sementes, e a
       variância de semente é comum a elas. Comparar médias com erro padrão não
       pareado joga essa informação fora e infla o erro padrão da diferença.
    2. **Limiar corrigido pelo número de pares**, que é C(k,2) e não k. Com três
       configurações são três pares, e o limiar bilateral de 5% vai de 1,960
       para 2,394.
    3. **Os três eixos**, e não só o WMAPE. O viés de nível entrou como eixo
       próprio porque o WMAPE soma erro absoluto, minimizado pela mediana
       condicional, enquanto a receita que a etapa 8 maximiza é uma esperança,
       que pede a média: uma régua pode ganhar no WMAPE justamente por mirar
       abaixo, e sem o nível ao lado a leitura do WMAPE engana.
    """
    cfg = consolidado["configuracoes"]
    nomes = sorted(cfg)
    pares = list(itertools.combinations(nomes, 2))
    limiar = statistics.NormalDist().inv_cdf(1 - alfa / (2 * len(pares))) if pares else None

    saida = {"n_configuracoes": len(nomes), "n_pares": len(pares), "alfa": alfa,
             "limiar_z_corrigido": limiar, "eixos": {}}
    for eixo, campo in EIXOS.items():
        series = {k: cfg[k].get(campo) for k in nomes}
        if any(v is None for v in series.values()):
            continue
        sementes = sorted(set.intersection(*(set(v) for v in series.values())), key=int)
        if len(sementes) < 2:
            continue
        linhas = {}
        for a, b in pares:
            d = np.array([series[a][s] - series[b][s] for s in sementes], float)
            ep = float(d.std(ddof=1) / np.sqrt(len(d)))
            if ep > 0:
                z = float(d.mean() / ep)
            elif d.mean() == 0.0:
                z = 0.0          # as duas configuracoes sao a mesma coisa
            else:
                z = math.copysign(float("inf"), d.mean())
            linhas[f"{a}__menos__{b}"] = {
                "diferenca_media": float(d.mean()),
                "erro_padrao_pareado": ep,
                "z": z,
                "passa_limiar": bool(abs(z) > limiar),
            }
        medias = {k: float(np.mean([series[k][s] for s in sementes])) for k in nomes}
        saida["eixos"][eixo] = {
            "n_sementes": len(sementes),
            "medias": medias,
            "pares": linhas,
            "poder": poder_da_comparacao(medias, linhas, limiar, len(sementes),
                                        chave=ORDENACAO.get(eixo)),
        }
    return saida


def poder_da_comparacao(medias: dict, pares: dict, limiar: float,
                        n_sementes: int, chave=None, topo: int = 5) -> dict:
    """Quanto esta comparação consegue distinguir, e não só o que ela achou.

    Uma tabela de médias com erro padrão responde "a diferença é grande o
    bastante?". Ela **não** responde a pergunta que vem antes, e que é o critério
    de invalidação declarado em D37: *este desenho consegue distinguir as
    diferenças que a decisão precisa distinguir?* Sem ela, "nenhum par passou o
    limiar" é ambíguo entre duas leituras opostas: as configurações empatam de
    verdade, ou a medição é cega e empataria de qualquer jeito.

    O que se calcula:

    - **DMD**, a diferença mínima detectável, `limiar × ep_pareado`. É a menor
      diferença que este desenho conseguiria declarar. Sai por par, e a mediana
      resume.
    - **A amplitude que a decisão precisa resolver**: a distância entre a melhor
      e a segunda, e a amplitude dentro do topo. É o alvo.
    - **O veredito**: se a DMD é maior que o alvo, o desenho é cego para a
      escolha em questão, e a saída honesta é mais sementes (ou outro desenho),
      não escolher assim mesmo.
    - **Quantas sementes faltam**: o erro padrão cai com `1/√n`, então para
      levar a DMD até o alvo são `n × (DMD/alvo)²` sementes. O número é
      aproximado porque o desvio pareado é ele próprio estimado com `n`
      observações, e com `n` pequeno essa estimativa é ruidosa: leia como ordem
      de grandeza, não como promessa.
    """
    if not medias or not pares or limiar is None:
        return {}
    eps_todos = sorted(d["erro_padrao_pareado"] for d in pares.values())
    base = {"ep_pareado_mediano": float(np.median(eps_todos)),
            "diferenca_minima_detectavel_mediana":
                float(limiar * np.median(eps_todos)),
            "amplitude_das_medias": float(max(medias.values()) - min(medias.values())),
            "n_pares_que_passam": sum(1 for d in pares.values() if d["passa_limiar"]),
            "n_pares": len(pares)}
    if chave is None:
        # Eixo sem "melhor" definido: entrega o piso de ruído e para. Ver
        # ORDENACAO.
        return {**base, "tem_ordem_de_melhor": False}
    ordem = sorted(medias, key=lambda k: chave(medias[k]))
    melhor, segunda = ordem[0], ordem[1] if len(ordem) > 1 else ordem[0]

    def ep_do_par(a, b):
        return (pares.get(f"{a}__menos__{b}") or pares.get(f"{b}__menos__{a}")
                or {}).get("erro_padrao_pareado")

    ep_1_2 = ep_do_par(melhor, segunda)
    alvo = abs(chave(medias[segunda]) - chave(medias[melhor]))
    grupo = ordem[:topo]
    amplitude_topo = abs(chave(medias[grupo[-1]]) - chave(medias[grupo[0]]))

    dmd_1_2 = float(limiar * ep_1_2) if ep_1_2 else None
    decidivel = bool(dmd_1_2 is not None and alvo > dmd_1_2)
    necessarias = None
    if dmd_1_2 and alvo > 0:
        necessarias = int(np.ceil(n_sementes * (dmd_1_2 / alvo) ** 2))

    return {
        **base,
        "tem_ordem_de_melhor": True,
        "melhor": melhor, "segunda": segunda,
        "distancia_melhor_para_segunda": float(alvo),
        "amplitude_do_topo": float(amplitude_topo),
        "n_do_topo": len(grupo),
        "diferenca_minima_detectavel_do_par_1_2": dmd_1_2,
        "escolha_entre_1_e_2_e_decidivel": decidivel,
        "sementes_para_decidir_1_contra_2": necessarias,
    }


def consolidar(categoria: str, frag: Path, saida: Path) -> dict:
    """Junta os fragmentos num artefato só, com média, desvio e erro padrão.

    O erro padrão **da média** é o que compara configurações; a amplitude
    descreve a dispersão de uma execução única e serve para dizer o que uma
    comparação de semente única teria arriscado.
    """
    por_config: dict[str, dict] = {}
    conjuntos: set[str] = set()
    for arquivo in sorted(frag.glob(f"{categoria}__*.json")):
        d = json.loads(arquivo.read_text())
        # Em que conjunto o número foi medido entra no artefato. Sem isso, um
        # WMAPE de validação e um de teste ficam indistinguíveis dentro do JSON,
        # e quem lê depois não tem como saber qual dos dois está citando. As
        # famílias anteriores a 16/09/2026 não gravavam o campo e são de teste.
        conjuntos.add(d.get("medido_em", "teste"))
        alvo = por_config.setdefault(d["config"], {"n_atributos": d["n_atributos"],
                                                   "wmape": {}, "elast_h10": {},
                                                   "vies": {}})
        alvo["wmape"][str(d["semente"])] = d["wmape_pct"]
        if d.get("vies_de_nivel") is not None:
            alvo["vies"][str(d["semente"])] = d["vies_de_nivel"]
        if d.get("elast_h10") is not None:
            alvo["elast_h10"][str(d["semente"])] = d["elast_h10"]

    resumo = {}
    for nome, d in sorted(por_config.items()):
        v = np.array([d["wmape"][k] for k in sorted(d["wmape"], key=int)])
        linha = {
            "n_atributos": d["n_atributos"],
            "n_sementes": int(len(v)),
            "wmape_media": float(v.mean()),
            "wmape_desvio_padrao": float(v.std(ddof=1)) if len(v) > 1 else None,
            "wmape_erro_padrao_da_media":
                float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
            "wmape_amplitude": float(v.max() - v.min()),
            "wmape_min": float(v.min()),
            "wmape_max": float(v.max()),
            "wmape_por_semente": {k: d["wmape"][k] for k in sorted(d["wmape"], key=int)},
        }
        if d.get("vies"):
            b = np.array([d["vies"][k] for k in sorted(d["vies"], key=int)])
            linha.update({
                "vies_de_nivel_media": float(b.mean()),
                "vies_de_nivel_erro_padrao_da_media":
                    float(b.std(ddof=1) / np.sqrt(len(b))) if len(b) > 1 else None,
                "vies_de_nivel_por_semente":
                    {k: d["vies"][k] for k in sorted(d["vies"], key=int)},
            })
        if d["elast_h10"]:
            e = np.array([d["elast_h10"][k] for k in sorted(d["elast_h10"], key=int)])
            linha.update({
                "elast_h10_media": float(e.mean()),
                "elast_h10_desvio_padrao": float(e.std(ddof=1)) if len(e) > 1 else None,
                "elast_h10_erro_padrao_da_media":
                    float(e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 1 else None,
                "elast_h10_amplitude": float(e.max() - e.min()),
                # As tres series por semente existem para o teste PAREADO. As
                # configuracoes correm sobre as MESMAS sementes, e a variancia
                # de semente e comum a elas: comparar medias com erro padrao
                # nao pareado joga fora essa informacao e infla o erro padrao
                # da diferenca. Sem a serie gravada, o artefato nao permite
                # refazer o teste certo, e foi o que aconteceu com a derivada.
                "elast_h10_por_semente":
                    {k: d["elast_h10"][k] for k in sorted(d["elast_h10"], key=int)},
            })
        resumo[nome] = linha

    if len(conjuntos) > 1:
        raise SystemExit(
            f"\nFRAGMENTOS DE CONJUNTOS DIFERENTES em {frag}: {sorted(conjuntos)}\n"
            "  Médias sobre validação e teste misturados não significam nada.\n")

    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps({"categoria": categoria,
                                 "medido_em": (conjuntos.pop() if conjuntos
                                               else "teste"),
                                 "configuracoes": resumo},
                                indent=2, ensure_ascii=False))
    return resumo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--familia", default="sazonalidade", choices=sorted(FAMILIAS))
    ap.add_argument("--config", nargs="+", default=[])
    ap.add_argument("--sementes", default="0-2", help='ex.: "0-49" ou "0,5,9"')
    ap.add_argument("--elasticidade", action="store_true",
                    help="mede também a elasticidade implícita de passo 10%%")
    ap.add_argument("--consolidar", action="store_true")
    ap.add_argument("--comparar", action="store_true",
                    help="le o consolidado e faz o teste pareado de D26 nos "
                         "tres eixos, com limiar corrigido pelo numero de pares")
    ap.add_argument("--fragmentos", default=str(FRAGMENTOS_PADRAO))
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--saida", default="reports")
    ap.add_argument("--frac-teste", type=float, default=0.20)
    ap.add_argument("--zona", type=int, default=8)
    ap.add_argument("--arvores", type=int, default=800)
    ap.add_argument("--n-jobs", type=int, default=0, help="0 = padrão do LightGBM")
    a = ap.parse_args()

    familia = FAMILIAS[a.familia]
    frag = Path(a.fragmentos) / a.familia
    destino = Path(a.saida) / f"ruido_semente_{a.categoria}_{a.familia}.json"
    if a.familia == "sazonalidade":  # nomes herdados, preservados
        frag = Path(a.fragmentos)
        destino = Path(a.saida) / f"ruido_semente_{a.categoria}.json"

    if a.comparar:
        alvo = Path(a.saida) / f"comparacao_{a.categoria}_{a.familia}.json"
        saida = comparar(json.loads(destino.read_text()))
        alvo.write_text(json.dumps(saida, indent=2, ensure_ascii=False))
        print(f"{saida['n_pares']} pares, limiar de z corrigido "
              f"{saida['limiar_z_corrigido']:.3f}\n")
        for eixo, bloco in saida["eixos"].items():
            print(f"  --- {eixo} ({bloco['n_sementes']} sementes) ---")
            for nome, m in bloco["medias"].items():
                print(f"      {nome:18s} {m:10.4f}")
            for par, d in bloco["pares"].items():
                print(f"      {par:42s} dif {d['diferenca_media']:+9.4f} "
                      f"ep {d['erro_padrao_pareado']:8.4f} z {d['z']:+9.2f} "
                      f"{'passa' if d['passa_limiar'] else 'NAO passa'}")
            print()
        print(f"gravado em {alvo}")
        return

    if a.consolidar:
        resumo = consolidar(a.categoria, frag, destino)
        print(f"{'configuração':18s} {'n':>3} {'WMAPE médio':>12} {'ep da média':>12} "
              f"{'amplitude':>10} {'nível':>8} {'elast h10':>10}")
        for nome, d in resumo.items():
            ep = d["wmape_erro_padrao_da_media"]
            nivel = d.get("vies_de_nivel_media")
            el = d.get("elast_h10_media")
            print(f"{nome:18s} {d['n_sementes']:3d} {d['wmape_media']:11.3f}% "
                  f"{(ep if ep is not None else float('nan')):12.3f} "
                  f"{d['wmape_amplitude']:10.3f} "
                  f"{(nivel if nivel is not None else float('nan')):8.4f} "
                  f"{(el if el is not None else float('nan')):10.4f}")
        print(f"\ngravado em {destino}")
        return

    import lightgbm as lgb
    sementes = faixa(a.sementes)
    frag.mkdir(parents=True, exist_ok=True)

    # Ordem por semente e não por configuração: se a execução for interrompida,
    # todas as configurações ficam com o mesmo número de sementes, e o parcial
    # já é comparável. Ordenar por configuração deixaria a primeira completa e
    # a última vazia.
    pendentes = [(c, s) for s in sementes for c in a.config
                 if not (frag / f"{a.categoria}__{c}__{s:03d}.json").exists()]
    if not pendentes:
        print("nada pendente")
        return

    pa = Path(a.painel)
    n = json.loads((pa / f"{a.categoria}_upcs.json").read_text())["n"]
    longo = familia.preparar(pd.read_parquet(pa / f"{a.categoria}_celulas.parquet"),
                             pd.read_parquet(pa / f"{a.categoria}_bem_externo.parquet"), n)
    treino, teste, _ = particionar(longo, a.frac_teste, a.zona)
    base = [c for c in colunas_atributos(n, sem_promo_proprio=True)
            if c != "semana_do_ano"]
    cfg = familia.configuracoes(3)

    # Uma família pode trocar a REGRA de treino e não só o conjunto de
    # atributos. Só `perda` faz isso hoje; as outras caem no padrão de D20.
    espec_de = getattr(familia, "especificacao", lambda _nome: {})

    for nome, semente in pendentes:
        colunas = colunas_da_config(nome, n, base, cfg,
                                    getattr(familia, "COMPLETO", False))
        espec = espec_de(nome)
        alvo = espec.get("alvo")
        modelo = lgb.LGBMRegressor(
            objective=espec.get("objective", "poisson"), n_estimators=a.arvores,
            learning_rate=0.05, num_leaves=127, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=semente, verbose=-1,
            **({"n_jobs": a.n_jobs} if a.n_jobs else {}))
        y = treino["alvo"].to_numpy(float)
        modelo.fit(treino[colunas], alvo(y) if alvo else y,
                   categorical_feature=CATEGORICAS)
        avaliavel = modelo
        if espec.get("inverso"):
            avaliavel = _Destransformado(
                modelo, espec["inverso"](modelo, treino, colunas, alvo))
        pred = avaliavel.predict(teste[colunas])
        w = metricas(teste["alvo"], pred)["wmape_pct"]
        # Viés de NÍVEL, e ele não é redundante com o WMAPE. O WMAPE soma erro
        # ABSOLUTO, que é minimizado pela mediana condicional; a receita que a
        # etapa 8 maximiza é uma esperança, que pede a média. Uma régua pode
        # ganhar no WMAPE justamente por mirar a mediana, e aí ela subestima o
        # volume de forma sistemática. Sem este número a troca fica invisível.
        vies = float(np.sum(np.maximum(pred, 0.0)) / np.sum(teste["alvo"]))
        e = None
        if a.elasticidade:
            e = derivada_por_diferencas(
                avaliavel, teste, colunas, n, 0.10)["elasticidade_implicita_mediana"]
        (frag / f"{a.categoria}__{nome}__{semente:03d}.json").write_text(json.dumps(
            {"config": nome, "semente": semente, "n_atributos": len(colunas),
             "wmape_pct": float(w), "vies_de_nivel": vies,
             "elast_h10": float(e) if e is not None else None}))
        print(f"{nome} semente {semente}: WMAPE {w:.3f}%, nível {vies:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
