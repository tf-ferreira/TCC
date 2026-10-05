"""Sonda do item 8: a primeira execução do solver, e o que ela decide.

Não é o resultado do item 8. É a medição que dimensiona o item 8, e ela responde
quatro perguntas que nenhum documento do projeto responde hoje:

1. **Quantas células o item 8 de fato otimiza** (pendências 10.1 e 10.2). Os
   números de escopo de D38 saíram de comando ad hoc em conversa; aqui eles
   ganham produtor, que é o que `numeros_oficiais.py` existe para exigir.
2. **Quanto custa um `solve`**, e portanto se a multipartida de D15 (K entre 30 e
   50 por célula) cabe na janela de teste inteira ou só em amostra declarada.
3. **Onde a solução para**: quantas coordenadas na borda inferior, quantas na
   superior, quantas no interior. É a pergunta de existência do item 8, agora no
   problema CONJUNTO, e não a coordenada a coordenada de 8.13.
4. **Se margem e receita param em lugares diferentes**, que é o contraste que
   D21 promete como resultado de pesquisa.

## O que este script NÃO mede, e a distinção é a do item 9

O ganho reportado aqui é `M(p*) / M(p⁰) − 1`, com os dois termos vindo do
**próprio modelo**. Não é o contrafactual do objetivo específico 5, que compara
contra a **receita histórica real**. A diferença importa: a MSLE encolhe o nível
em 14 pontos em amostra (D35, emenda), e esse encolhimento em boa parte se
**cancela** numa razão entre duas avaliações do mesmo modelo, mas não cancela
contra o histórico. Misturar os dois é o erro que o item 9 tem de não cometer.

A otimização aqui também é **míope**, no sentido exato de D27: as defasagens de
preço ficam congeladas no valor histórico. Congelá-las é o cenário declarado, e
tornar a simulação sequencial é da etapa 9, não desta sonda.

Uso:

    python3 src/optimization/sonda_item8.py frj --celulas 0
    python3 src/optimization/sonda_item8.py frj --celulas 0 --comparar
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
for sub in ("models", "data", "experiments", "optimization"):
    sys.path.insert(0, str(RAIZ / "src" / sub))

import hierarquia as HIER  # noqa: E402
import problema as P  # noqa: E402
import rede  # noqa: E402

OBJETIVOS = ("margem", "receita")


def carregar_modelo(arquivo: Path, dados: dict):
    """Rebuild da rede adotada a partir do artefato canônico, com conferências.

    Três checagens antes de usar, e cada uma pega um modo de falha real:

    - a configuração salva é a de `varredura_rede.MELHOR`, como `suavidade_d16`
      já exige: um artefato treinado com hiperparâmetro digitado à mão pode
      divergir do que foi medido sem que nada acuse;
    - `centro_de_preco` do artefato bate com o recomputado do painel. Se o painel
      mudou depois do treino, `u` passa a ser medido de outro ponto e toda a
      matriz de elasticidades se desloca em silêncio;
    - o escalonador de contexto, idem.
    """
    import torch
    from varredura_rede import MELHOR

    pt = torch.load(arquivo, map_location="cpu", weights_only=False)
    cfg = pt["cfg"]
    divergentes = {k: (cfg.get(k), v) for k, v in MELHOR.items() if cfg.get(k) != v}
    if divergentes or not cfg.get("restrita", False):
        raise SystemExit(f"o artefato {arquivo} não é o ponto adotado: "
                         f"{divergentes}. Refaça com "
                         "`python3 src/models/rede.py frj --adotada`.")

    # D41/D42: a especificação do contexto tem de ser a ativa. Artefato sem a
    # chave é anterior a 23/09/2026 e, portanto, da especificação "adotada".
    espec = cfg.get("especificacao", "adotada")
    if espec != rede.ESPECIFICACAO_ATIVA:
        raise SystemExit(f"o artefato {arquivo} foi treinado na especificação "
                         f"'{espec}', e a ativa é '{rede.ESPECIFICACAO_ATIVA}'. "
                         "Chame rede.usar_especificacao antes de preparar os dados.")

    centro_art = np.asarray(pt["centro_de_preco"], float)
    centro_now = np.asarray(dados["centro_de_preco"], float)
    if not np.allclose(centro_art, centro_now, rtol=0, atol=1e-10):
        raise SystemExit("centro_de_preco do artefato não bate com o painel "
                         f"atual: max |dif| = {np.abs(centro_art - centro_now).max():.3e}. "
                         "O painel mudou depois do treino; retreine.")
    for nome in ("ctx_celula", "ctx_sku"):
        for chave in ("media", "desvio"):
            a = np.asarray(pt["escalonador"][nome][chave], float)
            b = np.asarray(dados["escalonador"][nome][chave], float)
            if not np.allclose(a, b, rtol=0, atol=1e-10):
                raise SystemExit(f"escalonador {nome}/{chave} divergiu do painel")

    dim_ctx = (len(rede.CONTEXTO_CELULA)
               + len(rede.CONTEXTO_SKU) * dados["n"])
    mods = rede.construir_modulos()
    modelo = mods["RedeDemanda"](
        n=dados["n"], dim_ctx=dim_ctx, n_lojas=dados["n_lojas"],
        largura=cfg["largura"], profundidade=cfg["profundidade"],
        dim_emb=cfg["dim_emb"], h_monotona=cfg["h_monotona"],
        restrita=cfg["restrita"], intercepto=cfg["intercepto"],
        posto=cfg.get("posto"),
        gama_contextual=cfg.get("gama_contextual", False),
        semente=cfg["semente"])
    modelo.load_state_dict(pt["estado"])
    modelo.eval()
    return modelo, cfg, centro_now


def demanda_e_matriz(modelo, esc: dict, indices: np.ndarray, U: np.ndarray):
    """`V̂` e a matriz de elasticidades em lote, num vetor de preços qualquer.

    Em lote e não célula a célula porque a checagem de coerência usa
    `rede.identidade_de_agregacao`, que é o **ponto único** daquela fórmula
    (pendência 2.8), e ela opera sobre o lote inteiro.
    """
    import torch
    dtype = next(modelo.parameters()).dtype
    with torch.no_grad():
        ctx = torch.as_tensor(esc["ctx"][indices], dtype=dtype)
        loja = torch.as_tensor(esc["loja_idx"][indices].astype("int64"),
                               dtype=torch.long)
        u = torch.as_tensor(np.asarray(U), dtype=dtype)
        v = torch.exp(modelo.log_demanda(ctx, u, loja))
        eps = modelo.elasticidades_fechadas(u)
    return v, eps


def _percentis(v: np.ndarray) -> dict:
    return {"mediana": float(np.median(v)), "p10": float(np.quantile(v, 0.10)),
            "p90": float(np.quantile(v, 0.90)), "media": float(np.mean(v))}


def restricoes_do_escopo(categoria: str, painel, bruto, centro, esc: dict,
                         forma: str):
    """`(A, B)` da hierarquia de marca para TODAS as células do escopo (D39).

    `B` tem uma linha por célula, na ordem de `esc`, mesmo na forma `absoluta`,
    em que o lado direito é igual em toda célula: uma forma só na saída evita um
    `if` de forma dentro do laço de otimização, que é onde ele erraria.
    """
    info = HIER.identificar(categoria, painel, bruto)
    pares = HIER.pares(info)
    n = esc["n"]
    A, b = HIER.restricoes(pares, centro, n, forma=forma,
                           u0=(esc["u"] if forma == "nao_piorar" else None))
    B = np.tile(np.asarray(b, float), (len(esc["u"]), 1)) if np.ndim(b) == 1 else b
    # Viabilidade ANTES do solver. Sem esta checagem, uma celula sem ponto viavel
    # na caixa recebe do scipy o ponto menos inviavel que ele achou, e aquilo entra
    # no relatorio como se fosse solucao.
    baixo, alto = P.caixa_em_u(esc["u"])
    inviavel = HIER.inviaveis_na_caixa(A, B, baixo, alto)
    return A, B, pares, info, inviavel


def rodar(modelo, centro, esc: dict, objetivo: str, indices: np.ndarray,
          metodo: str, limites=P.LIMITES_CAIXA, restr=None) -> dict:
    """Uma otimização por célula, a partir do ponto HISTÓRICO.

    `limites` existe para que `varredura_caixa.py` reutilize esta função em vez
    de reimplementar a checagem de coerência. A fórmula da incoerência mora
    **aqui e em nenhum outro lugar**, pela mesma razão que
    `rede.identidade_de_agregacao` mora num lugar só: fórmula duplicada é
    fórmula que diverge.
    """
    n = esc["n"]
    custos = (esc["custos"] if objetivo == "margem"
              else np.zeros_like(esc["custos"]))
    tempos, ganhos, interior = [], [], []
    baixo, alto, sucesso = [], [], 0
    total_ini = total_fim = 0.0
    u_otimo = np.empty((len(indices), n), dtype=float)
    ativas, viol = [], []
    reserva = inviaveis = nao_viavel = 0
    for pos, c in enumerate(indices):
        if restr is not None and restr[2][c]:
            inviaveis += 1
            u_otimo[pos] = esc["u"][c]
            continue
        r_celula = None if restr is None else (restr[0], restr[1][c])
        prob = P.ProblemaDeCelula(modelo, esc["ctx"][c], int(esc["loja_idx"][c]),
                                 centro, custos[c], esc["u"][c],
                                 limites=limites, restricoes=r_celula)
        t0 = time.perf_counter()
        r = P.resolver(prob, metodo=metodo)
        tempos.append(time.perf_counter() - t0)
        sucesso += int(r["sucesso"])
        ganhos.append(r["valor"] / r["valor_inicial"] - 1.0)
        interior.append(r["n_interior"])
        baixo.append(int(r["na_borda_inferior"].sum()))
        alto.append(int(r["na_borda_superior"].sum()))
        total_ini += r["valor_inicial"]
        total_fim += r["valor"]
        u_otimo[pos] = r["u"]
        ativas.append(r["n_restricoes_ativas"])
        viol.append(r["violacao_maxima_das_restricoes"])
        reserva += int(r["usou_reserva"])
        nao_viavel += int(not r["viavel"])
    b, a, i = np.array(baixo), np.array(alto), np.array(interior)

    # ---- A CHECAGEM DE COERENCIA, e ela tem DOIS objetos distintos ----------
    #
    # E o motivo de esta sonda existir antes do resultado, e a primeira versao
    # dela, de 17/09/2026, media a coisa errada: comparava a media SIMPLES das
    # variacoes de preco contra o volume total, e chamava de incoerencia os
    # casos em que os dois sobem juntos. Isso confunde incoerencia com
    # COMPOSICAO: baixar 15% o preco de um SKU de giro alto e subir 19 de giro
    # baixo faz a media simples subir e o volume total subir, e nao ha nada de
    # incoerente nisso, e substituicao funcionando.
    #
    # Objeto 1, a coerencia do MODELO no ponto escolhido. E a identidade da
    # pendencia 2.8, que fala de subida UNIFORME de 1% em todos os precos, e por
    # isso e imune a composicao. `rede.identidade_de_agregacao` e o ponto unico
    # dela, e aqui ela e avaliada no OTIMO e na BASE, para que a comparacao seja
    # contra 8.16 e nao contra nada.
    #
    # Objeto 2, a resposta agregada da POLITICA. Aqui o indice de preco tem de
    # ser ponderado pelas quantidades da base (tipo Laspeyres), e nao simples:
    #
    #     dlnP = sum_i s_i0 * (u_i* - u_i0)      com s_i0 = V_i0 / sum V_0
    #
    # Mesmo ponderado, a razao dlnV/dlnP NAO e uma elasticidade quando os vinte
    # precos se movem em direcoes diferentes: ela e um resumo da politica, e esta
    # rotulada como tal.
    u0 = esc["u"][indices]
    v0, eps0 = demanda_e_matriz(modelo, esc, indices, u0)
    v1, eps1 = demanda_e_matriz(modelo, esc, indices, u_otimo)
    identidade_base = rede.identidade_de_agregacao(eps0, v0)
    identidade_otimo = rede.identidade_de_agregacao(eps1, v1)

    s0 = (v0 / v0.sum(dim=1, keepdim=True)).numpy().astype(float)
    dlnP = (s0 * (u_otimo - u0)).sum(axis=1)
    dlnV = np.log(v1.sum(dim=1).numpy().astype(float)
                  / v0.sum(dim=1).numpy().astype(float))
    dp, dv = np.expm1(dlnP), np.expm1(dlnV)
    coord = float((len(indices) - inviaveis) * n)
    # A elasticidade agregada implícita da POLÍTICA, por célula. Ela é positiva
    # exatamente quando preço e volume andam no mesmo sentido, de modo que a
    # fração positiva É a taxa de incoerência, sem precisar dividir. A mediana
    # do valor sai só das células em que o preço se move o bastante para a razão
    # ter significado.
    lp, lv = np.log1p(dp), np.log1p(dv)
    usavel = np.abs(lp) > 1e-4
    implicita = lv[usavel] / lp[usavel]
    incoerentes = ((dp > 0) & (dv > 0)) | ((dp < 0) & (dv < 0))
    return {
        "objetivo": objetivo,
        "metodo": metodo,
        "raio_caixa": [float(limites[0]), float(limites[1])],
        "n_restricoes_lineares": 0 if restr is None else int(restr[0].shape[0]),
        "restricoes_ativas_por_celula": _percentis(np.array(ativas, float)),
        "pct_celulas_com_alguma_ativa": 100.0 * float((np.array(ativas) > 0).mean()),
        "violacao_maxima_observada": float(np.max(viol)) if viol else 0.0,
        "violacao_maxima_em_fracao_de_preco": (
            float(np.expm1(np.max(viol))) if viol else 0.0),
        "tolerancia_declarada": P.TOL_VIABILIDADE,
        "acima_da_tolerancia": bool(viol and np.max(viol) > P.TOL_VIABILIDADE),
        "pct_celulas_nao_viaveis_na_tolerancia": 100.0 * nao_viavel / len(indices),
        "pct_celulas_com_metodo_de_reserva": 100.0 * reserva / len(indices),
        "n_celulas": int(len(indices)),
        "n_celulas_inviaveis_na_caixa": inviaveis,
        "pct_celulas_inviaveis_na_caixa": 100.0 * inviaveis / len(indices),
        "pct_sucesso": (100.0 * sucesso / (len(indices) - inviaveis)
                        if len(indices) > inviaveis else float("nan")),
        "tempo_por_solve_s": _percentis(np.array(tempos)),
        "tempo_total_s": float(np.sum(tempos)),
        "ganho_pct_por_celula": {k: 100.0 * v
                                 for k, v in _percentis(np.array(ganhos)).items()},
        "ganho_pct_agregado": 100.0 * (total_fim / total_ini - 1.0),
        "pct_coord_borda_inferior": 100.0 * b.sum() / coord,
        "pct_coord_borda_superior": 100.0 * a.sum() / coord,
        "pct_coord_interior": 100.0 * i.sum() / coord,
        "pct_celulas_todas_na_borda_inferior": 100.0 * float((b == n).mean()),
        "pct_celulas_sem_nenhuma_interior": 100.0 * float((i == 0).mean()),
        "interior_por_celula": _percentis(i.astype(float)),
        "variacao_pct_preco_medio": {k: 100.0 * v for k, v in _percentis(dp).items()},
        "variacao_pct_volume_total": {k: 100.0 * v for k, v in _percentis(dv).items()},
        # Células em que preço médio e volume total sobem JUNTOS: incoerência do
        # modelo sendo explorada pelo otimizador, não achado econômico.
        "pct_celulas_preco_e_volume_sobem": 100.0 * float(((dp > 0) & (dv > 0)).mean()),
        "pct_celulas_preco_e_volume_descem": 100.0 * float(((dp < 0) & (dv < 0)).mean()),
        # Resumo da POLITICA com sinal positivo. NAO e por si incoerencia: pode
        # ser composicao. Quem mede incoerencia e a identidade, abaixo.
        "pct_celulas_resposta_agregada_positiva": 100.0 * float(incoerentes.mean()),
        "resposta_agregada_da_politica_mediana": (float(np.median(implicita))
                                                  if implicita.size else float("nan")),
        "pct_celulas_com_razao_usavel": 100.0 * float(usavel.mean()),
        # Objeto 1: a coerencia do MODELO, imune a composicao (pendencia 2.8).
        "identidade_na_base": identidade_base,
        "identidade_no_otimo": identidade_otimo,
        "pct_violacao_identidade_base": identidade_base["pct_celulas_com_violacao"],
        "pct_violacao_identidade_otimo": identidade_otimo["pct_celulas_com_violacao"],
        "elast_agregada_uniforme_base": identidade_base["elast_agregada_mediana"],
        "elast_agregada_uniforme_otimo": identidade_otimo["elast_agregada_mediana"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--artefato", default=None)
    ap.add_argument("--celulas", type=int, default=200,
                    help="tamanho da amostra de células; 0 usa TODAS")
    ap.add_argument("--semente-amostra", type=int, default=0)
    ap.add_argument("--metodo", default="SLSQP",
                    help="SLSQP ou trust-constr (pontos interiores)")
    ap.add_argument("--hierarquia", default="nao_piorar",
                    choices=("nenhuma",) + HIER.FORMAS,
                    help="forma da hierarquia de marca (D39). `nao_piorar` e a "
                         "adotada; `nenhuma` reproduz o problema so com caixa")
    ap.add_argument("--bruto", default="data/raw")
    ap.add_argument("--comparar", action="store_true",
                    help="roda TODAS as formas de hierarquia na mesma amostra e "
                         "imprime a tabela de custo e beneficio de D39. E assim "
                         "que aquela tabela ganha produtor, em vez de sair de duas "
                         "execucoes separadas que podem diferir em amostra")
    ap.add_argument("--sufixo", default="")
    ap.add_argument("--saida", default="reports")
    a = ap.parse_args()

    painel = Path(a.painel)
    dados = rede.preparar(a.categoria, painel)
    arq = Path(a.artefato or RAIZ / "models_artifacts" / f"rede_{a.categoria}.pt")
    modelo, cfg, centro = carregar_modelo(arq, dados)

    esc = P.escopo(dados, painel, a.categoria, "teste")
    c = esc["contagens"]

    total = len(esc["u"])
    if a.celulas and a.celulas < total:
        rng = np.random.default_rng(a.semente_amostra)
        indices = np.sort(rng.choice(total, size=a.celulas, replace=False))
        amostra = {"tipo": "aleatoria_declarada", "n": int(a.celulas),
                   "semente": a.semente_amostra}
    else:
        indices = np.arange(total)
        amostra = {"tipo": "todas", "n": total}

    formas = (("nenhuma",) + HIER.FORMAS) if a.comparar else (a.hierarquia,)
    resultados, pares = [], []
    for forma in formas:
        restr = None
        if forma != "nenhuma":
            A, B, pares, _, inviavel = restricoes_do_escopo(
                a.categoria, painel, a.bruto, centro, esc, forma)
            restr = (A, B, inviavel)
        for obj in OBJETIVOS:
            linha = rodar(modelo, centro, esc, obj, indices, a.metodo, restr=restr)
            linha["hierarquia"] = forma
            resultados.append(linha)

    saida = {"categoria": a.categoria, "artefato": arq.name,
             "configuracao": cfg, "caixa": list(P.LIMITES_CAIXA),
             "hierarquia": "todas" if a.comparar else a.hierarquia,
             "formas": list(formas), "pares": [list(x) for x in pares],
             "escopo": c, "amostra": amostra,
             "resultados": resultados}

    # Projeção do custo da multipartida de D15, com o tempo medido aqui.
    mediana = max(r["tempo_por_solve_s"]["mediana"] for r in saida["resultados"])
    saida["projecao_d15"] = {
        "segundos_por_solve_mediana": mediana,
        "celulas_no_escopo": c["n_celulas_com_custo_completo"],
        "horas_uma_partida_dois_objetivos":
            2 * mediana * c["n_celulas_com_custo_completo"] / 3600.0,
        "horas_30_partidas_dois_objetivos":
            2 * 30 * mediana * c["n_celulas_com_custo_completo"] / 3600.0,
        "horas_50_partidas_dois_objetivos":
            2 * 50 * mediana * c["n_celulas_com_custo_completo"] / 3600.0,
    }

    rotulo = "todas" if a.comparar else a.hierarquia
    destino = (Path(a.saida)
               / f"sonda_item8_{a.categoria}_{rotulo}{a.sufixo}.json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))

    print(f"escopo (D38), partição {c['particao']}:")
    print(f"  células da partição            {c['n_celulas_da_particao']:>8d}")
    print(f"  com custo completo             {c['n_celulas_com_custo_completo']:>8d}")
    print(f"  excluídas por custo            {c['n_celulas_excluidas_por_custo']:>8d}"
          f"  ({c['pct_excluidas_por_custo']:.2f}%)")
    print(f"  semanas {c['semana_min']} a {c['semana_max']}, "
          f"{c['n_lojas']} lojas, margem relativa mediana "
          f"{c['margem_relativa_mediana']:.4f}")
    print(f"\namostra: {amostra['tipo']}, {amostra['n']} células, método {a.metodo}"
          f", hierarquia {rotulo} ({len(pares)} pares)\n")
    cab = (f"{'hierarquia':12s} {'objetivo':9s} {'s/solve':>8} {'sucesso':>8} "
           f"{'ganho ag.':>10} {'inferior':>9} {'superior':>9} {'interior':>9} "
           f"{'todas inf.':>11}")
    print(cab)
    for r in saida["resultados"]:
        print(f"{r['hierarquia']:12s} "
              f"{r['objetivo']:9s} {r['tempo_por_solve_s']['mediana']:8.4f} "
              f"{r['pct_sucesso']:7.1f}% {r['ganho_pct_agregado']:9.2f}% "
              f"{r['pct_coord_borda_inferior']:8.2f}% "
              f"{r['pct_coord_borda_superior']:8.2f}% "
              f"{r['pct_coord_interior']:8.2f}% "
              f"{r['pct_celulas_todas_na_borda_inferior']:10.2f}%")
    print("\ncoerência do MODELO (identidade de 2.8, subida uniforme, imune a composição):")
    print(f"  {'hierarquia':12s} {'objetivo':9s} {'violação base':>14} "
          f"{'violação ótimo':>15} {'ε agr. base':>12} {'ε agr. ótimo':>13}")
    for r in saida["resultados"]:
        print(f"  {r['hierarquia']:12s} "
              f"{r['objetivo']:9s} {r['pct_violacao_identidade_base']:13.2f}% "
              f"{r['pct_violacao_identidade_otimo']:14.2f}% "
              f"{r['elast_agregada_uniforme_base']:12.3f} "
              f"{r['elast_agregada_uniforme_otimo']:13.3f}")
    print("\nresumo da POLÍTICA (índice de preço ponderado pela base; não é elasticidade):")
    for r in saida["resultados"]:
        print(f"  {r['objetivo']:9s} preço {r['variacao_pct_preco_medio']['mediana']:+7.2f}%"
              f"  volume {r['variacao_pct_volume_total']['mediana']:+8.2f}%"
              f"  razão mediana {r['resposta_agregada_da_politica_mediana']:+7.2f}"
              f"  com sinal positivo {r['pct_celulas_resposta_agregada_positiva']:6.2f}%")
    if any(r["n_restricoes_lineares"] for r in saida["resultados"]):
        print("\nhierarquia de marca (D39):")
        for r in (x for x in saida["resultados"] if x["n_restricoes_lineares"]):
            print(f"  {r['hierarquia']:12s} {r['objetivo']:9s} "
                  f"{r['n_restricoes_lineares']} linhas"
                  f"  células com alguma ativa {r['pct_celulas_com_alguma_ativa']:6.2f}%"
                  f"  ativas por célula (mediana) {r['restricoes_ativas_por_celula']['mediana']:.1f}"
                  f"  violação máxima {r['violacao_maxima_observada']:.2e}"
                  f" = {100 * r['violacao_maxima_em_fracao_de_preco']:.4f}% do preço"
                  f"  fora da tolerância {r['pct_celulas_nao_viaveis_na_tolerancia']:5.2f}%"
                  f"{'  ACIMA DA TOLERÂNCIA' if r['acima_da_tolerancia'] else ''}"
                  f"  reserva {r['pct_celulas_com_metodo_de_reserva']:5.2f}%"
                  f"  INVIÁVEIS {r['pct_celulas_inviaveis_na_caixa']:5.2f}%")
    p = saida["projecao_d15"]
    print(f"\nprojeção de D15 no escopo inteiro ({p['celulas_no_escopo']} células, "
          "dois objetivos):")
    print(f"  1 partida    {p['horas_uma_partida_dois_objetivos']:8.2f} h")
    print(f"  30 partidas  {p['horas_30_partidas_dois_objetivos']:8.2f} h")
    print(f"  50 partidas  {p['horas_50_partidas_dois_objetivos']:8.2f} h")
    print(f"\n-> {destino}")


if __name__ == "__main__":
    main()
