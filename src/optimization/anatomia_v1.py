"""Anatomia da política da v1: o que ela faz com os preços, e a miopia medida.

Complementa a R3 (`sementes_v1.py`), que guarda só os agregados do objetivo 5.
Por semente, na especificação pedida (v1 por padrão) e com a hierarquia "não
piorar" de D39:

1. **Duas avaliações da mesma decisão.** A passada **estática** resolve cada
   célula com a defasagem histórica: é o problema de uma semana que o otimizador
   de fato resolve, e nela a política da rede é, por construção, a melhor no
   mundo μ = 1, a menos de ótimo local. A passada **sequencial** é a de D27, a
   que dá o objetivo 5. Se a política robusta ganhar da política da rede em μ = 1
   só na sequencial, a diferença é o custo da miopia; se ganhar também na
   estática, é o solver que encontra ótimos locais piores com o objetivo da rede.
   Duas avaliações cruzadas separam o **canal das defasagens** da adaptação da
   decisão: as decisões estáticas avaliadas com a defasagem que elas mesmas
   produziriam (o B do item 9), e as sequenciais avaliadas com a defasagem
   histórica. Com a decisão fixa, a diferença entre as duas avaliações é o quanto
   do ganho passa pela resposta da rede às defasagens, que o item 9 mostrou não
   ser identificada; por isso a resposta sustentada às defasagens também é
   gravada por semente.
2. **A direção.** Por SKU e por tipo de marca: variação média do log preço
   contra o histórico, fração de coordenadas que sobem, que descem e que param
   em cada borda da caixa, nas células elegíveis da passada sequencial.
3. **O suporte.** Fração dos preços recomendados acima do maior preço que o
   mesmo produto teve na mesma loja na janela de treino. É onde a rede
   extrapola, e onde a calibração de D40 não tem como ser conferida.
4. **A própria por SKU**, mediana no teste, ao lado do repasse do primeiro
   estágio de D42: nos SKUs sem repasse, a própria vem da forma funcional e da
   regularização, e não do instrumento.

Nada aqui muda decisão: é descrição da política que a R3 avalia, para o texto.

Uso:
    python3 src/optimization/anatomia_v1.py frj --sementes 1 --sufixo _sonda
    python3 src/optimization/anatomia_v1.py frj --sementes 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
LOG_BAIXO, LOG_ALTO = float(np.log(0.85)), float(np.log(1.15))


def direcao(du: np.ndarray, baixo: float = LOG_BAIXO, alto: float = LOG_ALTO,
            tol: float = 1e-4) -> dict:
    """Resumo da variação `du = u_política − u_histórico`, por coluna e no total.

    `du` tem uma linha por célula e uma coluna por SKU. Uma coordenada "sobe" se
    `du > tol`, "desce" se `du < −tol`, e está numa borda se fica a menos de
    `tol` dela. A caixa de D38 é ancorada no preço histórico da célula, então as
    bordas em `du` são as mesmas em toda célula.
    """
    du = np.asarray(du, float)
    if du.ndim != 2:
        raise ValueError("du precisa ser (células, SKUs)")
    # Registrada e não levantada: uma exceção aqui mataria uma rodada de horas
    # por uma violação que o texto pode simplesmente declarar.
    fora_da_caixa = float(max(0.0, baixo - du.min(), du.max() - alto))
    sobe, desce = du > tol, du < -tol
    b_alto, b_baixo = np.abs(du - alto) < tol, np.abs(du - baixo) < tol

    def _frac(m, eixo=None):
        return m.mean(axis=eixo)

    return {
        "media_por_sku": du.mean(0).tolist(),
        "sobe_por_sku": _frac(sobe, 0).tolist(),
        "desce_por_sku": _frac(desce, 0).tolist(),
        "borda_alta_por_sku": _frac(b_alto, 0).tolist(),
        "borda_baixa_por_sku": _frac(b_baixo, 0).tolist(),
        "media": float(du.mean()),
        "sobe": float(_frac(sobe)),
        "desce": float(_frac(desce)),
        "borda_alta": float(_frac(b_alto)),
        "borda_baixa": float(_frac(b_baixo)),
        "interior": float(1.0 - _frac(b_alto) - _frac(b_baixo)),
        "maior_saida_da_caixa": fora_da_caixa,
    }


def tetos_do_treino(cel: pd.DataFrame, n: int, ultima_semana: int,
                    chaves: pd.DataFrame) -> np.ndarray:
    """O maior preço de cada SKU em cada loja na janela de treino, alinhado a `chaves`.

    Devolve uma matriz (linhas de `chaves`, n). Loja que não aparece no treino
    recebe o maior preço do SKU entre as lojas do treino, e isso é declarado no
    resultado, porque é o análogo do índice de reserva do embedding.
    """
    tr = cel[cel["week"] <= ultima_semana]
    cols = [f"preco_{i:02d}" for i in range(n)]
    precos = tr[["store"] + cols].copy()
    precos[cols] = precos[cols].where(precos[cols] > 0)
    por_loja = precos.groupby("store")[cols].max()
    geral = precos[cols].max()
    alinhado = por_loja.reindex(chaves["store"].to_numpy())
    alinhado = alinhado.fillna(geral)
    return alinhado.to_numpy(float)


def acima_do_teto(p: np.ndarray, teto: np.ndarray, rel: float = 1e-9) -> dict:
    """Fração de coordenadas com preço acima do teto do treino, por SKU e no total."""
    p, teto = np.asarray(p, float), np.asarray(teto, float)
    if p.shape != teto.shape:
        raise ValueError("preço e teto com formas diferentes")
    acima = p > teto * (1.0 + rel)
    return {"por_sku": acima.mean(0).tolist(), "total": float(acima.mean())}


def _medias(linhas: list, chave) -> dict:
    x = np.array([chave(l) for l in linhas], float)
    n = len(x)
    dp = float(x.std(ddof=1)) if n > 1 else float("nan")
    return {"media": float(x.mean()), "desvio": dp,
            "erro_padrao": dp / np.sqrt(n) if n > 1 else float("nan"), "n": n}


def main() -> None:
    for sub in ("models", "data", "experiments", "optimization"):
        sys.path.insert(0, str(RAIZ / "src" / sub))
    import contrafactual_item9 as C9
    import diagnostico_item9 as D9
    import hierarquia as HIER
    import mundos as MU
    import problema as P
    import rede
    import torch
    from sementes_v1 import objetivo5_no_mundo
    from sonda_item8 import OBJETIVOS, restricoes_do_escopo
    from varredura_rede import MELHOR

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--bruto", default="data/raw")
    ap.add_argument("--especificacao", default="v1", choices=sorted(rede.ESPECIFICACOES))
    ap.add_argument("--sementes", type=int, default=5)
    ap.add_argument("--semente-inicial", type=int, default=0)
    ap.add_argument("--metodo", default="SLSQP")
    ap.add_argument("--saida", default="reports")
    ap.add_argument("--sufixo", default="")
    a = ap.parse_args()

    rede.usar_especificacao(a.especificacao)
    painel = Path(a.painel)
    dados = rede.preparar(a.categoria, painel)
    centro = np.asarray(dados["centro_de_preco"], float)
    esc = P.escopo(dados, painel, a.categoria, "teste")
    n = esc["n"]
    esc["escalas_dos_lags"] = C9.escalas_dos_lags(dados, n)
    mapa = C9.mapa_das_corridas(esc["chaves"])
    A, B, pares, _, inviavel = restricoes_do_escopo(
        a.categoria, painel, a.bruto, centro, esc, "nao_piorar")
    restr = (A, B, inviavel)
    S = mapa["indices_elegiveis"]

    info = HIER.identificar(a.categoria, painel, a.bruto).sort_values("pos")
    propria = info["marca_propria"].to_numpy(bool)
    cel = pd.read_parquet(painel / f"{a.categoria}_celulas.parquet")
    ultima = int(dados["meta"]["janela_treino"][1])
    teto = tetos_do_treino(cel, n, ultima, esc["chaves"])[S]
    lojas_sem_treino = int((~np.isin(esc["chaves"]["store"].to_numpy()[S],
                                     cel.loc[cel["week"] <= ultima, "store"].unique())).sum())
    receita_hist = (esc["precos"][S] * esc["alvo"][S]).sum(0)
    peso = receita_hist / receita_hist.sum()

    print(f"especificação {a.especificacao}; {len(S)} elegíveis; {len(pares)} pares; "
          f"{propria.sum()} SKUs de marca própria\n", flush=True)
    linhas, t0 = [], time.perf_counter()
    for i in range(a.sementes):
        s = a.semente_inicial + i
        cfg = dict(MELHOR, semente=s, restrita=True, especificacao=a.especificacao)
        cfg.setdefault("gama_contextual", False)
        t = time.perf_counter()
        modelo = rede.treinar(dados, cfg)["modelo"].eval()
        decisores = {"rede": modelo, "robusta": MU.politica_robusta_v1(modelo)}
        avaliadores = MU.mundos_v1(modelo)
        te = dados["teste"]
        with torch.no_grad():
            eps = modelo.elasticidades_fechadas(torch.tensor(te["u"], dtype=torch.float32))
        propria_sku = torch.diagonal(eps, dim1=1, dim2=2).median(0).values.numpy()
        linha = {"semente": s, "propria_por_sku": propria_sku.tolist()}
        # A resposta à defasagem, como no item 9: a elasticidade da demanda total a
        # uma variação uniforme das defasagens, nas células elegíveis.
        for chave in ("lag1", "lag2"):
            J, V = D9.jacobiana_da_defasagem(modelo, esc, S, chave)
            linha[f"defasagem_{chave}_agregada"] = D9.resumo_da_jacobiana(J, V)["agregada_uniforme_ponderada"]
        linha["defasagem_sustentada"] = (linha["defasagem_lag1_agregada"]
                                         + linha["defasagem_lag2_agregada"])
        for obj in OBJETIVOS:
            custos = esc["custos"] if obj == "margem" else np.zeros_like(esc["custos"])
            linha[obj] = {}
            for pol in ("rede", "robusta"):
                linha[obj][pol] = {}
                passadas = {}
                for nome_passada, seq in (("estatica", False), ("sequencial", True)):
                    ps = C9.rodar_passada(decisores[pol], centro, esc, mapa, custos,
                                          restr, 1, 0, a.metodo, seq, progresso=0)
                    passadas[nome_passada] = ps
                    aval = {w: objetivo5_no_mundo(m, centro, esc, mapa, custos, ps)
                            for w, m in avaliadores.items()}
                    bloco = {w: {"ganho_pct": x["ganho_modelo_contra_modelo_pct"],
                                 "objetivo5_pct": x["objetivo5_contra_o_observado_pct"]}
                             for w, x in aval.items()}
                    # direção e suporte nas duas passadas: a estática é a decisão do número
                    # principal (D47), a sequencial a do cenário com o canal das defasagens
                    du = ps["u"][S] - esc["u"][S]
                    d = direcao(du)
                    d["media_ponderada_pela_receita"] = float((du.mean(0) * peso).sum())
                    d["propria"] = direcao(du[:, propria])
                    d["nacional"] = direcao(du[:, ~propria])
                    pol_p = np.exp(ps["u"][S] + centro)
                    bloco["direcao"] = d
                    bloco["acima_do_teto_do_treino"] = acima_do_teto(pol_p, teto)
                    linha[obj][pol][nome_passada] = bloco
                # O canal das defasagens com a DECISÃO FIXA: as decisões estáticas
                # avaliadas com a defasagem que elas mesmas produziriam (o B do
                # item 9) e as sequenciais avaliadas com a defasagem histórica.
                est, sq = passadas["estatica"], passadas["sequencial"]
                cruzadas = {
                    "estatica_com_defasagem_propria": {
                        "ctx": C9.ctx_honesto_da_politica(esc, mapa, est["u"], centro),
                        "u": est["u"]},
                    "sequencial_com_defasagem_historica": {"ctx": esc["ctx"], "u": sq["u"]}}
                for nome_c, ps in cruzadas.items():
                    linha[obj][pol][nome_c] = {
                        w: {"ganho_pct": x["ganho_modelo_contra_modelo_pct"],
                            "objetivo5_pct": x["objetivo5_contra_o_observado_pct"]}
                        for w, x in ((w, objetivo5_no_mundo(m, centro, esc, mapa, custos, ps))
                                     for w, m in avaliadores.items())}
        linha["tempo_s"] = time.perf_counter() - t
        linhas.append(linha)
        m = linha["margem"]; r = linha["receita"]
        print(f"  semente {s}: μ=1 estática margem rede {m['rede']['estatica']['mu=1']['ganho_pct']:.2f}"
              f" robusta {m['robusta']['estatica']['mu=1']['ganho_pct']:.2f}; sequencial rede "
              f"{m['rede']['sequencial']['mu=1']['ganho_pct']:.2f} robusta "
              f"{m['robusta']['sequencial']['mu=1']['ganho_pct']:.2f} | receita estática rede "
              f"{r['rede']['estatica']['mu=1']['ganho_pct']:.2f} robusta "
              f"{r['robusta']['estatica']['mu=1']['ganho_pct']:.2f}; sequencial rede "
              f"{r['rede']['sequencial']['mu=1']['ganho_pct']:.2f} robusta "
              f"{r['robusta']['sequencial']['mu=1']['ganho_pct']:.2f} | sobe (robusta, receita) "
              f"{r['robusta']['sequencial']['direcao']['sobe']:.2f} ({linha['tempo_s']:.0f} s)",
              flush=True)

    resumo = {}
    for obj in OBJETIVOS:
        resumo[obj] = {}
        for pol in ("rede", "robusta"):
            resumo[obj][pol] = {}
            for pas in ("estatica", "sequencial"):
                resumo[obj][pol][pas] = {
                    w: _medias(linhas, lambda l, o=obj, p=pol, q=pas, w=w: l[o][p][q][w]["ganho_pct"])
                    for w in MU.NOMES_V1}
            seqs = [l[obj][pol]["sequencial"] for l in linhas]
            resumo[obj][pol]["direcao"] = {
                k: _medias(seqs, lambda x, k=k: x["direcao"][k])
                for k in ("media", "media_ponderada_pela_receita", "sobe", "desce",
                          "borda_alta", "borda_baixa", "interior")}
            resumo[obj][pol]["direcao_propria_sobe"] = _medias(seqs, lambda x: x["direcao"]["propria"]["sobe"])
            resumo[obj][pol]["direcao_nacional_sobe"] = _medias(seqs, lambda x: x["direcao"]["nacional"]["sobe"])
            resumo[obj][pol]["acima_do_teto"] = _medias(seqs, lambda x: x["acima_do_teto_do_treino"]["total"])
            # as mesmas estatísticas na passada estática (D47), com o sufixo "_estatica"
            ests = [l[obj][pol]["estatica"] for l in linhas]
            resumo[obj][pol]["direcao_estatica"] = {
                k: _medias(ests, lambda x, k=k: x["direcao"][k])
                for k in ("media", "media_ponderada_pela_receita", "sobe", "desce",
                          "borda_alta", "borda_baixa", "interior")}
            resumo[obj][pol]["direcao_propria_sobe_estatica"] = _medias(
                ests, lambda x: x["direcao"]["propria"]["sobe"])
            resumo[obj][pol]["direcao_nacional_sobe_estatica"] = _medias(
                ests, lambda x: x["direcao"]["nacional"]["sobe"])
            resumo[obj][pol]["acima_do_teto_estatica"] = _medias(
                ests, lambda x: x["acima_do_teto_do_treino"]["total"])
        # o canal das defasagens, por política, em μ = 1: sequencial − (sequencial
        # com defasagem histórica), com a decisão sequencial fixa
        for pol in ("rede", "robusta"):
            resumo[obj][pol]["canal_das_defasagens_mu1"] = _medias(
                linhas, lambda l, o=obj, p=pol: l[o][p]["sequencial"]["mu=1"]["ganho_pct"]
                - l[o][p]["sequencial_com_defasagem_historica"]["mu=1"]["ganho_pct"])
        # a miopia, pareada por semente: robusta − rede em μ = 1, nas duas passadas
        for pas in ("estatica", "sequencial"):
            resumo[obj][f"robusta_menos_rede_mu1_{pas}"] = _medias(
                linhas, lambda l, o=obj, q=pas: l[o]["robusta"][q]["mu=1"]["ganho_pct"]
                - l[o]["rede"][q]["mu=1"]["ganho_pct"])
    resumo["propria_por_sku_media"] = np.mean([l["propria_por_sku"] for l in linhas], 0).tolist()
    resumo["defasagem_sustentada"] = _medias(linhas, lambda l: l["defasagem_sustentada"])
    fc = dados["meta"].get("funcao_de_controle") or {}

    bloco = {"categoria": a.categoria, "especificacao": a.especificacao,
             "hierarquia": "nao_piorar", "n_sementes": a.sementes,
             "n_elegiveis": int(len(S)), "marca_propria": propria.tolist(),
             "peso_receita_historica": peso.tolist(),
             "celulas_elegiveis_de_loja_sem_treino": lojas_sem_treino,
             "repasse_primeiro_estagio": fc.get("pi"), "f_primeiro_estagio": fc.get("f_parcial"),
             "resumo": resumo, "por_semente": linhas,
             "tempo_total_min": (time.perf_counter() - t0) / 60.0}
    destino = Path(a.saida) / f"anatomia_v1_{a.categoria}_{a.especificacao}_n{a.sementes}{a.sufixo}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(bloco, indent=2, ensure_ascii=False))
    print(f"\ntempo total {bloco['tempo_total_min']:.1f} min\n-> {destino}")


if __name__ == "__main__":
    main()
