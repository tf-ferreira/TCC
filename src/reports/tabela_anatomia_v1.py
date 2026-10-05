"""As tabelas do objetivo 5 da v1 a partir da anatomia (D47).

Lê a saída de `src/optimization/anatomia_v1.py` e produz, para cada objetivo:

- a **avaliação principal** (D47): o ganho de uma semana, com a decisão otimizada e
  avaliada com as defasagens históricas (`estatica`);
- o **cenário "com o canal das defasagens"**: o sequencial de D40 (`sequencial`);
- nas duas, por política e mundo, o objetivo 5 (`C/R − 1`) e o ganho modelo contra
  modelo, com média, desvio e erro padrão entre sementes, o pior dos três mundos da
  média por semente, e a diferença **pareada** robusta − rede;
- o **canal das defasagens com a decisão fixa**: a decisão sequencial avaliada com a
  defasagem própria menos a mesma decisão avaliada com a histórica, em pontos de ganho;
- a calibração no histórico, derivada da identidade `1 + objetivo 5 = (1 + ganho) ×
  calibração`, conferida igual em todas as passadas, políticas e mundos.

Uso:
    python3 src/reports/tabela_anatomia_v1.py reports/anatomia_v1_frj_v1_n50.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MUNDOS = ["mu=0", "mu=0.5", "mu=1", "so_substitutos"]
MEDIA = ["mu=0", "mu=0.5", "mu=1"]
POLITICAS = ["rede", "robusta"]
OBJETIVOS = ["margem", "receita"]
PASSADAS = {"principal": "estatica", "com_canal": "sequencial"}


def resumo(x) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    dp = float(x.std(ddof=1)) if n > 1 else float("nan")
    ep = dp / np.sqrt(n) if n > 1 else float("nan")
    return {"media": float(x.mean()), "desvio": dp, "erro_padrao": ep,
            "minimo": float(x.min()), "maximo": float(x.max()), "n": int(n),
            "n_negativos": int((x < 0).sum()), "n_positivos": int((x > 0).sum())}


def pareada(a, b) -> dict:
    r = resumo(np.asarray(a, float) - np.asarray(b, float))
    r["z"] = (r["media"] / r["erro_padrao"]
              if r["erro_padrao"] and r["erro_padrao"] > 0 else float("nan"))
    return r


def _calibracao(x: dict) -> float:
    return (1.0 + x["objetivo5_pct"] / 100.0) / (1.0 + x["ganho_pct"] / 100.0)


def montar(bloco: dict) -> dict:
    L = bloco["por_semente"]
    fora = {"especificacao": bloco["especificacao"], "n_sementes": len(L),
            "n_elegiveis": bloco.get("n_elegiveis")}
    for obj in OBJETIVOS:
        t = {}
        # calibração: uma por semente, igual em toda passada, política e mundo
        cal = np.array([[_calibracao(l[obj][p][q][w]) for p in POLITICAS
                         for q in ("estatica", "sequencial", "estatica_com_defasagem_propria",
                                   "sequencial_com_defasagem_historica")
                         for w in MUNDOS] for l in L])
        if np.abs(cal - cal[:, :1]).max() > 1e-6:
            raise ValueError(f"{obj}: a calibração varia entre passadas, políticas ou mundos")
        t["calibracao"] = resumo(cal[:, 0])
        for nome, q in PASSADAS.items():
            tt = {}
            for p in POLITICAS:
                tt[p] = {w: {"objetivo5": resumo([l[obj][p][q][w]["objetivo5_pct"] for l in L]),
                             "ganho": resumo([l[obj][p][q][w]["ganho_pct"] for l in L])}
                         for w in MUNDOS}
                tt[p]["pior_dos_tres"] = resumo(
                    [min(l[obj][p][q][w]["objetivo5_pct"] for w in MEDIA) for l in L])
            tt["robusta_menos_rede"] = {
                w: pareada([l[obj]["robusta"][q][w]["objetivo5_pct"] for l in L],
                           [l[obj]["rede"][q][w]["objetivo5_pct"] for l in L])
                for w in MUNDOS}
            tt["robusta_menos_rede"]["pior_dos_tres"] = pareada(
                [min(l[obj]["robusta"][q][w]["objetivo5_pct"] for w in MEDIA) for l in L],
                [min(l[obj]["rede"][q][w]["objetivo5_pct"] for w in MEDIA) for l in L])
            t[nome] = tt
        # o canal das defasagens com a decisão sequencial fixa, e a adaptação da decisão
        t["canal_das_defasagens"] = {
            p: {w: pareada([l[obj][p]["sequencial"][w]["ganho_pct"] for l in L],
                           [l[obj][p]["sequencial_com_defasagem_historica"][w]["ganho_pct"] for l in L])
                for w in MUNDOS} for p in POLITICAS}
        t["adaptacao_da_decisao"] = {
            p: {w: pareada([l[obj][p]["sequencial_com_defasagem_historica"][w]["ganho_pct"] for l in L],
                           [l[obj][p]["estatica"][w]["ganho_pct"] for l in L])
                for w in MUNDOS} for p in POLITICAS}
        t["direcao"] = bloco.get("resumo", {}).get(obj, {})
        fora[obj] = t
    fora["propria_por_sku_media"] = bloco.get("resumo", {}).get("propria_por_sku_media")
    fora["repasse_primeiro_estagio"] = bloco.get("repasse_primeiro_estagio")
    # a resposta agregada às defasagens, por semente: é ela que o canal das defasagens usa
    fora["defasagem_sustentada"] = (resumo([l["defasagem_sustentada"] for l in L])
                                    if all("defasagem_sustentada" in l for l in L) else None)
    # a própria média dos produtos que não repassam custo (primeiro estágio sem instrumento)
    pi = bloco.get("repasse_primeiro_estagio")
    prop = fora["propria_por_sku_media"]
    if pi and prop:
        sem = [i for i, x in enumerate(pi) if x < 0.1]
        fora["propria_dos_sem_repasse"] = {"skus": sem, "propria": [prop[i] for i in sem],
                                           "faixa_dos_demais": [min(prop[i] for i in range(len(prop)) if i not in sem),
                                                                max(prop[i] for i in range(len(prop)) if i not in sem)]}
    return fora


def markdown(t: dict) -> str:
    def c(r):
        return f"{r['media']:.2f} ± {r['erro_padrao']:.2f}".replace(".", ",")
    out = [f"especificação {t['especificacao']}, {t['n_sementes']} sementes; objetivo 5 "
           "(C/R − 1, %), média ± erro padrão"]
    for obj in OBJETIVOS:
        out.append(f"\n**{obj}** (calibração no histórico {t[obj]['calibracao']['media']:.4f})")
        for nome in PASSADAS:
            tt = t[obj][nome]
            out.append(f"\n{nome}\n")
            out.append("| política | μ = 0 | μ = 0,5 | μ = 1 | só substitutos | pior dos três |")
            out.append("|---|---:|---:|---:|---:|---:|")
            for p in POLITICAS:
                out.append(f"| {p} | " + " | ".join(c(tt[p][w]["objetivo5"]) for w in MUNDOS)
                           + f" | {c(tt[p]['pior_dos_tres'])} |")
            d = tt["robusta_menos_rede"]
            out.append("| robusta − rede | " + " | ".join(
                f"{d[w]['media']:+.2f} (z {d[w]['z']:+.1f})".replace(".", ",")
                for w in MUNDOS + ["pior_dos_tres"]) + " |")
        out.append("\ncanal das defasagens, decisão sequencial fixa (pontos de ganho):")
        for p in POLITICAS:
            out.append(f"  {p}: " + "  ".join(
                f"{w} {t[obj]['canal_das_defasagens'][p][w]['media']:+.2f}"
                for w in MUNDOS))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("entrada")
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()
    entrada = Path(a.entrada)
    t = montar(json.loads(entrada.read_text()))
    destino = Path(a.saida) if a.saida else entrada.with_name(
        entrada.name.replace("anatomia_v1_", "tabela_anatomia_v1_"))
    destino.write_text(json.dumps(t, indent=2, ensure_ascii=False))
    print(markdown(t))
    print(f"\n-> {destino}")


if __name__ == "__main__":
    main()
