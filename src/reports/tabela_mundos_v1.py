"""A tabela política × mundo do objetivo 5 (D44), com as diferenças pareadas.

Lê a saída de `src/optimization/sementes_v1.py` e produz, para cada objetivo:

- por política e mundo: o objetivo 5 (`C/R − 1`), o ganho modelo contra modelo
  (`C/M̂ − 1`) e a calibração no histórico (`M̂/R`), com média, desvio e erro
  padrão entre sementes;
- a diferença **pareada por semente** robusta − rede em cada mundo, com z;
- o pior caso de cada política entre os três mundos da média (μ = 0; 0,5; 1).

A diferença pareada é o que D26 pede para comparar duas políticas: as duas
decidem sobre a mesma rede treinada em cada semente, e o ruído de treino que elas
compartilham sai da comparação.

A calibração é a mesma em todos os mundos por construção (a âncora de
`mundos.py`), e o script confere em vez de assumir.

Uso:
    python3 src/reports/tabela_mundos_v1.py reports/sementes_v1_frj_v1_nao_piorar_n50.json
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
CAMPOS = {"objetivo5": "objetivo5_contra_o_observado_pct",
          "ganho": "ganho_modelo_contra_modelo_pct",
          "calibracao": "calibracao_no_historico"}


def resumo(x) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    dp = float(x.std(ddof=1)) if n > 1 else float("nan")
    ep = dp / np.sqrt(n) if n > 1 else float("nan")
    return {"media": float(x.mean()), "desvio": dp, "erro_padrao": ep,
            "minimo": float(x.min()), "maximo": float(x.max()), "n": int(n)}


def pareada(a, b) -> dict:
    r = resumo(np.asarray(a, float) - np.asarray(b, float))
    r["z"] = (r["media"] / r["erro_padrao"]
              if r["erro_padrao"] and r["erro_padrao"] > 0 else float("nan"))
    return r


def montar(bloco: dict) -> dict:
    linhas = bloco["por_semente"]
    fora = {"especificacao": bloco["especificacao"], "n_sementes": len(linhas),
            "hierarquia": bloco.get("hierarquia")}
    for obj in OBJETIVOS:
        t = {}
        for pol in POLITICAS:
            t[pol] = {}
            for w in MUNDOS:
                t[pol][w] = {nome: resumo([l[obj][pol][w][campo] for l in linhas])
                             for nome, campo in CAMPOS.items()}
            # pior caso POR SEMENTE entre os mundos da média, depois a média dele
            t[pol]["pior_dos_tres"] = resumo(
                [min(l[obj][pol][w][CAMPOS["objetivo5"]] for w in MEDIA) for l in linhas])
        # a calibração não pode variar entre mundos nem entre políticas
        cal = np.array([[l[obj][pol][w][CAMPOS["calibracao"]] for pol in POLITICAS
                         for w in MUNDOS] for l in linhas])
        if np.abs(cal - cal[:, :1]).max() > 1e-9:
            raise ValueError(f"{obj}: a calibração varia entre mundos ou políticas")
        t["robusta_menos_rede"] = {
            w: pareada([l[obj]["robusta"][w][CAMPOS["objetivo5"]] for l in linhas],
                       [l[obj]["rede"][w][CAMPOS["objetivo5"]] for l in linhas])
            for w in MUNDOS}
        t["robusta_menos_rede"]["pior_dos_tres"] = pareada(
            [min(l[obj]["robusta"][w][CAMPOS["objetivo5"]] for w in MEDIA) for l in linhas],
            [min(l[obj]["rede"][w][CAMPOS["objetivo5"]] for w in MEDIA) for l in linhas])
        fora[obj] = t
    return fora


def markdown(t: dict) -> str:
    def c(r):
        return f"{r['media']:.2f} ± {r['erro_padrao']:.2f}".replace(".", ",")
    out = [f"especificação {t['especificacao']}, {t['n_sementes']} sementes; "
           "objetivo 5 (C/R − 1, %), média ± erro padrão\n"]
    for obj in OBJETIVOS:
        out.append(f"\n**{obj}** (calibração no histórico {t[obj]['rede']['mu=1']['calibracao']['media']:.4f})\n")
        out.append("| política | μ = 0 | μ = 0,5 | μ = 1 | só substitutos | pior dos três |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for pol in POLITICAS:
            out.append(f"| {pol} | " + " | ".join(c(t[obj][pol][w]["objetivo5"]) for w in MUNDOS)
                       + f" | {c(t[obj][pol]['pior_dos_tres'])} |")
        d = t[obj]["robusta_menos_rede"]
        out.append("| robusta − rede (pareado) | " + " | ".join(
            f"{d[w]['media']:+.2f} (z {d[w]['z']:+.1f})".replace(".", ",") for w in MUNDOS)
            + f" | {d['pior_dos_tres']['media']:+.2f} (z {d['pior_dos_tres']['z']:+.1f})".replace(".", ",")
            + " |")
        out.append("\nganho modelo contra modelo (C/M̂ − 1):")
        for pol in POLITICAS:
            out.append(f"  {pol}: " + "  ".join(
                f"{w} {t[obj][pol][w]['ganho']['media']:.2f}" for w in MUNDOS))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("entrada")
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()
    entrada = Path(a.entrada)
    t = montar(json.loads(entrada.read_text()))
    destino = Path(a.saida) if a.saida else entrada.with_name(
        entrada.name.replace("sementes_v1_", "tabela_mundos_v1_"))
    destino.write_text(json.dumps(t, indent=2, ensure_ascii=False))
    print(markdown(t))
    print(f"\n-> {destino}")


if __name__ == "__main__":
    main()
