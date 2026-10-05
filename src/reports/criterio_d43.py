"""O critério de D43: o código de promoção muda o objetivo 5 em 2 pontos ou mais?

D43 declarou, antes da rodada, que o código vira dimensão de cenário da faixa se o
objetivo 5 da política robusta, no mundo μ = 1, mudar 2 pontos ou mais em qualquer dos
dois objetivos entre `v1_codigo` e `v1`; se não, ele fica como linha de sensibilidade.

A comparação é **pareada por semente**: a mesma semente inicializa as duas redes, e a
diferença de cada semente tira o ruído que as duas especificações compartilham.

São duas bases, porque D47 mudou o número principal depois de D43 ter sido declarada:

- `sequencial`: a avaliação de D40, a única que a R3b (`sementes_v1.py`) mede;
- `estatica`: a avaliação principal de D47, o ganho de uma semana, que vem da anatomia
  (`anatomia_v1.py`) das duas especificações. Sem a anatomia da `v1_codigo`, ela sai
  como pendente.

Uso:
    python3 src/reports/criterio_d43.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MUNDOS = ["mu=0", "mu=0.5", "mu=1", "so_substitutos"]
POLITICAS = ["rede", "robusta"]
OBJETIVOS = ["margem", "receita"]
LIMIAR = 2.0


def pareada(a: dict, b: dict) -> dict:
    """Diferença a − b nas sementes comuns, com média, erro padrão e z."""
    comuns = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in comuns], float)
    n = len(d)
    ep = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    media = float(d.mean()) if n else float("nan")
    return {"media": media, "erro_padrao": ep,
            "z": media / ep if n > 1 and ep > 0 else float("nan"),
            "n": n, "sementes": comuns}


def por_semente_sementes(bloco: dict, obj: str, pol: str, mundo: str) -> dict:
    return {l["semente"]: l[obj][pol][mundo]["objetivo5_contra_o_observado_pct"]
            for l in bloco["por_semente"]}


def por_semente_anatomia(bloco: dict, obj: str, pol: str, passada: str, mundo: str) -> dict:
    return {l["semente"]: l[obj][pol][passada][mundo]["objetivo5_pct"]
            for l in bloco["por_semente"]}


def decidir(base: dict) -> dict:
    """Aplica o critério declarado na robusta em μ = 1."""
    difs = {obj: base[obj]["robusta"]["mu=1"]["media"] for obj in OBJETIVOS}
    passa = any(abs(v) >= LIMIAR for v in difs.values())
    return {"diferencas_robusta_mu1": difs, "limiar": LIMIAR,
            "passa": passa, "leitura": "cenario" if passa else "sensibilidade"}


def comparar(fn_codigo, fn_v1) -> dict:
    base = {obj: {pol: {w: pareada(fn_codigo(obj, pol, w), fn_v1(obj, pol, w))
                        for w in MUNDOS} for pol in POLITICAS} for obj in OBJETIVOS}
    base["criterio"] = decidir(base)
    return base


def montar(v1_sem: dict, cod_sem: dict, v1_anat: dict | None, cod_anat: dict | None) -> dict:
    fora = {"limiar_pontos": LIMIAR, "bases": {}}
    fora["bases"]["sequencial"] = comparar(
        lambda o, p, w: por_semente_sementes(cod_sem, o, p, w),
        lambda o, p, w: por_semente_sementes(v1_sem, o, p, w))
    fora["bases"]["sequencial"]["produtor"] = "sementes_v1.py (R3 e R3b)"
    # de onde vem a diferença: 1 + objetivo 5 = (1 + ganho) × calibração
    campo = lambda bloco, obj, k: {l["semente"]: l[obj]["robusta"]["mu=1"][k] for l in bloco["por_semente"]}
    fora["bases"]["sequencial"]["decomposicao_robusta_mu1"] = {
        obj: {"ganho_pontos": pareada(campo(cod_sem, obj, "ganho_modelo_contra_modelo_pct"),
                                      campo(v1_sem, obj, "ganho_modelo_contra_modelo_pct")),
              "calibracao": pareada(campo(cod_sem, obj, "calibracao_no_historico"),
                                    campo(v1_sem, obj, "calibracao_no_historico"))}
        for obj in OBJETIVOS}
    if v1_anat is not None and cod_anat is not None:
        fora["bases"]["estatica"] = comparar(
            lambda o, p, w: por_semente_anatomia(cod_anat, o, p, "estatica", w),
            lambda o, p, w: por_semente_anatomia(v1_anat, o, p, "estatica", w))
        fora["bases"]["estatica"]["produtor"] = "anatomia_v1.py (noite2 e noite3)"
        # o canal das defasagens (decisão sequencial fixa), com e sem o código, nas
        # mesmas sementes: onde mora o efeito do código
        def canal(bloco, obj, pol, w):
            return {l["semente"]: l[obj][pol]["sequencial"][w]["ganho_pct"]
                    - l[obj][pol]["sequencial_com_defasagem_historica"][w]["ganho_pct"]
                    for l in bloco["por_semente"]}
        fora["canal_das_defasagens"] = {
            obj: {pol: {w: {"v1": pareada(canal(v1_anat, obj, pol, w), {k: 0.0 for k in canal(cod_anat, obj, pol, w)}),
                            "v1_codigo": pareada(canal(cod_anat, obj, pol, w), {k: 0.0 for k in canal(v1_anat, obj, pol, w)}),
                            "codigo_menos_v1": pareada(canal(cod_anat, obj, pol, w), canal(v1_anat, obj, pol, w))}
                        for w in MUNDOS} for pol in POLITICAS} for obj in OBJETIVOS}
    else:
        fora["bases"]["estatica"] = {"pendente": "falta a anatomia da v1_codigo (noite3)"}
    # a passada sequencial da anatomia tem de reproduzir a R3, semente a semente
    if v1_anat is not None:
        dif = []
        for obj in OBJETIVOS:
            for pol in POLITICAS:
                for w in MUNDOS:
                    a = por_semente_anatomia(v1_anat, obj, pol, "sequencial", w)
                    b = por_semente_sementes(v1_sem, obj, pol, w)
                    dif += [abs(a[s] - b[s]) for s in set(a) & set(b)]
        fora["anatomia_reproduz_r3_max_dif_pontos"] = float(max(dif)) if dif else None
    return fora


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--categoria", default="frj")
    ap.add_argument("--reports", default="reports")
    a = ap.parse_args()
    r = Path(a.reports)
    ler = lambda nome: json.loads((r / nome).read_text()) if (r / nome).exists() else None
    v1_sem = ler(f"sementes_v1_{a.categoria}_v1_nao_piorar_n50.json")
    cod_sem = ler(f"sementes_v1_{a.categoria}_v1_codigo_nao_piorar_n20.json")
    v1_anat = ler(f"anatomia_v1_{a.categoria}_v1_n50.json")
    cod_anat = ler(f"anatomia_v1_{a.categoria}_v1_codigo_n20.json")
    if v1_sem is None or cod_sem is None:
        raise SystemExit("faltam as saídas da R3 ou da R3b")
    t = montar(v1_sem, cod_sem, v1_anat, cod_anat)
    destino = r / f"criterio_d43_{a.categoria}.json"
    destino.write_text(json.dumps(t, indent=2, ensure_ascii=False))
    for nome, base in t["bases"].items():
        if "pendente" in base:
            print(f"{nome}: pendente ({base['pendente']})")
            continue
        print(f"{nome}: v1_codigo − v1, objetivo 5 pareado, {base['margem']['robusta']['mu=1']['n']} sementes")
        for obj in OBJETIVOS:
            for pol in POLITICAS:
                print(f"  {obj:7s} {pol:7s} " + "  ".join(
                    f"{w} {base[obj][pol][w]['media']:+6.2f} (ep {base[obj][pol][w]['erro_padrao']:.2f})"
                    for w in MUNDOS))
        c = base["criterio"]
        print(f"  critério (robusta, μ = 1, limiar {LIMIAR:g}): {c['diferencas_robusta_mu1']} -> {c['leitura']}")
    if t.get("anatomia_reproduz_r3_max_dif_pontos") is not None:
        print(f"a sequencial da anatomia reproduz a R3 com diferença máxima de "
              f"{t['anatomia_reproduz_r3_max_dif_pontos']:.2e} ponto")
    print(f"-> {destino}")


if __name__ == "__main__":
    main()
