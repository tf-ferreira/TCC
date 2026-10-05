"""Compara o registro de números oficiais contra uma referência do git.

## Por que este arquivo existe

O projeto já tinha dois mecanismos de verdade numérica: `numeros_oficiais.py`,
que diz quais números existem e quem os produz, e `verificar_documentos.py`, que
pergunta se os documentos citam os números que os artefatos produzem. Faltava o
terceiro: **os artefatos voltam iguais quando tudo é refeito do zero?**

Sem ele, "reproduzível" é afirmação. Com ele, é medida, e a medida tem o mesmo
formato das outras: número, produtor, e o que invalidaria.

O caso concreto que motivou: o ambiente de trabalho passou a ser Python 3.14 na
máquina do autor, enquanto os artefatos commitados tinham sido produzidos em
Python 3.10 com outro conjunto de versões. Se algum número se move, é preciso
saber **qual** e **quanto**, e não apenas que o arquivo mudou.

## O que ele compara

Chave por chave, o campo `valor` de cada entrada, que é o número de ponto
flutuante antes de qualquer formatação. Compara também quais chaves apareceram e
quais sumiram, porque uma chave que some é tão grave quanto um valor que muda: o
produtor dela deixou de rodar.

O campo `gerado_em` é ignorado de propósito. Ele muda todo dia por construção e
não é resultado.

## Como ler a saída

Diferença **exata em zero** é o esperado num mesmo ambiente. Diferença pequena
mas não nula, na casa de 1e-12, costuma ser ordem de soma em ponto flutuante, e
aparece quando a versão de uma biblioteca muda a implementação de uma redução.
Diferença grande é mudança de resultado, e aí não é questão de ambiente.

Uso:
    python3 src/reports/comparar_registro.py                    # contra HEAD
    python3 src/reports/comparar_registro.py --ref HEAD~3
    python3 src/reports/comparar_registro.py --tol 1e-9 --falhar
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def carregar_do_git(ref: str, caminho: str) -> dict:
    """Lê a versão de um arquivo como ela está num commit, sem tocar na árvore."""
    try:
        bruto = subprocess.run(
            ["git", "show", f"{ref}:{caminho}"],
            capture_output=True, check=True, text=True).stdout
    except subprocess.CalledProcessError as erro:
        raise SystemExit(
            f"não consegui ler {caminho} em {ref}: {erro.stderr.strip()}")
    return json.loads(bruto)


def diferenca_relativa(a: float, b: float) -> float:
    """Relativa quando há escala, absoluta quando a referência é zero."""
    if b == 0.0:
        return abs(a - b)
    return abs(a - b) / abs(b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registro", default="reports/numeros_oficiais.json")
    ap.add_argument("--ref", default="HEAD",
                    help="commit de referência (padrão: HEAD)")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="diferença relativa tolerada (padrão: 0, exato)")
    ap.add_argument("--falhar", action="store_true",
                    help="sai com código 1 se houver diferença acima da tolerância")
    a = ap.parse_args()

    atual = json.loads(Path(a.registro).read_text())
    antes = carregar_do_git(a.ref, a.registro)

    ea, eb = atual["entradas"], antes["entradas"]
    sumiram = sorted(set(eb) - set(ea))
    surgiram = sorted(set(ea) - set(eb))
    comuns = sorted(set(ea) & set(eb))

    mudaram = []
    for chave in comuns:
        va, vb = float(ea[chave]["valor"]), float(eb[chave]["valor"])
        d = diferenca_relativa(va, vb)
        if d > a.tol:
            mudaram.append((chave, vb, va, d, ea[chave]["descricao"]))

    print(f"registro atual : {a.registro} (gerado em {atual['gerado_em']})")
    print(f"referência     : {a.ref} (gerado em {antes['gerado_em']})")
    print(f"chaves         : {len(ea)} agora, {len(eb)} na referência, "
          f"{len(comuns)} em comum")
    print(f"tolerância     : {a.tol:g} (relativa)\n")

    if sumiram:
        print(f"!! {len(sumiram)} chave(s) SUMIRAM. O produtor delas não rodou, "
              f"ou mudou de campo:")
        for c in sumiram:
            print(f"    {c:28s} {eb[c]['texto']:>12s}  {eb[c]['produtor']}")
        print()

    if surgiram:
        print(f"   {len(surgiram)} chave(s) novas desde a referência:")
        for c in surgiram:
            print(f"    {c:28s} {ea[c]['texto']:>12s}  {ea[c]['descricao']}")
        print()

    if mudaram:
        print(f"!! {len(mudaram)} número(s) MUDARAM:\n")
        print(f"    {'chave':28s} {'referência':>16s} {'agora':>16s} {'dif. rel.':>12s}")
        for chave, vb, va, d, desc in sorted(mudaram, key=lambda x: -x[3]):
            print(f"    {chave:28s} {vb:16.8g} {va:16.8g} {d:12.2e}")
            print(f"    {'':28s} {desc}")
    else:
        print(f"nenhum número mudou: os {len(comuns)} valores comuns batem "
              f"dentro da tolerância.")

    problemas = len(sumiram) + len(mudaram)
    print(f"\nresumo: {len(mudaram)} número(s) fora da tolerância, "
          f"{len(sumiram)} chave(s) perdida(s), {len(surgiram)} nova(s).")
    if a.falhar and problemas:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
