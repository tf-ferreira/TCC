# shellcheck shell=bash
# Funções comuns aos scripts de execução. É carregado por eles, não é para rodar sozinho.
#
# Variáveis de ambiente aceitas por todos os scripts:
#   PYTHON          interpretador a usar (padrão: python3 do ambiente ativo)
#   TCC_CATEGORIA   categoria do Dominick's (padrão: frj, sucos congelados)
#   TCC_SIMULAR=1   só mostra os comandos, sem executar nada
#
# Compatível com o bash 3.2 que vem no macOS.

set -uo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RAIZ" || exit 1

PY="${PYTHON:-python3}"
CAT="${TCC_CATEGORIA:-frj}"
SIMULAR="${TCC_SIMULAR:-0}"
INICIO=$SECONDS

titulo() { printf '\n\033[1m>>> %s\033[0m\n' "$*"; }
falha()  { printf '\n\033[1;31m!!! %s\033[0m\n' "$*" >&2; exit 1; }

decorrido() {
  local s=$((SECONDS - INICIO))
  printf '%dh%02dmin%02ds' $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
}

# roda <comando...>: mostra o comando e o executa; com TCC_SIMULAR=1 só mostra.
roda() {
  printf '    $ %s\n' "$*"
  if [ "$SIMULAR" = "1" ]; then return 0; fi
  "$@" || falha "falhou: $*"
}

# avisa (sem interromper) quando nenhum ambiente virtual está ativo
confere_ambiente() {
  if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    echo "aviso: nenhum ambiente virtual ativo; usando $("$PY" -V 2>&1) em $(command -v "$PY")"
  fi
  echo "python   : $("$PY" -V 2>&1)"
  echo "categoria: $CAT"
}

# exige_arquivo <caminho...>: para com mensagem clara se faltar algum insumo
# (no modo de simulação só avisa, para que os comandos possam ser vistos antes)
exige_arquivo() {
  local f
  for f in "$@"; do
    if [ ! -e "$f" ]; then
      if [ "$SIMULAR" = "1" ]; then
        echo "aviso: falta $f (necessário para executar de fato)"
      else
        falha "falta $f (veja data/README.md e a ordem de execução no README)"
      fi
    fi
  done
}
