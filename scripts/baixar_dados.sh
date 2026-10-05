#!/usr/bin/env bash
# Baixa os dados brutos do Dominick's Finer Foods do site do Kilts Center for
# Marketing (Chicago Booth) e confere o SHA-256 de cada arquivo contra o dos
# arquivos que produziram os resultados deste repositório.
#
# Os dados não são redistribuídos aqui: o Kilts Center os libera "para fins de
# pesquisa acadêmica" e pede citação. Ver data/README.md.
#
# Uso:
#   bash scripts/baixar_dados.sh             # o que o pipeline do TCC usa (~63 MB)
#   bash scripts/baixar_dados.sh --triagem   # mais as 26 outras categorias (~830 MB),
#                                            # só para refazer a triagem de categorias
#
# Se o download automático falhar, baixe à mão na página oficial e coloque cada
# arquivo no caminho indicado em data/README.md:
#   https://www.chicagobooth.edu/research/kilts/research-data/dominicks

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BASE="https://www.chicagobooth.edu/research/kilts/research-data/-/media/enterprise/centers/kilts/datasets/dominicks-dataset"
DEMOS="https://www.chicagobooth.edu/boothsitecore/docs/dff/store-demos-customer-count"
RAW="data/raw"

if command -v sha256sum > /dev/null 2>&1; then
  sha() { sha256sum "$1" | cut -d' ' -f1; }
else
  sha() { shasum -a 256 "$1" | cut -d' ' -f1; }   # macOS
fi

# baixa <url> <destino relativo a data/raw>
baixa() {
  local url="$1" destino="$RAW/$2"
  if [ -s "$destino" ]; then
    echo "  já existe: $destino"
    return 0
  fi
  mkdir -p "$(dirname "$destino")"
  echo "  baixando:  $destino"
  if ! curl -fL --retry 3 --retry-delay 5 -o "$destino.parcial" "$url"; then
    rm -f "$destino.parcial"
    echo "  !! falhou: $url"
    return 1
  fi
  mv "$destino.parcial" "$destino"
}

# confere <arquivo de somas> <estrito: 1 ou 0>
confere() {
  local lista="$1" estrito="$2" esperado arquivo obtido ruins=0
  while read -r esperado arquivo; do
    [ -n "$arquivo" ] || continue
    if [ ! -e "$RAW/$arquivo" ]; then
      echo "  FALTA      $arquivo"; ruins=$((ruins + 1)); continue
    fi
    obtido=$(sha "$RAW/$arquivo")
    if [ "$obtido" = "$esperado" ]; then
      echo "  ok         $arquivo"
    else
      echo "  DIFERENTE  $arquivo"; ruins=$((ruins + 1))
    fi
  done < "$lista"
  if [ "$ruins" -gt 0 ]; then
    if [ "$estrito" = "1" ]; then
      echo
      echo "!! $ruins arquivo(s) ausente(s) ou diferente(s) dos usados no trabalho."
      echo "   Os números podem não reproduzir. Confira a origem dos arquivos."
      return 1
    fi
    echo "  aviso: $ruins arquivo(s) da triagem fora do esperado (não afeta o pipeline do TCC)."
  fi
  return 0
}

echo ">>> dados do pipeline do TCC: sucos congelados (frj), contagem de clientes e demografia"
falhou=0
baixa "$BASE/movement-files/wfrj.zip"   "frj/wfrj.zip"                      || falhou=1
baixa "$BASE/upc_csv-files/upcfrj.csv"  "frj/upcfrj.csv"                    || falhou=1
baixa "$DEMOS/ccount_stata.zip"         "customer_count/ccount_stata.zip"   || falhou=1
baixa "$DEMOS/demo_stata.zip"           "store_demographics/demo_stata.zip" || falhou=1

if [ "${1:-}" = "--triagem" ]; then
  echo
  echo ">>> arquivos de movimento das outras 26 categorias (triagem)"
  for c in ana bat ber bjc cer che cig coo cra cso did fec frd fre fsf gro lnd oat ptw sdr sha sna soa tbr tpa tti; do
    baixa "$BASE/movement-files/w$c.zip" "$c/w$c.zip" || falhou=1
  done
fi

echo
echo ">>> conferência do SHA-256"
confere data/SHA256SUMS 1 || falhou=1
if [ "${1:-}" = "--triagem" ]; then
  # A categoria ana não tem soma registrada: o arquivo usado na triagem foi
  # baixado com outro nome e não há como garantir que é byte a byte o mesmo.
  confere data/SHA256SUMS.triagem 0
fi

exit "$falhou"
