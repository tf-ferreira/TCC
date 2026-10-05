#!/usr/bin/env bash
# Triagem das 27 categorias do Dominick's que têm dados de movimento, e a escolha
# dos sucos congelados (frj).
#
#   1. estatísticas por série e cobertura por produto, em cada categoria
#   2. o sortimento de 20 produtos por completude conjunta nas oito finalistas
#   3. a tabela comparativa versionada (reports/comparacao_categorias_n20.json)
#   4. as figuras da triagem (reports/figures/), que leem o resumo por categoria
#      em data/interim/screening/_payload.json (ver data/README.md)
#
# Opcional: o pipeline do TCC só precisa da categoria escolhida, e o
# scripts/reproduzir.sh já faz a triagem dela. Este script exige os arquivos
# de todas as categorias:  bash scripts/baixar_dados.sh --triagem  (~830 MB)
#
# Uso:
#   bash scripts/triagem.sh
#   TCC_SIMULAR=1 bash scripts/triagem.sh   # só mostra os comandos

# shellcheck source=scripts/_comum.sh
source "$(dirname "$0")/_comum.sh"
confere_ambiente

TODAS="ana bat ber bjc cer che cig coo cra cso did fec frd fre frj fsf gro lnd oat ptw sdr sha sna soa tbr tpa tti"
FINALISTAS="tti frj cso frd bjc che sdr fre"

titulo "1/4 triagem de cada categoria"
for c in $TODAS; do
  [ "$SIMULAR" = "1" ] || ls "data/raw/$c"/w*.zip > /dev/null 2>&1 || falha "falta o arquivo de movimento de $c em data/raw/$c/"
  roda "$PY" src/data/screening.py "$c"
done

titulo "2/4 sortimento das finalistas (N = 20)"
for c in $FINALISTAS; do
  roda "$PY" src/data/nucleo.py "$c" --n 20
done

titulo "3/4 tabela comparativa"
roda "$PY" src/reports/comparacao_categorias.py --n 20

titulo "4/4 figuras da triagem"
roda "$PY" src/reports/figuras_triagem.py

printf '\n\033[1mfim\033[0m em %s.\n' "$(decorrido)"
