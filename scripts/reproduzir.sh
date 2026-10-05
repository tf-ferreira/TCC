#!/usr/bin/env bash
# Pipeline base: dos dados brutos ao painel de modelagem e ao registro de números,
# na ordem de dependência, e a conferência de que os números voltam iguais aos
# versionados.
#
# Cobre a triagem da categoria escolhida, o sortimento e o painel de células, o
# bem externo, as elasticidades de referência, a referência de árvores, as
# defasagens, a partição e o painel final de modelagem. As rodadas longas (rede,
# otimização, avaliação contrafactual e versão final) têm scripts próprios; a
# ordem completa está no README.
#
# Uso:
#   bash scripts/reproduzir.sh            # compara o registro novo contra HEAD
#   bash scripts/reproduzir.sh HEAD~1     # contra outro commit
#   TCC_SIMULAR=1 bash scripts/reproduzir.sh   # só mostra os comandos
#
# Pré-requisito: os dados brutos em data/raw/ (bash scripts/baixar_dados.sh).
#
# As etapas 12 e 13 repetem a configuração da 11 de propósito: a 11 é a
# referência de árvores e não muda de forma; o par 12 e 13 mede a diferença
# entre duas especificações (com e sem o indicador de promoção do próprio
# produto) e por isso precisa sair do mesmo ambiente, em arquivos próprios.

# shellcheck source=scripts/_comum.sh
source "$(dirname "$0")/_comum.sh"

REF="${1:-HEAD}"
N=20

etapa() {
  local rotulo="$1"; shift
  local t0=$SECONDS
  titulo "$rotulo"
  roda "$@"
  printf '    (%ss)\n' "$((SECONDS - t0))"
}

confere_ambiente
echo "referência de comparação: $REF"
exige_arquivo "data/raw/$CAT/w$CAT.zip" "data/raw/$CAT/upc$CAT.csv" \
              data/raw/customer_count/ccount_stata.zip data/raw/store_demographics/demo_stata.zip

etapa " 1/20 triagem da categoria: estatísticas por série e cobertura por produto" \
  "$PY" src/data/screening.py "$CAT"
etapa " 2/20 núcleo da categoria: sortimento por completude conjunta (N = $N)" \
  "$PY" src/data/nucleo.py "$CAT" --n "$N"
etapa " 3/20 painel de células (loja, semana)" \
  "$PY" src/data/painel.py "$CAT" --n "$N"
etapa " 4/20 bem externo, índice principal" \
  "$PY" src/data/bem_externo.py "$CAT"
etapa " 5/20 bem externo, pesos truncados no p95 (robustez)" \
  "$PY" src/data/bem_externo.py "$CAT" --truncar-peso 95
etapa " 6/20 robustez da busca gulosa do sortimento (76 sementes)" \
  "$PY" src/experiments/robustez_selecao.py "$CAT"
etapa " 7/20 sensibilidade do sortimento ao piso de suporte" \
  "$PY" src/experiments/sensibilidade_piso.py "$CAT" --n "$N"
etapa " 8/20 elasticidades de referência, erros padrão agrupados e bootstrap" \
  "$PY" src/experiments/elasticidades.py "$CAT"
etapa " 9/20 elasticidades sobre o índice truncado" \
  "$PY" src/experiments/elasticidades.py "$CAT" --sufixo _p95
etapa "10/20 instrumentos alternativos" \
  "$PY" src/experiments/instrumentos.py "$CAT"
etapa "11/20 referência de árvores (LightGBM) e varredura do passo da derivada" \
  "$PY" src/models/baseline_arvores.py "$CAT"
etapa "12/20 a mesma referência COM o indicador de promoção do próprio produto" \
  "$PY" src/models/baseline_arvores.py "$CAT" --sufixo _com_promo
etapa "13/20 a mesma referência SEM o indicador de promoção do próprio produto" \
  "$PY" src/models/baseline_arvores.py "$CAT" --sem-promo-proprio --sufixo _sem_promo
etapa "14/20 cobertura das defasagens" \
  "$PY" src/experiments/varredura_defasagens.py "$CAT"
etapa "15/20 diagnóstico da partição para a rede" \
  "$PY" src/experiments/diagnostico_particao.py "$CAT"
etapa "16/20 agrupamento de preços entre lojas" \
  "$PY" src/experiments/agrupamento_de_precos.py "$CAT"
etapa "17/20 contagem de clientes nos dois estimandos" \
  "$PY" src/experiments/trafego.py "$CAT"
etapa "18/20 painel final de modelagem" \
  "$PY" src/data/atributos.py "$CAT"
etapa "19/20 unidade de observação e células incompletas" \
  "$PY" src/experiments/unidade_de_observacao.py "$CAT"
etapa "20/20 registro de números oficiais" \
  "$PY" src/reports/numeros_oficiais.py

[ "$SIMULAR" = "1" ] && exit 0

titulo "conferência: os números batem com $REF?"
if git rev-parse --verify --quiet "$REF" > /dev/null; then
  "$PY" src/reports/comparar_registro.py --ref "$REF" || true
else
  echo "    sem repositório git ou referência $REF: comparação pulada"
fi

titulo "testes"
bash scripts/testes.sh || true

if git rev-parse --git-dir > /dev/null 2>&1; then
  titulo "arquivos versionados que mudaram"
  git status --short -- reports/ src/ || true
fi

printf '\n\033[1mfim\033[0m em %s.\n' "$(decorrido)"
echo "Próximas etapas, nesta ordem: scripts/ruido_semente.sh, scripts/item6_rede.sh,"
echo "scripts/item7_avaliacao.sh, scripts/item8_otimizacao.sh, scripts/item9_contrafactual.sh"
echo "e scripts/rodar_v1.sh. O registro (etapa 20) lê também os resultados versionados"
echo "dessas rodadas longas; refazê-las atualiza os números correspondentes."
