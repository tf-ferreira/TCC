#!/usr/bin/env bash
# As rodadas da versão final: os números das seções de Resultados.
#
#   r2  referência de variáveis instrumentais da elasticidade própria   Tabela 2
#   r1  checagem preditiva das especificações, 50 sementes              Tabela 1
#   r3c objetivo 5 na especificação anterior ("adotada"), 20 sementes
#   r3  objetivo 5 na especificação final (v1), 50 sementes:
#       a avaliação sequencial por política e mundo                     Tabela 4
#   r3b objetivo 5 com o código de promoção (v1_codigo), 20 sementes
#   s19 quanto da variação de custo de um produto é comum aos outros 19
#   anatomia_v1         a avaliação de uma semana e a parte do ganho que passa
#                       pelo canal das defasagens, 50 sementes          Tabelas 3 e 4
#   anatomia_v1_codigo  a mesma anatomia com o código de promoção, 20 sementes
#   s01, s22, s21       sondas oficiais: presença contra venda, deriva do código
#                       de promoção, elasticidade sustentada
#   anatomia_v1_direcao a anatomia com a direção e o suporte dos preços
#                       recomendados também nas decisões de uma semana
#   tabelas             tabelas por política e mundo, critério do código de
#                       promoção e o registro de números
#
# Tempo de referência, no ambiente oficial: cerca de seis horas, quase todo nas
# três anatomias e nas rodadas de sementes (cada etapa registra o próprio tempo
# no JSON de saída). A saída de cada etapa vai para reports/logs_v1/<etapa>.log,
# e o resumo (início, fim e status) para a tela.
#
# Uso:
#   bash scripts/rodar_v1.sh               # tudo, na ordem
#   bash scripts/rodar_v1.sh anatomia_v1   # uma etapa só, pelo nome
#   TCC_SIMULAR=1 bash scripts/rodar_v1.sh # só mostra os comandos
#
# Pré-requisito: scripts/reproduzir.sh. No macOS, rode com o Mac na tomada e
# sem dormir:  caffeinate -dimsu bash scripts/rodar_v1.sh

# shellcheck source=scripts/_comum.sh
source "$(dirname "$0")/_comum.sh"
confere_ambiente
exige_arquivo "data/processed/${CAT}_modelagem.parquet" "data/raw/${CAT}/w${CAT}.zip"

SO="${1:-}"
LOGS="reports/logs_v1"
[ "$SIMULAR" = "1" ] || mkdir -p "$LOGS"
carimbo() { date "+%d/%m %H:%M"; }

# etapa <nome> <comando...>: roda com a saída num log próprio
etapa() {
  local nome="$1"; shift
  if [ -n "$SO" ] && [ "$SO" != "$nome" ]; then return 0; fi
  printf '    $ %s\n' "$*"
  [ "$SIMULAR" = "1" ] && return 0
  echo "[$(carimbo)] início  $nome"
  "$@" > "$LOGS/$nome.log" 2>&1
  local st=$?
  echo "[$(carimbo)] fim     $nome (status $st)"
  [ "$st" -eq 0 ] || falha "a etapa $nome falhou; veja $LOGS/$nome.log"
}

O="src/optimization"
if [ -z "$SO" ]; then
  etapa testes bash scripts/testes.sh
fi
etapa r2_referencia_iv     "$PY" src/experiments/referencia_iv.py "$CAT"
etapa r1_checagem          "$PY" src/experiments/checagem_v1.py "$CAT" --sementes 50
etapa r3c_adotada          "$PY" "$O/sementes_v1.py" "$CAT" --especificacao adotada --sementes 20
etapa r3_v1                "$PY" "$O/sementes_v1.py" "$CAT" --especificacao v1 --sementes 50
etapa r3b_v1_codigo        "$PY" "$O/sementes_v1.py" "$CAT" --especificacao v1_codigo --sementes 20
etapa s19_independencia_do_custo  bash -c "cd sondas_diagnostico && $PY s19_independencia_do_custo.py"
etapa anatomia_v1          "$PY" "$O/anatomia_v1.py" "$CAT" --sementes 50
etapa anatomia_v1_codigo   "$PY" "$O/anatomia_v1.py" "$CAT" --especificacao v1_codigo --sementes 20
etapa s01_presenca_vs_venda       bash -c "cd sondas_diagnostico && $PY s01_presenca_vs_venda.py"
etapa s22_deriva_do_codigo        bash -c "cd sondas_diagnostico && $PY s22_deriva_do_codigo.py"
etapa s21_elasticidade_sustentada bash -c "cd sondas_diagnostico && $PY s21_elasticidade_sustentada.py 5"
etapa anatomia_v1_direcao  "$PY" "$O/anatomia_v1.py" "$CAT" --sementes 50 --sufixo _direcao

if [ -z "$SO" ] || [ "$SO" = "tabelas" ]; then
  titulo "tabelas e registro"
  roda "$PY" src/reports/tabela_mundos_v1.py "reports/sementes_v1_${CAT}_v1_nao_piorar_n50.json"
  roda "$PY" src/reports/tabela_mundos_v1.py "reports/sementes_v1_${CAT}_adotada_nao_piorar_n20.json"
  roda "$PY" src/reports/tabela_anatomia_v1.py "reports/anatomia_v1_${CAT}_v1_n50.json"
  roda "$PY" src/reports/tabela_anatomia_v1.py "reports/anatomia_v1_${CAT}_v1_n50_direcao.json"
  roda "$PY" src/reports/tabela_anatomia_v1.py "reports/anatomia_v1_${CAT}_v1_codigo_n20.json"
  roda "$PY" src/reports/criterio_d43.py
  roda "$PY" src/reports/numeros_oficiais.py
fi

printf '\n\033[1mfim\033[0m em %s.\n' "$(decorrido)"
