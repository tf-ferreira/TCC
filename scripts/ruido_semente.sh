#!/usr/bin/env bash
# Piso de ruído entre sementes e as varreduras da engenharia de atributos.
#
# Uma árvore ou rede treinada com outra semente é outro modelo; por isso toda
# comparação preditiva do trabalho é entre médias sobre sementes, pareadas, com
# erro padrão. Este script refaz essas varreduras na ordem em que os resultados
# versionados foram produzidos, com o mesmo número de sementes de cada um:
#
#   sazonalidade           5 representações do tempo, 50 sementes cada
#   par com/sem promoção   3 sementes (registrado como pendência no próprio registro)
#   defasagens             20 sementes; sem defasagem e defasagem de volume, 5
#   tráfego                20 sementes
#   painel final           20 sementes
#   perda (Poisson, MSLE)  50 sementes
#
# É retomável: cada par (configuração, semente) é um fragmento próprio em
# ~/.tcc_fragmentos_ruido (ou em $TCC_FRAGMENTOS), e semente já calculada não é
# refeita. Pode interromper e rodar de novo.
#
# Tempo de referência, no ambiente oficial: cerca de 40 minutos só para as 250
# execuções da sazonalidade.
#
# Uso:
#   bash scripts/ruido_semente.sh
#   TCC_SIMULAR=1 bash scripts/ruido_semente.sh   # só mostra os comandos
#
# Pré-requisito: scripts/reproduzir.sh (o painel de células e o de modelagem).

# shellcheck source=scripts/_comum.sh
source "$(dirname "$0")/_comum.sh"
confere_ambiente
exige_arquivo "data/interim/painel/${CAT}_celulas.parquet"
R="src/experiments/ruido_semente.py"

titulo "sazonalidade: a melhor representação do tempo, semente única"
roda "$PY" src/experiments/sazonalidade.py "$CAT"

titulo "sazonalidade, 50 sementes por representação"
roda "$PY" "$R" "$CAT" --config modulo52 harm1 harm2 harm3 harm3_feriados --sementes 0-49

titulo "par com e sem o indicador de promoção do próprio produto, com elasticidade"
roda "$PY" "$R" "$CAT" --config d25_com d25_sem --sementes 0-2 --elasticidade
roda "$PY" "$R" "$CAT" --consolidar

titulo "defasagens"
roda "$PY" "$R" "$CAT" --familia defasagens --config sem_lag lagp1_volume --sementes 0-4 --elasticidade
roda "$PY" "$R" "$CAT" --familia defasagens --config lagp1 lagp12 lagp1a4 lagp1a8 --sementes 0-19 --elasticidade
roda "$PY" "$R" "$CAT" --familia defasagens --consolidar

titulo "contagem de clientes"
roda "$PY" "$R" "$CAT" --familia trafego --config sem_trafego com_trafego --sementes 0-19 --elasticidade
roda "$PY" "$R" "$CAT" --familia trafego --consolidar

titulo "painel final contra o piso da referência"
roda "$PY" "$R" "$CAT" --familia item5 --config piso final final_sem_tempo piso_com_tempo --sementes 0-19 --elasticidade
roda "$PY" "$R" "$CAT" --familia item5 --consolidar

titulo "perda de treino: Poisson, MSLE e MSLE com correção de Duan"
roda "$PY" "$R" "$CAT" --familia perda --config poisson msle msle_smearing --sementes 0-49 --elasticidade
roda "$PY" "$R" "$CAT" --familia perda --consolidar
roda "$PY" "$R" "$CAT" --familia perda --comparar
roda "$PY" src/experiments/varredura_perda.py "$CAT"

titulo "registro de números oficiais"
roda "$PY" src/reports/numeros_oficiais.py

printf '\n\033[1mfim\033[0m em %s.\n' "$(decorrido)"
