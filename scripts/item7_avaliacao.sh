#!/usr/bin/env bash
# Avaliação preditiva da rede: a derivada que o otimizador usa confere com o dado?
#
#   alinhamento   alinha o estimando da referência de variáveis instrumentais ao
#                 objeto que a rede reporta
#   espelho       aplica o mesmo estimador ao log da previsão da rede, 50 sementes
#                 (grava as previsões por semente em data/interim/espelho_d16/, ~760 MB)
#   decomposição  separa a diferença do espelho nos canais próprio, cruzado e contexto
#   suavidade     a derivada própria ao longo da faixa de preço, nos pesos do artefato
#   por produto   o espelho produto a produto
#   segmentos     onde a rede erra (produto, loja, semana) e quem responde pelo nível
#
# Uso:
#   bash scripts/item7_avaliacao.sh
#   TCC_SIMULAR=1 bash scripts/item7_avaliacao.sh   # só mostra os comandos
#
# Pré-requisito: scripts/item6_rede.sh (ou o artefato versionado em models_artifacts/).

# shellcheck source=scripts/_comum.sh
source "$(dirname "$0")/_comum.sh"
confere_ambiente
exige_arquivo "data/processed/${CAT}_modelagem.parquet" "models_artifacts/rede_${CAT}.pt"

titulo "alinhamento do estimando"
roda "$PY" src/experiments/alinhamento_d16.py "$CAT"
titulo "regressão-espelho, 50 sementes"
roda "$PY" src/experiments/espelho_d16.py "$CAT"
titulo "decomposição do espelho"
roda "$PY" src/experiments/decomposicao_espelho_d16.py "$CAT"
titulo "suavidade da derivada própria"
roda "$PY" src/experiments/suavidade_d16.py "$CAT"
titulo "espelho por produto"
roda "$PY" src/experiments/espelho_por_sku_d16.py "$CAT"
titulo "erro por segmento"
roda "$PY" src/experiments/erro_por_segmento_item7.py "$CAT"

titulo "registro de números oficiais"
roda "$PY" src/reports/numeros_oficiais.py

printf '\n\033[1mfim\033[0m em %s.\n' "$(decorrido)"
