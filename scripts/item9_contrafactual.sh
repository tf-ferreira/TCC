#!/usr/bin/env bash
# Avaliação contrafactual sequencial na especificação anterior à versão final.
#
#   diagnóstico   sinal da resposta da rede à defasagem de preço
#   sonda         custo e consistência do contrafactual numa amostra
#   contrafactual o artefato canônico com 50 partidas por célula
#   sementes      50 redes (~1h30), mais 25 extras (~45 min) pela regra de parada
#                 declarada, e a junção das 75
#
# A versão final do trabalho refaz esta avaliação na especificação com função de
# controle, em scripts/rodar_v1.sh, reaproveitando o código daqui.
#
# Uso:
#   bash scripts/item9_contrafactual.sh
#   TCC_SIMULAR=1 bash scripts/item9_contrafactual.sh   # só mostra os comandos
#
# Pré-requisito: scripts/item6_rede.sh (ou o artefato versionado em models_artifacts/).

# shellcheck source=scripts/_comum.sh
source "$(dirname "$0")/_comum.sh"
confere_ambiente
exige_arquivo "data/interim/painel/${CAT}_celulas.parquet" "models_artifacts/rede_${CAT}.pt"
O="src/optimization"
H="nao_piorar"

titulo "resposta da rede à defasagem de preço"
roda "$PY" "$O/diagnostico_item9.py" "$CAT"
titulo "sonda e contrafactual do artefato canônico"
roda "$PY" "$O/contrafactual_item9.py" "$CAT" --sonda
roda "$PY" "$O/contrafactual_item9.py" "$CAT" -k 50
titulo "sementes: sonda de custo, 50, 25 extras e a junção"
roda "$PY" "$O/sementes_item9.py" "$CAT" --sementes 2 --sufixo _sonda
roda "$PY" "$O/sementes_item9.py" "$CAT" --sementes 50
roda "$PY" "$O/sementes_item9.py" "$CAT" --sementes 25 --semente-inicial 50 --sufixo _extra
roda "$PY" "$O/sementes_item9.py" "$CAT" --juntar \
  "reports/sementes_item9_${CAT}_${H}_n50.json" "reports/sementes_item9_${CAT}_${H}_n25_extra.json"

printf '\n\033[1mfim\033[0m em %s.\n' "$(decorrido)"
