#!/usr/bin/env bash
# Formulação e validação do problema de otimização (PNL) sobre a rede treinada.
#
#   hierarquia     auditoria da hierarquia de marca, célula a célula
#   sonda          primeira execução do solver nas 4.022 células do teste, nas três
#                  formas de hierarquia (nenhuma, absoluta, "não piorar")
#   caixa          varredura do raio da caixa de preços (1% a 15%)
#   multipartida   50 partidas por célula, com e sem a hierarquia (~15 min cada)
#   sementes       o resultado sobre 50 redes de sementes diferentes (~1 h)
#   sensibilidade  ganho sob reescala das elasticidades (~7 min)
#   gap global     o gap até o ótimo global num modelo reduzido com certificado,
#                  na caixa de 15% e na de 90%
#
# Os tempos são os registrados nos próprios resultados, no ambiente oficial.
#
# Uso:
#   bash scripts/item8_otimizacao.sh
#   TCC_SIMULAR=1 bash scripts/item8_otimizacao.sh   # só mostra os comandos
#
# Pré-requisito: scripts/item6_rede.sh (ou o artefato versionado em models_artifacts/).

# shellcheck source=scripts/_comum.sh
source "$(dirname "$0")/_comum.sh"
confere_ambiente
exige_arquivo "data/interim/painel/${CAT}_celulas.parquet" "models_artifacts/rede_${CAT}.pt"
O="src/optimization"

titulo "auditoria da hierarquia de marca"
roda "$PY" "$O/hierarquia.py" "$CAT"
titulo "sonda do solver, todas as células e as três formas de hierarquia"
roda "$PY" "$O/sonda_item8.py" "$CAT" --celulas 0 --comparar
titulo "varredura do raio da caixa"
roda "$PY" "$O/varredura_caixa.py" "$CAT"
titulo "multipartida, 50 partidas, com e sem hierarquia"
roda "$PY" "$O/multipartida.py" "$CAT" -k 50
roda "$PY" "$O/multipartida.py" "$CAT" -k 50 --hierarquia nenhuma
titulo "50 sementes (antes, uma sonda de custo com 2)"
roda "$PY" "$O/sementes_item8.py" "$CAT" --sementes 2 --sufixo _sonda
roda "$PY" "$O/sementes_item8.py" "$CAT" --sementes 50
titulo "sensibilidade às elasticidades"
roda "$PY" "$O/sensibilidade_item8.py" "$CAT"
titulo "gap até o ótimo global, caixa de 15% e de 90%"
roda "$PY" "$O/sonda_d15.py" "$CAT" --h 16 --produtos 2,3,4,5,8,10
roda "$PY" "$O/sonda_d15.py" "$CAT" --h 16 --produtos 4,8 --raio 0.90 --sufixo _r090

printf '\n\033[1mfim\033[0m em %s.\n' "$(decorrido)"
