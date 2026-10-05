#!/usr/bin/env bash
# Rede de demanda: seleção do ponto, famílias de comparação e o artefato canônico.
#
# A ordem importa:
#   1. hiper_v3   triagem de hiperparâmetros na VALIDAÇÃO, 50 sementes; escolhe o ponto
#   2. perda      MSLE contra Poisson, sozinha e antes das outras, porque é a única
#                 que poderia derrubar a escolha da perda e invalidar o que vem depois
#   3. estrutura, bemexterno, cabeca, cruzada   as famílias que reportam, 50 sementes
#   4. fechamento correlação intra-célula do resíduo e identidade de agregação
#   5. artefato   partição de validação e os dois modelos salvos em models_artifacts/
#                 (a rede adotada e a mesma rede sem a restrição de monotonicidade)
#
# São cerca de 1.900 treinos, algo como dez horas no ambiente oficial. É
# retomável: cada par (configuração, semente) é um fragmento em
# ~/.tcc_fragmentos_rede (ou em $TCC_FRAGMENTOS_REDE), semente já calculada não é
# refeita, e nenhum fragmento é apagado. Se uma família recusar fragmentos de
# outra configuração, ela para e diz o que fazer, em vez de apagar em silêncio.
#
# Uso:
#   bash scripts/item6_rede.sh                 # tudo, na ordem
#   bash scripts/item6_rede.sh cabeca          # uma etapa só, pelo nome
#   bash scripts/item6_rede.sh historicas      # as duas triagens antigas, medidas no
#                                              # teste e mantidas só como registro
#   TCC_SIMULAR=1 bash scripts/item6_rede.sh   # só mostra os comandos
#
# Pré-requisito: scripts/reproduzir.sh. No macOS, rode com o Mac na tomada e
# sem dormir:  caffeinate -dimsu bash scripts/item6_rede.sh

# shellcheck source=scripts/_comum.sh
source "$(dirname "$0")/_comum.sh"

SEMENTES="${TCC_SEMENTES:-0-49}"
SO="${1:-}"
V="src/experiments/varredura_rede.py"

confere_ambiente
exige_arquivo "data/processed/${CAT}_modelagem.parquet"

quer() { [ -z "$SO" ] || [ "$SO" = "$1" ]; }

familia() {
  local nome="$1" sementes="$2"
  quer "$nome" || return 0
  titulo "família $nome, sementes $sementes  [$(decorrido)]"
  roda "$PY" "$V" "$CAT" --familia "$nome" --sementes "$sementes"
  roda "$PY" "$V" "$CAT" --familia "$nome" --consolidar
  roda "$PY" "$V" "$CAT" --familia "$nome" --comparar
}

if [ -z "$SO" ]; then
  titulo "suíte de testes (nada roda se ela falhar)"
  if [ "$SIMULAR" = "1" ]; then echo "    $ bash scripts/testes.sh"
  else bash scripts/testes.sh || falha "a suíte falhou"; fi
fi

familia hiper_v3   "$SEMENTES"
familia perda      "$SEMENTES"
familia estrutura  "$SEMENTES"
familia bemexterno "$SEMENTES"
familia cabeca     "$SEMENTES"
familia cruzada    "$SEMENTES"

if quer fechamento; then
  titulo "fechamento: correlação intra-célula e identidade de agregação  [$(decorrido)]"
  roda "$PY" src/experiments/fechamento_item6.py "$CAT" --sementes "$SEMENTES"
fi

if quer artefato; then
  # --adotada lê a configuração de varredura_rede.MELHOR, o mesmo objeto que as
  # famílias usaram: é isso que impede o modelo salvo de divergir do medido.
  titulo "artefato canônico e partição de validação  [$(decorrido)]"
  roda "$PY" src/models/rede.py "$CAT" --so-particao
  roda "$PY" src/models/rede.py "$CAT" --adotada
  roda "$PY" src/models/rede.py "$CAT" --adotada --irrestrita --sufixo _irrestrita
fi

if [ "$SO" = "historicas" ]; then
  for f in hiper hiper_msle; do
    titulo "triagem histórica $f, sementes 0-4, medida no teste  [$(decorrido)]"
    roda "$PY" "$V" "$CAT" --familia "$f" --sementes 0-4
    roda "$PY" "$V" "$CAT" --familia "$f" --consolidar
    roda "$PY" "$V" "$CAT" --familia "$f" --comparar
  done
fi

if [ -z "$SO" ]; then
  titulo "registro de números oficiais"
  roda "$PY" src/reports/numeros_oficiais.py
fi

printf '\n\033[1mfim\033[0m em %s.\n' "$(decorrido)"
