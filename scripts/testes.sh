#!/usr/bin/env bash
# Suíte de testes, em DOIS processos.
#
# No macOS arm64, treinar um LightGBM depois de `import torch` no mesmo processo
# termina em segmentation fault na construção do Dataset do LightGBM
# (lightgbm/basic.py, __init_from_np2d). Importar os dois, em qualquer ordem,
# funciona; o que quebra é o uso conjunto. Nem OMP_NUM_THREADS=1 nem
# KMP_DUPLICATE_LIB_OK=TRUE evitam, então não é duplicação de libomp. A causa
# raiz não foi identificada.
#
# A consequência vale além dos testes: nenhum script deste repositório treina
# árvore e rede no mesmo processo.
#
# Uso:
#   bash scripts/testes.sh

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python3}"

falhou=0

echo ">>> 1/2 suíte sem a rede (lightgbm, pandas, numpy)"
"$PY" -m pytest tests/ -q \
  --ignore=tests/test_rede.py \
  --ignore=tests/test_problema_pnl.py \
  --ignore=tests/test_multipartida.py \
  --ignore=tests/test_sementes_item8.py \
  --ignore=tests/test_sonda_d15.py \
  --ignore=tests/test_sensibilidade_item8.py \
  --ignore=tests/test_contrafactual_item9.py \
  --ignore=tests/test_mundos.py || falhou=1

echo
echo ">>> 2/2 suíte da rede e da otimização (torch, scipy)"
"$PY" -m pytest -q \
  tests/test_rede.py \
  tests/test_problema_pnl.py \
  tests/test_multipartida.py \
  tests/test_sementes_item8.py \
  tests/test_sonda_d15.py \
  tests/test_sensibilidade_item8.py \
  tests/test_contrafactual_item9.py \
  tests/test_mundos.py || falhou=1

echo
if [ "$falhou" -eq 0 ]; then
  echo "tudo passou, nos dois processos."
else
  echo "alguma suíte falhou; veja acima."
fi
exit "$falhou"
