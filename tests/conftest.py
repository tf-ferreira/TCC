"""Configuração da suíte de testes.

Quase todos os testes rodam sobre dados sintéticos. Quatro testes de integração,
em `tests/test_rede.py`, leem o painel real em `data/interim/painel/`, que
`scripts/reproduzir.sh` produz a partir dos dados brutos (não versionados, ver
`data/README.md`). Num clone sem esses arquivos, eles são pulados com o motivo
declarado, em vez de falharem por arquivo ausente. Com o painel no lugar, rodam
normalmente.

O critério é o próprio código do teste: é pulado o teste que usa a constante
`PAINEL`, o caminho do painel real.
"""

import inspect
from pathlib import Path

import pytest

PAINEL = Path(__file__).resolve().parents[1] / "data" / "interim" / "painel"
EXIGIDOS = ("frj_upcs.json", "frj_celulas.parquet", "frj_bem_externo.parquet")


def pytest_collection_modifyitems(config, items):
    if all((PAINEL / nome).exists() for nome in EXIGIDOS):
        return
    pular = pytest.mark.skip(
        reason="requer o painel real em data/interim/painel/ (rode scripts/reproduzir.sh)")
    for item in items:
        funcao = getattr(item, "function", None)
        if funcao is None:
            continue
        try:
            fonte = inspect.getsource(funcao)
        except (OSError, TypeError):
            continue
        if "PAINEL" in fonte:
            item.add_marker(pular)
