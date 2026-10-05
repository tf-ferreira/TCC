"""Todo módulo do projeto importa sem erro.

Existe porque o refactor da fase 1.1 moveu `extrair_longo` de `painel.py` para
`sortimento.py` e deixou `bem_externo.py` com um import morto. Nada falhou até
alguém rodar o script, e o verificador de documentos não pega isso, porque ele
compara números e não código.

    python3 -m pytest tests/test_imports.py
"""
import importlib, sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
for pasta in ("data", "experiments", "reports"):
    sys.path.insert(0, str(RAIZ / "src" / pasta))

MODULOS = [p.stem for pasta in ("data", "experiments", "reports")
           for p in sorted((RAIZ / "src" / pasta).glob("*.py"))
           if p.stem != "__init__"]


@pytest.mark.parametrize("modulo", MODULOS)
def test_importa(modulo):
    importlib.import_module(modulo)
