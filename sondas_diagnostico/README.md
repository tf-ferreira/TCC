# Sondas da versão final

Três scripts curtos que fazem parte da rodada oficial da versão final (`scripts/rodar_v1.sh`) e cujos resultados entram no registro de números ([`src/reports/numeros_oficiais.py`](../src/reports/numeros_oficiais.py)). Eles rodam sobre o código de `src/` sem alterá-lo: importam os módulos da rede, do problema e da avaliação e, quando precisam de uma variante, a fazem só na memória do processo.

| Sonda | Pergunta | Onde aparece | Custo |
|---|---|---|---|
| [`s01_presenca_vs_venda.py`](s01_presenca_vs_venda.py) | existe preço registrado sem venda? | O conjunto de dados: a presença definida por preço positivo | segundos |
| [`s19_independencia_do_custo.py`](s19_independencia_do_custo.py) | quanto da variação de custo de um produto é comum aos outros 19? (a identificação das cruzadas sob a função de controle) | Resultados: o modelo de demanda e o objetivo 5 na avaliação de uma semana | segundos |
| [`s22_deriva_do_codigo.py`](s22_deriva_do_codigo.py) | a marcação do código de promoção mudou ao longo dos anos? | O conjunto de dados: a regra de marcação do código de promoção | segundos |

Cada sonda imprime o resultado e grava um JSON com o mesmo nome ao lado do script. Os JSON versionados são os da rodada oficial.

## Como rodar

De dentro da pasta, depois de `scripts/reproduzir.sh`:

```bash
cd sondas_diagnostico
python3 s01_presenca_vs_venda.py
python3 s19_independencia_do_custo.py
python3 s22_deriva_do_codigo.py
```

O `scripts/rodar_v1.sh` roda as três na ordem oficial.
