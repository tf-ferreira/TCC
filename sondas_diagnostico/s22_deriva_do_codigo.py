"""Sonda 22: a regra de marcação do código de promoção (`sale`) mudou ao longo do painel?

O manual da Kilts Center diz que o código "não é marcado pela DFF de forma
consistente" (D43). Esta sonda mede a deriva nas células completas do sortimento
(as 25.234 células com os 20 produtos presentes), por ano do painel (blocos de 52
semanas a partir de set/1989, como nas outras sondas):

- **desconto de preço:** linha com preço pelo menos 10% abaixo do preço modal do
  mesmo produto, na mesma loja, no mesmo ano (moda dos preços arredondados ao centavo);
- **fração sem código entre os descontos:** se o código acompanhasse o desconto de
  forma estável, essa fração seria estável;
- **fração das linhas com código,** qualquer código.

Grava `s22_deriva_do_codigo.json` ao lado deste arquivo.
"""
from pathlib import Path as _P
D = str(_P(__file__).resolve().parents[1] / "data")
import json, zipfile
import numpy as np, pandas as pd

z = zipfile.ZipFile(D + '/raw/frj/wfrj.zip')
df = pd.read_csv(z.open('wfrj.csv'), usecols=['STORE', 'UPC', 'WEEK', 'QTY', 'PRICE', 'SALE', 'OK'])
df.columns = [c.lower() for c in df.columns]
df = df[(df.ok == 1) & (df.price > 0)]
upcs = json.load(open(D + '/interim/painel/frj_upcs.json'))['ordem']
cel = pd.read_parquet(D + '/interim/painel/frj_celulas.parquet')
precos = [c for c in cel.columns if c.startswith('preco_')]
completas = cel[cel[precos].gt(0).all(axis=1)][['store', 'week']]

s = df[df.upc.isin(upcs)].merge(completas, on=['store', 'week'])
s['p'] = s.price / s.qty
s['sale'] = s.sale.fillna('').astype(str).str.strip()
s['ano'] = (s.week - 1) // 52
moda = s.groupby(['store', 'upc', 'ano']).p.agg(lambda x: x.round(2).mode().iloc[0])
s = s.join(moda.rename('p_moda'), on=['store', 'upc', 'ano'])
s['desconto'] = s.p <= 0.9 * s.p_moda
s['com_codigo'] = s.sale != ''

por_ano = s.groupby('ano').agg(linhas=('p', 'size'), com_codigo=('com_codigo', 'mean'))
desc = s[s.desconto].groupby('ano').agg(descontos=('p', 'size'),
                                        com_codigo_nos_descontos=('com_codigo', 'mean'))
por_ano = por_ano.join(desc)
por_ano['sem_codigo_nos_descontos'] = 1 - por_ano.com_codigo_nos_descontos

anos = sorted(por_ano.index)
saida = {
    "celulas_completas": int(len(completas)),
    "linhas": int(len(s)),
    "por_ano": {int(a): {k: (float(v) if k != "linhas" and k != "descontos" else int(v))
                         for k, v in por_ano.loc[a].items()} for a in anos},
    "resumo": {
        "sem_codigo_nos_descontos_4_primeiros": [float(por_ano.loc[anos[:4], 'sem_codigo_nos_descontos'].min()),
                                                 float(por_ano.loc[anos[:4], 'sem_codigo_nos_descontos'].max())],
        "sem_codigo_nos_descontos_3_ultimos": [float(por_ano.loc[anos[-3:], 'sem_codigo_nos_descontos'].min()),
                                               float(por_ano.loc[anos[-3:], 'sem_codigo_nos_descontos'].max())],
        "com_codigo_primeiro_ano": float(por_ano.loc[anos[0], 'com_codigo']),
        "com_codigo_2_ultimos": [float(por_ano.loc[anos[-2:], 'com_codigo'].min()),
                                 float(por_ano.loc[anos[-2:], 'com_codigo'].max())],
    },
}
print(f"{saida['celulas_completas']} células completas, {saida['linhas']} linhas")
print(por_ano.round(3).to_string())
print(json.dumps(saida["resumo"], indent=1))
destino = _P(__file__).with_name("s22_deriva_do_codigo.json")
destino.write_text(json.dumps(saida, indent=1))
print(f"-> {destino}")
