"""Sonda 1: presença (price>0) é independente de venda (move>0)?

Se quase toda linha com price>0 tem move>0, então 'presente' equivale a
'vendeu pelo menos uma unidade', e a exigência de célula completa (D12, D31)
condiciona no próprio desfecho. Mede também a mesma coisa para o 'resto',
porque n_resto e vol_resto entram na rede.
"""
from pathlib import Path as _P
D = str(_P(__file__).resolve().parents[1] / "data")
import zipfile, json
import numpy as np, pandas as pd

z = zipfile.ZipFile(D + '/raw/frj/wfrj.zip')
df = pd.read_csv(z.open('wfrj.csv'), usecols=['STORE','UPC','WEEK','MOVE','QTY','PRICE','SALE','PROFIT','OK'])
df.columns = [c.lower() for c in df.columns]
df = df[df.ok == 1]
upcs = json.load(open(D + '/interim/painel/frj_upcs.json'))['ordem']
sort = df[df.upc.isin(upcs)]
resto = df[~df.upc.isin(upcs)]
saida = {}


def tab(d, rot):
    pos = d.price > 0
    ven = d.move > 0
    saida[rot.split(" ")[0]] = {"linhas": int(len(d)),
                                "preco_positivo_sem_venda": int((pos & ~ven).sum()),
                                "venda_sem_preco": int((~pos & ven).sum()),
                                "p_venda_dado_preco": float(ven[pos].mean())}
    print(f"--- {rot}: {len(d):,} linhas ok==1")
    print(f"price>0 & move>0 : {np.mean(pos & ven):.4%}")
    print(f"price>0 & move==0: {np.mean(pos & ~ven):.4%}  ({(pos & ~ven).sum():,} linhas)")
    print(f"price==0 & move>0: {np.mean(~pos & ven):.4%}  ({(~pos & ven).sum():,} linhas)")
    print(f"price==0 & move==0: {np.mean(~pos & ~ven):.4%}")
    print(f"P(move>0 | price>0) = {ven[pos].mean():.5f}")
tab(sort, 'sortimento (20 SKUs)')
tab(resto, 'resto (155 SKUs)')
destino = _P(__file__).with_name("s01_presenca_vs_venda.json")
destino.write_text(json.dumps(saida, indent=1))
print(f"-> {destino}")
