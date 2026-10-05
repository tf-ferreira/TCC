# Dados

Os dados não são versionados. Vêm do [Dominick's Finer Foods](https://www.chicagobooth.edu/research/kilts/research-data/dominicks), registro de leitura de código de barras de uma rede de supermercados da região de Chicago entre 1989 e 1997, distribuído pelo James M. Kilts Center for Marketing da University of Chicago Booth School of Business. O Kilts Center libera os dados apenas para pesquisa acadêmica e exige que trabalhos e publicações que os utilizem citem o Kilts Center.

## O que baixar

```bash
bash scripts/baixar_dados.sh             # o que o pipeline do TCC usa
bash scripts/baixar_dados.sh --triagem   # mais as outras 26 categorias, só para a triagem
```

O script baixa da página oficial e confere o SHA-256 de cada arquivo contra o dos arquivos que produziram os resultados ([`SHA256SUMS`](SHA256SUMS)). Se o download automático falhar, baixe à mão na página do Kilts Center e coloque cada arquivo no caminho abaixo.

| Arquivo | Conteúdo | Tamanho | Usado por |
|---|---|---|---|
| `raw/frj/wfrj.zip` | movimento semanal dos sucos congelados (produto, loja, semana) | 22 MB | todo o pipeline |
| `raw/frj/upcfrj.csv` | cadastro dos produtos (descrição, tamanho) | 9 KB | hierarquia de marca |
| `raw/customer_count/ccount_stata.zip` | contagem de clientes por loja e dia | 41 MB | atributo de tráfego, testado e excluído |
| `raw/store_demographics/demo_stata.zip` | demografia das lojas | 165 KB | agrupamento de preços entre lojas |
| `raw/<cat>/w<cat>.zip` | movimento das outras 26 categorias | ~830 MB no total | só `scripts/triagem.sh` |

Na triagem, a categoria `ana` não tem soma registrada em [`SHA256SUMS.triagem`](SHA256SUMS.triagem): o arquivo usado no trabalho foi baixado com outro nome, e não há como garantir que seja byte a byte o da página.

## Organização

```
data/
├── raw/         os arquivos como baixados, nunca editados à mão
├── interim/     saídas intermediárias, todas regeneráveis pelos scripts
│   ├── screening/   triagem: estatísticas por série, cobertura por produto, núcleo
│   ├── painel/      painel de células, bem externo e diagnósticos do pipeline base
│   └── espelho_d16/ previsões por semente da avaliação preditiva (~760 MB)
└── processed/   o painel final de modelagem: frj_modelagem.parquet e frj_modelagem.json
```

O painel final tem 504.680 linhas: 25.234 células completas vezes 20 produtos.

## A única exceção versionada: `interim/screening/_payload.json`

É o resumo das 27 categorias que alimenta [`src/reports/figuras_triagem.py`](../src/reports/figuras_triagem.py), as figuras da triagem e a correlação de postos entre os critérios citada no texto. Ele junta os 27 arquivos `<cat>_resumo.json` produzidos por [`src/data/screening.py`](../src/data/screening.py) (os 1.755 campos em comum são idênticos) e acrescenta seis campos derivados: nome de exibição, número de produtos com cobertura, tamanho do núcleo por limiar de regularidade, histograma do coeficiente de variação e curvas de sobrevivência por regularidade e por longevidade.

Esses seis campos foram montados fora do código versionado, e por isso o arquivo fica no repositório: sem ele, as figuras da triagem não poderiam ser refeitas. São estatísticas agregadas por categoria, não dados de venda.

## Fragmentos das varreduras

As varreduras entre sementes gravam cada par (configuração, semente) como um fragmento fora do repositório, para serem retomáveis:

| Pasta | Variável para trocar | Usada por |
|---|---|---|
| `~/.tcc_fragmentos_ruido` | `TCC_FRAGMENTOS` | `src/experiments/ruido_semente.py` |
| `~/.tcc_fragmentos_rede` | `TCC_FRAGMENTOS_REDE` | `src/experiments/varredura_rede.py` |

Cada fragmento carrega uma impressão digital da configuração; um fragmento de outra configuração é recusado, e nenhum é apagado automaticamente.
