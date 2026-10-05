# Resultados

Cada medição do trabalho grava um JSON nesta pasta, com os números e a configuração que os produziu (sementes, especificação, hierarquia, escopo). Os arquivos são versionados para que os números do texto possam ser conferidos sem refazer as rodadas, que somam dezenas de horas.

## O registro de números

[`numeros_oficiais.json`](numeros_oficiais.json) é a ponte entre estes arquivos e o texto: 821 números, cada um com o valor, o texto formatado, uma descrição, o script que o produz e o arquivo e o campo de origem. Ele é montado por [`src/reports/numeros_oficiais.py`](../src/reports/numeros_oficiais.py), que só lê artefatos (nenhum valor é digitado à mão), e comparado contra qualquer commit por [`src/reports/comparar_registro.py`](../src/reports/comparar_registro.py). Parte das entradas do registro vem de diagnósticos do pipeline base gravados em `data/interim/painel/` e `data/processed/`, que não são versionados e são refeitos por `scripts/reproduzir.sh`.

## As tabelas do texto

| Tabela | Arquivo |
|---|---|
| Tabela 1: ajuste e elasticidades no teste, 50 sementes | [`checagem_v1_frj_n50.json`](checagem_v1_frj_n50.json) |
| Tabela 2: referência de variáveis instrumentais da elasticidade própria | [`referencia_iv_frj.json`](referencia_iv_frj.json) |
| Tabela 3: objetivo 5 na avaliação de uma semana | [`tabela_anatomia_v1_frj_v1_n50.json`](tabela_anatomia_v1_frj_v1_n50.json) |
| Tabela 4: cenário com o canal das defasagens | [`tabela_mundos_v1_frj_v1_nao_piorar_n50.json`](tabela_mundos_v1_frj_v1_nao_piorar_n50.json) e [`tabela_anatomia_v1_frj_v1_n50.json`](tabela_anatomia_v1_frj_v1_n50.json) |

## Convenção dos nomes

`<medição>_<categoria>[_<especificação>][_<hierarquia>][_n<sementes>][_<sufixo>].json`: por exemplo, `sementes_v1_frj_v1_nao_piorar_n50.json` é o objetivo 5 na especificação final, com a hierarquia de marca na forma "não piorar", sobre 50 sementes. O sufixo `_sonda` marca uma rodada curta de custo, feita antes da rodada completa e mantida como registro.

## Todos os arquivos, por etapa

A coluna "No registro" marca os arquivos que alimentam o registro de números. Os demais são medições intermediárias das etapas, mantidas como registro do que foi feito.

### Pipeline base (`scripts/reproduzir.sh`)

| Arquivo | Produzido por | No registro |
|---|---|:---:|
| [`agrupamento_precos_frj.json`](agrupamento_precos_frj.json) | [`src/experiments/agrupamento_de_precos.py`](../src/experiments/agrupamento_de_precos.py) | ✓ |
| [`baseline_arvores_frj.json`](baseline_arvores_frj.json) | [`src/models/baseline_arvores.py`](../src/models/baseline_arvores.py) |  |
| [`baseline_arvores_frj_com_promo.json`](baseline_arvores_frj_com_promo.json) | [`src/models/baseline_arvores.py`](../src/models/baseline_arvores.py) | ✓ |
| [`baseline_arvores_frj_sem_promo.json`](baseline_arvores_frj_sem_promo.json) | [`src/models/baseline_arvores.py`](../src/models/baseline_arvores.py) | ✓ |
| [`defasagens_frj.json`](defasagens_frj.json) | [`src/experiments/varredura_defasagens.py`](../src/experiments/varredura_defasagens.py) | ✓ |
| [`diagnostico_particao_frj.json`](diagnostico_particao_frj.json) | [`src/experiments/diagnostico_particao.py`](../src/experiments/diagnostico_particao.py) | ✓ |
| [`sensibilidade_piso_frj_n20.json`](sensibilidade_piso_frj_n20.json) | [`src/experiments/sensibilidade_piso.py`](../src/experiments/sensibilidade_piso.py) |  |
| [`trafego_frj.json`](trafego_frj.json) | [`src/experiments/trafego.py`](../src/experiments/trafego.py) | ✓ |
| [`unidade_de_observacao_frj.json`](unidade_de_observacao_frj.json) | [`src/experiments/unidade_de_observacao.py`](../src/experiments/unidade_de_observacao.py) | ✓ |

### Ruído entre sementes e atributos (`scripts/ruido_semente.sh`)

| Arquivo | Produzido por | No registro |
|---|---|:---:|
| [`comparacao_frj_perda.json`](comparacao_frj_perda.json) | [`src/experiments/ruido_semente.py`](../src/experiments/ruido_semente.py) | ✓ |
| [`decomposicao_perda_frj.json`](decomposicao_perda_frj.json) | [`src/experiments/varredura_perda.py`](../src/experiments/varredura_perda.py) | ✓ |
| [`ruido_semente_frj.json`](ruido_semente_frj.json) | [`src/experiments/ruido_semente.py`](../src/experiments/ruido_semente.py) | ✓ |
| [`ruido_semente_frj_defasagens.json`](ruido_semente_frj_defasagens.json) | [`src/experiments/ruido_semente.py`](../src/experiments/ruido_semente.py) | ✓ |
| [`ruido_semente_frj_item5.json`](ruido_semente_frj_item5.json) | [`src/experiments/ruido_semente.py`](../src/experiments/ruido_semente.py) | ✓ |
| [`ruido_semente_frj_perda.json`](ruido_semente_frj_perda.json) | [`src/experiments/ruido_semente.py`](../src/experiments/ruido_semente.py) | ✓ |
| [`ruido_semente_frj_trafego.json`](ruido_semente_frj_trafego.json) | [`src/experiments/ruido_semente.py`](../src/experiments/ruido_semente.py) | ✓ |
| [`sazonalidade_frj.json`](sazonalidade_frj.json) | [`src/experiments/sazonalidade.py`](../src/experiments/sazonalidade.py) |  |

### Rede de demanda (`scripts/item6_rede.sh`)

| Arquivo | Produzido por | No registro |
|---|---|:---:|
| [`comparacao_rede_frj_bemexterno.json`](comparacao_rede_frj_bemexterno.json) | [`src/experiments/varredura_rede.py`](../src/experiments/varredura_rede.py) | ✓ |
| [`comparacao_rede_frj_cabeca.json`](comparacao_rede_frj_cabeca.json) | [`src/experiments/varredura_rede.py`](../src/experiments/varredura_rede.py) | ✓ |
| [`comparacao_rede_frj_cruzada.json`](comparacao_rede_frj_cruzada.json) | [`src/experiments/varredura_rede.py`](../src/experiments/varredura_rede.py) | ✓ |
| [`comparacao_rede_frj_estrutura.json`](comparacao_rede_frj_estrutura.json) | [`src/experiments/varredura_rede.py`](../src/experiments/varredura_rede.py) | ✓ |
| [`comparacao_rede_frj_hiper.json`](comparacao_rede_frj_hiper.json) | [`src/experiments/varredura_rede.py`](../src/experiments/varredura_rede.py) |  |
| [`comparacao_rede_frj_hiper_msle.json`](comparacao_rede_frj_hiper_msle.json) | [`src/experiments/varredura_rede.py`](../src/experiments/varredura_rede.py) |  |
| [`comparacao_rede_frj_hiper_v3.json`](comparacao_rede_frj_hiper_v3.json) | [`src/experiments/varredura_rede.py`](../src/experiments/varredura_rede.py) | ✓ |
| [`comparacao_rede_frj_perda.json`](comparacao_rede_frj_perda.json) | [`src/experiments/varredura_rede.py`](../src/experiments/varredura_rede.py) | ✓ |
| [`fechamento_item6_frj.json`](fechamento_item6_frj.json) | [`src/experiments/fechamento_item6.py`](../src/experiments/fechamento_item6.py) | ✓ |
| [`particao_validacao_frj.json`](particao_validacao_frj.json) | [`src/models/rede.py`](../src/models/rede.py) | ✓ |
| [`rede_frj.json`](rede_frj.json) | [`src/models/rede.py`](../src/models/rede.py) |  |
| [`rede_frj_irrestrita.json`](rede_frj_irrestrita.json) | [`src/models/rede.py`](../src/models/rede.py) |  |
| [`rede_varredura_frj_bemexterno.json`](rede_varredura_frj_bemexterno.json) | [`src/experiments/varredura_rede.py`](../src/experiments/varredura_rede.py) | ✓ |
| [`rede_varredura_frj_cabeca.json`](rede_varredura_frj_cabeca.json) | [`src/experiments/varredura_rede.py`](../src/experiments/varredura_rede.py) | ✓ |
| [`rede_varredura_frj_cruzada.json`](rede_varredura_frj_cruzada.json) | [`src/experiments/varredura_rede.py`](../src/experiments/varredura_rede.py) | ✓ |
| [`rede_varredura_frj_estrutura.json`](rede_varredura_frj_estrutura.json) | [`src/experiments/varredura_rede.py`](../src/experiments/varredura_rede.py) | ✓ |
| [`rede_varredura_frj_hiper.json`](rede_varredura_frj_hiper.json) | [`src/experiments/varredura_rede.py`](../src/experiments/varredura_rede.py) | ✓ |
| [`rede_varredura_frj_hiper_msle.json`](rede_varredura_frj_hiper_msle.json) | [`src/experiments/varredura_rede.py`](../src/experiments/varredura_rede.py) |  |
| [`rede_varredura_frj_hiper_v3.json`](rede_varredura_frj_hiper_v3.json) | [`src/experiments/varredura_rede.py`](../src/experiments/varredura_rede.py) | ✓ |
| [`rede_varredura_frj_perda.json`](rede_varredura_frj_perda.json) | [`src/experiments/varredura_rede.py`](../src/experiments/varredura_rede.py) | ✓ |

### Avaliação preditiva (`scripts/item7_avaliacao.sh`)

| Arquivo | Produzido por | No registro |
|---|---|:---:|
| [`alinhamento_d16_frj.json`](alinhamento_d16_frj.json) | [`src/experiments/alinhamento_d16.py`](../src/experiments/alinhamento_d16.py) | ✓ |
| [`decomposicao_espelho_d16_frj.json`](decomposicao_espelho_d16_frj.json) | [`src/experiments/decomposicao_espelho_d16.py`](../src/experiments/decomposicao_espelho_d16.py) |  |
| [`erro_por_segmento_item7_frj.json`](erro_por_segmento_item7_frj.json) | [`src/experiments/erro_por_segmento_item7.py`](../src/experiments/erro_por_segmento_item7.py) |  |
| [`espelho_d16_frj.json`](espelho_d16_frj.json) | [`src/experiments/espelho_d16.py`](../src/experiments/espelho_d16.py) |  |
| [`espelho_por_sku_d16_frj.json`](espelho_por_sku_d16_frj.json) | [`src/experiments/espelho_por_sku_d16.py`](../src/experiments/espelho_por_sku_d16.py) |  |
| [`suavidade_d16_frj.json`](suavidade_d16_frj.json) | [`src/experiments/suavidade_d16.py`](../src/experiments/suavidade_d16.py) |  |

### Problema de otimização (`scripts/item8_otimizacao.sh`)

| Arquivo | Produzido por | No registro |
|---|---|:---:|
| [`auditoria_hierarquia_frj.json`](auditoria_hierarquia_frj.json) | [`src/optimization/hierarquia.py`](../src/optimization/hierarquia.py) |  |
| [`multipartida_item8_frj_k50_nao_piorar.json`](multipartida_item8_frj_k50_nao_piorar.json) | [`src/optimization/multipartida.py`](../src/optimization/multipartida.py) |  |
| [`multipartida_item8_frj_k50_nenhuma.json`](multipartida_item8_frj_k50_nenhuma.json) | [`src/optimization/multipartida.py`](../src/optimization/multipartida.py) |  |
| [`sementes_item8_frj_nao_piorar_n2_sonda.json`](sementes_item8_frj_nao_piorar_n2_sonda.json) | [`src/optimization/sementes_item8.py`](../src/optimization/sementes_item8.py) |  |
| [`sementes_item8_frj_nao_piorar_n50.json`](sementes_item8_frj_nao_piorar_n50.json) | [`src/optimization/sementes_item8.py`](../src/optimization/sementes_item8.py) |  |
| [`sensibilidade_item8_frj.json`](sensibilidade_item8_frj.json) | [`src/optimization/sensibilidade_item8.py`](../src/optimization/sensibilidade_item8.py) |  |
| [`sonda_d15_frj_h16.json`](sonda_d15_frj_h16.json) | [`src/optimization/sonda_d15.py`](../src/optimization/sonda_d15.py) |  |
| [`sonda_d15_frj_h16_r090.json`](sonda_d15_frj_h16_r090.json) | [`src/optimization/sonda_d15.py`](../src/optimization/sonda_d15.py) |  |
| [`sonda_item8_frj_todas.json`](sonda_item8_frj_todas.json) | [`src/optimization/sonda_item8.py`](../src/optimization/sonda_item8.py) |  |
| [`varredura_caixa_item8_frj.json`](varredura_caixa_item8_frj.json) | [`src/optimization/varredura_caixa.py`](../src/optimization/varredura_caixa.py) |  |

### Avaliação contrafactual, especificação anterior (`scripts/item9_contrafactual.sh`)

| Arquivo | Produzido por | No registro |
|---|---|:---:|
| [`contrafactual_item9_frj_k50_nao_piorar.json`](contrafactual_item9_frj_k50_nao_piorar.json) | [`src/optimization/contrafactual_item9.py`](../src/optimization/contrafactual_item9.py) |  |
| [`contrafactual_item9_frj_k50_nao_piorar.npz`](contrafactual_item9_frj_k50_nao_piorar.npz) | [`src/optimization/contrafactual_item9.py`](../src/optimization/contrafactual_item9.py) |  |
| [`diagnostico_item9_frj.json`](diagnostico_item9_frj.json) | [`src/optimization/diagnostico_item9.py`](../src/optimization/diagnostico_item9.py) |  |
| [`sementes_item9_frj_nao_piorar_n25_extra.json`](sementes_item9_frj_nao_piorar_n25_extra.json) | [`src/optimization/sementes_item9.py`](../src/optimization/sementes_item9.py) |  |
| [`sementes_item9_frj_nao_piorar_n2_sonda.json`](sementes_item9_frj_nao_piorar_n2_sonda.json) | [`src/optimization/sementes_item9.py`](../src/optimization/sementes_item9.py) |  |
| [`sementes_item9_frj_nao_piorar_n50.json`](sementes_item9_frj_nao_piorar_n50.json) | [`src/optimization/sementes_item9.py`](../src/optimization/sementes_item9.py) |  |
| [`sementes_item9_frj_nao_piorar_n75.json`](sementes_item9_frj_nao_piorar_n75.json) | [`src/optimization/sementes_item9.py`](../src/optimization/sementes_item9.py) |  |
| [`sonda_item9_frj.json`](sonda_item9_frj.json) | [`src/optimization/contrafactual_item9.py`](../src/optimization/contrafactual_item9.py) |  |

### Versão final (`scripts/rodar_v1.sh`)

| Arquivo | Produzido por | No registro |
|---|---|:---:|
| [`anatomia_v1_frj_v1_codigo_n20.json`](anatomia_v1_frj_v1_codigo_n20.json) | [`src/optimization/anatomia_v1.py`](../src/optimization/anatomia_v1.py) |  |
| [`anatomia_v1_frj_v1_n50.json`](anatomia_v1_frj_v1_n50.json) | [`src/optimization/anatomia_v1.py`](../src/optimization/anatomia_v1.py) |  |
| [`anatomia_v1_frj_v1_n50_direcao.json`](anatomia_v1_frj_v1_n50_direcao.json) | [`src/optimization/anatomia_v1.py`](../src/optimization/anatomia_v1.py) |  |
| [`checagem_v1_frj_n50.json`](checagem_v1_frj_n50.json) | [`src/experiments/checagem_v1.py`](../src/experiments/checagem_v1.py) | ✓ |
| [`criterio_d43_frj.json`](criterio_d43_frj.json) | [`src/reports/criterio_d43.py`](../src/reports/criterio_d43.py) | ✓ |
| [`referencia_iv_frj.json`](referencia_iv_frj.json) | [`src/experiments/referencia_iv.py`](../src/experiments/referencia_iv.py) | ✓ |
| [`sementes_v1_frj_adotada_nao_piorar_n20.json`](sementes_v1_frj_adotada_nao_piorar_n20.json) | [`src/optimization/sementes_v1.py`](../src/optimization/sementes_v1.py) |  |
| [`sementes_v1_frj_v1_codigo_nao_piorar_n20.json`](sementes_v1_frj_v1_codigo_nao_piorar_n20.json) | [`src/optimization/sementes_v1.py`](../src/optimization/sementes_v1.py) |  |
| [`sementes_v1_frj_v1_nao_piorar_n50.json`](sementes_v1_frj_v1_nao_piorar_n50.json) | [`src/optimization/sementes_v1.py`](../src/optimization/sementes_v1.py) |  |
| [`tabela_anatomia_v1_frj_v1_codigo_n20.json`](tabela_anatomia_v1_frj_v1_codigo_n20.json) | [`src/reports/tabela_anatomia_v1.py`](../src/reports/tabela_anatomia_v1.py) |  |
| [`tabela_anatomia_v1_frj_v1_n50.json`](tabela_anatomia_v1_frj_v1_n50.json) | [`src/reports/tabela_anatomia_v1.py`](../src/reports/tabela_anatomia_v1.py) | ✓ |
| [`tabela_anatomia_v1_frj_v1_n50_direcao.json`](tabela_anatomia_v1_frj_v1_n50_direcao.json) | [`src/reports/tabela_anatomia_v1.py`](../src/reports/tabela_anatomia_v1.py) | ✓ |
| [`tabela_mundos_v1_frj_adotada_nao_piorar_n20.json`](tabela_mundos_v1_frj_adotada_nao_piorar_n20.json) | [`src/reports/tabela_mundos_v1.py`](../src/reports/tabela_mundos_v1.py) | ✓ |
| [`tabela_mundos_v1_frj_v1_nao_piorar_n50.json`](tabela_mundos_v1_frj_v1_nao_piorar_n50.json) | [`src/reports/tabela_mundos_v1.py`](../src/reports/tabela_mundos_v1.py) | ✓ |

### Triagem das categorias (`scripts/triagem.sh`)

| Arquivo | Produzido por | No registro |
|---|---|:---:|
| [`comparacao_categorias_n20.json`](comparacao_categorias_n20.json) | [`src/reports/comparacao_categorias.py`](../src/reports/comparacao_categorias.py) |  |

### Registro

| Arquivo | Produzido por | No registro |
|---|---|:---:|
| [`numeros_oficiais.json`](numeros_oficiais.json) | [`src/reports/numeros_oficiais.py`](../src/reports/numeros_oficiais.py) |  |

## Figuras

[`figures/`](figures/) tem as cinco figuras da triagem de categorias (posicionamento, redundância entre critérios, sortimento, distribuição do coeficiente de variação e sobrevivência por limiar de regularidade), em PNG e PDF, produzidas por [`src/reports/figuras_triagem.py`](../src/reports/figuras_triagem.py) a partir do resumo descrito em [`data/README.md`](../data/README.md).

## O que não fica aqui

Os registros de execução das rodadas longas (`logs_v1/`) não são versionados.
