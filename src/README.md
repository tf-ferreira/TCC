# Código

Os scripts rodam a partir da raiz do repositório (`python3 src/<pasta>/<arquivo>.py frj ...`) e importam os módulos irmãos pelo nome. Cada um declara o uso na própria docstring e grava a saída em `data/` (intermediários) ou em `reports/` (resultados versionados). A ordem de execução está nos scripts de [`scripts/`](../scripts/) e no [README principal](../README.md#como-executar).

## `data/`: dos arquivos do Dominick's ao painel de modelagem

| Módulo | O que faz |
|---|---|
| [`screening.py`](data/screening.py) | triagem de uma categoria: lê o arquivo de movimento em blocos e resume preço, promoção, regularidade e cobertura por série (produto, loja) e por produto |
| [`nucleo.py`](data/nucleo.py) | resumo de uma categoria sob o critério vigente de sortimento, para comparar as finalistas |
| [`sortimento.py`](data/sortimento.py) | seleção do sortimento por completude conjunta (o único ponto onde a regra existe) e extração do painel longo, com preço unitário e custo pela margem |
| [`painel.py`](data/painel.py) | painel no nível da célula (loja, semana) |
| [`bem_externo.py`](data/bem_externo.py) | agregado dos produtos fora do sortimento, como contexto exógeno da célula |
| [`calendario.py`](data/calendario.py) | da semana corrida do Dominick's para data, ângulo do ano, harmônicas e feriados |
| [`defasagens.py`](data/defasagens.py) | preço e volume da mesma série k semanas atrás, por junção de calendário |
| [`preparo.py`](data/preparo.py) | preparo das entradas da rede: preenchimento, escalonamento, vocabulário de lojas e truncamento do tempo, sempre ajustados só no treino |
| [`atributos.py`](data/atributos.py) | o painel final de modelagem, `data/processed/frj_modelagem.parquet` |
| [`funcao_de_controle.py`](data/funcao_de_controle.py) | primeiro estágio da função de controle do custo e o resíduo que entra no contexto da rede |

## `models/`: os modelos de demanda

| Módulo | O que faz |
|---|---|
| [`baseline_arvores.py`](models/baseline_arvores.py) | ensemble de árvores (LightGBM) como referência obrigatória de previsão, a partição temporal com zona morta e a lista única de atributos |
| [`rede.py`](models/rede.py) | a rede de demanda: saída vetorial das 20 demandas da célula, sub-rede monótona, ligação log, matriz de elasticidades em forma fechada, especificações, treino e o artefato canônico |

## `experiments/`: medições

| Módulo | O que faz |
|---|---|
| [`elasticidades.py`](experiments/elasticidades.py) | as elasticidades de referência, com as amostras declaradas, erros padrão agrupados e bootstrap |
| [`instrumentos.py`](experiments/instrumentos.py) | instrumentos alternativos para o preço: relevância e deslocamento do estimando |
| [`referencia_iv.py`](experiments/referencia_iv.py) | referência de variáveis instrumentais da elasticidade própria (Tabela 2) |
| [`robustez_selecao.py`](experiments/robustez_selecao.py) | robustez da busca gulosa do sortimento e o sinal por tamanho de sortimento |
| [`sensibilidade_piso.py`](experiments/sensibilidade_piso.py) | sensibilidade do sortimento ao piso de suporte |
| [`agrupamento_de_precos.py`](experiments/agrupamento_de_precos.py) | quanto do agrupamento de preços entre lojas a demografia explica |
| [`trafego.py`](experiments/trafego.py), [`varredura_trafego.py`](experiments/varredura_trafego.py) | quanto da resposta ao preço passa pela contagem de clientes |
| [`sazonalidade.py`](experiments/sazonalidade.py) | como o calendário entra no modelo |
| [`varredura_defasagens.py`](experiments/varredura_defasagens.py) | defasagens: cobertura e valor preditivo |
| [`diagnostico_particao.py`](experiments/diagnostico_particao.py) | a partição vista pelas entradas da rede: lojas nunca vistas, extrapolação temporal, cobertura |
| [`unidade_de_observacao.py`](experiments/unidade_de_observacao.py) | correlação intra-célula, tamanho efetivo da amostra e células incompletas |
| [`ruido_semente.py`](experiments/ruido_semente.py) | piso de ruído de uma comparação entre sementes e o teste pareado, por famílias de configurações, retomável |
| [`varredura_item5.py`](experiments/varredura_item5.py) | o painel final contra o piso da referência de árvores |
| [`varredura_perda.py`](experiments/varredura_perda.py) | Poisson contra MSLE como perda de treino |
| [`varredura_rede.py`](experiments/varredura_rede.py) | varredura da rede: hiperparâmetros na validação e as famílias de comparação, com impressão digital de configuração |
| [`fechamento_item6.py`](experiments/fechamento_item6.py) | correlação intra-célula do resíduo da rede e identidade de agregação |
| [`checagem_v1.py`](experiments/checagem_v1.py) | checagem preditiva das especificações da versão final, 50 sementes (Tabela 1) |
| [`alinhamento_d16.py`](experiments/alinhamento_d16.py) | alinha o estimando da referência ao objeto que a rede reporta |
| [`espelho_d16.py`](experiments/espelho_d16.py) | regressão-espelho: o estimador da referência aplicado à previsão da rede |
| [`decomposicao_espelho_d16.py`](experiments/decomposicao_espelho_d16.py) | em qual canal (próprio, cruzado, contexto) está a diferença do espelho |
| [`espelho_por_sku_d16.py`](experiments/espelho_por_sku_d16.py) | a regressão-espelho produto a produto |
| [`suavidade_d16.py`](experiments/suavidade_d16.py) | a derivada própria da rede ao longo da faixa de preço |
| [`erro_por_segmento_item7.py`](experiments/erro_por_segmento_item7.py) | onde a rede erra (produto, loja, semana) e quem responde pelo nível |
| [`previsoes_item7.py`](experiments/previsoes_item7.py) | carregador único das previsões da avaliação preditiva, sem retreinar |

## `optimization/`: o problema de otimização e a avaliação contrafactual

| Módulo | O que faz |
|---|---|
| [`problema.py`](optimization/problema.py) | o problema de uma célula: objetivo, gradiente em forma fechada, caixa em log-preço e o solver com método de reserva |
| [`hierarquia.py`](optimization/hierarquia.py) | hierarquia de marca: identificação dos produtos, pares, auditoria por célula e restrições |
| [`mundos.py`](optimization/mundos.py) | mundos de resposta cruzada e a política robusta |
| [`contrafactual_item9.py`](optimization/contrafactual_item9.py) | avaliação contrafactual de uma semana e sequencial, contra o valor observado |
| [`sementes_v1.py`](optimization/sementes_v1.py) | o objetivo 5 da versão final por mundo, política da rede e política robusta, sobre sementes |
| [`anatomia_v1.py`](optimization/anatomia_v1.py) | o que a política da versão final faz com os preços, e a parte do ganho que passa pelo canal das defasagens |
| [`sonda_item8.py`](optimization/sonda_item8.py) | primeira execução do solver e comparação das formas de hierarquia |
| [`varredura_caixa.py`](optimization/varredura_caixa.py) | varredura do raio da caixa de preços |
| [`multipartida.py`](optimization/multipartida.py) | multipartida: limite inferior do ótimo e dispersão entre bacias |
| [`sementes_item8.py`](optimization/sementes_item8.py) | o resultado da otimização sobre 50 sementes |
| [`sensibilidade_item8.py`](optimization/sensibilidade_item8.py) | sensibilidade do ganho à escala das elasticidades |
| [`sonda_d15.py`](optimization/sonda_d15.py) | gap até o ótimo global num modelo reduzido, com certificado (requer `pyscipopt`) |
| [`diagnostico_item9.py`](optimization/diagnostico_item9.py) | sinal da resposta da rede à defasagem de preço |
| [`sementes_item9.py`](optimization/sementes_item9.py) | o contrafactual sequencial sobre 50 sementes, mais 25 extras |

## `reports/`: tabelas, figuras e registro

| Módulo | O que faz |
|---|---|
| [`numeros_oficiais.py`](reports/numeros_oficiais.py) | registro de todos os números produzidos, cada um com produtor e arquivo de origem |
| [`comparar_registro.py`](reports/comparar_registro.py) | compara o registro contra uma referência do git |
| [`tabela_anatomia_v1.py`](reports/tabela_anatomia_v1.py) | tabelas do objetivo 5 a partir da anatomia |
| [`tabela_mundos_v1.py`](reports/tabela_mundos_v1.py) | tabela política × mundo do objetivo 5, com as diferenças pareadas |
| [`criterio_d43.py`](reports/criterio_d43.py) | critério fixado antes da rodada: o código de promoção muda o objetivo 5 em 2 pontos ou mais? |
| [`comparacao_categorias.py`](reports/comparacao_categorias.py) | tabela comparativa das categorias finalistas |
| [`figuras_triagem.py`](reports/figuras_triagem.py) | figuras da triagem de categorias |

## Convenções

- Os nomes de variáveis são legíveis, e as docstrings remetem à notação do trabalho ($p$, $\hat V$, $u$, $\gamma$, $m$) para que código e texto fiquem rastreáveis um ao outro.
- Toda função de engenharia de atributos e toda restrição do problema de otimização tem teste unitário em [`tests/`](../tests/).
- Os comentários citam decisões pelo código D1 a D47 e as etapas do cronograma ("item 5" a "item 9"); o significado das etapas está no [README principal](../README.md#do-texto-ao-código).
