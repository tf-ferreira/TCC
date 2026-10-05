# Precificação dinâmica no varejo

**Rede neural identificada pelo custo e otimização não linear**

Código do Trabalho de Conclusão de Curso de **Thiago Ferreira** no MBA em Data Science e Analytics da USP/ESALQ, com orientação da **Profa. Dra. Patrícia Belfiore Fávero**.

O trabalho desenvolve um arcabouço prescritivo de precificação para o varejo supermercadista: uma rede neural de demanda, com a elasticidade própria identificada pelo custo de aquisição, alimenta um otimizador não linear com gradiente exato, e o ganho que ele teria produzido é estimado contra o histórico de uma rede de supermercados. Os dados são semanais, de 20 sucos congelados em 93 lojas do conjunto Dominick's Finer Foods, ao longo de 396 semanas.

Este repositório reúne tudo o que é necessário para executar o trabalho e conferir seus números: o código, os testes, os scripts de execução na ordem de dependência, os resultados versionados e a rede treinada. O texto do trabalho e os dados brutos não estão aqui (os dados são baixados da fonte original por um script).

---

## Sumário

- [O método em uma página](#o-método-em-uma-página)
- [Principais resultados](#principais-resultados)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Do texto ao código](#do-texto-ao-código)
- [Como executar](#como-executar)
- [Reprodutibilidade](#reprodutibilidade)
- [Dados e termos de uso](#dados-e-termos-de-uso)
- [Como citar](#como-citar)
- [Licença](#licença)
- [English summary](#english-summary)

---

## O método em uma página

**Dados e painel.** A categoria de sucos congelados (`frj`) foi escolhida por uma triagem das 27 categorias do Dominick's com dados disponíveis. O sortimento de 20 produtos foi selecionado por completude conjunta, e a unidade de observação é a célula (loja, semana): 25.234 células completas, 504.680 observações e 55,2% do volume da categoria. Os 155 produtos fora do sortimento entram como bem externo, contexto exógeno da célula.

**Partição.** Temporal, nunca por sorteio: as últimas 20% das semanas formam o teste (semanas 319 a 399), separado do treino por uma zona morta de oito semanas; dentro do treino, uma janela de validação escolhe os hiperparâmetros. Toda comparação entre modelos é feita entre médias sobre 50 sementes, pareadas por semente.

**Modelo de demanda.** Uma rede com saída vetorial dá o log das 20 demandas da célula:

$$\ln \hat V_i = g_i(x) + \sum_{j \neq i} \gamma_{ij}\, u_j - m_i(u_i), \qquad u_i = \ln p_i - c_i$$

O contexto $g_i(x)$ não vê os preços da semana; eles entram só pela matriz $\gamma$ e por $m_i$, uma sub-rede monótona e convexa por reparametrização. Por isso a elasticidade própria é negativa por construção e a matriz de elasticidades sai em forma fechada dos parâmetros. A elasticidade própria é identificada pelo custo de aquisição, por função de controle: o resíduo do primeiro estágio entra no contexto da rede e fica congelado no contrafactual. Um ensemble de árvores (LightGBM) é a referência obrigatória de previsão, mas não entra no otimizador, porque é constante por partes e não tem derivada em relação ao preço.

**Otimização.** Para cada célula do teste, o otimizador maximiza a margem e, separadamente, a receita dos 20 produtos, com caixa de ±15% em torno do preço histórico e restrições de hierarquia de marca. O gradiente é escrito em forma fechada a partir da matriz de elasticidades e conferido contra a diferenciação automática.

**Mundos e política robusta.** Como as elasticidades cruzadas são fracamente identificadas, o resultado é reportado por hipótese sobre elas ("mundos" de resposta cruzada), sob uma política robusta que otimiza o conjunto de mundos.

**Avaliação.** O número principal é o ganho de uma semana contra o valor observado, com as defasagens históricas; a avaliação sequencial, em que o preço recomendado vira a defasagem da semana seguinte, fica como cenário.

## Principais resultados

Conforme o resumo do trabalho:

- a função de controle melhorou o ajuste e tornou a elasticidade própria mediana menos negativa, de −1,91 para −1,45;
- com isso a categoria ficou inelástica e o ótimo se inverteu: a política recomendada eleva os preços;
- a margem aumenta entre 9% e 26% e a receita varia entre −4% e 7%, conforme a hipótese sobre as elasticidades cruzadas.

O registro [`reports/numeros_oficiais.json`](reports/numeros_oficiais.json) reúne 824 números produzidos pelos scripts, cada um com o valor, o script que o produz e o arquivo de origem: é a ponte entre os resultados e o texto. O mapa das tabelas para os arquivos está em [Do texto ao código](#do-texto-ao-código) e em [`reports/README.md`](reports/README.md).

## Estrutura do repositório

```
.
├── src/
│   ├── data/              leitura do Dominick's, triagem, sortimento, painel, atributos, função de controle
│   ├── models/            referência de árvores (LightGBM) e a rede de demanda
│   ├── experiments/       medições: elasticidades, varreduras entre sementes, avaliação preditiva
│   ├── optimization/      problema de otimização, hierarquia de marca, mundos, contrafactual
│   └── reports/           tabelas, figuras, registro e comparação de números
├── scripts/               execução, na ordem de dependência (ver Como executar)
├── tests/                 suíte pytest, em dois processos
├── reports/               resultados versionados (um JSON por medição) e figuras
├── models_artifacts/      a rede canônica treinada e a mesma rede sem restrição de monotonicidade
├── sondas_diagnostico/    as quatro sondas da rodada oficial da versão final
├── data/                  dados (não versionados): instruções e somas SHA-256
├── requirements.txt       versões mínimas das dependências
└── requirements.lock.txt  o conjunto exato do ambiente em que os resultados foram produzidos
```

Cada pasta tem um README próprio: [`src/`](src/README.md), [`data/`](data/README.md), [`reports/`](reports/README.md) e [`sondas_diagnostico/`](sondas_diagnostico/README.md).

## Do texto ao código

Os caminhos citados no trabalho escrito são os deste repositório. As listagens de código do texto (Figuras 1 a 6) são versões condensadas, para leitura; a nota de cada uma aponta o arquivo e as funções reais, todas presentes nos mesmos caminhos.

**Implementação**

| Seção do trabalho | Código |
|---|---|
| O conjunto de dados | [`src/data/screening.py`](src/data/screening.py) (leitura e regras de formato), [`src/data/sortimento.py`](src/data/sortimento.py) (preço unitário e custo pela margem) |
| O conjunto de dados: a escolha da categoria | [`src/data/screening.py`](src/data/screening.py), [`src/data/nucleo.py`](src/data/nucleo.py), [`src/reports/comparacao_categorias.py`](src/reports/comparacao_categorias.py), [`src/reports/figuras_triagem.py`](src/reports/figuras_triagem.py) |
| O sortimento e o painel de células | [`src/data/sortimento.py`](src/data/sortimento.py), [`src/data/painel.py`](src/data/painel.py), [`src/experiments/robustez_selecao.py`](src/experiments/robustez_selecao.py) |
| O resto da categoria como bem externo | [`src/data/bem_externo.py`](src/data/bem_externo.py) |
| Os atributos | [`src/data/atributos.py`](src/data/atributos.py), [`src/data/calendario.py`](src/data/calendario.py), [`src/data/defasagens.py`](src/data/defasagens.py), [`src/data/preparo.py`](src/data/preparo.py) |
| Partição, referência e protocolo de comparação | [`src/models/baseline_arvores.py`](src/models/baseline_arvores.py) (`particionar` e o LightGBM), [`src/models/rede.py`](src/models/rede.py) (`particionar_selecao`), [`src/experiments/ruido_semente.py`](src/experiments/ruido_semente.py) e [`src/experiments/varredura_rede.py`](src/experiments/varredura_rede.py) (comparação pareada entre sementes) |
| A rede de demanda (Figura 1) | [`src/models/rede.py`](src/models/rede.py): `SubRedeMonotona`, `RedeDemanda` |
| A função de controle (Figura 2) | [`src/data/funcao_de_controle.py`](src/data/funcao_de_controle.py): `ajustar`, `aplicar` |
| O treino (Figura 3) | [`src/models/rede.py`](src/models/rede.py): `treinar`; escolha de hiperparâmetros em [`src/experiments/varredura_rede.py`](src/experiments/varredura_rede.py) |
| O problema de otimização (Figura 4) | [`src/optimization/problema.py`](src/optimization/problema.py): `valor_e_gradiente`, `resolver`; [`src/optimization/hierarquia.py`](src/optimization/hierarquia.py): `restricoes` |
| Mundos de resposta cruzada e política robusta (Figura 5) | [`src/optimization/mundos.py`](src/optimization/mundos.py): `MundoAncorado`, `Conjunto` |
| A avaliação contrafactual (Figura 6) | [`src/optimization/contrafactual_item9.py`](src/optimization/contrafactual_item9.py): `rodar_passada`, `agregar`; [`src/optimization/sementes_v1.py`](src/optimization/sementes_v1.py): `objetivo5_no_mundo` |

**Resultados e discussão**

| Resultado | Produzido por | Arquivo |
|---|---|---|
| Tabela 1: ajuste e elasticidades no teste, 50 sementes | `src/experiments/checagem_v1.py` | [`reports/checagem_v1_frj_n50.json`](reports/checagem_v1_frj_n50.json) |
| Tabela 2: referência de variáveis instrumentais da elasticidade própria | `src/experiments/referencia_iv.py` | [`reports/referencia_iv_frj.json`](reports/referencia_iv_frj.json) |
| Tabela 3: objetivo 5 na avaliação de uma semana, por política e mundo | `src/optimization/anatomia_v1.py` e `src/reports/tabela_anatomia_v1.py` | [`reports/tabela_anatomia_v1_frj_v1_n50.json`](reports/tabela_anatomia_v1_frj_v1_n50.json) |
| Tabela 4: cenário com o canal das defasagens (avaliação sequencial) | `src/optimization/sementes_v1.py` e `src/reports/tabela_mundos_v1.py`, mais a anatomia | [`reports/tabela_mundos_v1_frj_v1_nao_piorar_n50.json`](reports/tabela_mundos_v1_frj_v1_nao_piorar_n50.json), [`reports/tabela_anatomia_v1_frj_v1_n50.json`](reports/tabela_anatomia_v1_frj_v1_n50.json) |
| O que a política recomendada faz com os preços | `src/optimization/anatomia_v1.py --sufixo _direcao` | [`reports/anatomia_v1_frj_v1_n50_direcao.json`](reports/anatomia_v1_frj_v1_n50_direcao.json) (campo `resumo`) |
| A especificação inicial e o cenário do código de promoção | `src/optimization/sementes_v1.py`, `src/optimization/anatomia_v1.py`, `src/reports/criterio_d43.py` | [`reports/criterio_d43_frj.json`](reports/criterio_d43_frj.json) |
| Presença por preço positivo e deriva do código de promoção (em O conjunto de dados) | sondas 1 e 22 | [`sondas_diagnostico/`](sondas_diagnostico/) |
| Parcela da variação de custo comum aos outros produtos | sonda 19 | [`sondas_diagnostico/s19_independencia_do_custo.json`](sondas_diagnostico/s19_independencia_do_custo.json) |

**Vocabulário do código.** Os nomes de arquivo e os comentários seguem as etapas do cronograma do projeto e um registro de decisões numeradas (D1 a D47) que é material de trabalho do autor e não faz parte deste repositório; o texto do trabalho é a referência pública do método. As etapas:

| No código | Etapa |
|---|---|
| triagem, `nucleo` | itens 3 e 4: triagem das categorias e escolha dos sucos congelados |
| item 5 | painel e engenharia de atributos |
| item 6 | rede de demanda |
| item 7, `*_d16` | avaliação preditiva: a derivada da rede confere com o dado? |
| item 8 | problema de otimização |
| item 9 | avaliação contrafactual |
| v1 | versão final do trabalho: especificação com função de controle e objetivo 5 por mundo |
| especificação `adotada` | a especificação anterior à função de controle, mantida reproduzível |

## Como executar

### 1. Ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # ou requirements.lock.txt, para o conjunto exato
```

No macOS, o LightGBM precisa da OpenMP, que o pip não instala: `brew install libomp`. Sem ela, `import lightgbm` falha com `Library not loaded: @rpath/libomp.dylib`, o que parece erro de versão do Python e não é.

Tudo roda em CPU; não há uso de GPU.

### 2. Dados

```bash
bash scripts/baixar_dados.sh             # ~63 MB: sucos congelados, contagem de clientes e demografia
```

O script baixa os arquivos do site do Kilts Center e confere o SHA-256 contra o dos arquivos que produziram os resultados. Para refazer também a triagem das 27 categorias, use `--triagem` (~830 MB a mais). Detalhes e download manual em [`data/README.md`](data/README.md).

### 3. Testes

```bash
bash scripts/testes.sh
```

A suíte roda em dois processos (ver [Reprodutibilidade](#reprodutibilidade)). Quatro testes de integração leem o painel real e são pulados até o `scripts/reproduzir.sh` produzi-lo.

### 4. Reprodução, na ordem

| # | Script | O que faz | Tempo de referência |
|---|---|---|---|
| 1 | [`scripts/reproduzir.sh`](scripts/reproduzir.sh) | pipeline base: da triagem da categoria ao painel de modelagem e ao registro de números, e confere se os números voltam iguais aos versionados | minutos |
| 2 | [`scripts/ruido_semente.sh`](scripts/ruido_semente.sh) | piso de ruído entre sementes e as varreduras da engenharia de atributos | 40 min só a sazonalidade |
| 3 | [`scripts/item6_rede.sh`](scripts/item6_rede.sh) | rede: seleção do ponto na validação, famílias de comparação e o artefato canônico | ~10 h |
| 4 | [`scripts/item7_avaliacao.sh`](scripts/item7_avaliacao.sh) | avaliação preditiva da derivada da rede | |
| 5 | [`scripts/item8_otimizacao.sh`](scripts/item8_otimizacao.sh) | problema de otimização: solver, caixa, multipartida, sementes, gap global | ≥ 1 h 30 |
| 6 | [`scripts/item9_contrafactual.sh`](scripts/item9_contrafactual.sh) | avaliação contrafactual na especificação anterior | ≥ 2 h 15 |
| 7 | [`scripts/rodar_v1.sh`](scripts/rodar_v1.sh) | versão final: todos os números da seção de Resultados (Tabelas 1 a 4) | ~6 h |
| | [`scripts/triagem.sh`](scripts/triagem.sh) | opcional: triagem das 27 categorias e figuras | |

Os tempos são os registrados nos próprios resultados ou nas notas de execução, no ambiente oficial (Mac com Apple Silicon); "≥" indica a soma só das etapas com tempo registrado, e o branco, ausência de registro.

Todos os passos dependem do 1. Os passos 4 a 6 usam a rede de `models_artifacts/`, que já está versionada, então não exigem refazer o passo 3. O passo 7 treina as próprias redes, uma por semente, e depende só do passo 1.

Recursos comuns aos scripts:

- `TCC_SIMULAR=1 bash scripts/<script>.sh` mostra os comandos exatos, sem executar nada;
- os scripts longos aceitam o nome de uma etapa para rodar só ela (por exemplo, `bash scripts/rodar_v1.sh anatomia_v1`);
- as varreduras são retomáveis: cada par (configuração, semente) é gravado como fragmento em `~/.tcc_fragmentos_ruido` e `~/.tcc_fragmentos_rede` (ou nas pastas de `TCC_FRAGMENTOS` e `TCC_FRAGMENTOS_REDE`), e semente já calculada não é refeita;
- no macOS, rodadas longas devem rodar com o Mac na tomada e sem dormir: `caffeinate -dimsu bash scripts/<script>.sh`.

### 5. Conferência dos números

O registro [`reports/numeros_oficiais.json`](reports/numeros_oficiais.json) é montado por [`src/reports/numeros_oficiais.py`](src/reports/numeros_oficiais.py) a partir dos arquivos de saída de cada medição: nenhum valor é digitado à mão, e um número sem artefato que o produza não entra. Depois de refazer qualquer etapa:

```bash
python3 src/reports/numeros_oficiais.py
python3 src/reports/comparar_registro.py            # contra o registro versionado (HEAD)
```

O `scripts/reproduzir.sh` faz as duas coisas ao final.

## Reprodutibilidade

**Ambiente oficial.** Os resultados versionados foram produzidos em Python 3.14.7, macOS arm64, com o conjunto exato de [`requirements.lock.txt`](requirements.lock.txt).

**Medida, não afirmada.** O pipeline base foi refeito do zero no ambiente oficial e comparado contra os artefatos produzidos antes em Python 3.10 com pandas 2.3: 109 de 109 números bateram, com diferença relativa máxima de 3,33e-13, atravessando uma mudança de versão maior do pandas. O único resíduo é nas métricas de teste da referência de árvores, que se movem 2,5e-5 em termos relativos sem mudar nenhum número na precisão em que é citado.

**Verificação deste repositório, fora do ambiente oficial.** A partir de um clone limpo e só dos quatro arquivos brutos, em Linux x86-64 com Python 3.13 e as versões do lock:

- `scripts/reproduzir.sh` rodou inteiro em 11 minutos, e a suíte de testes passou inteira, inclusive os quatro testes de integração com o painel real;
- dos 824 números do registro, 158 dependem do pipeline base e foram recalculados: 148 voltaram iguais a menos de 1e-9 em termos relativos (92 idênticos), e 10 mudaram até 0,8%. Os 10 vêm de modelos LightGBM (a referência de árvores sem o indicador de promoção e o resíduo usado na correlação intra-célula). No texto, isso só move o tamanho efetivo da amostra, de cerca de 34.900 para cerca de 34.700; a correlação intra-célula continua 0,078;
- a referência de variáveis instrumentais (Tabela 2) e a sonda 19 voltaram iguais a menos de 2e-13; as sondas 1 e 22 e a auditoria da hierarquia de marca, idênticas byte a byte;
- a checagem preditiva da Tabela 1, refeita com 2 das 50 sementes, treinou as redes das quatro especificações e reproduziu os valores por semente do arquivo versionado com diferença relativa máxima de 1,6e-4;
- o objetivo 5 da versão final, refeito para a semente 0 na avaliação sequencial e na anatomia que dá as Tabelas 3 e 4, ficou a menos de 0,02 ponto percentual do valor oficial na margem, em todos os mundos e nas duas políticas, e a menos de 0,25 ponto na receita, cuja superfície tem várias bacias por célula ([`reports/multipartida_item8_frj_k50_nao_piorar.json`](reports/multipartida_item8_frj_k50_nao_piorar.json)).

A reprodução exata é garantida no ambiente oficial; em outra plataforma, os números que vêm de árvores ou de redes podem mudar na terceira casa significativa.

**Redes neurais.** `torch.manual_seed` fixa a inicialização e a ordem dos lotes, mas o resultado não é idêntico bit a bit entre plataformas. Por isso nenhum número de rede vem de uma execução única: toda leitura é uma média sobre sementes, com erro padrão.

**PyTorch e LightGBM no mesmo processo.** No macOS arm64, treinar um LightGBM depois de importar o PyTorch no mesmo processo termina em falha de segmentação. A causa raiz não foi identificada. Nenhum script treina os dois no mesmo processo, e a suíte de testes roda em dois processos ([`scripts/testes.sh`](scripts/testes.sh)).

**Versões que importam.** O pandas, porque o comportamento de `groupby` com `observed` mudou entre versões e a absorção de efeitos fixos depende dele; e o LightGBM, porque o remapeamento de categóricas por valor na previsão é coberto por um teste próprio.

**Dependência opcional.** [`src/optimization/sonda_d15.py`](src/optimization/sonda_d15.py) (gap até o ótimo global, com certificado) usa o solver SCIP via `pyscipopt`; o teste correspondente é pulado sem ele.

## Dados e termos de uso

Os dados são do [Dominick's Finer Foods](https://www.chicagobooth.edu/research/kilts/research-data/dominicks), distribuídos pelo James M. Kilts Center for Marketing da University of Chicago Booth School of Business. O Kilts Center os libera apenas para pesquisa acadêmica e exige que trabalhos e publicações que os utilizem citem o Kilts Center. Por isso eles não são redistribuídos aqui: [`scripts/baixar_dados.sh`](scripts/baixar_dados.sh) os baixa da fonte original.

A única exceção é um resumo agregado por categoria, que alimenta as figuras da triagem e está documentado em [`data/README.md`](data/README.md).

## Como citar

> Ferreira, T. 2027. Precificação dinâmica no varejo: rede neural identificada pelo custo e otimização não linear. Trabalho de Conclusão de Curso (MBA em Data Science e Analytics), USP/ESALQ. Orientadora: Patrícia Belfiore Fávero.

```bibtex
@misc{ferreira2027precificacao,
  author = {Ferreira, Thiago},
  title  = {Precificação dinâmica no varejo: rede neural identificada pelo custo e otimização não linear},
  year   = {2027},
  note   = {Trabalho de Conclusão de Curso, MBA em Data Science e Analytics, USP/ESALQ. Orientadora: Patrícia Belfiore Fávero},
  url    = {https://github.com/tf-ferreira/TCC}
}
```

## Licença

O código está sob a [licença MIT](LICENSE). Os dados seguem os termos do Kilts Center descritos acima.

## English summary

This repository contains the code for an MBA thesis (USP/ESALQ) on dynamic pricing in grocery retail. A neural demand model with vector output for 20 frozen-juice products, a monotone own-price term and a closed-form elasticity matrix is identified through the acquisition cost by a control function, and feeds a nonlinear optimizer with exact gradients that maximizes margin or revenue for each store-week of the test period, under ±15% price bounds and brand-hierarchy constraints. Because cross elasticities are weakly identified, gains are reported per hypothesis ("world") under a robust policy. Data: Dominick's Finer Foods (Kilts Center), which must be downloaded from the source with `scripts/baixar_dados.sh`. Run order, tests and the registry of the numbers produced by the scripts are described above.
