"""A varredura da rede: hiperparâmetros, a estrutura de D33, e o bem externo.

Três famílias numa máquina só, porque as três comparam **redes** e compartilham
o laço de treino, o mesmo painel pivotado e os mesmos diagnósticos.

    hiper       centro mais UM EIXO POR VEZ. Profundidade, largura, dimensão do
                embedding de loja, L2, épocas, largura da sub-rede monótona e o
                intercepto por (loja, produto). São quatro decisões que D33
                deixou em aberto, e elas entram como UMA varredura porque
                nenhuma tem mecanismo que a escada construa: são
                hiperparâmetro, e hiperparâmetro se mede.
    estrutura   a rede restrita contra a SEM restrição, no melhor ponto da
                varredura de hiperparâmetros. É o critério de invalidação de
                D33 (pendência 8.12) e a medida do custo da estrutura.
    bemexterno  com e sem as quatro variáveis do bem externo, na restrita. É o
                critério de invalidação de D23, pendência 8.10, aberta desde
                11/09/2026 e a única que só esta fase pode fechar.

## Por que grade de um eixo por vez, e não grade completa

Grade completa de 7 eixos seria centenas de configurações, e D26 exige médias
sobre sementes em cada uma. O que interessa aqui não é o ótimo global de
hiperparâmetro, é **saber se algum eixo move o resultado além do ruído de
semente**. Um eixo por vez responde isso com um número de execuções que cabe, e
declara o que não responde: interações entre eixos ficam fora, e isso está dito
em vez de escondido.

## Os três eixos de leitura são os de D26 emendada

WMAPE, **viés de nível** e elasticidade própria. O nível não é opcional: a
varredura da perda de D32 mostrou que 63,6% de uma diferença de WMAPE podia ser
puro deslocamento de nível, e sem o terceiro eixo isso fica invisível.

Uso, retomável por fragmentos como as varreduras de árvore:
    python3 src/experiments/varredura_rede.py frj --familia hiper --sementes 0-4
    python3 src/experiments/varredura_rede.py frj --familia hiper --consolidar
    python3 src/experiments/varredura_rede.py frj --familia hiper --comparar
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src" / "models"))
sys.path.insert(0, str(RAIZ / "src" / "data"))
sys.path.insert(0, str(RAIZ / "src" / "experiments"))

import rede  # noqa: E402
from atributos import VERSAO_PIPELINE as VERSAO_PIPELINE_ATUAL  # noqa: E402
from baseline_arvores import metricas  # noqa: E402
from ruido_semente import comparar, consolidar, faixa  # noqa: E402

FRAGMENTOS_PADRAO = Path(os.environ.get("TCC_FRAGMENTOS_REDE",
                                        Path.home() / ".tcc_fragmentos_rede"))

# O CENTRO da triagem. Ele é o ponto de partida da varredura de um eixo por
# vez, e fica congelado: mudá-lo depois da triagem mudaria retroativamente o
# significado da configuração chamada "centro" no artefato já gravado.
CENTRO = {"epocas": 30, "lote": 256, "lr": 3e-3, "l2": 1e-4, "largura": 64,
          "profundidade": 2, "dim_emb": 8, "h_monotona": 8, "p_reserva": 0.05,
          "intercepto": False, "restrita": True}

# O PONTO ADOTADO, saído da triagem de 16/09/2026, e ele é quem as famílias
# `estrutura` e `bemexterno` usam.
#
# HISTÓRICO, e ele importa porque o ponto MUDOU e o motivo é instrutivo.
#
# A primeira triagem rodou sob POISSON e escolheu profundidade 3 com 30 épocas.
# Refeita sob MSLE (família `hiper_msle`), esse ponto cai para DÉCIMO lugar, e
# perde para três configurações acima do limiar corrigido. O mecanismo: sob
# MSLE, 60 épocas é a segunda pior de 23 configurações, enquanto sob Poisson era
# indiferente. **A MSLE sobreajusta mais rápido**, e por isso épocas passou a
# interagir com profundidade, coisa que sob Poisson não acontecia.
#
# No topo sob MSLE, as cinco primeiras empatam nos TRÊS eixos declarados: zero
# de dez pares passam o limiar no WMAPE e zero no nível, e a elasticidade delas
# vai de −1,911 a −1,951. O desempate veio da **identidade de agregação**, que é
# a pendência 2.8, declarada em 14/09 e portanto anterior a esta medição: não é
# critério escolhido depois de ver o resultado. `prof4_ep15` tem folga de
# +0,1273 e violação de 25,50%, contra 52,90% da profundidade 5.
#
# Ressalva medida que limita o uso desse desempate: a correlação entre
# |elasticidade própria| e folga, nas 23 configurações, é **+0,739**. A folga é
# parcialmente mecânica. Dentro do top 5 isso não contamina, porque as próprias
# são iguais dentro do erro; fora dele contamina.
# O CENTRO da terceira triagem, CONGELADO como os outros dois, e pelo mesmo
# motivo, que eu quase esqueci de novo: `hiper_v3` estava centrada em `MELHOR`,
# e atualizar `MELHOR` para a vencedora **redefiniria retroativamente a grade
# que a elegeu**. Um centro de triagem é um artefato histórico; o ponto adotado
# é uma conclusão. Confundir os dois faz o registro se apagar sozinho.
CENTRO_V3 = dict(CENTRO, profundidade=4, epocas=15, perda="msle")

# O PONTO ADOTADO desde 16/09/2026, saído de `hiper_v3` na VALIDAÇÃO com 50
# sementes, pela regra de `escolher_ponto`: dentro de 1 erro padrão pareado da
# melhor (4 elegíveis de 19), depois menos parâmetros. É `prof3_ep15_emb32`,
# com 21.092 parâmetros e WMAPE de validação 50,357 contra 50,346 da melhor.
#
# É quem as famílias que REPORTAM usam: `estrutura`, `bemexterno`, `perda`,
# `cabeca`, `cruzada` e o fechamento. Nenhuma família de triagem depende dele.
MELHOR = dict(CENTRO, profundidade=3, epocas=15, dim_emb=32, perda="msle")

# O centro da triagem sob a régua de D35, segunda rodada. `CENTRO` fica congelado porque
# o artefato de Poisson já está gravado e mudá-lo reescreveria retroativamente o
# que a configuração chamada "centro" significa lá. A triagem sob MSLE é família
# PRÓPRIA, `hiper_msle`, e as duas coexistem como registro.
CENTRO_MSLE = dict(CENTRO, perda="msle")

# Um eixo por vez em torno do centro. O nome da configuração diz o eixo e o
# valor, para que a tabela do consolidado seja legível sem consultar o código.
EIXOS = {
    # Profundidade e épocas foram ESTENDIDAS na segunda rodada de 16/09: na
    # primeira, os dois melhores valores eram os das BORDAS do que se testou
    # (profundidade 3 de {1,2,3} e 15 épocas de {15,30,60}), e parar ali seria
    # reportar como ótimo o limite da grade. Os demais eixos não se moveram
    # além do ruído de semente e ficaram como estavam.
    "profundidade": [1, 3, 4, 5],
    "largura": [32, 128],
    "dim_emb": [4, 16, 32],
    "l2": [1e-5, 1e-3, 1e-2],
    "epocas": [8, 15, 20, 60],
    "h_monotona": [4, 16],
    "intercepto": [True],
}

# A varredura é de um eixo por vez, e por construção ela NÃO vê interação. Os
# combinados abaixo existem para testar a única interação que a primeira rodada
# tornou plausível: os dois eixos que se moveram, juntos.
COMBINADOS = {
    "prof3_ep15": {"profundidade": 3, "epocas": 15},
    "prof4_ep15": {"profundidade": 4, "epocas": 15},
    "prof3_ep15_emb32": {"profundidade": 3, "epocas": 15, "dim_emb": 32},
}


def _grade(base: dict) -> dict[str, dict]:
    """Um eixo por vez em torno de `base`, SEM configurações repetidas.

    A deduplicação entrou em 16/09/2026 e não é cosmética. Com o centro em
    `MELHOR` = (profundidade 4, 15 épocas), quatro nomes da grade descrevem a
    MESMA configuração: `centro`, `profundidade=4`, `epocas=15` e `prof4_ep15`.
    Rodá-las gasta quatro vezes o mesmo treino, e, pior, infla `k` na correção
    de Bonferroni de D26: com 23 nomes o limiar sai de C(23,2)=253 pares em vez
    de C(19,2)=171, e um limiar mais alto **reduz o poder** do teste contra
    diferenças reais. Pagar poder estatístico por nome repetido é prejuízo puro.

    Quando `base` é `CENTRO` ou `CENTRO_MSLE` (profundidade 2, 30 épocas) não há
    colisão nenhuma, de modo que as grades já gravadas de `hiper` e
    `hiper_msle` não mudam. Isso é conferido pelo teste
    `test_a_grade_antiga_nao_muda_com_a_deduplicacao`.
    """
    bruta = {"centro": dict(base)}
    for eixo, valores in EIXOS.items():
        for v in valores:
            bruta[f"{eixo}={v}"] = dict(base, **{eixo: v})
    for nome, ajustes in COMBINADOS.items():
        bruta[nome] = dict(base, **ajustes)

    cfgs, visto = {}, {}
    for nome, cfg in bruta.items():
        chave = impressao(cfg)
        if chave in visto:
            APELIDOS.setdefault(visto[chave], []).append(nome)
            continue
        visto[chave] = nome
        cfgs[nome] = cfg
    return cfgs


# Nomes que colapsaram em outro na deduplicação, para o consolidado poder dizer
# "centro (= profundidade=4, epocas=15, prof4_ep15)" em vez de esconder.
APELIDOS: dict[str, list[str]] = {}


def configuracoes(familia: str) -> dict[str, dict]:
    if familia == "hiper":
        return _grade(CENTRO)
    if familia == "hiper_msle":
        # Pendência 8.15: a triagem que escolheu profundidade 3 rodou sob
        # POISSON, antes de D35. Esta é a mesma grade sob a régua adotada.
        return _grade(CENTRO_MSLE)
    if familia == "hiper_v3":
        # Terceira rodada, centrada em CENTRO_V3 e rodada na VALIDAÇÃO.
        # Ela responde a duas perguntas de uma vez: se algum eixo que não se
        # separou no centro anterior se separa aqui, e se o ponto sobrevive
        # quando a escolha deixa de olhar o teste.
        return _grade(CENTRO_V3)
    if familia == "estrutura":
        return {"restrita": dict(MELHOR, restrita=True),
                "irrestrita": dict(MELHOR, restrita=False)}
    if familia == "bemexterno":
        return {"com_bem_externo": dict(MELHOR, sem_bem_externo=False),
                "sem_bem_externo": dict(MELHOR, sem_bem_externo=True)}
    if familia == "perda":
        # D32 REFEITA SOBRE A REDE, pendência 8.6. A de 15/09 é de árvore: o
        # mecanismo dos dois eixos (ponderação entre SKUs, e média contra
        # mediana) é propriedade da PERDA e deve transferir; a magnitude não
        # tem por que. As três réguas usam a MESMA arquitetura e a MESMA
        # previsão `exp(z)`, de modo que só a perda muda, como no desenho
        # original de D32.
        # `msle_smearing` SAIU da grade e isso não é abandono da pergunta: ela
        # treina idêntico a `msle` (ver `rede.treinar`), de modo que rodá-la
        # como configuração separada gastaria 50 ajustes para reproduzir os
        # mesmos 50 pesos. O efeito da suavização passa a ser gravado como
        # diagnóstico em TODA execução de TODA família, o que responde a mesma
        # pergunta em mais lugares e por menos.
        return {nome: dict(MELHOR, perda=nome) for nome in ("poisson", "msle")}
    if familia == "cabeca":
        # Pendência 8.3: a cabeça de N saídas é o ÚNICO lugar em que a
        # arquitetura vetorial de D31 tem mais parâmetros que a por produto, e
        # é o único gatilho de invalidação de D31 interno ao desenho atual.
        # `posto` fatoriza a cabeça por um gargalo: se o sobreajuste estiver
        # nela, reduzir o posto MELHORA o teste. Se não melhorar, a suspeita
        # cai, e cai com número.
        return {"posto_cheio": dict(MELHOR),
                "posto=8": dict(MELHOR, posto=8),
                "posto=4": dict(MELHOR, posto=4),
                "posto=2": dict(MELHOR, posto=2)}
    if familia == "cruzada":
        # A ESTRUTURA CRUZADA é o suspeito que a medição apontou duas vezes de
        # forma independente: a rede sem restrição tem desvio de cruzada entre
        # células de 0,0826, e a restrita com γ constante tem folga de
        # agregação praticamente nula (52% de violação sob MSLE). γ contextual
        # é a primeira hipótese, e ela não toca a garantia de D33: γ depende do
        # CONTEXTO e não dos preços, de modo que a diagonal continua zerada.
        return {"gama_constante": dict(MELHOR),
                "gama_contextual": dict(MELHOR, gama_contextual=True)}
    raise KeyError(familia)


def impressao(cfg: dict, conjunto: str = "teste") -> str:
    """Impressão digital da configuração, sem a semente.

    Existe por causa de um risco concreto e não hipotético. O fragmento é
    nomeado por `categoria__configuração__semente`, e o NOME da configuração é
    estável enquanto o seu CONTEÚDO não é: quando `MELHOR` mudou de Poisson
    para MSLE em 16/09/2026, a configuração chamada `restrita` passou a
    significar outra coisa, e as 50 sementes já gravadas seriam reaproveitadas
    em silêncio com o significado errado.

    Gravar a impressão no fragmento e recusar a leitura quando ela diverge
    transforma essa disciplina em invariante executável. É a mesma ideia de
    `comparar_registro.py`: o artefato tem de provar que é do que diz ser.

    O `conjunto` entra na impressão pelo mesmo motivo, e o motivo é concreto
    também: `hiper` e `hiper_msle` foram medidas no TESTE, antes de 16/09/2026.
    Se elas passassem a rodar na validação com o mesmo nome de configuração, a
    impressão bateria e as sementes antigas seriam reaproveitadas medindo outra
    coisa. Duas medições da mesma configuração em conjuntos diferentes são
    números diferentes, e o que separa um do outro não é a configuração.
    """
    # A versão do pipeline entra desde 17/09/2026 (pendência 9.6): o vocabulário
    # de loja mudou sem que nenhuma configuração mudasse.
    from atributos import VERSAO_PIPELINE
    itens = sorted((k, repr(v)) for k, v in cfg.items() if k != "semente")
    return hashlib.sha256(repr([itens, conjunto, VERSAO_PIPELINE]).encode()).hexdigest()[:12]


def preparar_dados(categoria: str, painel: Path, sem_bem_externo: bool) -> dict:
    """O painel pivotado. `sem_bem_externo` zera as quatro colunas de D23.

    Zerar em vez de remover mantém a **dimensão da entrada** idêntica entre as
    duas configurações, de modo que a comparação não mistura efeito do atributo
    com efeito de contagem de parâmetros. As colunas zeradas entram no
    escalonamento como constantes, e o desvio zero vira 1 por guarda.
    """
    dados = rede.preparar(categoria, painel)
    if sem_bem_externo:
        alvo = [rede.CONTEXTO_CELULA.index(c)
                for c in ("idx_preco_resto", "vol_resto", "promo_resto", "n_resto")]
        # As QUATRO partes, e não só treino e teste: `bemexterno` roda no teste
        # hoje, mas deixar treino_reduzido e validacao fora seria uma armadilha
        # armada para o dia em que alguma família de seleção precisar zerar o
        # bem externo, e ela zeraria em silêncio só metade do pipeline.
        for parte in (dados["treino"], dados["teste"],
                      dados["treino_reduzido"], dados["validacao"]):
            parte["ctx_celula"][:, alvo] = 0.0
    return dados


# Famílias cuja pergunta é de SELEÇÃO rodam na validação; as que reportam
# rodam no teste. A separação é a regra de D28 aplicada à escolha de
# hiperparâmetro, e não só ao preparo de atributos.
#
# `hiper` e `hiper_msle` NÃO estão aqui, e a ausência é deliberada: elas já
# foram medidas no teste, e os fragmentos existem. Marcá-las como seleção não
# desfaria o vazamento retroativamente, só produziria uma pasta com metades
# incomparáveis. Elas ficam como estão, com o defeito declarado em 8.21, e quem
# responde a pergunta de seleção de agora em diante é `hiper_v3`.
FAMILIAS_DE_SELECAO = {"hiper_v3"}


def conjunto_de(familia: str) -> str:
    return "validacao" if familia in FAMILIAS_DE_SELECAO else "teste"


def em_validacao(dados: dict) -> dict:
    """A mesma estrutura de `preparar`, com treino reduzido e validação no
    lugar de treino e teste. Nada do período de teste entra."""
    return dict(dados, treino=dados["treino_reduzido"],
                teste=dados["validacao"],
                n_lojas=dados["n_lojas_validacao"])


def uma_execucao(dados: dict, cfg: dict) -> dict:
    """Treina, mede os três eixos de D26 e os diagnósticos de D19/8.12/8.13."""
    cfg = {k: v for k, v in cfg.items() if k != "sem_bem_externo"}
    saida = rede.treinar(dados, cfg)
    y = dados["teste"]["alvo"].ravel()
    p = saida["pred_teste"].ravel()
    m = metricas(y, p)
    T = saida["tensores"]
    diag = rede.diagnosticos(saida["modelo"], T["ctx_teste"], T["u_teste"],
                             T["loja_teste"], dados["teste"]["precos"])
    n_par = sum(t.numel() for t in saida["modelo"].parameters())

    # DECOMPOSIÇÃO DO VIÉS DE NÍVEL, acrescentada em 16/09/2026 e o motivo é uma
    # correção minha. O nível medido na janela de avaliação é produto de duas
    # coisas que estavam somadas sem serem separadas:
    #
    #     nível_avaliação = encolhimento × transferência
    #
    # O **encolhimento** é o nível medido DENTRO do treino. Sob MSLE a previsão
    # é `exp(z)` e `exp(E[ln y]) < E[y]` para alvo assimétrico, de modo que o
    # modelo subprevê a média já em amostra: é desigualdade de Jensen, é
    # propriedade da perda, e não some mudando de janela.
    #
    # A **transferência** é o resto: o quanto a janela de avaliação difere da de
    # treino. Neste painel o volume cai monotonicamente, o que empurra o nível
    # para CIMA, no sentido oposto ao encolhimento.
    #
    # Sem separar os dois, um nível de 0,9985 no teste parece calibração e é
    # coincidência: encolhimento de 0,59 dividido por uma queda de volume de
    # 0,64. O mesmo modelo na validação dá 0,72, e não porque piorou.
    nivel_treino = float(saida["pred_treino"].sum() / dados["treino"]["alvo"].sum())
    nivel_aval = float(p.sum() / y.sum())

    # O QUE A CORREÇÃO DE DUAN FARIA, de graça, em toda execução. `msle` e
    # `msle_smearing` treinam idêntico (ver `rede.treinar`), de modo que a
    # suavização é multiplicação pós-treino e não precisa de ajuste próprio.
    # Quando a perda já É `msle_smearing`, o fator já está aplicado em `p` e
    # estas colunas repetem as de cima, o que é a conferência de que a conta
    # bate dos dois lados.
    f = saida["fator_de_duan"]
    aplicado = cfg.get("perda") == "msle_smearing"
    p_suave = p if aplicado else p * f
    m_suave = metricas(y, p_suave)
    diag = dict(diag,
                vies_de_nivel_treino=nivel_treino,
                razao_de_transferencia_de_nivel=(nivel_aval / nivel_treino
                                                 if nivel_treino else float("nan")),
                fator_de_duan=float(f),
                wmape_com_suavizacao=float(m_suave["wmape_pct"]),
                vies_de_nivel_com_suavizacao=float(p_suave.sum() / y.sum()))
    return {"wmape_pct": float(m["wmape_pct"]),
            "vies_de_nivel": nivel_aval,
            "vies_de_nivel_treino": nivel_treino,
            "elast_h10": float(diag["elast_propria_mediana"]),
            "n_atributos": int(n_par),
            "rmse": float(m["rmse"]), "rmse_log": float(m["rmse_log"]),
            "diagnosticos": diag}


EXTRAS = ["vies_de_nivel_treino", "razao_de_transferencia_de_nivel",
          "fator_de_duan", "wmape_com_suavizacao",
          "vies_de_nivel_com_suavizacao",
          "pct_propria_positiva", "pct_otimo_interior",
          "dp_da_cruzada_entre_celulas", "dp_da_propria_entre_celulas",
          "elast_propria_p10", "elast_propria_p90", "pct_cruzada_positiva",
          # Acrescentados em 16/09 com a identidade de agregação (2.8). Um
          # fragmento antigo não os tem, e por isso a consolidação é TOLERANTE
          # a chave ausente: ela omite em vez de inventar zero.
          "folga_de_agregacao_mediana", "pct_violacao_agregacao",
          "elast_agregada_mediana", "propria_ponderada_mediana"]


def escolher_ponto(consolidado: dict, comparacao: dict, cfgs: dict) -> dict:
    """A REGRA DE ESCOLHA DO PONTO ADOTADO, declarada em 16/09/2026.

    Ela existe como código, e não como parágrafo, por um motivo só: um critério
    escrito depois de ver a tabela é indistinguível de um critério escolhido
    para produzir a linha que se queria. Este está escrito **antes** de a
    tabela de 50 sementes existir, tem teste, e roda sozinho.

    ## A regra

    1. **Grupo elegível**: as configurações cujo `|z|` pareado contra a melhor
       em WMAPE é **no máximo 1**, isto é, que estão dentro de UM erro padrão
       pareado da melhor.

       **EMENDA DE 16/09/2026, e ela corrige a versão original desta regra.**
       O degrau 1 dizia "não passa o limiar corrigido de D26", e isso estava
       errado por uma inversão de propósito. A correção de Bonferroni existe
       para controlar **descoberta falsa**: ela torna difícil *declarar uma
       diferença* que não existe. Usar o limiar dela como definição de
       "indistinguível" a transforma em licença para *declarar equivalência*, e
       nessa direção ela é anticonservadora. Pior, o defeito cresce com o
       tamanho da grade: quanto mais configurações eu comparo, mais largo o
       limiar, mais fácil chamar uma configuração ruim de empatada com a melhor.
       Comparar mais coisas não pode tornar mais fácil declarar que duas são
       iguais.

       O sintoma apareceu medido: sob o limiar de 3,622, `profundidade=1`
       entrava no grupo com `z = 3,32`, isto é, uma diferença a mais de três
       erros padrão de zero, e ganhava por ser a menor rede. Chamar isso de
       empate é abuso de linguagem.

       A banda de **um erro padrão** é a regra clássica de seleção de modelo
       (a "1-SE rule" da validação cruzada), não foi inventada aqui, e é
       **mais restritiva** e não menos: ela leva o grupo elegível de 10 para 4.
       Na formulação pareada ela é `|z| ≤ 1`, que é a que fica, por coerência
       com D26, que fez pareado tudo neste projeto.

       **Esta emenda é pós-hoc, e isso fica dito.** Eu já tinha visto que a
       versão original escolhia `profundidade=1`. O que sustenta a troca não é
       o desfecho: é que o argumento acima não depende de qual configuração
       ganha, e que a mudança restringe a minha liberdade de escolha em vez de
       ampliá-la.
    2. Dentro do grupo, **menos parâmetros** ganha.
    3. Empate exato em parâmetros: **menos épocas** ganha.
    4. Empate ainda: **menor WMAPE médio** ganha.
    5. Empate ainda: ordem alfabética do nome, só para ser determinístico.

    ## Por que parcimônia, e não os outros desempates

    Os dois candidatos naturais estão queimados, e isso está medido:

    - **Identidade de agregação.** Nas 19 configurações da validação,
      `corr(|elast|, %violação) = −0,443`: quem tem própria mais negativa viola
      menos, mecanicamente. Desempatar por agregação é escolher a elasticidade
      maior por via indireta.
    - **Proximidade do estimando de D18.** Seria circular. D16, no item 7,
      existe para VALIDAR a elasticidade da rede contra essa referência;
      escolher por ela transforma a validação em tautologia.

    A parcimônia tem uma propriedade que nenhum critério calculado a partir da
    saída do modelo tem: **a contagem de parâmetros não depende do resultado**.
    Ela não pode ficar confundida com nada, porque é lida da configuração e não
    da medição. Some-se a isso o precedente de D30, que excluiu a contagem de
    clientes por parcimônia depois de ela se mostrar inerte, e o de D36, em que
    menos liberdade na cruzada é o que mantém a própria perto do estimando
    independente sob a colinearidade medida em D29.

    ## A objeção a esta regra, dita por mim e não escondida

    **Contagem de parâmetros não mede capacidade efetiva quando `epocas` é um
    dos eixos da grade.** Uma configuração com 21.684 parâmetros e 60 épocas tem
    mais capacidade usada que outra com os mesmos 21.684 e 8 épocas. O degrau 3
    endereça isso em parte, e só em parte: ele só age em empate exato.

    A objeção é real e fica registrada. A razão de aceitar a regra mesmo assim é
    que a alternativa não é um critério melhor, é um critério contaminado.

    ## A ressalva de honestidade, e ela importa

    Eu **já projetei** o grupo elegível de 50 sementes (por `1/√n` sobre os `z`
    de 5 sementes) e portanto consigo antecipar o vencedor desta regra. Declarar
    um critério cujo desfecho provável já se conhece é mais fraco do que
    declará-lo às cegas, e seria desonesto apresentar isto como se fosse cego.
    O que sustenta a regra não é o desconhecimento do resultado, é o fato de que
    as quantidades que ela usa não dependem dele.

    Devolve o vencedor e o grupo inteiro, para que o artefato registre quem
    estava empatado e não só quem ganhou.
    """
    eixo = comparacao["eixos"]["wmape"]
    limiar = comparacao["limiar_z_corrigido"]
    cfg = consolidado["configuracoes"]
    medias = eixo["medias"]
    melhor = min(medias, key=lambda k: medias[k])

    def z_contra(a, b):
        d = (eixo["pares"].get(f"{a}__menos__{b}")
             or eixo["pares"].get(f"{b}__menos__{a}"))
        return 0.0 if d is None else abs(d["z"])

    BANDA = 1.0          # um erro padrão pareado; ver o degrau 1 da docstring
    grupo = [k for k in medias
             if k == melhor or z_contra(melhor, k) <= BANDA]

    # `epocas` vem da GRADE e não do consolidado, que não guarda a configuração.
    # Ler do consolidado devolveria 0 para todo mundo e deixaria o degrau 3
    # inerte em silêncio, que é o modo de falhar em que a regra documentada e a
    # regra executada divergem sem ninguém notar.
    faltando = [k for k in grupo if k not in cfgs]
    if faltando:
        raise SystemExit(f"configurações fora da grade atual: {faltando}")

    def chave(nome):
        return (cfg[nome]["n_atributos"], cfgs[nome]["epocas"],
                medias[nome], nome)

    ordenado = sorted(grupo, key=chave)
    escolhido = ordenado[0]
    return {
        "regra": "dentro de 1 erro padrao pareado da melhor no WMAPE, depois "
                 "menos parametros, depois menos epocas, depois menor WMAPE, "
                 "depois nome",
        "declarada_em": "2026-09-16",
        "banda_em_erros_padrao": BANDA,
        "limiar_de_bonferroni_nao_usado_na_banda": limiar,
        "medido_em": consolidado.get("medido_em", "teste"),
        "melhor_no_wmape": melhor,
        "limiar_z_corrigido": limiar,
        "n_no_grupo": len(grupo),
        "grupo": [{"config": k, "wmape_media": medias[k],
                   "n_atributos": cfg[k]["n_atributos"],
                   "epocas": cfgs[k]["epocas"],
                   "z_contra_o_melhor": z_contra(melhor, k)}
                  for k in ordenado],
        "escolhido": escolhido,
    }


def conferir_fragmentos(categoria: str, frag: Path, cfgs: dict,
                        conjunto: str) -> None:
    """Recusa qualquer fragmento cuja impressão não seja a da família atual.

    Rodava só no caminho de treino, e faltava no de consolidação, que é o mais
    importante dos dois: é ali que fragmento vira número de relatório. Um
    fragmento órfão, de uma configuração que saiu da grade, passava despercebido
    e entrava na média.
    """
    for arquivo in sorted(frag.glob(f"{categoria}__*.json")):
        d = json.loads(arquivo.read_text())
        nome, semente = d.get("config"), d.get("semente")
        if nome not in cfgs:
            raise SystemExit(
                f"\nFRAGMENTO ÓRFÃO: {arquivo.name}\n"
                f"  a configuração '{nome}' não existe mais nesta família.\n")
        esperada = impressao(dict(cfgs[nome], semente=semente), conjunto)
        if d.get("impressao") is None:
            raise SystemExit(
                f"\nFRAGMENTO SEM IMPRESSÃO: {arquivo.name}\n"
                f"  Anterior ao invariante de 16/09/2026, e por isso não consegue\n"
                f"  provar de que configuração é. Ele NÃO pode ser recarimbado:\n"
                f"  recarimbar sem ter o que conferir é o 'confie em mim' que o\n"
                f"  invariante existe para não precisar. Duas saídas honestas:\n"
                f"    (a) recalcular a família:  rm -rf {frag}\n"
                f"    (b) manter o consolidado que já existe e NÃO reconsolidar.\n")
        if d.get("impressao") != esperada:
            raise SystemExit(
                f"\nFRAGMENTO DE OUTRA CONFIGURAÇÃO: {arquivo.name}\n"
                f"  gravado com impressão {d.get('impressao')} "
                f"(medido em {d.get('medido_em', 'não registrado')}),\n"
                f"  esperado {esperada} (medido em {conjunto}).\n"
                f"  Apague a pasta da família e recalcule:  rm -rf {frag}\n")


def restampar(categoria: str, frag: Path, cfgs: dict, conjunto: str) -> int:
    """Recarimba fragmentos antigos com a impressão nova, uma vez.

    Existe porque acrescentar `conjunto` à impressão mudou o hash de TODOS os
    fragmentos já gravados, inclusive os de `hiper` e `hiper_msle`, que são
    medições válidas e caras que não quero refazer só por causa de metadado.

    A migração não pode lavar fragmento errado, e não lava: ela recalcula a
    impressão a partir da `configuracao` GRAVADA no próprio fragmento e depois
    exige que o resultado bata com a impressão da grade ATUAL. Se o conteúdo da
    configuração mudou de verdade, o recarimbo falha em vez de silenciar.
    """
    n = 0
    for arquivo in sorted(frag.glob(f"{categoria}__*.json")):
        d = json.loads(arquivo.read_text())
        nome = d.get("config")
        if nome not in cfgs:
            raise SystemExit(f"fragmento órfão, não recarimbo: {arquivo.name}")
        from atributos import VERSAO_PIPELINE
        if d.get("versao_pipeline") != VERSAO_PIPELINE:
            raise SystemExit(
                f"\nRECARIMBO RECUSADO: {arquivo.name}\n"
                f"  calculado com o pipeline '{d.get('versao_pipeline', 'anterior a 17/09')}',\n"
                f"  o atual é '{VERSAO_PIPELINE}'. O modelo mudou sem a configuração\n"
                f"  mudar, então recarimbar lavaria fragmento errado. Recalcule.\n")
        if "configuracao" not in d:
            raise SystemExit(
                f"\nSEM CONFIGURAÇÃO GRAVADA: {arquivo.name}\n"
                f"  Este fragmento é anterior a 16/09/2026 e não guarda a\n"
                f"  configuração com que foi calculado. Não há o que conferir, e\n"
                f"  carimbar às cegas seria inventar proveniência. Recalcule a\n"
                f"  família, ou mantenha o consolidado existente sem reconsolidar.\n")
        propria = impressao(dict(d["configuracao"], semente=d["semente"]), conjunto)
        da_grade = impressao(dict(cfgs[nome], semente=d["semente"]), conjunto)
        if propria != da_grade:
            raise SystemExit(
                f"\nRECARIMBO RECUSADO: {arquivo.name}\n"
                f"  a configuração gravada no fragmento não é a da grade atual.\n"
                f"  Isto é divergência de conteúdo, não de metadado: recalcule.\n")
        if d.get("impressao") == propria and d.get("medido_em") == conjunto:
            continue
        d["impressao"], d["medido_em"] = propria, conjunto
        arquivo.write_text(json.dumps(d, ensure_ascii=False))
        n += 1
    return n


def consolidar_extras(categoria: str, frag: Path) -> dict:
    """Média sobre sementes dos diagnósticos, que `consolidar` não conhece."""
    por: dict[str, dict[str, list]] = {}
    for arquivo in sorted(frag.glob(f"{categoria}__*.json")):
        d = json.loads(arquivo.read_text())
        alvo = por.setdefault(d["config"], {k: [] for k in EXTRAS})
        for k in EXTRAS:
            if k in d["diagnosticos"]:
                alvo[k].append(d["diagnosticos"][k])
    # O `n` por diagnóstico não é redundante com `n_sementes`, e a diferença é
    # um modo de falhar real: quando um diagnóstico NOVO é acrescentado, os
    # fragmentos antigos não o têm, e a média sairia sobre um subconjunto das
    # sementes sem nada reclamar. Gravar o n por chave transforma isso em algo
    # que o artefato mostra, e o aviso abaixo em algo que a execução grita.
    saida = {nome: {k: {"media": float(np.mean(v)),
                        "n": len(v),
                        "erro_padrao_da_media":
                            float(np.std(v, ddof=1) / np.sqrt(len(v)))
                            if len(v) > 1 else None}
                    for k, v in d.items() if v}
             for nome, d in sorted(por.items())}
    for nome, d in saida.items():
        ns = {x["n"] for x in d.values()}
        if len(ns) > 1:
            print(f"AVISO: em '{nome}' os diagnósticos têm n diferentes {sorted(ns)}. "
                  f"Fragmentos de épocas distintas do código estão misturados; "
                  f"apague a pasta da família e recalcule.", flush=True)
    return saida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--familia", default="hiper",
                    choices=["hiper", "hiper_msle", "hiper_v3",
                             "estrutura", "bemexterno", "perda", "cabeca",
                             "cruzada"])
    ap.add_argument("--config", nargs="+", default=[])
    ap.add_argument("--sementes", default="0-4")
    ap.add_argument("--consolidar", action="store_true")
    ap.add_argument("--restampar", action="store_true",
                    help="migração de uma vez: recarimba fragmentos antigos "
                         "com a impressão que inclui o conjunto de medição")
    ap.add_argument("--comparar", action="store_true")
    ap.add_argument("--fragmentos", default=str(FRAGMENTOS_PADRAO))
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--saida", default="reports")
    a = ap.parse_args()

    frag = Path(a.fragmentos) / a.familia
    destino = Path(a.saida) / f"rede_varredura_{a.categoria}_{a.familia}.json"

    if a.restampar:
        n = restampar(a.categoria, frag, configuracoes(a.familia),
                      conjunto_de(a.familia))
        print(f"{n} fragmentos recarimbados em {frag}")
        return

    if a.comparar:
        alvo = Path(a.saida) / f"comparacao_rede_{a.categoria}_{a.familia}.json"
        consolidado = json.loads(destino.read_text())
        saida = comparar(consolidado)
        if a.familia in FAMILIAS_DE_SELECAO:
            saida["escolha_do_ponto"] = escolher_ponto(
                consolidado, saida, configuracoes(a.familia))
        alvo.write_text(json.dumps(saida, indent=2, ensure_ascii=False))
        print(f"{saida['n_pares']} pares, limiar de z corrigido "
              f"{saida['limiar_z_corrigido']:.3f}\n")
        for eixo, bloco in saida["eixos"].items():
            print(f"  --- {eixo} ({bloco['n_sementes']} sementes) ---")
            for nome, m in sorted(bloco["medias"].items(), key=lambda kv: kv[1]):
                print(f"      {nome:26s} {m:10.4f}")
            for par, d in bloco["pares"].items():
                if d["passa_limiar"]:
                    print(f"      {par:52s} dif {d['diferenca_media']:+9.4f} "
                          f"z {d['z']:+9.2f}  passa")
            po = bloco.get("poder") or {}
            if po:
                print(f"      [poder] {po['n_pares_que_passam']}/{po['n_pares']} "
                      f"pares passam | ep pareado mediano "
                      f"{po['ep_pareado_mediano']:.4f} | DMD mediana "
                      f"{po['diferenca_minima_detectavel_mediana']:.4f} | "
                      f"amplitude das médias {po['amplitude_das_medias']:.4f}")
                if po.get("tem_ordem_de_melhor"):
                    veredito = ("DECIDÍVEL" if po["escolha_entre_1_e_2_e_decidivel"]
                                else "NÃO DECIDÍVEL")
                    dmd = po["diferenca_minima_detectavel_do_par_1_2"]
                    print(f"      [poder] 1ª {po['melhor']} vs 2ª {po['segunda']}: "
                          f"distância {po['distancia_melhor_para_segunda']:.4f}, "
                          f"DMD {dmd:.4f} -> {veredito}")
                    if not po["escolha_entre_1_e_2_e_decidivel"]:
                        print(f"      [poder] para decidir esse par seriam ~"
                              f"{po['sementes_para_decidir_1_contra_2']} sementes "
                              f"(ordem de grandeza)")
                    print(f"      [poder] amplitude do topo {po['n_do_topo']}: "
                          f"{po['amplitude_do_topo']:.4f}")
            print()
        esc = saida.get("escolha_do_ponto")
        if esc:
            print(f"  === REGRA DE ESCOLHA declarada em {esc['declarada_em']} ===")
            print(f"      melhor no WMAPE: {esc['melhor_no_wmape']}")
            print(f"      grupo indistinguível: {esc['n_no_grupo']} configurações")
            for g in esc["grupo"]:
                print(f"        {g['config']:22s} {g['n_atributos']:7d} params "
                      f"{g['epocas']:3d} ep  WMAPE {g['wmape_media']:7.3f}  "
                      f"z {g['z_contra_o_melhor']:5.2f}")
            print(f"      ESCOLHIDO por parcimônia: {esc['escolhido']}\n")
        print(f"gravado em {alvo}")
        return

    if a.consolidar:
        conferir_fragmentos(a.categoria, frag, configuracoes(a.familia),
                            conjunto_de(a.familia))
        resumo = consolidar(a.categoria, frag, destino)
        extras = consolidar_extras(a.categoria, frag)
        bloco = json.loads(destino.read_text())
        for nome in bloco["configuracoes"]:
            bloco["configuracoes"][nome]["diagnosticos"] = extras.get(nome, {})
        destino.write_text(json.dumps(bloco, indent=2, ensure_ascii=False))
        for nome, outros in sorted(APELIDOS.items()):
            print(f"nota: '{nome}' é a mesma configuração que "
                  f"{', '.join(outros)}; rodou uma vez só.")
        print(f"{'configuração':22s} {'n':>3} {'WMAPE':>9} {'ep':>7} "
              f"{'nível':>8} {'elast':>8} {'%pos':>6} {'%int':>6} "
              f"{'folga':>8} {'%viol':>7} {'params':>8}")
        for nome, d in sorted(resumo.items(), key=lambda kv: kv[1]["wmape_media"]):
            e = extras.get(nome, {})
            print(f"{nome:22s} {d['n_sementes']:3d} {d['wmape_media']:8.3f}% "
                  f"{d['wmape_erro_padrao_da_media'] or float('nan'):7.3f} "
                  f"{d.get('vies_de_nivel_media', float('nan')):8.4f} "
                  f"{d.get('elast_h10_media', float('nan')):8.4f} "
                  f"{e.get('pct_propria_positiva', {}).get('media', float('nan')):6.2f} "
                  f"{e.get('pct_otimo_interior', {}).get('media', float('nan')):6.2f} "
                  f"{e.get('folga_de_agregacao_mediana', {}).get('media', float('nan')):8.4f} "
                  f"{e.get('pct_violacao_agregacao', {}).get('media', float('nan')):7.2f} "
                  f"{d['n_atributos']:8d}")
        print(f"\ngravado em {destino}")
        return

    cfgs = configuracoes(a.familia)
    nomes = a.config or sorted(cfgs)
    sementes = faixa(a.sementes)
    frag.mkdir(parents=True, exist_ok=True)
    # Ordem por semente: se a execução for interrompida, todas as configurações
    # ficam com o mesmo número de sementes e o parcial já é comparável.
    # Um fragmento só é reaproveitado se a impressão digital bater. Se não
    # bater, a execução PARA e diz o que fazer, em vez de misturar.
    pendentes = []
    for s in sementes:
        for c in nomes:
            arq = frag / f"{a.categoria}__{c}__{s:03d}.json"
            if not arq.exists():
                pendentes.append((c, s))
                continue
            gravada = json.loads(arq.read_text()).get("impressao")
            esperada = impressao(dict(cfgs[c], semente=s),
                                 conjunto_de(a.familia))
            if gravada != esperada:
                raise SystemExit(
                    f"\nFRAGMENTO DE OUTRA CONFIGURAÇÃO: {arq.name}\n"
                    f"  gravado com impressão {gravada}, esperado {esperada}.\n"
                    f"  A configuração '{c}' mudou de conteúdo desde que este\n"
                    f"  fragmento foi calculado. Apague a pasta da família e\n"
                    f"  recalcule:  rm -rf {frag}\n")
    if not pendentes:
        print("nada pendente")
        return

    cache: dict[bool, dict] = {}
    for nome, semente in pendentes:
        cfg = dict(cfgs[nome], semente=semente)
        sbe = bool(cfg.get("sem_bem_externo", False))
        if sbe not in cache:
            cache[sbe] = preparar_dados(a.categoria, Path(a.painel), sbe)
        base = cache[sbe]
        conjunto = conjunto_de(a.familia)
        if conjunto == "validacao":
            base = em_validacao(base)
        r = uma_execucao(base, cfg)
        r.update({"config": nome, "semente": semente,
                  "impressao": impressao(cfg, conjunto), "configuracao": cfg,
                  "medido_em": conjunto, "versao_pipeline": VERSAO_PIPELINE_ATUAL})
        (frag / f"{a.categoria}__{nome}__{semente:03d}.json").write_text(
            json.dumps(r, ensure_ascii=False))
        print(f"{nome} semente {semente}: WMAPE {r['wmape_pct']:.3f}%, "
              f"nível {r['vies_de_nivel']:.4f}, elast {r['elast_h10']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
