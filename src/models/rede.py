"""A rede de demanda: saída vetorial, monotonicidade separável, ligação log.

Implementa **D31** (uma passada prevê o vetor de N demandas da célula), **D32**
(perda de Poisson), **D33** (monotonicidade por termo próprio separável) e
**D34** (ligação log na saída).

## A forma, e cada pedaço dela tem dono

    ln V̂ᵢ = gᵢ(x) + Σ_{j≠i} γᵢⱼ·uⱼ − mᵢ(uᵢ)

com `uᵢ = ln(pᵢ) − cᵢ`, onde `cᵢ` é a média de `ln pᵢ` **na janela de treino**
(D28: nada é ajustado fora dela).

| pedaço | o que é | por que assim |
|---|---|---|
| `gᵢ(x)` | MLP livre sobre o contexto | o nível pode depender de tudo |
| `γᵢⱼ` | matriz N×N com **diagonal zerada** | as cruzadas ficam livres em sinal |
| `mᵢ(uᵢ)` | sub-rede **monótona crescente** em `uᵢ` | `∂ln V̂ᵢ/∂uᵢ = −mᵢ′(uᵢ) ≤ 0` |

**`gᵢ` não vê `pᵢ`, e isso não é detalhe de implementação, é a garantia.** Se o
contexto contivesse o preço próprio, `∂gᵢ/∂uᵢ` entraria sem restrição e a
monotonicidade morreria em silêncio. É por isso que os 20 preços entram **apenas**
por `γ` e por `m`, e o contexto não tem nenhum preço da semana corrente. As
defasagens `preco_lag1` e `preco_lag2` entram no contexto porque são preços de
semanas passadas, e não variáveis de decisão da semana corrente.

**A garantia é global e não só na caixa viável.** `mᵢ` tem pesos não negativos e
ativação monótona, logo `mᵢ′ ≥ 0` em todo o domínio. A qualificação de caixa que
a emenda de D33 a D19 introduziu é o que a **forma desenhada** (a alternativa não
adotada) precisaria; a via separável entrega mais do que ela promete.

## A matriz de elasticidades sai dos parâmetros, e não de diferença finita

Como a saída é `ln V̂` e a entrada é `ln p`, a Jacobiana em log-log **é** a matriz
de elasticidades:

    diagonal      ∂ln V̂ᵢ/∂ln pᵢ = −mᵢ′(uᵢ)      varia com o preço próprio
    fora dela     ∂ln V̂ᵢ/∂ln pⱼ = γᵢⱼ           constante entre células

É isso que D16 valida e que a etapa 8 consome, e é o que torna a identidade de
agregação da pendência 2.8 uma conta e não uma estimativa.

**O custo declarado da matriz constante:** as elasticidades cruzadas não variam
entre células. É a mesma restrição que a forma de elasticidade constante tinha,
mas só na fora-diagonal, e a própria, que é quem cria o ótimo interior, continua
variando. Se a rede sem restrição mostrar que as cruzadas variam
sistematicamente, `γ` passa a depender do contexto, e isso é hiperparâmetro.

## A rede SEM restrição é artefato próprio, não descarte

`--irrestrita` treina `ln V̂ = MLP([x, u])`, sem estrutura nenhuma. Ela é o
instrumento de três medições que sem ela não saem (pendência 8.12): o critério de
invalidação de D33, a fração de derivadas com sinal economicamente absurdo que
D19 existe para evitar, e ρ intra-célula sem o confundimento da forma imposta.

Uso, e a primeira linha é a que produz o **artefato canônico** do item 6:

    python3 src/models/rede.py frj --adotada
    python3 src/models/rede.py frj --adotada --irrestrita --sufixo _irrestrita

`--adotada` lê a configuração de `varredura_rede.MELHOR`, que é o mesmo objeto
que as varreduras usaram. É ponto único de propósito: um modelo salvo com
hiperparâmetro digitado à mão pode divergir do que foi medido sem que nada
acuse, e o artefato canônico é o que **D16 valida no item 7**.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src" / "data"))
sys.path.insert(0, str(RAIZ / "src" / "models"))

from atributos import (CATEGORICAS, construir,  # noqa: E402
                       preparar_treino_teste)
from baseline_arvores import metricas  # noqa: E402
from preparo import INDICE_RESERVA, tempo_truncado  # noqa: E402

import funcao_de_controle  # noqa: E402

# ---------------------------------------------------------------------------
# As especificações do contexto
# ---------------------------------------------------------------------------
#
# "adotada" é a do item 9 (22/09/2026), e continua reproduzível byte a byte.
# "v1" é a da primeira versão escrita (D41 e D42, 23/09/2026): sai o que está
# entre o preço e o volume (`vol_resto` e `n_resto` são desfecho, D30) e entra a
# função de controle do custo (`v_cf`, `falta_v_cf`), que D42 emenda em D25.
#
# A especificação ATIVA mora nas duas listas abaixo, e elas são mutadas NO LUGAR
# por `usar_especificacao`, porque vários módulos guardam referência a elas.
# Quem precisa da largura do contexto deve lê-la na hora do uso, nunca na
# importação: `contrafactual_item9.fatias_dos_lags` foi corrigido por isso.
#
# As quatro defasagens são SEMPRE os quatro primeiros blocos de CONTEXTO_SKU,
# em qualquer especificação; o contrafactual sequencial depende disso e
# `test_as_defasagens_abrem_o_contexto_de_sku_em_toda_especificacao` fixa.
_LAGS = ["preco_lag1", "preco_lag2", "falta_preco_lag1", "falta_preco_lag2"]
ESPECIFICACOES = {
    "adotada": {
        "celula": ["sen_1", "cos_1", "sen_2", "cos_2", "tempo",
                   "idx_preco_resto", "vol_resto", "promo_resto", "n_resto"],
        "sku": list(_LAGS),
        "funcao_de_controle": False,
    },
    "v1": {
        "celula": ["sen_1", "cos_1", "sen_2", "cos_2", "tempo",
                   "idx_preco_resto", "promo_resto"],
        "sku": list(_LAGS) + list(funcao_de_controle.COLUNAS),
        "funcao_de_controle": True,
    },
    # As duas vizinhas da v1, que existem para os critérios declarados de D42 e
    # D43: a v1 SEM a função de controle (o WMAPE de D42 é comparado contra ela)
    # e a v1 COM o código de promoção declarado, congelado no contrafactual como
    # as outras colunas de contexto (D43 decide se ele vira cenário).
    "base_v1": {
        "celula": ["sen_1", "cos_1", "sen_2", "cos_2", "tempo",
                   "idx_preco_resto", "promo_resto"],
        "sku": list(_LAGS),
        "funcao_de_controle": False,
    },
    "v1_codigo": {
        "celula": ["sen_1", "cos_1", "sen_2", "cos_2", "tempo",
                   "idx_preco_resto", "promo_resto"],
        "sku": list(_LAGS) + list(funcao_de_controle.COLUNAS) + ["promo_proprio"],
        "funcao_de_controle": True,
    },
}

# Contexto por CÉLULA: uma cópia por (loja, semana).
CONTEXTO_CELULA = list(ESPECIFICACOES["adotada"]["celula"])
# Contexto por SKU: N cópias, uma por produto, achatadas na linha da célula.
CONTEXTO_SKU = list(ESPECIFICACOES["adotada"]["sku"])
# Nenhum preço da semana corrente entra no contexto. Ver o docstring. A única
# função do preço observado que entra é `v_cf`, e só na "v1", por D42.
ESPECIFICACAO_ATIVA = "adotada"


def usar_especificacao(nome: str) -> None:
    """Ativa uma especificação, mutando as listas no lugar."""
    global ESPECIFICACAO_ATIVA
    if nome not in ESPECIFICACOES:
        raise ValueError(f"especificação desconhecida: {nome}")
    CONTEXTO_CELULA[:] = ESPECIFICACOES[nome]["celula"]
    CONTEXTO_SKU[:] = ESPECIFICACOES[nome]["sku"]
    ESPECIFICACAO_ATIVA = nome


def com_funcao_de_controle() -> bool:
    return bool(ESPECIFICACOES[ESPECIFICACAO_ATIVA]["funcao_de_controle"])


# ---------------------------------------------------------------------------
# Pivô: o painel longo vira uma linha por célula
# ---------------------------------------------------------------------------

def pivotar(longo: pd.DataFrame, n: int) -> dict:
    """Painel longo (célula, SKU) para largo (célula), com alvo vetorial.

    Falha alto se alguma célula não tiver exatamente os N SKUs: o painel de
    modelagem só tem células completas por D31, e uma célula furada aqui viraria
    alvo com buraco silencioso.
    """
    d = longo.sort_values(["store", "week", "sku"]).reset_index(drop=True)
    chaves = d[["store", "week"]].drop_duplicates().reset_index(drop=True)
    c = len(chaves)
    if len(d) != c * n:
        raise ValueError(f"{len(d)} linhas não são {c} células × {n} SKUs")
    skus = d["sku"].to_numpy().reshape(c, n)
    if not np.all(skus == skus[0]):
        raise ValueError("as células não têm o mesmo conjunto de SKUs na mesma ordem")

    precos = np.stack([d[f"preco_{i:02d}"].to_numpy().reshape(c, n)[:, 0]
                       for i in range(n)], axis=1)
    return {
        "chaves": chaves,
        "precos": precos.astype("float64"),
        "alvo": d["alvo"].to_numpy(float).reshape(c, n),
        # O contexto de célula é o mesmo nas N linhas; a primeira basta, e o
        # teste `test_contexto_de_celula_e_constante_dentro_da_celula` fixa isso.
        "ctx_celula": np.stack(
            [d[col].to_numpy(float).reshape(c, n)[:, 0] for col in CONTEXTO_CELULA],
            axis=1),
        "ctx_sku": np.concatenate(
            [d[col].to_numpy(float).reshape(c, n) for col in CONTEXTO_SKU], axis=1),
        "loja": d["store"].to_numpy().reshape(c, n)[:, 0],
    }


def centro_de_preco(precos_treino: np.ndarray) -> np.ndarray:
    """`cᵢ`, a média de `ln pᵢ` na janela de treino. D28: só do treino."""
    return np.log(precos_treino).mean(axis=0)


def particionar_selecao(longo: pd.DataFrame, janela_treino: list) -> tuple:
    """O pipeline de SELEÇÃO, construído do painel CRU e não recortado do pronto.

    Existe porque escolher hiperparâmetro pelo WMAPE de **teste** é a segunda
    regra de D28 sendo violada, e a seleção é a forma pior dela: escolher a
    melhor entre 19 configurações e reportar o número dela no mesmo conjunto que
    a elegeu enviesa o número para baixo por construção.

    ## Por que do painel cru, e não recortando o treino já preparado

    A primeira versão desta função, de 16/09/2026 de manhã, recortava o treino
    **depois** de `preparar_treino_teste` ter rodado. Isso deixava três coisas do
    treino inteiro dentro do pipeline de seleção, e uma delas estragava a
    escolha:

    1. **`tempo` truncado na janela errada.** `atributos.py` trunca em
       [1, 310], o fim do treino inteiro. Recortado depois, o treino reduzido ia
       até 240 e a validação, de 249 a 310, recebia valores de `tempo` **acima
       de tudo que o treino reduzido viu**: 1,900 a 2,799 contra um máximo de
       1,767. É exatamente o que o truncamento de D28 existe para impedir, e a
       frase dela vale palavra por palavra: *"uma rede aprende tendência linear
       e a projeta sem freio"*. Medido numa sonda de 3 sementes, isso valia de
       **0,36 a 0,74 ponto de WMAPE**, e não o mesmo tanto para cada
       configuração: o espalhamento entre configurações era da ordem do
       espalhamento do grupo que a triagem estava tentando ordenar.
    2. A **mediana de preenchimento das defasagens**, estimada no treino inteiro.
    3. O **vocabulário de loja**, idem.

    As duas últimas são comuns a todas as configurações e por isso não torcem a
    ordem; a primeira interage com capacidade, e torce. Mesmo assim as três são
    consertadas juntas aqui, porque separar "vazamento que atrapalha" de
    "vazamento que não atrapalha" é o tipo de julgamento que já errei nesta
    fase.

    ## Como

    Chamando `preparar_treino_teste` de novo, sobre as linhas CRUAS do painel
    dentro da janela de treino. A função inteira reestima mediana, truncamento e
    vocabulário a partir do treino reduzido, e a partição interna dela é a mesma
    regra de D20: 20% no fim, com zona morta de 8 semanas. O pipeline de seleção
    passa a ser o pipeline de reporte aplicado a um painel mais curto, e não uma
    construção paralela que pode divergir.
    """
    dentro = longo[longo["week"] <= janela_treino[1]]
    red, val, _, vocab, meta = preparar_treino_teste(dentro)
    meta_val = {"semana_de_corte_validacao": meta["semana_de_corte"],
                "zona_morta_semanas": meta["zona_morta_semanas"],
                "janela_treino_reduzido": meta["janela_treino"],
                "n_treino_reduzido": meta["n_treino"],
                "n_validacao": meta["n_teste"]}
    return red, val, vocab, meta_val


def escalonar_contexto(treino: dict, *partes: dict) -> tuple:
    """Média e desvio do TREINO, aplicados em todas as partes (D28)."""
    saida = []
    estat = {}
    for nome in ("ctx_celula", "ctx_sku"):
        mu = treino[nome].mean(axis=0)
        sd = treino[nome].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        estat[nome] = {"media": mu.tolist(), "desvio": sd.tolist()}
        for parte in partes:
            parte[nome] = (parte[nome] - mu) / sd
    saida.append(estat)
    return tuple(saida)


def ajustar_escalas(base: dict, *partes: dict) -> tuple:
    """Centro de preço e escalonador estimados em `base`, aplicados em `partes`.

    Junta os dois ajustes de escala num só ponto porque eles têm a mesma regra
    (D28: estimar só no conjunto de ajuste) e porque agora existem dois
    pipelines, o de reporte (base = treino) e o de seleção (base =
    treino_reduzido). Ter os dois passos soltos era o convite a estimar um no
    conjunto certo e o outro no errado.

    Devolve `(centro, estat)` e escreve `u` e o contexto escalonado em `partes`.
    """
    centro = centro_de_preco(base["precos"])
    for parte in partes:
        parte["u"] = np.log(parte["precos"]) - centro
    (estat,) = escalonar_contexto(base, *partes)
    return centro, estat


# ---------------------------------------------------------------------------
# Os módulos
# ---------------------------------------------------------------------------

def _modulos():
    import torch
    from torch import nn
    return torch, nn


class _Lazy:
    """Adia o import do torch para quem de fato usa a rede.

    Os testes de pivô e de centro de preço não precisam dele, e o resto do
    repositório não deve passar a exigir torch para rodar `pytest`.
    """


def construir_modulos():
    torch, nn = _modulos()

    class SubRedeMonotona(nn.Module):
        """`mᵢ(uᵢ)` crescente, uma por produto, com derivada em forma fechada.

        A garantia é por **reparametrização** e não por projeção depois do passo
        do otimizador: `W = softplus(W_bruto) > 0` sempre, para qualquer valor
        que o Adam ponha em `W_bruto`. Projetar depois do passo deixaria uma
        janela em que o modelo avaliado viola a restrição, e um teste que só
        olhasse o fim do treino não veria.

            mᵢ(u)  = Σ_h W2[i,h] · softplus(W1[i,h]·u + b[i,h])
            mᵢ′(u) = Σ_h W2[i,h] · σ(W1[i,h]·u + b[i,h]) · W1[i,h]  ≥ 0

        Sem termo constante: ele seria exatamente redundante com o viés de `gᵢ`.
        """

        def __init__(self, n: int, h: int = 8, semente: int | None = None):
            super().__init__()
            g = torch.Generator().manual_seed(semente) if semente is not None else None
            # softplus(-0.5) ≈ 0,474: começa com inclinação modesta e positiva.
            self.w1_bruto = nn.Parameter(
                torch.randn(n, h, generator=g) * 0.2 - 0.5)
            self.w2_bruto = nn.Parameter(
                torch.randn(n, h, generator=g) * 0.2 - 0.5)
            self.b = nn.Parameter(torch.randn(n, h, generator=g) * 0.5)

        def pesos(self):
            return (nn.functional.softplus(self.w1_bruto),
                    nn.functional.softplus(self.w2_bruto))

        def forward(self, u):                      # u: (B, n)
            w1, w2 = self.pesos()
            z = u.unsqueeze(-1) * w1 + self.b      # (B, n, h)
            return (nn.functional.softplus(z) * w2).sum(-1)

        def derivada(self, u):
            """`mᵢ′(uᵢ)` em forma fechada, ≥ 0 por construção."""
            w1, w2 = self.pesos()
            z = u.unsqueeze(-1) * w1 + self.b
            return (torch.sigmoid(z) * w1 * w2).sum(-1)

    class Tronco(nn.Module):
        """MLP livre sobre o contexto, com embedding de loja."""

        def __init__(self, dim_ctx: int, n_lojas: int, n: int,
                     largura: int = 64, profundidade: int = 2,
                     dim_emb: int = 8, intercepto: bool = False,
                     posto: int | None = None, gama_contextual: bool = False):
            super().__init__()
            self.emb = nn.Embedding(n_lojas, dim_emb)
            # Intercepto por (loja, produto). E o analogo direto do que a
            # arvore faz com as categoricas cruzadas, e e o nivel que o
            # embedding de dimensao baixa nao consegue representar. Nao toca a
            # garantia de D33 porque nao depende de preco.
            self.intercepto = nn.Embedding(n_lojas, n) if intercepto else None
            if self.intercepto is not None:
                nn.init.zeros_(self.intercepto.weight)
            camadas, entrada = [], dim_ctx + dim_emb
            for _ in range(profundidade):
                camadas += [nn.Linear(entrada, largura), nn.Softplus()]
                entrada = largura
            self.corpo = nn.Sequential(*camadas)
            # `posto` fatoriza a cabeça: em vez de uma linha livre por produto,
            # a saída passa por um gargalo de dimensão `posto`. É o teste da
            # pendência 8.3, o único lugar em que a arquitetura de saída
            # vetorial tem MAIS parâmetros que a por produto. Posto cheio é o
            # padrão e equivale a `nn.Linear(entrada, n)`.
            if posto is None or posto >= min(entrada, n):
                self.saida = nn.Linear(entrada, n)
            else:
                self.saida = nn.Sequential(nn.Linear(entrada, posto, bias=False),
                                           nn.Linear(posto, n))
            # γ dependente do CONTEXTO: o tronco produz as N² entradas por
            # célula em vez de uma matriz fixa. A garantia de D33 sobrevive
            # porque γ depende do contexto e NÃO dos preços, de modo que a
            # derivada da parcela cruzada em relação a uᵢ continua sendo
            # γᵢᵢ(x), que a máscara zera.
            self.saida_gama = (nn.Linear(entrada, n * n)
                               if gama_contextual else None)
            if self.saida_gama is not None:
                nn.init.zeros_(self.saida_gama.weight)
                nn.init.zeros_(self.saida_gama.bias)
            self.n = n

        def forward(self, ctx, loja):
            h = self.corpo(torch.cat([ctx, self.emb(loja)], dim=1))
            z = self.saida(h)
            if self.intercepto is not None:
                z = z + self.intercepto(loja)
            if self.saida_gama is None:
                return z, None
            return z, self.saida_gama(h).view(-1, self.n, self.n)

    class RedeDemanda(nn.Module):
        """A rede de D31/D32/D33/D34, restrita ou não."""

        def __init__(self, n: int, dim_ctx: int, n_lojas: int,
                     largura: int = 64, profundidade: int = 2, dim_emb: int = 8,
                     h_monotona: int = 8, restrita: bool = True,
                     intercepto: bool = False, posto: int | None = None,
                     gama_contextual: bool = False, semente: int | None = None):
            super().__init__()
            self.n, self.restrita = n, restrita
            self.gama_contextual = gama_contextual
            if semente is not None:
                torch.manual_seed(semente)
            if restrita:
                self.tronco = Tronco(dim_ctx, n_lojas, n, largura, profundidade,
                                     dim_emb, intercepto, posto, gama_contextual)
                self.gama = (None if gama_contextual
                             else nn.Parameter(torch.zeros(n, n)))
                self.monotona = SubRedeMonotona(n, h_monotona, semente)
                self.register_buffer("fora_da_diagonal",
                                     1.0 - torch.eye(n))
            else:
                # Mesmo tronco, mas vendo TAMBÉM os preços, e sem estrutura
                # nenhuma depois dele. É o instrumento da pendência 8.12.
                self.tronco = Tronco(dim_ctx + n, n_lojas, n, largura,
                                     profundidade, dim_emb, intercepto, posto)

        def _vies_de_saida(self):
            s = self.tronco.saida
            return s.bias if isinstance(s, nn.Linear) else s[-1].bias

        def ajustar_vies_de_saida(self, media_por_produto):
            """Viés inicial em `ln(média do alvo)`, por produto.

            É a medida anti-estouro da ligação log: sem ela `exp(z)` parte de um
            ponto arbitrário e a perda de Poisson pode gerar gradiente enorme na
            primeira época.
            """
            with torch.no_grad():
                self._vies_de_saida().copy_(
                    torch.as_tensor(np.log(np.maximum(media_por_produto, 1e-6)),
                                    dtype=torch.float32))

        def log_demanda(self, ctx, u, loja):
            if not self.restrita:
                return self.tronco(torch.cat([ctx, u], dim=1), loja)[0]
            g, gama = self.tronco(ctx, loja)
            if gama is None:
                cruzado = u @ (self.gama * self.fora_da_diagonal).T
            else:
                gama = gama * self.fora_da_diagonal
                cruzado = torch.einsum("bij,bj->bi", gama, u)
            return g + cruzado - self.monotona(u)

        def forward(self, ctx, u, loja):
            return self.log_demanda(ctx, u, loja)

        def elasticidades_fechadas(self, u):
            """Matriz de elasticidades a partir dos PARÂMETROS, só na restrita.

            Devolve (B, n, n): diagonal `−mᵢ′(uᵢ)`, fora dela `γᵢⱼ`.
            """
            if not self.restrita:
                raise ValueError("a forma fechada só existe na rede restrita")
            if self.gama is None:
                raise ValueError("com γ contextual a forma fechada precisa do "
                                 "contexto; use elasticidades_por_autograd")
            b = u.shape[0]
            fora = (self.gama * self.fora_da_diagonal).unsqueeze(0).expand(b, -1, -1)
            diag = torch.diag_embed(-self.monotona.derivada(u))
            return fora + diag

    return {"SubRedeMonotona": SubRedeMonotona, "Tronco": Tronco,
            "RedeDemanda": RedeDemanda}


def elasticidades_por_autograd(modelo, ctx, u, loja):
    """A mesma matriz, por diferenciação automática, e ela vale para as duas.

    Existe por duas razões: é a única via na rede sem restrição, e na restrita
    ela **confere** a forma fechada. As duas concordarem é teste de que a
    álgebra escrita à mão bate com o que o grafo de fato calcula.
    """
    torch, _ = _modulos()
    u = u.clone().requires_grad_(True)
    ln_v = modelo.log_demanda(ctx, u, loja)
    n = ln_v.shape[1]
    linhas = []
    for i in range(n):
        (g,) = torch.autograd.grad(ln_v[:, i].sum(), u, retain_graph=True)
        linhas.append(g)
    return torch.stack(linhas, dim=1)


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def treinar(dados: dict, cfg: dict) -> dict:
    """Laço de treino. Perda de Poisson da ligação log (D32), L2 pelo Adam.

    O **sorteio do índice de reserva** (pendência 7.10, D28) mora aqui: com
    probabilidade `p_reserva`, o índice de loja do exemplo é trocado por 0
    durante o treino. Sem ele o índice 0 existe e nunca recebe gradiente, que é
    exatamente o problema que reservá-lo deveria resolver, e ele cobre as sete
    lojas que só aparecem no teste.

    Não há parada antecipada por desempenho no teste: isso usaria o período de
    teste para decidir, que é a segunda regra de D28. O número de épocas é
    hiperparâmetro e se mede.
    """
    torch, nn = _modulos()
    mods = construir_modulos()
    torch.manual_seed(cfg["semente"])

    tr, te = dados["treino"], dados["teste"]
    ctx_tr = torch.tensor(np.concatenate([tr["ctx_celula"], tr["ctx_sku"]], 1),
                          dtype=torch.float32)
    ctx_te = torch.tensor(np.concatenate([te["ctx_celula"], te["ctx_sku"]], 1),
                          dtype=torch.float32)
    u_tr = torch.tensor(tr["u"], dtype=torch.float32)
    u_te = torch.tensor(te["u"], dtype=torch.float32)
    y_tr = torch.tensor(tr["alvo"], dtype=torch.float32)
    loja_tr = torch.tensor(tr["loja_idx"], dtype=torch.long)
    loja_te = torch.tensor(te["loja_idx"], dtype=torch.long)

    modelo = mods["RedeDemanda"](
        n=dados["n"], dim_ctx=ctx_tr.shape[1], n_lojas=dados["n_lojas"],
        largura=cfg["largura"], profundidade=cfg["profundidade"],
        dim_emb=cfg["dim_emb"], h_monotona=cfg["h_monotona"],
        restrita=cfg["restrita"], intercepto=cfg["intercepto"],
        posto=cfg.get("posto"),
        gama_contextual=cfg.get("gama_contextual", False),
        semente=cfg["semente"])
    modelo.ajustar_vies_de_saida(tr["alvo"].mean(axis=0))

    otim = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["l2"])
    g = torch.Generator().manual_seed(cfg["semente"])
    c = len(u_tr)
    historico = []
    for epoca in range(cfg["epocas"]):
        modelo.train()
        ordem = torch.randperm(c, generator=g)
        perda_total = 0.0
        for k in range(0, c, cfg["lote"]):
            idx = ordem[k:k + cfg["lote"]]
            loja = loja_tr[idx].clone()
            if cfg["p_reserva"] > 0:
                sorteio = torch.rand(len(idx), generator=g) < cfg["p_reserva"]
                loja[sorteio] = INDICE_RESERVA
            z = modelo.log_demanda(ctx_tr[idx], u_tr[idx], loja)
            if cfg.get("perda", "poisson") == "poisson":
                perda = nn.functional.poisson_nll_loss(
                    z, y_tr[idx], log_input=True, full=False, reduction="mean")
            else:
                # MSLE na rede: erro quadrático em escala log. A previsão
                # continua `exp(z)` nas três réguas, de modo que o que muda é
                # SÓ a perda. Era esse o desenho de D32 na árvore, e mantê-lo
                # aqui é o que torna as duas medições comparáveis.
                perda = ((z - torch.log1p(y_tr[idx])) ** 2).mean()
            otim.zero_grad()
            perda.backward()
            otim.step()
            perda_total += float(perda.detach()) * len(idx)
        historico.append(perda_total / c)

    modelo.eval()
    with torch.no_grad():
        z_tr = modelo.log_demanda(ctx_tr, u_tr, loja_tr)
        pred_tr = torch.exp(z_tr).numpy()
        pred_te = torch.exp(modelo.log_demanda(ctx_te, u_te, loja_te)).numpy()
    # O FATOR DE DUAN É SEMPRE CALCULADO quando a perda é da família MSLE, e
    # não só quando `msle_smearing` foi pedida. O motivo é que `msle` e
    # `msle_smearing` **treinam de forma idêntica**: veja o laço acima, as duas
    # caem no mesmo `else`. O que as separa é uma multiplicação DEPOIS do
    # treino. Logo, rodar as duas como configurações separadas gasta dois
    # ajustes para produzir um modelo só, e D32 tratou a suavização como
    # terceira régua quando ela é, na rede, pós-processamento.
    #
    # Calculando sempre, a pergunta "quanto a correção de Duan mudaria este
    # resultado" passa a ser respondida de graça em TODA execução de TODA
    # família, em vez de exigir uma varredura própria. É a pendência 8.22 ficando
    # observável em todo lugar em vez de num experimento só.
    fator = 1.0
    if cfg.get("perda", "poisson") != "poisson":
        with torch.no_grad():
            fator = float(torch.exp(torch.log1p(y_tr) - z_tr).mean())
    if cfg.get("perda") == "msle_smearing":
        # Fator de Duan estimado NO TREINO (D28). Ele é constante, então não
        # muda a matriz de elasticidades: a decomposição de D32 previu isso e
        # aqui a previsão vira conferência.
        pred_tr, pred_te = pred_tr * fator, pred_te * fator
    return {"modelo": modelo, "historico": historico, "fator_de_duan": fator,
            "pred_treino": pred_tr, "pred_teste": pred_te,
            "tensores": {"ctx_treino": ctx_tr, "ctx_teste": ctx_te,
                         "u_treino": u_tr, "u_teste": u_te,
                         "loja_treino": loja_tr, "loja_teste": loja_te}}


def preparar(categoria: str, painel: Path) -> dict:
    """Do parquet de células ao par de dicionários pivotados e escalonados."""
    n = json.loads((painel / f"{categoria}_upcs.json").read_text())["n"]
    cel = pd.read_parquet(painel / f"{categoria}_celulas.parquet")
    fora = pd.read_parquet(painel / f"{categoria}_bem_externo.parquet")
    longo = construir(cel, fora, n)
    treino, teste, _, vocab, meta = preparar_treino_teste(longo)
    red_longo, val_longo, vocab_val, meta_val = particionar_selecao(
        longo, meta["janela_treino"])

    # D42: o primeiro estágio é ajustado no conjunto de ajuste de CADA pipeline
    # e aplicado nos dois lados dele. Ajustar no treino inteiro e aplicar na
    # validação seria o vazamento de D28 dentro do pipeline de seleção.
    if com_funcao_de_controle():
        treino, teste, meta["funcao_de_controle"] = funcao_de_controle.adicionar(
            treino, teste, cel=cel, n=n)
        red_longo, val_longo, meta_val["funcao_de_controle_selecao"] = \
            funcao_de_controle.adicionar(red_longo, val_longo, cel=cel, n=n)
    meta["especificacao"] = ESPECIFICACAO_ATIVA

    tr, te = pivotar(treino, n), pivotar(teste, n)

    # A janela de validação sai do TREINO e nunca do teste. Quem seleciona
    # hiperparâmetro usa `treino_reduzido` contra `validacao`; quem reporta usa
    # `treino` contra `teste`, uma vez. Os dois pipelines são independentes de
    # ponta a ponta: mediana de defasagem, truncamento de `tempo`, vocabulário,
    # centro de preço e escalonador de contexto, cada um estimado no seu próprio
    # conjunto de ajuste.
    tr_red, val = pivotar(red_longo, n), pivotar(val_longo, n)

    centro, estat = ajustar_escalas(tr, tr, te)
    centro_val, estat_val = ajustar_escalas(tr_red, tr_red, val)

    for partes, voc in (((tr, te), vocab["store"]),
                        ((tr_red, val), vocab_val["store"])):
        for parte in partes:
            parte["loja_idx"] = np.array([voc.get(v, INDICE_RESERVA)
                                          for v in parte["loja"]], dtype="int64")

    # Com vocabulário próprio, uma loja que só apareça na validação cai no
    # índice de RESERVA, como deve. O número fica medido e gravado de qualquer
    # modo, porque ele diz quanto do teste de reserva a validação exercita.
    meta_val["lojas_so_na_validacao"] = len(
        set(val["loja"].tolist()) - set(tr_red["loja"].tolist()))

    # Cada pipeline tem o TAMANHO de tabela do seu próprio vocabulário. Usar o
    # do reporte na seleção deixaria linhas de embedding que a seleção nunca
    # treina, e `n_atributos` contaria parâmetros mortos, que é justamente a
    # quantidade que a regra de escolha do ponto usa para desempatar.
    return {"n": n, "treino": tr, "teste": te, "meta": {**meta, **meta_val},
            "treino_reduzido": tr_red, "validacao": val,
            "centro_de_preco": centro.tolist(), "escalonador": estat,
            "centro_de_preco_validacao": centro_val.tolist(),
            "escalonador_validacao": estat_val,
            "n_lojas": len(vocab["store"]) + 1,
            "n_lojas_validacao": len(vocab_val["store"]) + 1,
            "categoricas": CATEGORICAS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--saida", default="reports")
    ap.add_argument("--artefatos", default="models_artifacts")
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--lote", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--largura", type=int, default=64)
    ap.add_argument("--profundidade", type=int, default=3)
    ap.add_argument("--dim-emb", type=int, default=8)
    ap.add_argument("--h-monotona", type=int, default=8)
    ap.add_argument("--p-reserva", type=float, default=0.05)
    ap.add_argument("--intercepto", action="store_true",
                    help="liga o intercepto por (loja, produto). PADRAO DESLIGADO, "
                         "e o padrao e provisorio: uma semente diz que ele piora, "
                         "e quem decide e a varredura de hiperparametros")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--perda", default="msle",
                    choices=["msle", "poisson", "msle_smearing"])
    ap.add_argument("--gama-contextual", action="store_true")
    ap.add_argument("--irrestrita", action="store_true",
                    help="treina a rede SEM restrição de monotonicidade, que é "
                         "o instrumento da pendência 8.12 e artefato próprio")
    ap.add_argument("--adotada", action="store_true",
                    help="usa a CONFIGURAÇÃO ADOTADA do item 6, lida de "
                         "varredura_rede.MELHOR, ignorando os demais "
                         "hiperparâmetros da linha de comando. É assim que o "
                         "artefato canônico é produzido, e é o que impede o "
                         "modelo salvo de divergir do que foi medido.")
    ap.add_argument("--sufixo", default="")
    ap.add_argument("--so-particao", action="store_true",
                    help="grava reports/particao_validacao_<cat>.json e para. "
                         "Existe para que os números da partição de D37 "
                         "citados em documento tenham PRODUTOR, como todo "
                         "número deste projeto: eles saem do painel e não da "
                         "minha memória.")
    ap.add_argument("--especificacao", default="adotada",
                    choices=sorted(ESPECIFICACOES),
                    help="'adotada' é a do item 9; 'v1' é a de D41 e D42. Com "
                         "especificação diferente da adotada e sem --sufixo, o "
                         "sufixo vira '_<especificação>', para que o artefato "
                         "canônico do item 9 nunca seja sobrescrito.")
    a = ap.parse_args()
    usar_especificacao(a.especificacao)
    if a.especificacao != "adotada" and not a.sufixo:
        a.sufixo = f"_{a.especificacao}"

    if a.so_particao:
        dados = preparar(a.categoria, Path(a.painel))
        bloco = {"categoria": a.categoria,
                 "particao": {k: v for k, v in dados["meta"].items()
                              if k != "escalonador"}}
        for nome in ("treino", "teste", "treino_reduzido", "validacao"):
            parte = dados[nome]
            semanas = parte["chaves"]["week"].to_numpy()
            bloco[nome] = {"n_celulas": int(len(parte["alvo"])),
                           "semana_min": int(semanas.min()),
                           "semana_max": int(semanas.max())}
        frac = (bloco["treino_reduzido"]["n_celulas"]
                / bloco["treino"]["n_celulas"])
        bloco["fracao_treino_reduzido"] = frac
        bloco["pct_treino_reduzido"] = 100.0 * frac
        Path(a.saida).mkdir(parents=True, exist_ok=True)
        destino = Path(a.saida) / f"particao_validacao_{a.categoria}.json"
        destino.write_text(json.dumps(bloco, indent=2, ensure_ascii=False))
        print(json.dumps(bloco, indent=2, ensure_ascii=False))
        print(f"\ngravado em {destino}")
        return

    import torch

    dados = preparar(a.categoria, Path(a.painel))
    cfg = {"epocas": a.epocas, "lote": a.lote, "lr": a.lr, "l2": a.l2,
           "largura": a.largura, "profundidade": a.profundidade,
           "dim_emb": a.dim_emb, "h_monotona": a.h_monotona,
           "p_reserva": a.p_reserva, "semente": a.semente,
           "intercepto": a.intercepto, "perda": a.perda,
           "gama_contextual": a.gama_contextual,
           "restrita": not a.irrestrita}
    if a.adotada:
        # Ponto único: a configuração adotada vive em varredura_rede.MELHOR,
        # que é o mesmo objeto que as famílias de varredura usaram. Copiá-la
        # para cá seria criar uma segunda fonte que pode divergir em silêncio.
        sys.path.insert(0, str(RAIZ / "src" / "experiments"))
        from varredura_rede import MELHOR  # noqa: E402
        cfg = dict(MELHOR, semente=a.semente,
                   restrita=not a.irrestrita)
        cfg.setdefault("gama_contextual", False)
    # Gravada no artefato: `sonda_item8.carregar_modelo` recusa carregar uma rede
    # numa especificação diferente da que a treinou, porque a largura do
    # contexto e o significado de cada coluna mudam entre elas.
    cfg["especificacao"] = ESPECIFICACAO_ATIVA
    saida_treino = treinar(dados, cfg)

    y_tr = dados["treino"]["alvo"].ravel()
    y_te = dados["teste"]["alvo"].ravel()
    p_tr = saida_treino["pred_treino"].ravel()
    p_te = saida_treino["pred_teste"].ravel()

    registro = {
        "categoria": a.categoria, "n_sortimento": dados["n"],
        "restrita": cfg["restrita"], "configuracao": cfg,
        "particao": {k: v for k, v in dados["meta"].items()
                     if k != "escalonador"},
        "n_celulas": {"treino": int(len(dados["treino"]["u"])),
                      "teste": int(len(dados["teste"]["u"]))},
        "perda_por_epoca": saida_treino["historico"],
        "desempenho": {
            "treino": metricas(y_tr, p_tr),
            "teste": metricas(y_te, p_te),
        },
        # D26 emendada: o viés de nível é terceiro eixo obrigatório.
        "vies_de_nivel": {
            "treino": float(p_tr.sum() / y_tr.sum()),
            "teste": float(p_te.sum() / y_te.sum()),
        },
    }
    Path(a.saida).mkdir(parents=True, exist_ok=True)
    destino = Path(a.saida) / f"rede_{a.categoria}{a.sufixo}.json"
    destino.write_text(json.dumps(registro, indent=2, ensure_ascii=False))

    Path(a.artefatos).mkdir(parents=True, exist_ok=True)
    torch.save({"estado": saida_treino["modelo"].state_dict(), "cfg": cfg,
                "centro_de_preco": dados["centro_de_preco"],
                "escalonador": dados["escalonador"]},
               Path(a.artefatos) / f"rede_{a.categoria}{a.sufixo}.pt")

    d = registro["desempenho"]
    print(f"{'':10s} {'RMSE':>9} {'MAE':>9} {'RMSE log':>9} {'WMAPE':>8} {'nível':>8}")
    for rot in ("treino", "teste"):
        m = d[rot]
        print(f"{rot:10s} {m['rmse']:9.2f} {m['mae']:9.2f} {m['rmse_log']:9.4f} "
              f"{m['wmape_pct']:7.2f}% {registro['vies_de_nivel'][rot]:8.4f}")
    print(f"\ngravado em {destino}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Diagnósticos que fecham pendências do item 6
# ---------------------------------------------------------------------------

LIMITES_CAIXA = (0.85, 1.15)   # a faixa de ±15% da seção 4 do CLAUDE.md


def identidade_de_agregacao(eps, v) -> dict:
    """A conta da pendência 2.8, por célula, e ela mora AQUI e em nenhum outro
    lugar.

    A restrição é

        |ε_agregada|  ≤  |média ponderada das próprias|

    e ela existe porque, para substitutos, as cruzadas positivas compensam parte
    do efeito próprio: a cesta inteira é menos elástica que a média dos seus
    itens. Com a matriz da rede a conta é direta, com `sᵢ = V̂ᵢ / Σ V̂`:

        ε_agregada        = Σᵢ sᵢ Σⱼ εᵢⱼ    (subida uniforme de 1% em todos)
        própria ponderada = Σᵢ sᵢ εᵢᵢ

    Violação não é erro de conta: é a rede dizendo que a cesta é mais elástica
    que seus itens, o que exige cruzadas líquidas negativas, isto é, marcas
    concorrentes tratadas como complementares.

    **Ponto único de propósito.** A fórmula estava escrita duas vezes, em
    `diagnosticos` e em `fechamento_item6`, com nomes de chave diferentes, e uma
    fórmula duplicada é uma fórmula que diverge. Os dois chamam esta.
    """
    torch, _ = _modulos()
    s = v / v.sum(dim=1, keepdim=True)
    agregada = (s * eps.sum(dim=2)).sum(dim=1)
    propria = (s * torch.diagonal(eps, dim1=1, dim2=2)).sum(dim=1)
    folga = propria.abs() - agregada.abs()
    return {
        "elast_agregada_mediana": float(agregada.median()),
        "propria_ponderada_mediana": float(propria.median()),
        "folga_mediana": float(folga.median()),
        "pct_celulas_com_violacao": float(100.0 * (folga < 0).float().mean()),
    }


def diagnosticos(modelo, ctx, u, loja, precos) -> dict:
    """Os números que o item 6 deve a três pendências, num lugar só.

    **Elasticidades (8.12, D16, 2.8).** A Jacobiana em log-log É a matriz de
    elasticidades, porque saída e entrada estão as duas em log. Sai por autograd
    para valer nas duas redes, e na restrita ela é conferida contra a forma
    fechada pelo teste `test_forma_fechada_bate_com_autograd`.

    **Fração de derivada própria positiva (D19).** Na rede restrita é zero por
    construção; na sem restrição é a medida de quanto a imposição estrutural
    estava evitando, que D19 afirma citando literatura e nunca mediu aqui.

    **Fração de ótimo interior, coordenada a coordenada (8.13).** Para a receita
    `Z = Σ pⱼ V̂ⱼ`,

        ∂Z/∂pᵢ  tem o sinal de  Rᵢ + Σⱼ Rⱼ·εⱼᵢ      com Rⱼ = pⱼ V̂ⱼ

    Avaliado nas duas bordas da caixa com os demais preços no valor histórico:
    sinais diferentes indicam estacionário interior naquela coordenada. **Não é
    o ótimo conjunto nos N preços**, que é do item 8 e exige o solver, e tem de
    ser reportado com esse nome.
    """
    torch, _ = _modulos()
    n = u.shape[1]

    eps0 = elasticidades_por_autograd(modelo, ctx, u, loja).detach()
    propria = torch.diagonal(eps0, dim1=1, dim2=2)                # (B, n)
    fora = eps0 * (1.0 - torch.eye(n))

    sinais = []
    for lim in LIMITES_CAIXA:
        s = []
        for i in range(n):
            u_ = u.clone()
            u_[:, i] = u_[:, i] + np.log(lim)
            eps = elasticidades_por_autograd(modelo, ctx, u_, loja).detach()
            with torch.no_grad():
                v = torch.exp(modelo.log_demanda(ctx, u_, loja))
            p = torch.tensor(precos, dtype=torch.float32).clone()
            p[:, i] = p[:, i] * lim
            r = p * v                                             # (B, n)
            # d Z / d p_i tem o sinal de R_i + soma_j R_j eps_ji
            s.append(r[:, i] + (r * eps[:, :, i]).sum(1))
        sinais.append(torch.stack(s, dim=1))                      # (B, n)
    interior = (torch.sign(sinais[0]) != torch.sign(sinais[1]))

    # Identidade de agregação (pendência 2.8), por célula. Sai daqui para que
    # TODA varredura a reporte, em vez de só o script de fechamento.
    with torch.no_grad():
        v0 = torch.exp(modelo.log_demanda(ctx, u, loja))
    agregacao = identidade_de_agregacao(eps0, v0)

    return {
        "elast_agregada_mediana": agregacao["elast_agregada_mediana"],
        "propria_ponderada_mediana": agregacao["propria_ponderada_mediana"],
        "folga_de_agregacao_mediana": agregacao["folga_mediana"],
        "pct_violacao_agregacao": agregacao["pct_celulas_com_violacao"],
        "elast_propria_mediana": float(propria.median()),
        "elast_propria_p10": float(propria.quantile(0.10)),
        "elast_propria_p90": float(propria.quantile(0.90)),
        "pct_propria_positiva": float(100.0 * (propria > 0).float().mean()),
        "elast_cruzada_media": float(fora.sum() / (fora.shape[0] * n * (n - 1))),
        "elast_cruzada_dp_entre_pares": float(fora[0].flatten().std()),
        "pct_cruzada_positiva": float(
            100.0 * (fora > 0).float().sum() / (fora.shape[0] * n * (n - 1))),
        # "constante mas mostre": o desvio da MESMA cruzada entre células. Na
        # rede restrita é zero por construção; na sem restrição é a medida de
        # quanto a matriz constante está deixando na mesa.
        "dp_da_cruzada_entre_celulas": float(fora.std(dim=0).mean()),
        "dp_da_propria_entre_celulas": float(propria.std(dim=0).mean()),
        "pct_otimo_interior": float(100.0 * interior.float().mean()),
        "n_celulas_avaliadas": int(u.shape[0]),
    }
