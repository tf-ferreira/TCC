"""Testes da rede de demanda: o pivô, a garantia de D33 e o sorteio de D28.

O teste central deste arquivo é `test_a_derivada_propria_e_nao_positiva_para_
qualquer_peso`, e ele é de natureza diferente dos outros. D33 exige que a
monotonicidade seja **garantida por construção**, não verificada depois do
treino. Um teste que treinasse a rede e conferisse o sinal no fim pode passar
por sorte, e passaria também numa implementação sem garantia nenhuma que por
acaso convergisse para a região certa.

O que se quer é um **invariante executável**: ele sorteia pesos brutos
arbitrários, inclusive muito negativos, e exige que a derivada continue não
positiva. Uma implementação que troque a reparametrização por projeção depois do
passo do otimizador, ou que esqueça o `softplus`, falha aqui mesmo sem treinar.

O segundo invariante é `test_o_contexto_nao_tem_preco_da_semana_corrente`. A
garantia de D33 depende de `gᵢ` não ver `pᵢ`, e essa dependência é invisível no
código: ela mora numa LISTA de nomes de coluna. Um atributo novo acrescentado
sem atenção quebraria a garantia sem quebrar nada que se veja.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

RAIZ = Path(__file__).resolve().parents[1]
for pasta in ("data", "models", "experiments"):
    sys.path.insert(0, str(RAIZ / "src" / pasta))

import rede  # noqa: E402

PAINEL = Path(__file__).resolve().parents[1] / "data" / "interim" / "painel"


# ---------------------------------------------------------------------------
# Os dois invariantes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("semente", [0, 1, 2, 3, 4])
def test_a_derivada_propria_e_nao_positiva_para_qualquer_peso(semente):
    """A garantia de D33, com pesos brutos ARBITRÁRIOS e sem treino nenhum."""
    mods = rede.construir_modulos()
    n, h = 4, 6
    m = mods["SubRedeMonotona"](n, h)
    g = torch.Generator().manual_seed(semente)
    with torch.no_grad():
        # Escala grande e centrada em zero: metade dos pesos brutos é negativa,
        # e é exatamente esse caso que a reparametrização tem de absorver.
        for par in (m.w1_bruto, m.w2_bruto, m.b):
            par.copy_(torch.randn(par.shape, generator=g) * 8.0)
    u = torch.linspace(-3.0, 3.0, 61).unsqueeze(1).repeat(1, n)
    assert torch.all(m.derivada(u) >= 0.0)


@pytest.mark.parametrize("semente", [0, 1, 2])
def test_a_rede_restrita_tem_diagonal_nao_positiva_por_autograd(semente):
    """O mesmo invariante, agora na rede inteira e medido pelo GRAFO.

    A forma fechada podia estar certa e a montagem do `log_demanda` errada, por
    exemplo somando `mᵢ` em vez de subtrair. Só o autograd pega isso.
    """
    mods = rede.construir_modulos()
    n, dim_ctx = 4, 5
    modelo = mods["RedeDemanda"](n=n, dim_ctx=dim_ctx, n_lojas=3, largura=8,
                                 profundidade=1, dim_emb=2, h_monotona=4,
                                 restrita=True, semente=semente)
    g = torch.Generator().manual_seed(semente + 100)
    with torch.no_grad():
        for par in modelo.parameters():
            par.copy_(torch.randn(par.shape, generator=g) * 3.0)
    ctx = torch.randn(7, dim_ctx, generator=g)
    u = torch.randn(7, n, generator=g)
    loja = torch.randint(0, 3, (7,), generator=g)
    eps = rede.elasticidades_por_autograd(modelo, ctx, u, loja)
    assert torch.all(torch.diagonal(eps, dim1=1, dim2=2) <= 1e-6)


def test_o_contexto_nao_tem_preco_da_semana_corrente():
    """`gᵢ` não pode ver `pᵢ`, e a garantia mora numa lista de nomes."""
    proibido = [f"preco_{i:02d}" for i in range(40)] + ["preco_proprio"]
    for col in rede.CONTEXTO_CELULA + rede.CONTEXTO_SKU:
        assert col not in proibido, f"{col} é preço da semana corrente"
    # As defasagens SÃO preços, e entram de propósito: são de semanas passadas
    # e não variáveis de decisão da semana corrente.
    assert "preco_lag1" in rede.CONTEXTO_SKU


def test_as_defasagens_abrem_o_contexto_de_sku_em_toda_especificacao():
    """O contrafactual sequencial reescreve os quatro primeiros blocos de
    CONTEXTO_SKU. Se uma especificação nova pusesse outro atributo na frente, a
    cadeia escreveria preço no lugar dele sem que nada quebrasse."""
    for nome, e in rede.ESPECIFICACOES.items():
        assert e["sku"][:4] == ["preco_lag1", "preco_lag2",
                                "falta_preco_lag1", "falta_preco_lag2"], nome
        proibido = [f"preco_{i:02d}" for i in range(40)] + ["preco_proprio"]
        assert not set(e["celula"] + e["sku"]) & set(proibido), nome


def test_a_v1_tira_o_que_esta_entre_o_preco_e_o_volume():
    """D41: `vol_resto` e `n_resto` são desfecho (D30); a v1 não os tem."""
    v1 = rede.ESPECIFICACOES["v1"]
    assert "vol_resto" not in v1["celula"] and "n_resto" not in v1["celula"]
    assert v1["funcao_de_controle"] is True
    assert rede.ESPECIFICACOES["adotada"]["funcao_de_controle"] is False


def test_usar_especificacao_muta_no_lugar_e_volta():
    lista_celula, lista_sku = rede.CONTEXTO_CELULA, rede.CONTEXTO_SKU
    original = (list(lista_celula), list(lista_sku))
    try:
        rede.usar_especificacao("v1")
        assert rede.CONTEXTO_CELULA is lista_celula
        assert "v_cf" in rede.CONTEXTO_SKU and rede.com_funcao_de_controle()
    finally:
        rede.usar_especificacao("adotada")
    assert (rede.CONTEXTO_CELULA, rede.CONTEXTO_SKU) == original
    with pytest.raises(ValueError):
        rede.usar_especificacao("inexistente")


def test_a_rede_irrestrita_pode_violar_o_sinal():
    """Sem este teste a comparação de 8.12 seria vazia.

    Se a rede sem restrição também garantisse o sinal por algum acidente de
    implementação, medir nela a fração de derivadas absurdas não mediria nada.
    """
    mods = rede.construir_modulos()
    n, dim_ctx = 3, 4
    modelo = mods["RedeDemanda"](n=n, dim_ctx=dim_ctx, n_lojas=2, largura=8,
                                 profundidade=1, dim_emb=2, restrita=False,
                                 semente=0)
    g = torch.Generator().manual_seed(7)
    encontrou = False
    for _ in range(40):
        with torch.no_grad():
            for par in modelo.parameters():
                par.copy_(torch.randn(par.shape, generator=g) * 3.0)
        ctx = torch.randn(5, dim_ctx, generator=g)
        u = torch.randn(5, n, generator=g)
        loja = torch.randint(0, 2, (5,), generator=g)
        eps = rede.elasticidades_por_autograd(modelo, ctx, u, loja)
        if bool((torch.diagonal(eps, dim1=1, dim2=2) > 0).any()):
            encontrou = True
            break
    assert encontrou


# ---------------------------------------------------------------------------
# Forma fechada, pivô e sorteio
# ---------------------------------------------------------------------------

def test_forma_fechada_bate_com_autograd():
    """A álgebra escrita à mão contra o que o grafo de fato calcula."""
    mods = rede.construir_modulos()
    n, dim_ctx = 5, 4
    modelo = mods["RedeDemanda"](n=n, dim_ctx=dim_ctx, n_lojas=3, largura=8,
                                 profundidade=1, dim_emb=2, h_monotona=4,
                                 restrita=True, semente=3)
    g = torch.Generator().manual_seed(11)
    ctx = torch.randn(6, dim_ctx, generator=g)
    u = torch.randn(6, n, generator=g)
    loja = torch.randint(0, 3, (6,), generator=g)
    auto = rede.elasticidades_por_autograd(modelo, ctx, u, loja).detach()
    fechada = modelo.elasticidades_fechadas(u).detach()
    assert torch.allclose(auto, fechada, atol=1e-5), (auto - fechada).abs().max()


def test_a_diagonal_de_gama_e_zerada():
    """A cruzada de i com i não existe: ela é a própria, e é de `mᵢ`."""
    mods = rede.construir_modulos()
    modelo = mods["RedeDemanda"](n=4, dim_ctx=3, n_lojas=2, largura=8,
                                 profundidade=1, restrita=True, semente=0)
    with torch.no_grad():
        modelo.gama.copy_(torch.ones(4, 4) * 5.0)
    u = torch.zeros(2, 4)
    fora = modelo.elasticidades_fechadas(u).detach()
    assert torch.allclose(torch.diagonal(fora, dim1=1, dim2=2),
                          -modelo.monotona.derivada(u))


def _longo_sintetico(c: int = 3, n: int = 2) -> pd.DataFrame:
    linhas = []
    for k in range(c):
        for i in range(n):
            linha = {"store": 10 + k % 2, "week": 100 + k, "sku": i,
                     "alvo": 5.0 * (k + 1) + i}
            for j in range(n):
                linha[f"preco_{j:02d}"] = 1.0 + 0.1 * j + 0.01 * k
            for col in rede.CONTEXTO_CELULA:
                linha[col] = float(k)
            for col in rede.CONTEXTO_SKU:
                linha[col] = float(k * 10 + i)
            linhas.append(linha)
    return pd.DataFrame(linhas)


def test_pivotar_preserva_o_alvo_e_o_preco():
    n = 2
    d = _longo_sintetico(3, n)
    p = rede.pivotar(d, n)
    assert p["alvo"].shape == (3, n)
    assert p["alvo"][0].tolist() == [5.0, 6.0]
    # O vetor de preços da célula é o mesmo nas N linhas; o pivô toma um deles.
    assert np.allclose(p["precos"][0], [1.0, 1.1])
    assert p["ctx_sku"].shape == (3, n * len(rede.CONTEXTO_SKU))


def test_pivotar_falha_em_celula_furada():
    """Célula sem os N SKUs viraria alvo com buraco silencioso."""
    d = _longo_sintetico(3, 2).drop(index=1).reset_index(drop=True)
    with pytest.raises(ValueError):
        rede.pivotar(d, 2)


def test_centro_de_preco_e_a_media_do_log():
    precos = np.array([[1.0, 4.0], [1.0, 1.0]])
    c = rede.centro_de_preco(precos)
    assert np.allclose(c, [0.0, np.log(4.0) / 2])


def test_o_sorteio_treina_o_indice_de_reserva():
    """Pendência 7.10 e D28: sem o sorteio, o índice 0 nunca recebe gradiente.

    O teste compara duas execuções idênticas, uma com `p_reserva = 0` e outra
    com `p_reserva = 1`, e exige que a linha 0 do embedding só se mova na
    segunda. É a diferença entre reservar o índice e de fato treiná-lo.
    """
    c, n, dim_ctx = 40, 3, 4
    rng = np.random.default_rng(0)
    dados = {
        "n": n, "n_lojas": 4,
        "treino": {"ctx_celula": rng.normal(size=(c, dim_ctx)),
                   "ctx_sku": rng.normal(size=(c, 0)),
                   "u": rng.normal(size=(c, n)) * 0.1,
                   "alvo": rng.integers(1, 30, size=(c, n)).astype(float),
                   "loja_idx": rng.integers(1, 4, size=c)},
    }
    dados["teste"] = dados["treino"]
    base = {"epocas": 3, "lote": 16, "lr": 0.05, "l2": 0.0, "largura": 8,
            "profundidade": 1, "dim_emb": 3, "h_monotona": 3,
            "intercepto": False, "restrita": True, "semente": 0}

    def linha_zero(p_reserva):
        saida = rede.treinar(dados, dict(base, p_reserva=p_reserva))
        return saida["modelo"].tronco.emb.weight[0].detach().clone()

    sem, com = linha_zero(0.0), linha_zero(1.0)
    assert not torch.allclose(sem, com), "o sorteio não mudou o índice 0"


# ---------------------------------------------------------------------------
# Cabeça de posto reduzido (8.3) e a identidade de agregação (2.8)
# ---------------------------------------------------------------------------

def test_posto_reduz_parametros_e_preserva_a_garantia():
    """A fatoração da cabeça é teste de sobreajuste, não relaxamento de D33."""
    mods = rede.construir_modulos()
    n, dim_ctx = 8, 5
    cheia = mods["RedeDemanda"](n=n, dim_ctx=dim_ctx, n_lojas=3, largura=32,
                                profundidade=1, restrita=True, semente=0)
    curta = mods["RedeDemanda"](n=n, dim_ctx=dim_ctx, n_lojas=3, largura=32,
                                profundidade=1, restrita=True, posto=2,
                                semente=0)
    assert sum(t.numel() for t in curta.parameters()) < \
           sum(t.numel() for t in cheia.parameters())
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for par in curta.parameters():
            par.copy_(torch.randn(par.shape, generator=g) * 3.0)
    eps = rede.elasticidades_por_autograd(
        curta, torch.randn(4, dim_ctx, generator=g),
        torch.randn(4, n, generator=g), torch.randint(0, 3, (4,), generator=g))
    assert torch.all(torch.diagonal(eps, dim1=1, dim2=2) <= 1e-6)


def test_o_vies_de_saida_e_ajustado_tambem_com_posto():
    """Com cabeça fatorada o viés mora na ÚLTIMA camada, não na primeira."""
    mods = rede.construir_modulos()
    modelo = mods["RedeDemanda"](n=3, dim_ctx=4, n_lojas=2, largura=16,
                                 profundidade=1, restrita=True, posto=2,
                                 semente=0)
    modelo.ajustar_vies_de_saida(np.array([1.0, np.e, np.e ** 2]))
    assert torch.allclose(modelo._vies_de_saida(),
                          torch.tensor([0.0, 1.0, 2.0]), atol=1e-6)


def test_identidade_de_agregacao_em_caso_construido():
    """Substitutos puros satisfazem; complementares violam.

    Dois produtos de mesmo peso. Com própria −2 e cruzada +1, a cesta responde
    −1 e a média das próprias é −2: a desigualdade vale com folga 1. Trocando a
    cruzada para −1, a cesta responde −3 contra −2, e a desigualdade cai. É
    exatamente o diagnóstico que 2.8 quer: violação é cruzada com sinal errado.
    """
    from fechamento_item6 import identidade_de_agregacao
    v = torch.ones(1, 2)
    subs = torch.tensor([[[-2.0, 1.0], [1.0, -2.0]]])
    d = identidade_de_agregacao(subs, v)
    assert abs(d["elast_agregada_mediana"] + 1.0) < 1e-6
    assert abs(d["propria_ponderada_mediana"] + 2.0) < 1e-6
    assert abs(d["folga_mediana"] - 1.0) < 1e-6
    assert d["pct_celulas_com_violacao"] == 0.0

    compl = torch.tensor([[[-2.0, -1.0], [-1.0, -2.0]]])
    d2 = identidade_de_agregacao(compl, v)
    assert abs(d2["elast_agregada_mediana"] + 3.0) < 1e-6
    assert d2["pct_celulas_com_violacao"] == 100.0


def test_impressao_separa_configuracoes_que_compartilham_o_nome():
    """O invariante contra fragmento contaminado (16/09/2026).

    O fragmento é nomeado por `categoria__configuração__semente`, e o NOME é
    estável enquanto o CONTEÚDO não é. Quando a perda adotada mudou de Poisson
    para MSLE, a configuração chamada `restrita` passou a significar outra
    coisa; sem a impressão digital, as sementes já gravadas seriam reusadas em
    silêncio com o significado errado.
    """
    from varredura_rede import impressao
    base = {"epocas": 30, "profundidade": 3, "perda": "poisson"}
    assert impressao(base) == impressao(dict(base, semente=7)), \
        "a semente NÃO pode entrar na impressão: ela varia de propósito"
    assert impressao(base) != impressao(dict(base, perda="msle"))
    assert impressao(base) != impressao(dict(base, profundidade=4))
    # A ordem das chaves não pode mudar a impressão.
    assert impressao(base) == impressao(dict(reversed(list(base.items()))))


def test_a_impressao_separa_o_conjunto_de_medicao():
    """A mesma configuração medida na validação e no teste são dois números.

    Acrescentado em 16/09/2026 junto com a partição de validação. Sem isto,
    reaproveitar fragmentos de `hiper` (medida no teste) numa família de
    seleção rodada na validação passaria pela impressão digital sem alarme,
    porque a CONFIGURAÇÃO é a mesma; o que muda é onde o número foi medido.
    """
    from varredura_rede import impressao
    base = {"epocas": 15, "profundidade": 4, "perda": "msle"}
    assert impressao(base, "teste") != impressao(base, "validacao")
    assert impressao(base) == impressao(base, "teste"), \
        "o padrão tem de ser o teste, que é o que os fragmentos antigos mediram"


def test_o_tempo_da_validacao_e_truncado_na_janela_do_treino_reduzido():
    """O defeito de 16/09/2026 que a primeira triagem na validação carregou.

    `atributos.py` trunca `tempo` na janela do treino INTEIRO. A primeira versão
    da partição recortava o treino DEPOIS disso, e a validação ficava com
    valores de `tempo` acima de tudo que o treino reduzido viu: a rede era
    posta a extrapolar tendência exatamente onde D28 manda congelar.

    O invariante: no conjunto de avaliação de CADA pipeline, `tempo` é
    constante. Um valor só, dos dois lados. Se algum dia alguém trocar a ordem
    de `particionar_selecao` e `ajustar_escalas`, ou voltar a recortar o pronto,
    este teste cai.
    """
    d = rede.preparar("frj", PAINEL)
    i = rede.CONTEXTO_CELULA.index("tempo")
    for ajuste, avaliacao in (("treino", "teste"),
                              ("treino_reduzido", "validacao")):
        t_aj = d[ajuste]["ctx_celula"][:, i]
        t_av = d[avaliacao]["ctx_celula"][:, i]
        assert len(np.unique(np.round(t_av, 9))) == 1, \
            f"`tempo` varia em {avaliacao}: a rede está extrapolando tendência"
        assert abs(t_av[0] - t_aj.max()) < 1e-9, \
            f"`tempo` de {avaliacao} não congelou no máximo de {ajuste}"


def test_os_dois_pipelines_nao_compartilham_estatistica_de_ajuste():
    """Cada pipeline estima o SEU centro de preço, o SEU escalonador e o SEU
    vocabulário. Compartilhar qualquer um deles põe a validação dentro das
    estatísticas que treinam o modelo que ela vai julgar."""
    d = rede.preparar("frj", PAINEL)
    assert not np.allclose(d["centro_de_preco"], d["centro_de_preco_validacao"]), \
        "os dois centros de preço são o mesmo objeto: a janela não foi separada"
    assert (d["escalonador"]["ctx_celula"]["media"]
            != d["escalonador_validacao"]["ctx_celula"]["media"])
    assert d["n_lojas_validacao"] <= d["n_lojas"], \
        "o vocabulário de seleção não pode ser maior que o de reporte"
    # A média do contexto escalonado é zero no conjunto de AJUSTE de cada
    # pipeline, e não no de avaliação. É isso que prova de onde a estatística veio.
    for ajuste, avaliacao in (("treino", "teste"),
                              ("treino_reduzido", "validacao")):
        assert np.allclose(d[ajuste]["ctx_sku"].mean(axis=0), 0.0, atol=1e-9)
        assert not np.allclose(d[avaliacao]["ctx_sku"].mean(axis=0), 0.0, atol=1e-6)


def test_a_validacao_vem_do_fim_do_treino_com_zona_morta():
    """Temporal, do fim, com zona morta de 8 semanas (D20), e sem interseção.

    Os três correspondem a jeitos conhecidos de errar: sortear a validação
    responderia outra pergunta que não prever o futuro; sem zona morta uma
    defasagem de k semanas põe informação da validação dentro do treino; e
    semana nos dois lados é vazamento direto.
    """
    d = rede.preparar("frj", PAINEL)
    s_red = np.unique(d["treino_reduzido"]["chaves"]["week"])
    s_val = np.unique(d["validacao"]["chaves"]["week"])
    s_tre = np.unique(d["treino"]["chaves"]["week"])
    s_tes = np.unique(d["teste"]["chaves"]["week"])
    assert s_red.max() < s_val.min()
    assert set(s_red) & set(s_val) == set()
    assert s_val.min() - s_red.max() - 1 == 8
    assert s_val.max() <= s_tre.max(), "a validação NÃO pode passar do treino"
    assert s_val.max() < s_tes.min(), "a validação NÃO pode tocar o teste"


def test_a_grade_deduplica_nomes_da_mesma_configuracao():
    """Com o centro em `MELHOR`, quatro nomes descrevem o mesmo treino.

    Rodar os quatro gasta quatro vezes o mesmo ajuste e infla `k` na correção
    de Bonferroni de D26, o que ERGUE o limiar e reduz o poder contra
    diferenças reais. O teste exige que a grade não tenha duas configurações
    com a mesma impressão digital.
    """
    import varredura_rede as V
    V.APELIDOS.clear()
    g = V.configuracoes("hiper_v3")
    impressoes = [V.impressao(c) for c in g.values()]
    assert len(set(impressoes)) == len(impressoes), "grade com configuração repetida"
    assert "centro" in g
    colapsados = {n for lista in V.APELIDOS.values() for n in lista}
    assert colapsados, "com o centro em (prof 4, ep 15) há nomes repetidos"
    assert colapsados.isdisjoint(g), "nome colapsado não pode continuar na grade"


def test_o_centro_da_triagem_nao_segue_o_ponto_adotado():
    """`CENTRO_V3` é congelado; `MELHOR` é conclusão e se move.

    Se a grade de `hiper_v3` fosse centrada em `MELHOR`, atualizar o ponto
    adotado redefiniria retroativamente a grade que o elegeu, e os 950
    fragmentos gravados passariam a descrever outra coisa. É o mesmo motivo por
    que `CENTRO` e `CENTRO_MSLE` já eram congelados.
    """
    import varredura_rede as V
    assert V.CENTRO_V3 != V.MELHOR, \
        "o centro da triagem virou o ponto adotado: o registro se apaga sozinho"
    V.APELIDOS.clear()
    g = V.configuracoes("hiper_v3")
    assert V.impressao(g["centro"]) == V.impressao(V.CENTRO_V3)


def test_a_grade_antiga_nao_muda_com_a_deduplicacao():
    """As grades já medidas de `hiper` e `hiper_msle` têm de ficar intactas:
    o centro delas é profundidade 2 com 30 épocas, que não colide com nenhum
    valor de `EIXOS` nem de `COMBINADOS`. Se colidisse, a deduplicação apagaria
    retroativamente um nome que tem fragmento gravado."""
    import varredura_rede as V
    for familia in ("hiper", "hiper_msle"):
        g = V.configuracoes(familia)
        esperado = 1 + sum(len(v) for v in V.EIXOS.values()) + len(V.COMBINADOS)
        assert len(g) == esperado, f"{familia} perdeu configuração na dedup"


def test_o_poder_separa_empate_real_de_medicao_cega():
    """O critério de invalidação de D37, como invariante executável.

    "Nenhum par passou o limiar" é ambíguo entre duas leituras opostas: as
    configurações empatam de verdade, ou a medição empataria de qualquer jeito.
    O bloco de poder existe para desfazer essa ambiguidade, e o teste constrói
    os dois casos: mesmas médias, erros padrão de ordens diferentes.
    """
    from ruido_semente import poder_da_comparacao
    medias = {"a": 50.0, "b": 50.5, "c": 52.0}
    limiar = 2.394

    preciso = {"a__menos__b": {"erro_padrao_pareado": 0.05, "passa_limiar": True},
               "a__menos__c": {"erro_padrao_pareado": 0.05, "passa_limiar": True},
               "b__menos__c": {"erro_padrao_pareado": 0.05, "passa_limiar": True}}
    cego = {k: {"erro_padrao_pareado": 5.0, "passa_limiar": False} for k in preciso}

    p = poder_da_comparacao(medias, preciso, limiar, 5, chave=lambda v: v)
    c = poder_da_comparacao(medias, cego, limiar, 5, chave=lambda v: v)

    assert p["melhor"] == "a" and p["segunda"] == "b"
    assert p["escolha_entre_1_e_2_e_decidivel"] is True
    assert c["escolha_entre_1_e_2_e_decidivel"] is False
    # E o desenho cego diz DE QUANTO ele é cego: o ep cai com 1/sqrt(n), logo
    # levar a DMD de 11,97 ate 0,5 pede (11,97/0,5)^2 vezes as 5 sementes.
    assert c["sementes_para_decidir_1_contra_2"] > 2000
    assert p["sementes_para_decidir_1_contra_2"] <= 5


def test_no_vies_de_nivel_o_melhor_e_perto_de_um_e_nao_o_menor():
    """Ordenar o nível por menor elegeria quem mais subestima.

    O viés de nível é Σŷ/Σy: 1,0 é acerto, e 0,72 e 1,28 são igualmente ruins.
    A elasticidade própria não tem "melhor" nenhum, porque é diagnóstico contra
    o estimando de D18, e inventar uma ordem ali seria transformar diagnóstico
    em critério de seleção pelas costas.
    """
    from ruido_semente import ORDENACAO, poder_da_comparacao
    medias = {"subestima": 0.72, "acerta": 0.99, "superestima": 1.28}
    pares = {"subestima__menos__acerta": {"erro_padrao_pareado": 0.01, "passa_limiar": True},
             "subestima__menos__superestima": {"erro_padrao_pareado": 0.01, "passa_limiar": True},
             "acerta__menos__superestima": {"erro_padrao_pareado": 0.01, "passa_limiar": True}}
    p = poder_da_comparacao(medias, pares, 2.394, 5,
                            chave=ORDENACAO["vies_de_nivel"])
    assert p["melhor"] == "acerta"
    assert ORDENACAO["elast_h10"] is None
    sem_ordem = poder_da_comparacao(medias, pares, 2.394, 5, chave=None)
    assert sem_ordem["tem_ordem_de_melhor"] is False
    assert "melhor" not in sem_ordem


# Ruído de semente construído à mão, e não sorteado, por dois motivos que se
# contradizem sob sorteio: o erro padrão pareado precisa ser MAIOR que zero
# (ruído idêntico entre configurações daria z infinito e faria qualquer
# diferença passar o limiar), e a ordem das médias precisa ser EXATA (com
# ruído sorteado e 5 sementes, uma diferença pequena o bastante para empatar
# é pequena o bastante para inverter). Os dois padrões abaixo somam zero, de
# modo que a média é exatamente a base, e são antissimétricos entre si, de
# modo que a diferença pareada tem desvio.
_PADRAO_A = [0.3, -0.3, 0.3, -0.3, 0.0]
_PADRAO_B = [-0.3, 0.3, -0.3, 0.3, 0.0]


def _serie(base: float, padrao: list) -> list:
    return [base + r for r in padrao]


def _consolidado_falso(linhas):
    """linhas = {nome: (wmape_por_semente, n_atributos)}."""
    return {"medido_em": "validacao",
            "configuracoes": {n: {"n_atributos": p,
                                  "wmape_por_semente": {str(i): v for i, v in enumerate(w)},
                                  "vies_de_nivel_por_semente": {str(i): 1.0 for i in range(len(w))},
                                  "elast_h10_por_semente": {str(i): -1.9 for i in range(len(w))}}
                              for n, (w, p) in linhas.items()}}


def test_a_regra_de_escolha_prefere_parcimonia_dentro_do_empate():
    """A regra de escolha do ponto, declarada em 16/09/2026 ANTES da tabela.

    Dois degraus, e o teste exige os dois:

    1. o grupo elegível é quem NÃO separa do melhor no WMAPE, pelo limiar
       corrigido de D26, e não os `k` primeiros de uma lista;
    2. dentro do grupo, ganha quem tem menos parâmetros, ainda que não seja o
       de menor WMAPE. Se a regra escolhesse o menor WMAPE, ela seria a mesma
       coisa que não ter regra.
    """
    from ruido_semente import comparar
    import varredura_rede as V
    d = _consolidado_falso({
        "gorda": (_serie(50.0, _PADRAO_A), 90000),  # melhor WMAPE, enorme
        "magra": (_serie(50.2, _PADRAO_B), 9000),   # empata no ruído, minúscula
        "ruim":  (_serie(70.0, _PADRAO_A), 5000),   # separa, e é a menor
    })
    cfgs = {"gorda": dict(V.MELHOR), "magra": dict(V.MELHOR),
            "ruim": dict(V.MELHOR)}
    r = comparar(d)
    e = V.escolher_ponto(d, r, cfgs)

    assert e["melhor_no_wmape"] == "gorda"
    assert set(g["config"] for g in e["grupo"]) == {"gorda", "magra"}, \
        "'ruim' tem WMAPE 20 pontos pior e não pode entrar no grupo"
    assert e["escolhido"] == "magra", \
        "a regra tem de preferir parcimônia DENTRO do empate, não o menor WMAPE"


def test_a_regra_desempata_parametros_iguais_por_menos_epocas():
    """Degrau 3. Contagem de parâmetros não mede capacidade efetiva quando
    `epocas` é eixo da grade, e esse é o limite declarado da regra: em empate
    exato de parâmetros, menos treino é menos capacidade usada."""
    from ruido_semente import comparar
    import varredura_rede as V
    d = _consolidado_falso({
        "muitas_epocas": (_serie(50.0, _PADRAO_A), 20000),
        "poucas_epocas": (_serie(50.2, _PADRAO_B), 20000),
    })
    cfgs = {"muitas_epocas": dict(V.MELHOR, epocas=60),
            "poucas_epocas": dict(V.MELHOR, epocas=8)}
    e = V.escolher_ponto(d, comparar(d), cfgs)
    assert e["escolhido"] == "poucas_epocas"


def test_a_regra_recusa_configuracao_fora_da_grade():
    """Se o consolidado tiver um nome que a grade atual não tem, o degrau 3 não
    tem de onde ler `epocas`. Ler 0 em silêncio faria a regra executada divergir
    da documentada sem ninguém notar."""
    from ruido_semente import comparar
    import varredura_rede as V
    d = _consolidado_falso({"a": (_serie(50.0, _PADRAO_A), 100),
                            "b": (_serie(50.1, _PADRAO_B), 200)})
    with pytest.raises(SystemExit):
        V.escolher_ponto(d, comparar(d), {"a": dict(V.MELHOR)})


def test_a_banda_de_elegibilidade_e_de_um_erro_padrao_e_nao_a_de_bonferroni():
    """A emenda de 16/09/2026 ao degrau 1 da regra de escolha.

    A correção de Bonferroni controla **descoberta falsa**: ela torna difícil
    declarar uma diferença que não existe. Usá-la como definição de
    "indistinguível" a transforma em licença para declarar **equivalência**, e
    nessa direção ela é anticonservadora, e piora quanto maior a grade.

    O teste constrói o caso que apareceu medido: uma configuração a três erros
    padrão da melhor, e minúscula. Sob o limiar de Bonferroni ela entrava no
    grupo e ganhava por parcimônia; sob a banda de um erro padrão, não entra.
    """
    from ruido_semente import comparar
    import varredura_rede as V
    d = _consolidado_falso({
        "boa":     (_serie(50.0, _PADRAO_A), 30000),
        "media":   (_serie(50.1, _PADRAO_B), 25000),
        "distante":(_serie(50.45, _PADRAO_B), 3000),  # fora de 1 ep, dentro de Bonferroni
    })
    cfgs = {k: dict(V.MELHOR) for k in ("boa", "media", "distante")}
    r = comparar(d)
    e = V.escolher_ponto(d, r, cfgs)

    z = abs((r["eixos"]["wmape"]["pares"].get("boa__menos__distante")
             or r["eixos"]["wmape"]["pares"]["distante__menos__boa"])["z"])
    assert z > 1.0, "o caso construído precisa ter a distante fora de 1 ep"
    assert z < r["limiar_z_corrigido"], \
        "e DENTRO do limiar de Bonferroni, senão o teste não testa nada"
    assert "distante" not in {g["config"] for g in e["grupo"]}, \
        "a banda de 1 erro padrão tem de excluir quem está a 3 erros padrão"
    assert e["escolhido"] != "distante"
    assert e["banda_em_erros_padrao"] == 1.0


def test_a_suavizacao_e_pos_processamento_e_nao_um_ajuste_proprio():
    """`msle` e `msle_smearing` treinam IDÊNTICO; só a previsão difere.

    Isso não é detalhe de implementação, é o que permite tirar `msle_smearing`
    da grade da família `perda` sem abandonar a pergunta: o efeito da correção
    de Duan passa a ser gravado como diagnóstico em toda execução, calculado do
    mesmo ajuste, em vez de exigir 50 treinos que reproduziriam os mesmos pesos.

    O teste exige as duas metades: os pesos batem, e o número suavizado de uma
    bate com o número direto da outra.
    """
    from pathlib import Path
    import varredura_rede as V
    d = V.preparar_dados("frj", PAINEL, False)
    cfg = dict(V.MELHOR, epocas=2, semente=0)

    a = V.uma_execucao(d, dict(cfg, perda="msle"))
    b = V.uma_execucao(d, dict(cfg, perda="msle_smearing"))

    assert a["n_atributos"] == b["n_atributos"]
    assert abs(a["elast_h10"] - b["elast_h10"]) < 1e-9, \
        "a suavização é fator CONSTANTE: não pode mover a elasticidade"
    assert abs(a["diagnosticos"]["fator_de_duan"]
               - b["diagnosticos"]["fator_de_duan"]) < 1e-9
    # o que `msle` diz que a suavização faria é o que `msle_smearing` de fato faz
    assert abs(a["diagnosticos"]["wmape_com_suavizacao"] - b["wmape_pct"]) < 1e-6
    assert abs(a["diagnosticos"]["vies_de_nivel_com_suavizacao"]
               - b["vies_de_nivel"]) < 1e-9
    # e, na que já aplicou, as colunas de diagnóstico repetem as principais
    assert abs(b["diagnosticos"]["wmape_com_suavizacao"] - b["wmape_pct"]) < 1e-9
