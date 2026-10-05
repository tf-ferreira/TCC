"""Mundos de resposta cruzada e política robusta (D44).

## O problema

O ganho do otimizador mora sobretudo nas elasticidades cruzadas (sondas 15 e 16
do diagnóstico de 23/09/2026: 70% do ganho de margem e 44% do de receita), e é
justamente a parte que o painel não identifica (8.16, D36). A varredura λ de D21
escala próprias e cruzadas juntas e por isso não expõe essa dependência.

## O mundo

Um **mundo** é a mesma rede com a resposta cruzada a **desvios** do preço
histórico passada por uma função `f` de `γ`, e a previsão no ponto histórico
**ancorada**, como na varredura λ:

    ln V̂ᵢ = gᵢ(x) + Σⱼ γᵢⱼ·uⱼ⁰ + Σⱼ f(γ)ᵢⱼ·(uⱼ − uⱼ⁰) − mᵢ(uᵢ)

Com `u = u⁰` todos os mundos preveem igual (a âncora), de modo que o
denominador do ganho, `M̂_hist`, e a calibração de D40 não mudam entre mundos; o
que muda é só a resposta a preço novo. Dois tipos de `f`:

- **escala**: `f(γ) = μ·γ`, com μ ∈ {0; 0,5; 1}. μ = 1 é a rede como está; μ = 0
  é "nenhuma resposta cruzada";
- **só substitutos**: `f(γ) = max(γ, 0)`, que zera as complementaridades entre
  marcas do mesmo suco, implausíveis como efeito de preço.

A garantia de D33 sobrevive em todo mundo: a diagonal continua `−mᵢ′(uᵢ) ≤ 0`, e
`f` só mexe fora dela.

## A política robusta

A política que o otimizador escolhe acreditando em `γ` rende pouco se as
cruzadas verdadeiras forem menores (sonda 16: 3,1% de margem com μ = 0, contra
12,8% da política que as ignora). A **política robusta** maximiza a **média** do
objetivo entre mundos, pelo mesmo argumento da política de conjunto entre
sementes: decidir sob incerteza declarada sobre o modelo. `Conjunto` é esse
objetivo médio, e `problema.valor_e_gradiente` o reconhece.

Robustez só vale sobre os mundos que se declaram: a sonda 17 mostrou que a
política robusta em μ perde da política da rede no mundo "só substitutos", que
ficou fora da média. O conjunto de mundos é decisão registrada, não detalhe.
"""

from __future__ import annotations

import numpy as np


def _torch():
    import torch
    return torch


def _gama(modelo):
    if getattr(modelo, "gama", None) is None:
        raise ValueError("mundos exigem a rede restrita com γ constante (D36)")
    return (modelo.gama * modelo.fora_da_diagonal).detach()


class Mundo:
    """A rede com a resposta cruzada a desvios do histórico passada por `f`.

    `escala` e `so_substitutos` definem `f`. Sem âncora ele não prevê nada:
    `ancorado(u0)` devolve o objeto que o problema de PNL usa.
    """

    def __init__(self, modelo, escala: float = 1.0, so_substitutos: bool = False,
                 nome: str | None = None):
        if so_substitutos and escala != 1.0:
            raise ValueError("'só substitutos' não combina com escala")
        self.modelo = modelo
        self.escala = float(escala)
        self.so_substitutos = bool(so_substitutos)
        self.nome = nome or (f"mu={self.escala:g}" if not so_substitutos
                             else "so_substitutos")

    def gama_da_resposta(self):
        torch = _torch()
        G = _gama(self.modelo)
        if self.so_substitutos:
            return torch.clamp(G, min=0.0)
        return self.escala * G

    def ancorado(self, u0) -> "MundoAncorado":
        return MundoAncorado(self, u0)

    def parameters(self):
        return self.modelo.parameters()


class MundoAncorado:
    """Um mundo com a âncora `u⁰` fixada, com a interface que o PNL usa.

    `u0` tem forma (B, n) ou (n,): uma linha por célula, na MESMA ordem das
    células que serão avaliadas. Num problema de célula, B = 1.
    """

    def __init__(self, mundo: Mundo, u0):
        torch = _torch()
        self.mundo = mundo
        dtype = next(mundo.modelo.parameters()).dtype
        u0 = torch.as_tensor(np.asarray(u0, dtype=float), dtype=dtype)
        self.u0 = u0.unsqueeze(0) if u0.ndim == 1 else u0
        self.G = _gama(mundo.modelo).to(dtype)
        self.Gf = mundo.gama_da_resposta().to(dtype)

    def parameters(self):
        return self.mundo.modelo.parameters()

    def log_demanda(self, ctx, u, loja):
        m = self.mundo.modelo
        g, gama_ctx = m.tronco(ctx, loja)
        if gama_ctx is not None:
            raise ValueError("mundos exigem γ constante (D36)")
        if self.u0.shape[0] not in (1, u.shape[0]):
            raise ValueError(f"âncora com {self.u0.shape[0]} linhas para {u.shape[0]} células")
        return g + self.u0 @ self.G.T + (u - self.u0) @ self.Gf.T - m.monotona(u)

    def elasticidades_fechadas(self, u):
        torch = _torch()
        b = u.shape[0]
        fora = self.Gf.unsqueeze(0).expand(b, -1, -1)
        return fora + torch.diag_embed(-self.mundo.modelo.monotona.derivada(u))


class Conjunto:
    """Objetivo MÉDIO sobre membros (mundos ou redes). É a política robusta."""

    def __init__(self, membros: list, nome: str = "robusta"):
        if not membros:
            raise ValueError("conjunto vazio")
        self.membros = list(membros)
        self.nome = nome

    def parameters(self):
        return self.membros[0].parameters()

    def ancorado(self, u0) -> "Conjunto":
        return Conjunto([ancorar(m, u0) for m in self.membros], self.nome)


def ancorar(modelo, u0):
    """Âncora de um mundo ou conjunto; a rede pura passa sem mudança."""
    return modelo.ancorado(u0) if hasattr(modelo, "ancorado") else modelo


# Os mundos da v1 (D44). A política robusta é a média dos três de escala.
ESCALAS_V1 = (0.0, 0.5, 1.0)
NOMES_V1 = [f"mu={mu:g}" for mu in ESCALAS_V1] + ["so_substitutos"]


def mundos_v1(modelo) -> dict:
    """Os quatro mundos de avaliação da v1, por nome."""
    fora = {f"mu={mu:g}": Mundo(modelo, escala=mu) for mu in ESCALAS_V1}
    fora["so_substitutos"] = Mundo(modelo, so_substitutos=True)
    return fora


def politica_robusta_v1(modelo) -> Conjunto:
    return Conjunto([Mundo(modelo, escala=mu) for mu in ESCALAS_V1])
