"""A sensibilidade que D21 exige, e ela tem DUAS perguntas distintas.

D21 declarou um **critério de escalonamento**, e ele é uma promessa a cumprir e não
uma ressalva:

> se na etapa 7 a elasticidade agregada derivada da rede continuar com intervalo
> atravessando `|ε| = 1`, a etapa 8 **não deve** reportar direção ótima de margem, e o
> contraste de D21 fica restrito ao que é robusto, que é o efeito de um corte.

Isso exige o **intervalo**, não o ponto, e intervalo aqui é entre sementes: o ponto
−2,404 sai do artefato canônico, que as 50 sementes mostraram ser pessimista em ganho
(10.13). Daí a Parte 1 deste script.

A segunda pergunta é mais ampla e é a que o texto precisa: **quanto do resultado
depende de a magnitude das elasticidades estar certa?** Um ganho de 42% que virasse 8%
com a elasticidade 25% menor seria resultado frágil, mesmo com intervalo que não
atravessa 1. Daí a Parte 2.

## Parte 1: a elasticidade agregada entre sementes

Sob perturbação **uniforme** de 1% em todos os preços, a elasticidade agregada da
célula é `Σᵢ sᵢ Σⱼ εᵢⱼ` com `sᵢ = V̂ᵢ/ΣV̂`, que é a conta de `rede.identidade_de_agregacao`
(ponto único, pendência 2.8). Uniforme é o que a torna elasticidade: variação não
uniforme não tem elasticidade agregada, e foi por confundir isso que a primeira versão
do diagnóstico de coerência errou (10.10).

Treina 50 sementes e reporta média, erro padrão e o intervalo. O teste é se o intervalo
**cruza −1**.

## Parte 2: o ganho contra a magnitude das elasticidades

Reescalar todas as elasticidades por `λ`, mantendo a previsão no ponto histórico
**intacta**. No modelo adotado, com `ln V̂ = g(x) + Σγu − m(u)`, o canal de preço é a
parte que depende de `u`, de modo que

    ln V̂_λ(u) = ln V̂(u⁰) + λ·[ ln V̂(u) − ln V̂(u⁰) ]

Em `u = u⁰` nada muda, e a Jacobiana fica `λ·ε` exatamente, porque `g` não depende de
`u`. Isso é o que torna a conta uma sensibilidade **à elasticidade** e não uma
sensibilidade ao nível: mexer no nível mudaria a receita base e o ganho seria razão
entre coisas diferentes.

`λ = 1` reproduz o resultado adotado, e serve de conferência de que o invólucro não
mudou nada por acidente.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
for sub in ("models", "data", "experiments", "optimization"):
    sys.path.insert(0, str(RAIZ / "src" / sub))

import hierarquia as HIER  # noqa: E402
import problema as P  # noqa: E402
import rede  # noqa: E402
from sementes_item8 import resumo  # noqa: E402
from sonda_item8 import OBJETIVOS, carregar_modelo, restricoes_do_escopo  # noqa: E402

LAMBDAS = (0.5, 0.75, 1.0, 1.25, 1.5)


class ModeloEscalado:
    """Invólucro que multiplica todas as elasticidades por `lam`.

    Expõe a mesma interface que `problema.valor_e_gradiente` consome
    (`log_demanda`, `elasticidades_fechadas`, `parameters`), de modo que nada a
    jusante precisa saber que o modelo foi reescalado. É o ponto em que um
    invólucro é melhor que um parâmetro novo em cinco funções.
    """

    def __init__(self, modelo, lam: float, u_base):
        import torch
        self._m = modelo
        self.lam = float(lam)
        # `u_base` e o u0 de UMA celula, vetor de tamanho n. Guardado em (1, n) para
        # difundir sobre o lote que `log_demanda` receber.
        u0 = np.atleast_2d(np.asarray(u_base, dtype=float))
        if u0.shape[0] != 1:
            raise ValueError("u_base tem de ser o u0 de UMA celula")
        self._u0 = torch.as_tensor(u0, dtype=next(modelo.parameters()).dtype)

    def parameters(self):
        return self._m.parameters()

    def log_demanda(self, ctx, u, loja):
        u0 = self._u0.expand(u.shape[0], -1)
        base = self._m.log_demanda(ctx, u0, loja)
        return base + self.lam * (self._m.log_demanda(ctx, u, loja) - base)

    def elasticidades_fechadas(self, u):
        return self.lam * self._m.elasticidades_fechadas(u)


def elasticidade_agregada_por_semente(dados, cfg_base, esc, n_sementes: int,
                                      semente_inicial: int = 0) -> dict:
    """Parte 1: `Σᵢ sᵢ Σⱼ εᵢⱼ` no ponto histórico, por semente treinada."""
    import torch
    from varredura_rede import MELHOR  # noqa: F401  (documenta a origem de cfg)

    ctx = torch.as_tensor(esc["ctx"], dtype=torch.float32)
    loja = torch.as_tensor(esc["loja_idx"].astype("int64"), dtype=torch.long)
    u = torch.as_tensor(esc["u"], dtype=torch.float32)

    agregadas, violacoes, proprias = [], [], []
    for i in range(n_sementes):
        s = semente_inicial + i
        cfg = dict(cfg_base, semente=s, restrita=True)
        cfg.setdefault("gama_contextual", False)
        modelo = rede.treinar(dados, cfg)["modelo"]
        modelo.eval()
        with torch.no_grad():
            v = torch.exp(modelo.log_demanda(ctx, u, loja))
            eps = modelo.elasticidades_fechadas(u)
        ident = rede.identidade_de_agregacao(eps, v)
        agregadas.append(ident["elast_agregada_mediana"])
        proprias.append(ident["propria_ponderada_mediana"])
        violacoes.append(ident["pct_celulas_com_violacao"])
        print(f"  semente {s:2d}  ε agregada {agregadas[-1]:7.3f}  "
              f"própria ponderada {proprias[-1]:7.3f}  "
              f"violação {violacoes[-1]:5.2f}%", flush=True)

    ag = np.array(agregadas)
    r = resumo(ag)
    # Intervalo de 95% da MEDIA, que e o objeto do critério de D21: ele fala do
    # intervalo da elasticidade derivada da rede, e a rede tem uma por semente.
    r["ic95_media"] = [r["media"] - 1.96 * r["erro_padrao"],
                       r["media"] + 1.96 * r["erro_padrao"]]
    # E o intervalo entre SEMENTES, que e mais largo e responde "onde uma execucao
    # unica pode cair". Os dois vao no relatorio, pela licao de 10.13.
    r["faixa_entre_sementes"] = [r["min"], r["max"]]
    r["cruza_menos_um"] = bool(r["ic95_media"][1] > -1.0)
    r["alguma_semente_cruza_menos_um"] = bool(r["max"] > -1.0)
    return {"elast_agregada": r,
            "propria_ponderada": resumo(np.array(proprias)),
            "pct_violacao_identidade": resumo(np.array(violacoes)),
            "por_semente": [{"semente": semente_inicial + i,
                             "elast_agregada": agregadas[i],
                             "propria_ponderada": proprias[i],
                             "pct_violacao": violacoes[i]}
                            for i in range(n_sementes)]}


def ganho_com_lambda(modelo, centro, esc, lam: float, restr, metodo: str) -> dict:
    """Parte 2: reotimiza com todas as elasticidades multiplicadas por `lam`."""
    linha = {"lambda": lam}
    for objetivo in OBJETIVOS:
        custos = (esc["custos"] if objetivo == "margem"
                  else np.zeros_like(esc["custos"]))
        ini = fim = 0.0
        interior = inviaveis = 0
        d_preco = []
        for c in range(len(esc["u"])):
            if restr is not None and restr[2][c]:
                inviaveis += 1
                continue
            # O invólucro precisa do u⁰ DESTA célula, senão a ancoragem é da errada.
            esc_c = ModeloEscalado(modelo, lam, esc["u"][c])
            r_celula = None if restr is None else (restr[0], restr[1][c])
            prob = P.ProblemaDeCelula(esc_c, esc["ctx"][c], int(esc["loja_idx"][c]),
                                      centro, custos[c], esc["u"][c],
                                      restricoes=r_celula)
            r = P.resolver(prob, metodo=metodo)
            ini += r["valor_inicial"]
            fim += r["valor"]
            interior += r["n_interior"]
            d_preco.append(float(np.mean(r["u"] - esc["u"][c])))
        resolvidas = len(esc["u"]) - inviaveis
        linha[objetivo] = {
            "ganho_pct_agregado": 100.0 * (fim / ini - 1.0),
            "pct_coord_interior": 100.0 * interior / max(resolvidas * esc["n"], 1),
            "variacao_log_preco_media": float(np.mean(d_preco)),
            "pct_celulas_que_SOBEM_preco": 100.0 * float(np.mean(
                np.array(d_preco) > 0)),
        }
    return linha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categoria")
    ap.add_argument("--painel", default="data/interim/painel")
    ap.add_argument("--bruto", default="data/raw")
    ap.add_argument("--artefato", default=None)
    ap.add_argument("--sementes", type=int, default=50)
    ap.add_argument("--lambdas", default=",".join(str(x) for x in LAMBDAS))
    ap.add_argument("--hierarquia", default="nao_piorar",
                    choices=("nenhuma",) + HIER.FORMAS)
    ap.add_argument("--metodo", default="SLSQP")
    ap.add_argument("--so-parte", type=int, default=0, choices=(0, 1, 2),
                    help="0 roda as duas partes; 1 ou 2 roda só aquela")
    ap.add_argument("--saida", default="reports")
    ap.add_argument("--sufixo", default="")
    a = ap.parse_args()

    from varredura_rede import MELHOR

    painel = Path(a.painel)
    dados = rede.preparar(a.categoria, painel)
    centro = np.asarray(dados["centro_de_preco"], float)
    esc = P.escopo(dados, painel, a.categoria, "teste")

    restr, pares = None, []
    if a.hierarquia != "nenhuma":
        A, B, pares, _, inviavel = restricoes_do_escopo(
            a.categoria, painel, a.bruto, centro, esc, a.hierarquia)
        restr = (A, B, inviavel)

    bloco = {"categoria": a.categoria, "hierarquia": a.hierarquia,
             "escopo": esc["contagens"], "configuracao": dict(MELHOR)}
    t0 = time.perf_counter()

    if a.so_parte in (0, 1):
        print(f"PARTE 1: elasticidade agregada uniforme em {a.sementes} sementes\n")
        bloco["parte1"] = elasticidade_agregada_por_semente(
            dados, dict(MELHOR), esc, a.sementes)

    if a.so_parte in (0, 2):
        arq = Path(a.artefato or RAIZ / "models_artifacts" / f"rede_{a.categoria}.pt")
        modelo, cfg, centro_art = carregar_modelo(arq, dados)
        lams = [float(x) for x in a.lambdas.split(",")]
        print(f"\nPARTE 2: ganho contra λ, no artefato canônico\n")
        linhas = []
        for lam in lams:
            t = time.perf_counter()
            linha = ganho_com_lambda(modelo, centro, esc, lam, restr, a.metodo)
            linhas.append(linha)
            print(f"  λ = {lam:4.2f}  margem {linha['margem']['ganho_pct_agregado']:6.2f}%"
                  f"  receita {linha['receita']['ganho_pct_agregado']:6.2f}%"
                  f"  ({time.perf_counter() - t:.0f} s)", flush=True)
        bloco["parte2"] = {"lambdas": lams, "linhas": linhas}

    bloco["tempo_total_min"] = (time.perf_counter() - t0) / 60.0
    destino = (Path(a.saida)
               / f"sensibilidade_item8_{a.categoria}{a.sufixo}.json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(bloco, indent=2, ensure_ascii=False))

    if "parte1" in bloco:
        e = bloco["parte1"]["elast_agregada"]
        print(f"\n== PARTE 1, o critério de escalonamento de D21")
        print(f"  elasticidade agregada uniforme: média {e['media']:.4f}, "
              f"ep {e['erro_padrao']:.4f}, dp {e['desvio']:.4f}")
        print(f"  IC 95% da média:        [{e['ic95_media'][0]:.4f}, "
              f"{e['ic95_media'][1]:.4f}]")
        print(f"  faixa entre sementes:   [{e['faixa_entre_sementes'][0]:.4f}, "
              f"{e['faixa_entre_sementes'][1]:.4f}]")
        print(f"  o IC cruza −1?                  "
              f"{'SIM' if e['cruza_menos_um'] else 'NÃO'}")
        print(f"  alguma semente cruza −1?        "
              f"{'SIM' if e['alguma_semente_cruza_menos_um'] else 'NÃO'}")
        print(f"  -> D21 {'PROIBE' if e['cruza_menos_um'] else 'LIBERA'} "
              "reportar direção ótima de margem")

    if "parte2" in bloco:
        print(f"\n== PARTE 2, sensibilidade à magnitude das elasticidades")
        print(f"{'λ':>5} {'ganho margem':>13} {'interior':>9} {'sobem preço':>12}"
              f" | {'ganho receita':>13} {'interior':>9} {'sobem preço':>12}")
        for L in bloco["parte2"]["linhas"]:
            m, r = L["margem"], L["receita"]
            print(f"{L['lambda']:5.2f} {m['ganho_pct_agregado']:12.2f}% "
                  f"{m['pct_coord_interior']:8.2f}% "
                  f"{m['pct_celulas_que_SOBEM_preco']:11.2f}% | "
                  f"{r['ganho_pct_agregado']:12.2f}% "
                  f"{r['pct_coord_interior']:8.2f}% "
                  f"{r['pct_celulas_que_SOBEM_preco']:11.2f}%")
    print(f"\ntempo total {bloco['tempo_total_min']:.1f} min\n-> {destino}")


if __name__ == "__main__":
    main()
