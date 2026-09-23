"""Recoloca na fila os produtos que têm campos “Não encontrado”/vazios.

Lê o estado do agente (``saida/estado_parte_*.jsonl``), identifica os
registros ``ok`` com lacunas usando as mesmas regras da auditoria
(``auditar_enriquecimento.py``) e grava uma nova linha de estado com
``status: erro`` e ``tentativas: 0`` — o produto volta à fila na próxima
execução do agente.

Como o agente preserva campos já preenchidos ao reprocessar, apenas as
lacunas serão preenchidas; o que está bom permanece intocado.

Por segurança o padrão é simular; use ``--aplicar`` para gravar.

Uso::

    python resetar_lacunas.py                          # simula
    python resetar_lacunas.py --aplicar                # recoloca todos com lacuna
    python resetar_lacunas.py --aplicar --min-lacunas 3   # só os mais críticos
    python agent.py --limite 20                        # processa aos poucos
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from auditar_enriquecimento import ausente, fotos_de

PASTA_BASE = Path(__file__).resolve().parent

CAMPOS_TEXTO = ("descricao_produto", "especificacao_tecnica", "pagina_web", "manual")


def lacunas_do(registro: dict) -> list:
    """Campos sem conteúdo real no registro de saída (mesmas regras da auditoria)."""
    lacunas = [campo for campo in CAMPOS_TEXTO if ausente(registro.get(campo))]
    if not fotos_de(registro.get("fotos", "")):
        lacunas.append("fotos")
    return lacunas


def _ordem_natural(caminho: Path):
    return [
        int(trecho) if trecho.isdigit() else trecho
        for trecho in re.split(r"(\d+)", caminho.stem)
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Recoloca à fila os produtos 'ok' que têm campos não encontrados."
    )
    parser.add_argument(
        "--saida-dir", default=str(PASTA_BASE / "saida"), help="Pasta de estado do agente."
    )
    parser.add_argument(
        "--padrao", default="estado_parte_*.jsonl", help="Glob dos arquivos de estado."
    )
    parser.add_argument(
        "--min-lacunas",
        type=int,
        default=1,
        help="Só recoloca produtos com ao menos N lacunas (padrão: 1).",
    )
    parser.add_argument(
        "--aplicar", action="store_true", help="Grava o novo estado (sem isto, simula)."
    )
    args = parser.parse_args(argv)

    if args.min_lacunas < 1:
        parser.error("--min-lacunas precisa ser >= 1")

    saida_dir = Path(args.saida_dir).resolve()
    estados = sorted(saida_dir.glob(args.padrao), key=_ordem_natural)
    if not estados:
        print(f"❌ Nenhum estado encontrado em {saida_dir} com '{args.padrao}'.", flush=True)
        return 1

    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total_ok = 0
    total_reset = 0
    por_lacunas: dict = {}

    for arquivo in estados:
        # Última linha de cada id vence (mesma leitura do agente).
        registros = {}
        with arquivo.open("r", encoding="utf-8") as ponteiro:
            for linha in ponteiro:
                linha = linha.strip()
                if not linha:
                    continue
                try:
                    registro = json.loads(linha)
                except json.JSONDecodeError:
                    continue
                if registro.get("idsubproduto"):
                    registros[str(registro["idsubproduto"])] = registro

        alvo = []
        for registro in registros.values():
            if registro.get("status") != "ok":
                continue
            total_ok += 1
            lacunas = lacunas_do(registro)
            if len(lacunas) >= args.min_lacunas:
                alvo.append((registro, lacunas))
                por_lacunas[len(lacunas)] = por_lacunas.get(len(lacunas), 0) + 1

        print(
            f"📦 {arquivo.name}: {len(registros)} produtos | "
            f"{len(alvo)} com >= {args.min_lacunas} lacuna(s) voltam à fila",
            flush=True,
        )

        if args.aplicar:
            with arquivo.open("a", encoding="utf-8") as ponteiro:
                for registro, lacunas in alvo:
                    novo = dict(registro)
                    novo["status"] = "erro"
                    novo["erro"] = f"resetado p/ reprocessamento (lacunas: {';'.join(lacunas)})"
                    novo["tentativas"] = 0
                    novo["atualizado_em"] = agora
                    ponteiro.write(json.dumps(novo, ensure_ascii=False) + "\n")
        total_reset += len(alvo)

    distribuicao = ", ".join(f"{n} lacunas: {qtd}" for n, qtd in sorted(por_lacunas.items(), reverse=True))
    print(f"\n🧮 ok analisados: {total_ok} | com lacunas: {sum(por_lacunas.values())} ({distribuicao})", flush=True)

    if args.aplicar:
        print(
            f"✅ {total_reset} produto(s) recolocados na fila. Processe com:\n"
            f"   python agent.py --limite 20   (ou dispare o workflow no Actions)",
            flush=True,
        )
    else:
        print(
            f"👀 Simulação — nada foi gravado. Use --aplicar para recolocar "
            f"{total_reset} produto(s) na fila.",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
