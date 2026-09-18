"""Divide ``ListaProdutosCissAtivos.csv`` em partes menores.

Gera ``parte_1.csv``, ``parte_2.csv``, ... na pasta do script. A divisão é
determinística (mesma ordem, mesmo tamanho), o que mantém estáveis as chaves do
estado usado pelo ``agent.py`` entre execuções.

Idempotente: se as partes já existem, nada é regravado — a menos que se use
``--forcar``. Assim o passo pode rodar em todo job do GitHub Actions sem
desperdiçar trabalho nem sujar o diff.

Uso::

    python breaker.py                     # partes de 1000 linhas
    python breaker.py --chunk-size 500
    python breaker.py --forcar            # recria mesmo se já existir
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PASTA_BASE = Path(__file__).resolve().parent


def dividir(
    entrada: Path,
    prefixo: str,
    tamanho: int,
    saida_dir: Path,
    forcar: bool = False,
) -> list:
    """Quebra o CSV de entrada em blocos de ``tamanho`` linhas."""
    if not entrada.exists():
        raise SystemExit(f"❌ Arquivo de entrada não encontrado: {entrada}")

    df = pd.read_csv(entrada, dtype=str)
    total = len(df)
    quantidade = (total + tamanho - 1) // tamanho

    if quantidade == 0:
        print(f"⚠️ {entrada.name} não tem linhas; nada a dividir.", flush=True)
        return []

    saida_dir.mkdir(parents=True, exist_ok=True)
    gerados = []

    for indice in range(quantidade):
        destino = saida_dir / f"{prefixo}_{indice + 1}.csv"
        if destino.exists() and not forcar:
            print(f"↩️  {destino.name} já existe, pulando (use --forcar para recriar).", flush=True)
            gerados.append(destino)
            continue

        bloco = df.iloc[indice * tamanho : (indice + 1) * tamanho]
        bloco.to_csv(destino, index=False)
        gerados.append(destino)
        print(f"✂️  {destino.name}: {len(bloco)} produtos", flush=True)

    print(f"\n🎉 {total} produtos divididos em {quantidade} parte(s).", flush=True)
    return gerados


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Divide o CSV de produtos em partes.")
    parser.add_argument(
        "--entrada",
        default=str(PASTA_BASE / "ListaProdutosCissAtivos.csv"),
        help="CSV completo de origem.",
    )
    parser.add_argument(
        "--saida-dir",
        default=str(PASTA_BASE),
        help="Pasta onde as partes são gravadas (padrão: a pasta deste script).",
    )
    parser.add_argument("--prefixo", default="parte", help="Prefixo dos arquivos gerados.")
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Linhas por parte (padrão: 1000)."
    )
    parser.add_argument(
        "--forcar", action="store_true", help="Recria as partes mesmo que já existam."
    )
    args = parser.parse_args(argv)

    if args.chunk_size <= 0:
        parser.error("--chunk-size precisa ser maior que zero")

    dividir(
        entrada=Path(args.entrada).resolve(),
        prefixo=args.prefixo,
        tamanho=args.chunk_size,
        saida_dir=Path(args.saida_dir).resolve(),
        forcar=args.forcar,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
