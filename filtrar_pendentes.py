"""Remove das partes CSV os produtos já enriquecidos e gera um arquivo único
com o que ainda falta processar (``teste_final.csv`` por padrão).

Lê o estado gravado pelo agente (``saida/estado_parte_*.jsonl``) e considera
"já enriquecido" todo produto com ``status == "ok"``. Registros de ``erro``
continuam na fila — o agente os retoma automaticamente (até MAX_TENTATIVAS).

De quebra, devolve ao estado das partes os produtos processados via o arquivo
consolidado (``saida/estado_teste_final.jsonl``), para que uma execução futura
sobre as partes não pague de novo por eles.

Por segurança, o padrão é simular: nenhum arquivo é alterado. Use ``--aplicar``
para gravar.

Uso::

    python filtrar_pendentes.py              # só mostra o que seria feito
    python filtrar_pendentes.py --aplicar    # filtra as partes e grava o consolidado
    python agent.py --padrao teste_final.csv --limite 20   # teste final
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

PASTA_BASE = Path(__file__).resolve().parent


def _ordem_natural(caminho: Path):
    return [
        int(trecho) if trecho.isdigit() else trecho
        for trecho in re.split(r"(\d+)", caminho.stem)
    ]


def carregar_ok(caminho_estado: Path) -> dict:
    """Mapa idsubproduto → registro para os produtos com status 'ok'."""
    ok = {}
    if not caminho_estado.exists():
        return ok
    with caminho_estado.open("r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if not linha:
                continue
            try:
                registro = json.loads(linha)
            except json.JSONDecodeError:
                continue
            if registro.get("status") == "ok" and registro.get("idsubproduto"):
                ok[str(registro["idsubproduto"])] = registro
    return ok


def mesclar_estado_consolidado(saida_dir: Path, estados: dict, partes_df: dict) -> int:
    """Copia os 'ok' do consolidado para o estado da parte de origem.

    Sem isso, um produto processado via ``teste_final.csv`` seria processado
    de novo (e cobrado de novo) quando o pipeline rodar pelas partes.
    """
    consolidado = saida_dir / "estado_teste_final.jsonl"
    if not consolidado.exists():
        return 0

    onde_esta = {
        str(linha["idsubproduto"]): parte
        for parte, df in partes_df.items()
        if "idsubproduto" in df.columns
        for linha in df.to_dict(orient="records")
        if str(linha.get("idsubproduto", "")) != ""
    }

    mesclados = 0
    with consolidado.open("r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if not linha:
                continue
            try:
                registro = json.loads(linha)
            except json.JSONDecodeError:
                continue
            if registro.get("status") != "ok":
                continue
            chave = str(registro.get("idsubproduto", ""))
            parte = onde_esta.get(chave)
            if parte is None or chave in estados[parte]:
                continue
            with (saida_dir / f"estado_{parte.stem}.jsonl").open(
                "a", encoding="utf-8"
            ) as destino:
                destino.write(json.dumps(registro, ensure_ascii=False) + "\n")
            estados[parte][chave] = registro
            mesclados += 1

    if mesclados:
        print(f"🔀 {mesclados} produto(s) do consolidado devolvidos ao estado das partes.", flush=True)
    return mesclados


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove os produtos já enriquecidos das partes e gera o consolidado."
    )
    parser.add_argument(
        "--entrada-dir", default=str(PASTA_BASE), help="Pasta com os parte_*.csv."
    )
    parser.add_argument(
        "--saida-dir", default=str(PASTA_BASE / "saida"), help="Pasta de estado do agente."
    )
    parser.add_argument(
        "--consolidado",
        default=str(PASTA_BASE / "teste_final.csv"),
        help="Arquivo gerado com os produtos pendentes (padrão: teste_final.csv).",
    )
    parser.add_argument(
        "--padrao", default="parte_*.csv", help="Glob das partes (padrão: parte_*.csv)."
    )
    parser.add_argument(
        "--aplicar",
        action="store_true",
        help="Grava as alterações (sem isto, apenas simula).",
    )
    args = parser.parse_args(argv)

    entrada_dir = Path(args.entrada_dir).resolve()
    saida_dir = Path(args.saida_dir).resolve()

    partes = sorted(entrada_dir.glob(args.padrao), key=_ordem_natural)
    if not partes:
        print(f"❌ Nenhuma parte encontrada em {entrada_dir} com '{args.padrao}'.", flush=True)
        return 1

    partes_df = {}
    for parte in partes:
        partes_df[parte] = pd.read_csv(parte, dtype=str).fillna("")

    estados = {
        parte: carregar_ok(saida_dir / f"estado_{parte.stem}.jsonl") for parte in partes
    }

    mesclar_estado_consolidado(saida_dir, estados, partes_df)

    pendentes_total = []
    print("", flush=True)
    for parte in partes:
        df = partes_df[parte]
        ok = estados[parte]
        mascara = ~df["idsubproduto"].astype(str).isin(ok)
        restantes = df[mascara]
        removidos = len(df) - len(restantes)
        pendentes_total.append(restantes)
        print(
            f"📦 {parte.name}: {len(df)} produtos | "
            f"{removidos} já enriquecidos (saem) | {len(restantes)} restam",
            flush=True,
        )
        if args.aplicar and removidos:
            restantes.to_csv(parte, index=False)

    consolidado = pd.concat(pendentes_total, ignore_index=True) if pendentes_total else None
    total_restante = 0 if consolidado is None else len(consolidado)

    if args.aplicar:
        destino = Path(args.consolidado).resolve()
        if consolidado is None or consolidado.empty:
            if destino.exists():
                destino.unlink()
            print("\n🎉 Nenhum produto pendente; consolidado não foi gravado.", flush=True)
        else:
            consolidado.to_csv(destino, index=False)
            print(
                f"\n📝 Consolidado gravado: {destino} ({total_restante} produtos)", flush=True
            )
            print(
                f"   Teste com: python agent.py --padrao {destino.name} --limite 20",
                flush=True,
            )
    else:
        print(
            f"\n👀 Simulação — nada foi gravado. Total que restaria: "
            f"{total_restante} produtos. Use --aplicar para gravar.",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
