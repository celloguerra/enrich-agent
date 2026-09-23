"""Audita a cobertura dos campos nas planilhas enriquecidas.

Lê os ``saida/produtos_enriquecidos_parte_*.csv`` e resume, campo a campo,
o que está preenchido de verdade (URL válida, texto) versus "Não encontrado"
ou vazio. Gera também um relatório CSV só com os produtos que têm lacunas,
ordenados do mais ao menos crítico.

Uso::

    python auditar_enriquecimento.py                    # resume na tela
    python auditar_enriquecimento.py --relatorio -      # só na tela, sem CSV
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

PASTA_BASE = Path(__file__).resolve().parent

# Variações que a IA/pipeline usam para "não achei".
NAO_ENCONTRADO = re.compile(r"n[ãa]o\s+encontrad", re.IGNORECASE)


def _norm(valor) -> str:
    return (valor or "").strip()


def eh_url(valor: str) -> bool:
    valor = _norm(valor)
    if not valor.lower().startswith("http"):
        return False
    try:
        return bool(urlparse(valor).netloc)
    except ValueError:
        return False


def ausente(valor: str) -> bool:
    valor = _norm(valor)
    return (not valor) or bool(NAO_ENCONTRADO.search(valor))


def fotos_de(valor: str) -> list:
    return [u for u in _norm(valor).split(" | ") if u.startswith("http")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audita cobertura das planilhas enriquecidas.")
    parser.add_argument(
        "--saida-dir", default=str(PASTA_BASE / "saida"), help="Pasta com as planilhas."
    )
    parser.add_argument(
        "--relatorio",
        default=str(PASTA_BASE / "saida" / "auditoria_enriquecimento.csv"),
        help="CSV de saída com os produtos que têm lacunas ('-' para não gravar).",
    )
    args = parser.parse_args(argv)

    saida_dir = Path(args.saida_dir).resolve()
    planilhas = sorted(saida_dir.glob("produtos_enriquecidos_parte_*.csv"))
    if not planilhas:
        print(f"❌ Nenhuma planilha em {saida_dir}.", flush=True)
        return 1

    linhas = []
    fieldnames = None
    for planilha in planilhas:
        with planilha.open(encoding="utf-8-sig") as arquivo:
            leitor = csv.DictReader(arquivo)
            fieldnames = leitor.fieldnames
            linhas.extend(leitor)

    total = len(linhas)
    texto = {campo: 0 for campo in ("descricao_produto", "especificacao_tecnica")}
    link = {
        campo: {"url": 0, "ausente": 0, "outro": 0}
        for campo in ("pagina_web", "manual")
    }
    dist_fotos = Counter()
    dominios = Counter()
    sem_nenhum_link = 0
    com_lacunas = []

    for registro in linhas:
        lacunas = []

        for campo in texto:
            if ausente(registro.get(campo)):
                lacunas.append(campo)
            else:
                texto[campo] += 1

        tem_link = False
        for campo in link:
            valor = _norm(registro.get(campo))
            if ausente(valor):
                link[campo]["ausente"] += 1
                lacunas.append(campo)
            elif eh_url(valor):
                link[campo]["url"] += 1
                tem_link = True
                dominios[urlparse(valor).netloc] += 1
            else:
                link[campo]["outro"] += 1
                lacunas.append(campo + "?")

        fotos = fotos_de(registro.get("fotos"))
        dist_fotos[min(len(fotos), 3)] += 1
        if fotos:
            tem_link = True
        else:
            lacunas.append("fotos")

        if not tem_link:
            sem_nenhum_link += 1
        if lacunas:
            com_lacunas.append((len(lacunas), registro, lacunas))

    pct = lambda n: f"{100 * n / total:5.1f}%" if total else "  n/a"  # noqa: E731
    print(f"\n📊 Auditoria de enriquecimento — {total} produtos em {len(planilhas)} planilha(s)\n")
    for campo, n in texto.items():
        print(f"   {campo:<22} preenchido: {n:>5} ({pct(n)})")
    for campo, contagem in link.items():
        print(
            f"   {campo:<22} URL real: {contagem['url']:>5} ({pct(contagem['url'])}) | "
            f"Não encontrado: {contagem['ausente']:>5} ({pct(contagem['ausente'])}) | "
            f"preenchido sem URL: {contagem['outro']}"
        )
    print(
        "   fotos                  distribuição: "
        + " | ".join(
            f"{rotulo}: {dist_fotos.get(n, 0)}"
            for n, rotulo in ((0, "0"), (1, "1"), (2, "2"), (3, "3+"))
        )
    )
    print(f"\n   🔗 Produtos sem nenhum link (web/manual/foto): {sem_nenhum_link} ({pct(sem_nenhum_link)})")
    print(f"   ⚠️  Produtos com ao menos uma lacuna: {len(com_lacunas)} ({pct(len(com_lacunas))})")

    if dominios:
        print("\n   Top domínios em pagina_web:")
        for dominio, n in dominios.most_common(10):
            print(f"      {n:>5}  {dominio}")

    if args.relatorio != "-":
        com_lacunas.sort(key=lambda item: (-item[0], item[1].get("idsubproduto", "")))
        destino = Path(args.relatorio).resolve()
        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open("w", encoding="utf-8-sig", newline="") as arquivo:
            escritor = csv.writer(arquivo)
            escritor.writerow(["idsubproduto", "nome_produto", "n_lacunas", "lacunas"])
            for _, registro, lacunas in com_lacunas:
                escritor.writerow(
                    [
                        registro.get("idsubproduto", ""),
                        registro.get("nome_produto", ""),
                        len(lacunas),
                        ";".join(lacunas),
                    ]
                )
        print(f"\n   📝 Relatório de lacunas: {destino} ({len(com_lacunas)} produtos)")

        # Separação para reprocessamento: linhas com lacuna de um lado,
        # linhas completas do outro. "Não encontrado"/vazio NÃO contam como
        # preenchido — um produto só é "completo" se todos os campos têm
        # conteúdo real.
        ids_com_lacuna = {
            registro.get("idsubproduto", "") for _, registro, _ in com_lacunas
        }
        caminho_lacunas = destino.with_name("produtos_com_lacunas.csv")
        caminho_completos = destino.with_name("produtos_completos.csv")
        with caminho_lacunas.open("w", encoding="utf-8-sig", newline="") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=fieldnames or [])
            escritor.writeheader()
            for _, registro, _ in com_lacunas:
                escritor.writerow(registro)
        with caminho_completos.open("w", encoding="utf-8-sig", newline="") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=fieldnames or [])
            escritor.writeheader()
            for registro in linhas:
                if registro.get("idsubproduto", "") not in ids_com_lacuna:
                    escritor.writerow(registro)
        print(f"   ✂️  Separados: {caminho_lacunas.name} ({len(com_lacunas)} produtos) | "
              f"{caminho_completos.name} ({total - len(com_lacunas)} produtos)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
