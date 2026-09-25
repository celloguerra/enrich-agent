"""Agente de enriquecimento de produtos.

Lê as partes CSV (``parte_*.csv``) versionadas no repositório, pesquisa cada
produto na web (DuckDuckGo + extração das próprias páginas), extrai os dados
com IA (OpenRouter) e grava uma planilha enriquecida por parte em ``./saida/``.

Pensado para rodar sem intervenção humana no GitHub Actions:

* todos os caminhos são relativos à pasta do próprio script, então funcionam
  tanto na máquina local quanto num checkout do Actions;
* o estado é gravado em JSONL a cada produto concluído — a execução seguinte
  continua de onde parou. Isso é essencial: 4 mil produtos não cabem no limite
  de 6 horas de um job, então o trabalho avança a cada execução agendada;
* ``--max-minutos`` encerra a execução de forma limpa antes do limite do job,
  preservando o progresso;
* nada é lido do teclado e nenhuma etapa exige confirmação.

Uso::

    python agent.py                          # processa todas as partes
    python agent.py --partes parte_1,parte_2 # só algumas partes
    python agent.py --limite 5               # smoke test com 5 produtos
"""

import argparse
import itertools
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. CONFIGURAÇÃO
# ==========================================
PASTA_BASE = Path(__file__).resolve().parent
ENTRADA_DIR = Path(os.getenv("ENTRADA_DIR", PASTA_BASE))
SAIDA_DIR = Path(os.getenv("SAIDA_DIR", PASTA_BASE / "saida"))

MODELO = os.getenv("MODELO", "qwen/qwen-2.5-72b-instruct")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Cada produto consome créditos de API. Estas chaves permitem baratear a
# execução desligando as buscas complementares (manual e fotos).
BUSCAR_MANUAL = os.getenv("BUSCAR_MANUAL", "1").strip().lower() not in {"0", "false", "nao", "no"}
BUSCAR_FOTOS = os.getenv("BUSCAR_FOTOS", "1").strip().lower() not in {"0", "false", "nao", "no"}

# Quantas vezes um produto com erro volta para a fila em execuções seguintes.
MAX_TENTATIVAS = int(os.getenv("MAX_TENTATIVAS", "3"))

# Busca no DuckDuckGo: região dos resultados e intervalo mínimo global entre
# chamadas (segundos) — o DDG bloqueia clientes que buscam rápido demais.
REGIAO_BUSCA = os.getenv("REGIAO_BUSCA", "br-pt")
INTERVALO_BUSCA = float(os.getenv("INTERVALO_BUSCA", "1.0"))

# Colunas de negócio gravadas nas planilhas. Campos de controle (status, erro,
# tentativas, atualizado_em) vivem apenas no JSONL de estado.
COLUNAS_SAIDA = [
    "idsubproduto",
    "nome_produto",
    "descrsecao",
    "descrgrupo",
    "descrsubgrupo",
    "descricao_produto",
    "especificacao_tecnica",
    "pagina_web",
    "manual",
    "fotos",
]

PROMPT_SISTEMA = """
Você é um assistente especialista em extração de dados de e-commerce.
Sua tarefa é analisar o contexto de busca fornecido e extrair informações sobre um produto.
VOCÊ DEVE RESPONDER APENAS COM UM JSON VÁLIDO. Não escreva explicações, não use markdown.

O JSON deve seguir estritamente esta estrutura:
{
  "descricao_produto": "string (resumo claro do produto em 2 ou 3 frases)",
  "especificacao_tecnica": "string (liste as principais especificações técnicas encontradas)",
  "pagina_web": "string (link URL da página oficial ou principal do produto. Se não houver, use 'Não encontrado')",
  "manual": "string (link URL direto para o manual em PDF ou página de suporte. Se não houver, use 'Não encontrado')",
  "fotos": ["string (link URL da imagem 1)", "string (link URL da imagem 2)"]
}

REGRAS IMPORTANTES:
- Para 'pagina_web' e 'manual', extraia APENAS links que estejam explicitamente no contexto. Não invente URLs.
- Se a informação não estiver no contexto, use 'Não encontrado'.
"""

# Preenchidos por inicializar_clientes()
cliente_ia: Any = None
cliente_busca: Any = None


def _aplicar_patch_ipv4() -> None:
    """Força resolução IPv4 quando o ambiente tem IPv6 problemático.

    Desligado por padrão: nos runners do GitHub o IPv6 funciona normalmente.
    Ligue localmente com ``FORCAR_IPV4=1`` se precisar.
    """
    if os.getenv("FORCAR_IPV4", "").strip().lower() not in {"1", "true", "sim", "yes"}:
        return

    import socket

    getaddrinfo_original = socket.getaddrinfo

    def somente_ipv4(*args, **kwargs):
        resultados = getaddrinfo_original(*args, **kwargs)
        return [
            (socket.AF_INET, *info[1:])
            for info in resultados
            if info[0] == socket.AF_INET
        ]

    socket.getaddrinfo = somente_ipv4
    print("🌐 FORCAR_IPV4 ativo: resolução DNS restrita a IPv4.", flush=True)


def inicializar_clientes() -> None:
    """Cria os clientes de IA e de busca a partir das variáveis de ambiente."""
    global cliente_ia, cliente_busca
    from ddgs import DDGS
    from openai import OpenAI

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    faltando = [
        nome
        for nome, valor in (("OPENROUTER_API_KEY", openrouter_key),)
        if not valor
    ]
    if faltando:
        raise SystemExit(
            "⚠️ Erro: variáveis de ambiente não encontradas: "
            + ", ".join(faltando)
            + ". Defina-as no ambiente (ou num arquivo .env local)."
        )

    # O OpenRouter responde 401 "Missing Authentication header" quando o bearer
    # não é uma chave dele (ex.: placeholder exportado sem querer). O aviso
    # abaixo torna esse caso óbvio antes de qualquer chamada real.
    if "openrouter.ai" in OPENROUTER_BASE_URL and not openrouter_key.startswith("sk-or-"):
        print(
            "   ⚠️ OPENROUTER_API_KEY não parece uma chave do OpenRouter "
            f"(começa com '{openrouter_key[:8]}…'; esperado prefixo 'sk-or-'). "
            "Gere uma chave real em https://openrouter.ai/keys.",
            flush=True,
        )

    cliente_ia = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=openrouter_key,
        max_retries=2,
        timeout=120.0,
    )
    cliente_busca = DDGS()

    _preflight_openrouter()
    _preflight_busca()


def _preflight_openrouter() -> None:
    """Chamada mínima ao OpenRouter para falhar rápido com mensagem clara.

    Sem isso uma chave inválida só seria descoberta produto a produto, depois
    de minutos de retries (401 "Missing Authentication header").
    """
    try:
        _com_retry(
            lambda: cliente_ia.chat.completions.create(
                model=MODELO,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            ),
            "verificação do OpenRouter",
            tentativas=2,
            espera_inicial=1.0,
        )
    except Exception as erro:  # noqa: BLE001
        raise SystemExit(
            "⚠️ O OpenRouter recusou a chamada de teste — nenhum produto seria "
            "enriquecido nesta execução.\n"
            f"   erro: {erro}\n"
            "   Confira OPENROUTER_API_KEY (gere em https://openrouter.ai/keys) "
            f"e o identificador do modelo MODELO='{MODELO}'."
        ) from erro
    print("   🔑 OpenRouter: credencial e modelo OK.", flush=True)


def _preflight_busca() -> None:
    """Busca mínima no DuckDuckGo para falhar rápido (não precisa de chave).

    O DDG bloqueia temporariamente clientes que buscam rápido demais; é melhor
    descobrir isso no início do que produto a produto.
    """
    try:
        _com_retry(
            lambda: cliente_busca.text("parafuso", region=REGIAO_BUSCA, max_results=1),
            "verificação do DuckDuckGo",
            tentativas=2,
            espera_inicial=1.0,
        )
    except Exception as erro:  # noqa: BLE001
        raise SystemExit(
            "⚠️ O DuckDuckGo não respondeu à busca de teste — as buscas falhariam "
            "para todos os produtos.\n"
            f"   erro: {erro}\n"
            "   O DDG limita clientes muito rápidos: tente de novo em alguns minutos "
            "ou reduza WORKERS/INTERVALO_BUSCA."
        ) from erro
    print("   🔑 DuckDuckGo: busca OK (não precisa de chave).", flush=True)



# ==========================================
# 2. UTILITÁRIOS
# ==========================================

def _texto(valor) -> str:
    """Converte qualquer valor de DataFrame/JSON em texto limpo."""
    if valor is None:
        return ""
    if isinstance(valor, float) and pd.isna(valor):
        return ""
    if isinstance(valor, (list, tuple)):
        return " | ".join(_texto(item) for item in valor if _texto(item))
    texto = str(valor).strip()
    return "" if texto.lower() in {"nan", "none", "nat"} else texto


def campo_vazio(valor) -> bool:
    """Campo sem conteúdo real: vazio ou marcado como “Não encontrado”.

    Mesma regra da auditoria (``auditar_enriquecimento.py``): texto que anuncia
    ausência de dado não conta como preenchido.
    """
    texto = _texto(valor)
    return (not texto) or bool(re.search(r"n[ãa]o\s+encontrad", texto, re.IGNORECASE))


def _links(valor) -> list:
    """Normaliza o campo de fotos da IA para uma lista de URLs."""
    if valor is None:
        return []
    if isinstance(valor, str):
        candidatos = re.split(r"[|;\n]", valor)
    elif isinstance(valor, (list, tuple)):
        candidatos = list(valor)
    else:
        return []

    links = []
    for candidato in candidatos:
        url = _texto(candidato)
        if url.startswith("http") and url not in links:
            links.append(url)
    return links


def _com_retry(operacao, descricao, tentativas=3, espera_inicial=2.0):
    """Executa ``operacao`` com backoff exponencial. Relança o último erro."""
    atraso = espera_inicial
    ultimo_erro = None

    for tentativa in range(1, tentativas + 1):
        try:
            return operacao()
        except Exception as erro:  # noqa: BLE001 - rede/IA falham de muitos jeitos
            ultimo_erro = erro
            if tentativa >= tentativas:
                break
            print(
                f"   ⚠️ {descricao} falhou ({tentativa}/{tentativas}): {erro}. "
                f"Repetindo em {atraso:.0f}s...",
                flush=True,
            )
            time.sleep(atraso)
            atraso *= 2

    if ultimo_erro is None:
        raise RuntimeError(
            f"{descricao} não chegou a ser executada (tentativas={tentativas})"
        )
    raise ultimo_erro


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rel(caminho: Path) -> str:
    """Caminho relativo à pasta do projeto quando possível, absoluto caso não."""
    try:
        return str(Path(caminho).relative_to(PASTA_BASE))
    except ValueError:
        return str(caminho)


# ==========================================
# 3. BUSCA E ENRIQUECIMENTO
# ==========================================

def montar_consulta(linha: dict) -> str:
    """Monta a consulta de busca a partir da descrição e da taxonomia.

    As descrições do sistema trazem marcadores internos (``*``, ``#``) e são
    curtas demais para uma boa busca, então removemos o ruído e acrescentamos
    grupo/subgrupo como contexto.
    """
    nome = re.sub(r"[*#]+", " ", _texto(linha.get("descricaoproduto")))
    nome = re.sub(r"\s+", " ", nome).strip()

    partes = [nome] if nome else []
    for campo in ("descrsubgrupo", "descrgrupo"):
        valor = _texto(linha.get(campo))
        if len(valor) < 3 or valor.upper() in {"GERAL", "INATIVO"}:
            continue
        if valor not in partes:
            partes.append(valor)

    return " ".join(partes)


# Cadência global das buscas no DDG entre todas as threads.
_trava_busca = threading.Lock()
_proxima_busca_em = [0.0]


def _pautar_busca() -> None:
    """Garante um intervalo mínimo global entre chamadas ao DuckDuckGo.

    O DDG bloqueia temporariamente clientes que buscam rápido demais; com
    vários workers, o intervalo (``INTERVALO_BUSCA``) mantém a cadência
    coletiva abaixo do limite.
    """
    with _trava_busca:
        agora = time.monotonic()
        espera = _proxima_busca_em[0] - agora
        _proxima_busca_em[0] = max(agora, _proxima_busca_em[0]) + INTERVALO_BUSCA
    if espera > 0:
        time.sleep(espera)


def _extrair_texto_pagina(url: str, limite: int = 2000) -> str:
    """Baixa a página e extrai o texto principal ("" se não der).

    O DDG devolve só título/URL/resumo; o conteúdo que alimenta a IA vem daqui.
    Qualquer falha (rede, PDF, página gigante) devolve "" e o chamador cai
    para o resumo da busca.
    """
    url = _texto(url)
    if not url.startswith("http"):
        return ""
    try:
        import requests
        from bs4 import BeautifulSoup

        resposta = requests.get(
            url,
            timeout=8,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
                )
            },
        )
        resposta.raise_for_status()
        if "html" not in resposta.headers.get("Content-Type", "").lower():
            return ""
        if len(resposta.content) > 2_000_000:
            return ""
        sopa = BeautifulSoup(resposta.content, "html.parser")
        for tag in sopa(["script", "style", "noscript", "header", "footer", "nav", "form", "svg"]):
            tag.decompose()
        return re.sub(r"\s+", " ", sopa.get_text(" ")).strip()[:limite]
    except Exception:  # noqa: BLE001 - rede/parse falham de muitos jeitos
        return ""


def pesquisar_na_web(consulta: str) -> dict:
    """Busca textos e links usando DuckDuckGo (sem chave, sujeito a rate-limit).

    O DDG devolve apenas título/URL/resumo, então as páginas da busca
    principal são baixadas e têm o texto extraído para dar contexto à IA;
    quando uma página falha, usa-se o resumo. Buscas de manual e fotos
    devolvem só links, que são mais baratas.
    """

    def _buscar_texto(consulta_busca: str, maximo: int) -> list:
        _pautar_busca()
        return _com_retry(
            lambda: cliente_busca.text(
                consulta_busca, region=REGIAO_BUSCA, max_results=maximo
            )
            or [],
            "busca no DuckDuckGo",
        )

    contexto_texto = ""

    try:
        for resultado in _buscar_texto(f"{consulta} especificações técnicas manual", 5):
            titulo = _texto(resultado.get("title"))
            url = _texto(resultado.get("href") or resultado.get("url"))
            contexto_texto += f"\n--- Fonte: {titulo} (URL: {url}) ---\n"
            conteudo = _extrair_texto_pagina(url) or _texto(resultado.get("body"))
            contexto_texto += f"{conteudo[:2000]}\n"

        if BUSCAR_MANUAL:
            manuais = _buscar_texto(f"{consulta} manual do usuário PDF", 3)
            if manuais:
                contexto_texto += "\n=== POSSÍVEIS MANUAIS ===\n"
                for resultado in manuais:
                    titulo = _texto(resultado.get("title"))
                    url = _texto(resultado.get("href") or resultado.get("url"))
                    contexto_texto += f"- {titulo} (Link: {url})\n"

    except Exception as erro:  # noqa: BLE001
        print(f"   ⚠️ Erro na busca DuckDuckGo: {erro}", flush=True)
        contexto_texto = contexto_texto or "Erro ao buscar na web."

    fotos = []
    if BUSCAR_FOTOS:
        try:
            _pautar_busca()
            imagens = _com_retry(
                lambda: cliente_busca.images(
                    f"{consulta} produto", region=REGIAO_BUSCA, max_results=3
                )
                or [],
                "busca de fotos",
            )
            for imagem in imagens:
                url = _texto(
                    imagem.get("image") or imagem.get("thumbnail") or imagem.get("url")
                )
                if url and url not in fotos:
                    fotos.append(url)
        except Exception as erro:  # noqa: BLE001
            print(f"   ⚠️ Erro na busca de fotos: {erro}", flush=True)

    return {"texto": contexto_texto, "fotos": fotos}


def limpar_json(texto: str) -> str:
    """Remove cercas de markdown caso a IA insista em colocá-las."""
    texto = re.sub(r"^```(?:json)?\s*", "", texto.strip())
    texto = re.sub(r"\s*```$", "", texto.strip())
    return texto.strip()


def enriquecer_com_ia(nome_produto: str, contexto: dict) -> dict:
    """Chama o OpenRouter para processar os dados e retornar um JSON."""
    prompt_usuario = f"""
    Nome do Produto: {nome_produto}

    Contexto de Busca:
    {contexto['texto']}

    Links de páginas com possíveis fotos do produto:
    {json.dumps(contexto['fotos'], ensure_ascii=False)}
    """

    resposta = _com_retry(
        lambda: cliente_ia.chat.completions.create(
            model=MODELO,
            messages=[
                {"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user", "content": prompt_usuario},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        ),
        "chamada de IA",
    )

    dados = json.loads(limpar_json(_texto(resposta.choices[0].message.content)))
    if not isinstance(dados, dict):
        raise TypeError("a IA não retornou um objeto JSON")
    return dados


def processar_produto(linha: dict, tentativas: int, anterior: dict | None = None) -> dict:
    """Pesquisa + IA para um produto, já no formato final de gravação.

    Com ``anterior`` (o registro que o produto já tinha no estado), campos
    preenchidos de rodadas anteriores são preservados: a nova rodada só
    preenche o que está vazio ou marcado como “Não encontrado”. Fotos são
    somadas, sem duplicar.
    """
    anterior = anterior or {}
    registro = {
        "idsubproduto": _texto(linha.get("idsubproduto")),
        "nome_produto": _texto(linha.get("descricaoproduto")),
        "descrsecao": _texto(linha.get("descrsecao")),
        "descrgrupo": _texto(linha.get("descrgrupo")),
        "descrsubgrupo": _texto(linha.get("descrsubgrupo")),
        "descricao_produto": "",
        "especificacao_tecnica": "",
        "pagina_web": "Não encontrado",
        "manual": "Não encontrado",
        "fotos": "",
        "status": "ok",
        "erro": "",
        "tentativas": tentativas,
        "atualizado_em": _agora(),
    }
    for campo in (
        "descricao_produto",
        "especificacao_tecnica",
        "pagina_web",
        "manual",
        "fotos",
    ):
        preservado = _texto(anterior.get(campo))
        if not campo_vazio(preservado):
            registro[campo] = preservado

    consulta = montar_consulta(linha)
    if not consulta:
        registro.update(status="erro", erro="descrição vazia no CSV de origem")
        return registro

    try:
        contexto = pesquisar_na_web(consulta)
        dados = enriquecer_com_ia(registro["nome_produto"] or consulta, contexto)

        # Só preenche o que era lacuna: o que já veio de rodada anterior e
        # estava bom permanece intocado, mesmo que a IA repondra.
        for campo in (
            "descricao_produto",
            "especificacao_tecnica",
            "pagina_web",
            "manual",
        ):
            if campo_vazio(registro[campo]):
                novo = _texto(dados.get(campo))
                if not campo_vazio(novo):
                    registro[campo] = novo
        atuais = _links(registro["fotos"])
        registro["fotos"] = " | ".join(
            atuais + [foto for foto in _links(dados.get("fotos")) if foto not in atuais]
        )
    except Exception as erro:  # noqa: BLE001
        registro.update(status="erro", erro=f"{type(erro).__name__}: {erro}")

    return registro


# ==========================================
# 4. ESTADO (RETOMADA) E SAÍDA
# ==========================================

def descobrir_partes(entrada_dir: Path, padrao: str, filtro: str = "") -> list:
    """Lista as partes CSV em ordem natural (parte_1, parte_2, ..., parte_10)."""
    partes = sorted(
        entrada_dir.glob(padrao),
        key=lambda caminho: [
            int(trecho) if trecho.isdigit() else trecho
            for trecho in re.split(r"(\d+)", caminho.stem)
        ],
    )

    if filtro:
        desejadas = {nome.strip() for nome in filtro.split(",") if nome.strip()}
        partes = [
            parte
            for parte in partes
            if parte.stem in desejadas or parte.name in desejadas
        ]

    return partes


def carregar_estado(arquivo_estado: Path) -> dict:
    """Lê o JSONL de estado. Linhas corrompidas são ignoradas, não fatais."""
    estado = {}
    if not arquivo_estado.exists():
        return estado

    with arquivo_estado.open("r", encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, start=1):
            linha = linha.strip()
            if not linha:
                continue
            try:
                registro = json.loads(linha)
            except json.JSONDecodeError:
                print(
                    f"   ⚠️ {arquivo_estado.name}:{numero} inválida, ignorando.",
                    flush=True,
                )
                continue
            chave = _texto(registro.get("idsubproduto"))
            if chave:
                estado[chave] = registro

    return estado


def esta_pendente(registro_anterior) -> bool:
    """Um produto volta para a fila se nunca rodou ou se falhou poucas vezes."""
    if registro_anterior is None:
        return True
    if registro_anterior.get("status") == "ok":
        return False
    return int(registro_anterior.get("tentativas", 0)) < MAX_TENTATIVAS


def salvar_planilhas(registros: list, destino: Path) -> tuple:
    """Grava a parte enriquecida em XLSX e CSV (o CSV é amigável para diff)."""
    destino.parent.mkdir(parents=True, exist_ok=True)

    colunas = list(COLUNAS_SAIDA)
    linhas = [
        {coluna: registro.get(coluna, "") for coluna in colunas}
        for registro in registros
    ]
    df = pd.DataFrame(linhas, columns=colunas)

    caminho_csv = destino.with_suffix(".csv")
    df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")

    caminho_xlsx = destino.with_suffix(".xlsx")
    try:
        df.to_excel(caminho_xlsx, index=False)
    except Exception as erro:  # noqa: BLE001 - openpyxl ausente não pode derrubar o job
        print(f"   ⚠️ Não foi possível gravar {caminho_xlsx.name}: {erro}", flush=True)
        caminho_xlsx = None

    return caminho_xlsx, caminho_csv


# ==========================================
# 5. ORQUESTRAÇÃO
# ==========================================

def processar_parte(parte: Path, args, prazo: float | None) -> dict:
    """Processa uma parte CSV, retomando do estado já gravado."""
    arquivo_estado = SAIDA_DIR / f"estado_{parte.stem}.jsonl"
    estado = carregar_estado(arquivo_estado)

    df = pd.read_csv(parte, dtype=str).fillna("")
    linhas = df.to_dict(orient="records")

    pendentes = []
    for linha in linhas:
        chave = _texto(linha.get("idsubproduto"))
        if not chave:
            continue
        anterior = estado.get(chave)
        if esta_pendente(anterior):
            pendentes.append(
                (linha, int((anterior or {}).get("tentativas", 0)) + 1, anterior)
            )

    if args.limite:
        pendentes = pendentes[: args.limite]

    ids_no_csv = {
        chave
        for chave in (_texto(linha.get("idsubproduto")) for linha in linhas)
        if chave
    }
    # Produtos concluídos em execuções anteriores que não estão mais no CSV
    # (ex.: partes filtradas para conter só pendentes). Seguem na contagem e
    # na planilha — o estado é a memória completa da parte.
    registros_fora_do_csv = [
        registro for chave, registro in estado.items() if chave not in ids_no_csv
    ]

    total = len(linhas) + len(registros_fora_do_csv)
    print(
        f"\n📦 {parte.name}: {total} produtos | "
        f"{total - len(pendentes)} já prontos | {len(pendentes)} a processar",
        flush=True,
    )

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)
    trava = threading.Lock()
    contador = {"novos": 0, "erros": 0}
    esgotou_prazo = False

    if pendentes:
        arquivo_estado.parent.mkdir(parents=True, exist_ok=True)
        with arquivo_estado.open("a", encoding="utf-8") as arquivo:
            def gravar(registro):
                with trava:
                    arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
                    arquivo.flush()
                    os.fsync(arquivo.fileno())
                    estado[registro["idsubproduto"]] = registro
                    contador["novos"] += 1
                    if registro["status"] != "ok":
                        contador["erros"] += 1
                    if contador["novos"] % 10 == 0:
                        print(
                            f"   💾 {parte.stem}: {contador['novos']}/{len(pendentes)} "
                            "gravados no estado",
                            flush=True,
                        )

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                fila = iter(pendentes)
                futuros = {}

                for linha, tentativas, anterior in itertools.islice(fila, args.workers * 2):
                    futuros[pool.submit(processar_produto, linha, tentativas, anterior)] = linha

                while futuros:
                    concluidos, _ = wait(futuros, return_when=FIRST_COMPLETED)
                    for futuro in concluidos:
                        linha = futuros.pop(futuro)
                        nome = _texto(linha.get("descricaoproduto"))[:50]
                        try:
                            registro = futuro.result()
                        except Exception as erro:  # noqa: BLE001 - nunca perder o lote
                            registro = {
                                "idsubproduto": _texto(linha.get("idsubproduto")),
                                "nome_produto": _texto(linha.get("descricaoproduto")),
                                "status": "erro",
                                "erro": f"{type(erro).__name__}: {erro}",
                                "tentativas": 1,
                                "atualizado_em": _agora(),
                            }

                        marcador = "✅" if registro["status"] == "ok" else "❌"
                        print(
                            f"   {marcador} [{contador['novos'] + 1}/{len(pendentes)}] "
                            f"{registro['idsubproduto']} — {nome}",
                            flush=True,
                        )
                        if registro["status"] != "ok":
                            print(f"      motivo: {registro.get('erro', '')}", flush=True)
                        gravar(registro)

                    dentro_do_prazo = prazo is None or time.monotonic() < prazo
                    if dentro_do_prazo:
                        proximo = next(fila, None)
                        if proximo is not None:
                            linha, tentativas, anterior = proximo
                            futuros[
                                pool.submit(processar_produto, linha, tentativas, anterior)
                            ] = linha
                    elif not esgotou_prazo:
                        esgotou_prazo = True
                        print(
                            "   ⏰ Orçamento de tempo esgotado. Encerrando esta parte "
                            "e preservando o progresso.",
                            flush=True,
                        )

    # Reconstitui a parte inteira na ordem original do CSV, mais os produtos
    # que saíram do CSV mas já estavam prontos no estado. Feito fora do
    # "if pendentes" para a planilha e o resumo saírem mesmo numa execução
    # de retomada sem produtos novos.
    registros_finais = [
        estado[chave]
        for chave in (_texto(linha.get("idsubproduto")) for linha in linhas)
        if chave and chave in estado
    ]
    registros_finais += registros_fora_do_csv

    destino = SAIDA_DIR / f"produtos_enriquecidos_{parte.stem}"
    caminho_xlsx, caminho_csv = salvar_planilhas(registros_finais, destino)
    print(
        f"   💾 {caminho_csv.name}"
        + (f" + {caminho_xlsx.name}" if caminho_xlsx else "")
        + f" ({len(registros_finais)}/{total} produtos)",
        flush=True,
    )

    return {
        "parte": parte.name,
        "total": total,
        "enriquecidos": len(registros_finais),
        "ok": sum(1 for r in registros_finais if r.get("status") == "ok"),
        "com_erro": sum(1 for r in registros_finais if r.get("status") != "ok"),
        "novos_nesta_execucao": contador["novos"],
        "erros_nesta_execucao": contador["erros"],
        "arquivos": [_rel(caminho) for caminho in (caminho_xlsx, caminho_csv) if caminho],
    }


def interpretar_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Enriquece os produtos das partes CSV com dados de web + IA."
    )
    parser.add_argument(
        "--entrada-dir",
        default=str(ENTRADA_DIR),
        help="Pasta com os CSV de entrada (padrão: a pasta deste script).",
    )
    parser.add_argument(
        "--saida-dir",
        default=str(SAIDA_DIR),
        help="Pasta de saída (padrão: ./saida ao lado deste script).",
    )
    parser.add_argument(
        "--padrao",
        default=os.getenv("PADRAO_PARTES", "parte_*.csv"),
        help="Glob das partes de entrada (padrão: parte_*.csv).",
    )
    parser.add_argument(
        "--partes",
        default=os.getenv("PARTES", ""),
        help="Restringe a execução, ex.: parte_1,parte_3 (padrão: todas).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("WORKERS", "6")),
        help="Produtos processados em paralelo (padrão: 6).",
    )
    parser.add_argument(
        "--max-minutos",
        type=float,
        default=float(os.getenv("MAX_MINUTOS", "330")),
        help="Orçamento de tempo; 0 = sem limite (padrão: 330, cabe num job de 6h).",
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=int(os.getenv("LIMITE", "0") or 0),
        help="Processa no máximo N produtos por parte (0 = todos). Útil p/ teste.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = interpretar_args(argv)

    global SAIDA_DIR
    SAIDA_DIR = Path(args.saida_dir).resolve()
    entrada_dir = Path(args.entrada_dir).resolve()

    print("🚀 Agente de Enriquecimento de Produtos", flush=True)
    print(f"   entrada : {entrada_dir}", flush=True)
    print(f"   saída   : {SAIDA_DIR}", flush=True)
    print(f"   modelo  : {MODELO}", flush=True)
    print(f"   workers : {args.workers}", flush=True)
    print(f"   prazo   : {args.max_minutos:g} min", flush=True)

    partes = descobrir_partes(entrada_dir, args.padrao, args.partes)
    if not partes:
        print(
            f"❌ Nenhuma parte encontrada em {entrada_dir} com o padrão '{args.padrao}'.\n"
            "   Rode 'python breaker.py' para gerar as partes a partir de "
            "ListaProdutosCissAtivos.csv.",
            flush=True,
        )
        return 1

    print(f"   partes  : {', '.join(parte.name for parte in partes)}", flush=True)

    _aplicar_patch_ipv4()
    inicializar_clientes()
    SAIDA_DIR.mkdir(parents=True, exist_ok=True)

    inicio = time.monotonic()
    prazo = inicio + args.max_minutos * 60 if args.max_minutos > 0 else None

    resumos = []
    for parte in partes:
        if prazo is not None and time.monotonic() >= prazo:
            print(
                f"⏰ Orçamento esgotado antes de {parte.name}; ela fica para a "
                "próxima execução.",
                flush=True,
            )
            break
        try:
            resumos.append(processar_parte(parte, args, prazo))
        except Exception as erro:  # noqa: BLE001 - uma parte ruim não derruba as demais
            print(f"❌ Falha ao processar {parte.name}: {erro}", flush=True)
            resumos.append({"parte": parte.name, "erro_fatal": str(erro)})

    total = sum(resumo.get("total", 0) for resumo in resumos)
    enriquecidos = sum(resumo.get("enriquecidos", 0) for resumo in resumos)
    completo = total > 0 and enriquecidos >= total

    progresso = {
        "atualizado_em": _agora(),
        "duracao_segundos": round(time.monotonic() - inicio, 1),
        "completo": completo,
        "total_produtos": total,
        "enriquecidos": enriquecidos,
        "restantes": max(total - enriquecidos, 0),
        "partes": resumos,
    }
    arquivo_progresso = SAIDA_DIR / "progresso.json"
    arquivo_progresso.write_text(
        json.dumps(progresso, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 60, flush=True)
    print(f"📊 {enriquecidos}/{total} produtos enriquecidos", flush=True)
    for resumo in resumos:
        if "erro_fatal" in resumo:
            print(f"   ❌ {resumo['parte']}: {resumo['erro_fatal']}", flush=True)
        else:
            print(
                f"   {resumo['parte']}: {resumo['enriquecidos']}/{resumo['total']} "
                f"(+{resumo['novos_nesta_execucao']} agora, "
                f"{resumo['erros_nesta_execucao']} com erro)",
                flush=True,
            )
    print(f"   progresso: {_rel(arquivo_progresso)}", flush=True)

    if completo:
        print("🎉 Todas as partes concluídas!", flush=True)
    else:
        print(
            "⏳ Incompleto — a próxima execução retoma automaticamente do estado "
            f"em {_rel(SAIDA_DIR)}/.",
            flush=True,
        )
    print("=" * 60, flush=True)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(
            "\n⏹️ Interrompido pelo usuário — o progresso gravado em ./saida/ "
            "foi preservado e a próxima execução retoma de onde parou.",
            flush=True,
        )
        sys.exit(130)
