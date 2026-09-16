import socket

# 1. Salva a função original do sistema
_original_getaddrinfo = socket.getaddrinfo

# 2. Cria uma nova função que chama a original e filtra apenas IPv4
def _forcar_ipv4(*args, **kwargs):
    resultados = _original_getaddrinfo(*args, **kwargs)
    # Filtra para retornar apenas endereços IPv4 (AF_INET)
    return [(socket.AF_INET, *info[1:]) for info in resultados if info[0] == socket.AF_INET]

# 3. Substitui a função do socket pela nossa versão filtrada
socket.getaddrinfo = _forcar_ipv4

import json
import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient
import pandas as pd
# 2. Carregar as variáveis do arquivo .env
load_dotenv()

# ==========================================
# 1. CONFIGURAÇÕES E PROMPTS
# ==========================================
MODELO = "qwen/qwen-2.5-72b-instruct"
pasta_do_script = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_ENTRADA = os.path.join(pasta_do_script, "produtos_base.csv")
ARQUIVO_SAIDA = os.path.join(pasta_do_script, "produtos_enriquecidos.xlsx")

# 3. Pegar as chaves do ambiente (o Python vai ler do arquivo .env automaticamente)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENROUTER_API_KEY or not TAVILY_API_KEY:
    raise ValueError("⚠️ Erro: As chaves OPENROUTER_API_KEY e TAVILY_API_KEY não foram encontradas no arquivo .env")

# Inicializa os clientes
cliente_ia = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)
cliente_busca = TavilyClient(api_key=TAVILY_API_KEY)

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

# ==========================================
# 2. FUNÇÕES DO AGENTE
# ==========================================

def pesquisar_na_web(nome_produto: str) -> dict:
    """Busca textos e links usando Tavily."""
    print(f"   🔍 Pesquisando na web: {nome_produto}...")
    contexto_texto = ""

    try:
        # MUDOU: Tavily faz a busca e já retorna o conteúdo limpo das páginas
        resultado = cliente_busca.search(
            query=f"{nome_produto} especificações técnicas manual",
            max_results=5,
            search_depth="advanced",  # Busca mais profunda (lê o conteúdo das páginas)
            include_raw_content=False
        )

        # Formata os resultados para a IA
        for r in resultado.get("results", []):
            contexto_texto += f"\n--- Fonte: {r['title']} (URL: {r['url']}) ---\n"
            contexto_texto += r.get("content", "")[:2000]  # Limita a 2000 chars por fonte
            contexto_texto += "\n"

        # Busca extra focada em manuais PDF
        resultado_manual = cliente_busca.search(
            query=f"{nome_produto} manual do usuário PDF",
            max_results=3,
            search_depth="basic"
        )
        if resultado_manual.get("results"):
            contexto_texto += "\n=== POSSÍVEIS MANUAIS ===\n"
            for r in resultado_manual["results"]:
                contexto_texto += f"- {r['title']} (Link: {r['url']})\n"

    except Exception as e:
        print(f"   ⚠️ Erro na busca Tavily: {e}")
        contexto_texto = "Erro ao buscar na web."

    # MUDOU: Tavily não busca imagens nativamente.
    # Para fotos, usamos uma busca extra com foco em imagens.
    fotos = []
    try:
        resultado_fotos = cliente_busca.search(
            query=f"{nome_produto} produto imagem oficial",
            max_results=3,
            search_depth="basic"
        )
        # Extrai URLs que parecem ser de páginas de produto (que contêm fotos)
        for r in resultado_fotos.get("results", []):
            fotos.append(r["url"])
    except Exception:
        pass

    return {"texto": contexto_texto, "fotos": fotos}

def limpar_json(texto: str) -> str:
    """Remove marcas de markdown caso a IA teime em colocar."""
    texto = re.sub(r'^```json\s*|\s*```$', '', texto, flags=re.MULTILINE)
    return texto.strip()

def enriquecer_com_ia(nome_produto: str, contexto: dict) -> dict:
    """Chama o OpenRouter para processar os dados e retornar um JSON."""
    print(f"   🧠 Processando com IA ({MODELO}): {nome_produto}...")

    prompt_usuario = f"""
    Nome do Produto: {nome_produto}

    Contexto de Busca:
    {contexto['texto']}

    Links de páginas com possíveis fotos do produto:
    {json.dumps(contexto['fotos'])}
    """

    try:
        # MUDOU: Chamada no formato OpenAI
        resposta = cliente_ia.chat.completions.create(
            model=MODELO,
            messages=[
                {"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user", "content": prompt_usuario}
            ],
            response_format={"type": "json_object"}  # Força JSON
        )

        texto_bruto = resposta.choices[0].message.content
        texto_limpo = limpar_json(texto_bruto)
        dados = json.loads(texto_limpo)
        return dados

    except json.JSONDecodeError:
        print(f"   ❌ A IA não retornou um JSON válido.")
        return {"descricao_produto": "Erro", "especificacao_tecnica": "Erro",
                "pagina_web": "Erro", "manual": "Erro", "fotos": []}
    except Exception as e:
        print(f"   ❌ Erro ao chamar a IA: {e}")
        return {"descricao_produto": "Erro", "especificacao_tecnica": "Erro",
                "pagina_web": "Erro", "manual": "Erro", "fotos": []}

# ==========================================
# 3. ORQUESTRAÇÃO PRINCIPAL
# ==========================================

def main():
    print("🚀 Iniciando Agente de Enriquecimento (Cloud Edition)...\n")

    df_original = pd.read_csv(ARQUIVO_ENTRADA)
    dados_finais = []

    for index, row in df_original.iterrows():
        nome = row['nome_produto']
        print(f"📦 [{index + 1}/{len(df_original)}] Processando: {nome}")

        contexto = pesquisar_na_web(nome)
        dados_ia = enriquecer_com_ia(nome, contexto)

        linha_final = {
            "id_original": row['id'],
            "nome_produto": nome,
            "descricao_produto": dados_ia.get("descricao_produto", ""),
            "especificacao_tecnica": dados_ia.get("especificacao_tecnica", ""),
            "pagina_web": dados_ia.get("pagina_web", ""),
            "manual": dados_ia.get("manual", ""),
            "fotos": " | ".join(dados_ia.get("fotos", []))
        }

        dados_finais.append(linha_final)
        print(f"   ✅ Concluído!\n")
        time.sleep(1)  # Na nuvem podemos reduzir o sleep (Tavily não bloqueia)

    print("💾 Salvando planilha final...")
    df_final = pd.DataFrame(dados_finais)
    df_final.to_excel(ARQUIVO_SAIDA, index=False)
    print(f"\n🎉 Sucesso! Arquivo salvo como: {ARQUIVO_SAIDA}")

if __name__ == "__main__":
    main()
