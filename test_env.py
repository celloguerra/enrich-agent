import os
from dotenv import load_dotenv

load_dotenv()

key_openrouter = os.getenv("OPENROUTER_API_KEY")
key_tavily = os.getenv("TAVILY_API_KEY")

print("--- DEBUG DO .ENV ---")
print(f"OpenRouter está carregada? {bool(key_openrouter)}")
print(f"Tamanho da chave OpenRouter: {len(key_openrouter) if key_openrouter else 0}")
print(f"Conteúdo (parcial): {key_openrouter[:15]}... se houver")
print("---------------------")
