import ollama
import pandas as pd
import re

# 1. Carrega o arquivo
df = pd.read_csv("ListaProdutosCissAtivos.csv")

def gerar_chave_agrupamento(texto):
    if not isinstance(texto, str):
        return ""

    texto = texto.upper()

    # 1. Cores e acabamentos comuns (geralmente indicam variação)
    cores = r'\b(BRANCA|BRANCO|PRETA|PRETO|AZUL|VERMELHA|VERMELHO|VERDE|AMARELA|AMARELO|CINZA|PRATA|DOURADA|DOURADO|COBRE|INCOLOR|CRISTAL|TRANSPARENTE|ZINCADA|ZINCADO|GALVANIZADA|GALVANIZADO)\b'
    texto = re.sub(cores, '', texto)

    # 2. Medidas comuns
    medidas = r'\b\d+([.,]\d+)?\s*(MM|CM|M|MT|G|KG|GRAMAS|ML|L|POL|POLEGADAS|W|V|AMP|A|KVA|KA|HP|CV|UNIDADES?|PEÇAS?|PÇS?|PC|PCS)\b'
    texto = re.sub(medidas, '', texto)

    # 3. Dimensões NxN (Ex: 10X20, 1'' X 1/2) e formatos dependentes de X
    texto = re.sub(r'\b\d+([.,]\d+)?\s*X\s*\d+([.,]\d+)?\b', '', texto)
    texto = re.sub(r'\bX\s*\d+([.,]\d+)?(MM|CM|M|MT|G|KG|L|ML)?\b', '', texto)

    # 4. Polegadas e frações (Ex: 3/4, 1.1/2'')
    texto = re.sub(r'\b\d+\s*[\'\"”]', '', texto)
    texto = re.sub(r'\b\d+/\d+\s*[\'\"”]?', '', texto)

    # 5. Números que indicam bitolas/número de série (Ex: N.3, #20, = 1)
    texto = re.sub(r'\bN[.,]?\s*\d+', '', texto)
    texto = re.sub(r'#\s*\d+', '', texto)
    texto = re.sub(r'=\s*\d+', '', texto)

    # 6. Números isolados no final (Ex: tamanhos de calçado/roupa)
    texto = re.sub(r'\b\d{1,4}\b$', '', texto.strip())

    # 7. Limpeza do esqueleto: caracteres especiais residuais
    texto = re.sub(r'[\*\#\-\(\)\=\/]+', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto

# Cria a chave estrutural de forma "invisível" para o usuário
df['chave_agrupamento'] = df['descricaoproduto'].apply(gerar_chave_agrupamento)

# Remove as duplicatas baseadas na chave, mas MANTÉM a descrição original da primeira ocorrência
df_agrupado = df.drop_duplicates(subset=['chave_agrupamento'], keep='first').copy()

# Remove a coluna de cálculo, pois não será necessária no seu CSV final
df_agrupado = df_agrupado.drop(columns=['chave_agrupamento'])

# Salva o arquivo final
df_agrupado.to_csv("Produtos_Limpos_Sendo_Mantida_A_Descricao.csv", index=False)

print(f"Pronto! A lista foi reduzida mantendo as descrições intactas. Linhas restantes: {len(df_agrupado)}")
