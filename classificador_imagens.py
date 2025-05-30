# -*- coding: utf-8 -*-
"""
Classificador de Imagens - Inferência com Modelo Treinado
Otimizado para Mac M1

Este script permite classificar novas imagens usando um modelo de regressão logística
previamente treinado para reconhecimento de carros vs. não-carros.

Autor: Flávio Augusto Marques Adão
Data: Maio 2025
"""

import os
import sys
import numpy as np
from PIL import Image
import time
import argparse
import matplotlib.pyplot as plt

def carregar_e_preprocessar_imagem(caminho_imagem, tamanho_alvo=(64, 64)):
    """
    Carrega uma imagem, converte para escala de cinza, redimensiona, achata e normaliza.
    
    Parâmetros:
        caminho_imagem (str): Caminho absoluto para o arquivo de imagem
        tamanho_alvo (tuple): Dimensões para redimensionar a imagem (altura, largura)
    
    Retorna:
        numpy.ndarray: Vetor unidimensional normalizado representando a imagem,
                      ou None se ocorrer um erro
    """
    try:
        # Carrega a imagem e converte para escala de cinza
        imagem = Image.open(caminho_imagem).convert("L")
        
        # Redimensiona para o tamanho padrão
        imagem = imagem.resize(tamanho_alvo)
        
        # Converte para array NumPy
        matriz_imagem = np.array(imagem, dtype=np.float64)
        
        # Achata a matriz 2D para um vetor 1D
        imagem_achatada = matriz_imagem.flatten()
        
        # Normaliza os valores de pixel para o intervalo [0,1]
        imagem_normalizada = imagem_achatada / 255.0
        
        return imagem_normalizada
    
    except Exception as e:
        print(f"Erro ao processar a imagem {caminho_imagem}: {e}")
        return None

def normalizar_imagem(imagem, media=None, desvio_padrao=None):
    """
    Normaliza uma imagem usando padronização (z-score).
    
    Parâmetros:
        imagem (numpy.ndarray): Vetor de características da imagem
        media (numpy.ndarray): Média pré-calculada (opcional)
        desvio_padrao (numpy.ndarray): Desvio padrão pré-calculado (opcional)
    
    Retorna:
        numpy.ndarray: Imagem normalizada
    """
    # Se média e desvio padrão não forem fornecidos, usa valores padrão
    # Nota: idealmente, estes valores deveriam ser calculados no conjunto de treinamento
    if media is None:
        media = 0.5  # Valor aproximado para imagens normalizadas entre 0-1
    
    if desvio_padrao is None:
        desvio_padrao = 0.25  # Valor aproximado para imagens normalizadas entre 0-1
    
    # Aplica a normalização
    imagem_normalizada = (imagem - media) / desvio_padrao
    
    return imagem_normalizada

def sigmoide(z):
    """
    Calcula a função sigmoide: σ(z) = 1 / (1 + e^(-z))
    
    Parâmetros:
        z (numpy.ndarray): Entrada da função sigmoide
    
    Retorna:
        numpy.ndarray: Saída da função sigmoide, valores entre 0 e 1
    """
    # Limita z para evitar overflow/underflow
    z_limitado = np.clip(z, -500, 500)
    s = 1 / (1 + np.exp(-z_limitado))
    return s

def carregar_modelo(caminho_modelo):
    """
    Carrega os parâmetros do modelo treinado.
    
    Parâmetros:
        caminho_modelo (str): Caminho para o arquivo do modelo (.npz)
    
    Retorna:
        tuple: (pesos, bias) - Parâmetros do modelo
    """
    try:
        modelo = np.load(caminho_modelo)
        pesos = modelo['pesos']
        bias = modelo['bias'].item() if isinstance(modelo['bias'], np.ndarray) else modelo['bias']
        
        print(f"Modelo carregado com sucesso de {caminho_modelo}")
        print(f"Formato dos pesos: {pesos.shape}")
        
        return pesos, bias
    
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None, None

def classificar_imagem(imagem, pesos, bias):
    """
    Classifica uma imagem usando o modelo de regressão logística.
    
    Parâmetros:
        imagem (numpy.ndarray): Vetor de características da imagem (1D)
        pesos (numpy.ndarray): Pesos do modelo
        bias (float): Bias do modelo
    
    Retorna:
        tuple: (classe, probabilidade)
            - classe: 1 para carro, 0 para não-carro
            - probabilidade: Probabilidade de ser um carro (0-1)
    """
    # Garante que a imagem seja um vetor coluna para multiplicação matricial
    if imagem.ndim == 1:
        imagem = imagem.reshape(-1, 1)
    
    # Calcula a probabilidade: σ(w^T * x + b)
    z = np.dot(pesos.T, imagem) + bias
    probabilidade = sigmoide(z)[0, 0]
    
    # Classifica: 1 se probabilidade > 0.5, 0 caso contrário
    classe = 1 if probabilidade > 0.5 else 0
    
    return classe, probabilidade

def visualizar_resultado(caminho_imagem, classe, probabilidade):
    """
    Visualiza a imagem classificada com o resultado.
    
    Parâmetros:
        caminho_imagem (str): Caminho para a imagem original
        classe (int): Classe predita (1 para carro, 0 para não-carro)
        probabilidade (float): Probabilidade de ser um carro
    """
    # Carrega a imagem original (colorida)
    imagem = Image.open(caminho_imagem)
    
    # Configura o plot
    plt.figure(figsize=(8, 6))
    plt.imshow(imagem)
    
    # Remove os eixos
    plt.axis('off')
    
    # Adiciona título com o resultado
    classe_texto = "CARRO" if classe == 1 else "NÃO-CARRO"
    titulo = f"Classificação: {classe_texto} (Confiança: {probabilidade:.2%})"
    plt.title(titulo, fontsize=14, color='white', backgroundcolor='black', pad=10)
    
    # Adiciona borda colorida baseada na classificação
    cor_borda = 'green' if classe == 1 else 'red'
    plt.gca().spines['top'].set_color(cor_borda)
    plt.gca().spines['bottom'].set_color(cor_borda)
    plt.gca().spines['left'].set_color(cor_borda)
    plt.gca().spines['right'].set_color(cor_borda)
    plt.gca().spines['top'].set_linewidth(5)
    plt.gca().spines['bottom'].set_linewidth(5)
    plt.gca().spines['left'].set_linewidth(5)
    plt.gca().spines['right'].set_linewidth(5)
    
    # Salva a imagem com o resultado
    nome_arquivo = os.path.basename(caminho_imagem)
    nome_base, extensao = os.path.splitext(nome_arquivo)
    caminho_saida = f"{nome_base}_resultado{extensao}"
    plt.savefig(caminho_saida, bbox_inches='tight', dpi=300)
    print(f"Resultado salvo em: {caminho_saida}")
    
    # Mostra a imagem (opcional, descomente se estiver em ambiente interativo)
    # plt.show()
    
    return caminho_saida

def processar_imagem(caminho_imagem, caminho_modelo, visualizar=True):
    """
    Processa uma imagem e a classifica usando o modelo carregado.
    
    Parâmetros:
        caminho_imagem (str): Caminho para a imagem a ser classificada
        caminho_modelo (str): Caminho para o arquivo do modelo
        visualizar (bool): Se True, visualiza e salva o resultado
    
    Retorna:
        dict: Resultado da classificação
    """
    inicio = time.time()
    
    # Verifica se o arquivo de imagem existe
    if not os.path.isfile(caminho_imagem):
        print(f"Erro: Arquivo de imagem não encontrado: {caminho_imagem}")
        return {"erro": "Arquivo não encontrado"}
    
    # Carrega o modelo
    pesos, bias = carregar_modelo(caminho_modelo)
    if pesos is None or bias is None:
        return {"erro": "Falha ao carregar o modelo"}
    
    # Carrega e pré-processa a imagem
    print(f"Processando imagem: {caminho_imagem}")
    imagem = carregar_e_preprocessar_imagem(caminho_imagem)
    if imagem is None:
        return {"erro": "Falha ao processar a imagem"}
    
    # Normaliza a imagem (idealmente usando estatísticas do conjunto de treinamento)
    imagem_normalizada = normalizar_imagem(imagem)
    
    # Classifica a imagem
    classe, probabilidade = classificar_imagem(imagem_normalizada, pesos, bias)
    
    # Prepara o resultado
    resultado = {
        "caminho_imagem": caminho_imagem,
        "classe": classe,
        "classe_texto": "Carro" if classe == 1 else "Não-Carro",
        "probabilidade": probabilidade,
        "confianca": f"{probabilidade:.2%}"
    }
    
    # Exibe o resultado
    print(f"\nResultado da classificação:")
    print(f"  Imagem: {os.path.basename(caminho_imagem)}")
    print(f"  Classe: {resultado['classe_texto']}")
    print(f"  Confiança: {resultado['confianca']}")
    
    # Visualiza o resultado se solicitado
    if visualizar:
        caminho_resultado = visualizar_resultado(caminho_imagem, classe, probabilidade)
        resultado["imagem_resultado"] = caminho_resultado
    
    fim = time.time()
    tempo_total = fim - inicio
    print(f"Processamento concluído em {tempo_total:.2f} segundos.")
    
    return resultado

def processar_diretorio(caminho_diretorio, caminho_modelo, limite=None):
    """
    Processa todas as imagens em um diretório.
    
    Parâmetros:
        caminho_diretorio (str): Caminho para o diretório com imagens
        caminho_modelo (str): Caminho para o arquivo do modelo
        limite (int): Número máximo de imagens a processar (opcional)
    
    Retorna:
        list: Lista de resultados para cada imagem
    """
    # Verifica se o diretório existe
    if not os.path.isdir(caminho_diretorio):
        print(f"Erro: Diretório não encontrado: {caminho_diretorio}")
        return []
    
    # Lista todos os arquivos de imagem no diretório
    extensoes_imagem = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    arquivos_imagem = [f for f in os.listdir(caminho_diretorio) 
                      if f.lower().endswith(extensoes_imagem) and not f.startswith('.')]
    
    # Limita o número de imagens se especificado
    if limite is not None and limite > 0:
        arquivos_imagem = arquivos_imagem[:limite]
    
    print(f"Encontradas {len(arquivos_imagem)} imagens para processar.")
    
    # Processa cada imagem
    resultados = []
    for i, arquivo in enumerate(arquivos_imagem):
        caminho_completo = os.path.join(caminho_diretorio, arquivo)
        print(f"\nProcessando imagem {i+1}/{len(arquivos_imagem)}: {arquivo}")
        
        resultado = processar_imagem(caminho_completo, caminho_modelo)
        resultados.append(resultado)
    
    # Exibe estatísticas
    carros = sum(1 for r in resultados if "classe" in r and r["classe"] == 1)
    nao_carros = sum(1 for r in resultados if "classe" in r and r["classe"] == 0)
    
    print(f"\nResumo do processamento:")
    print(f"  Total de imagens: {len(resultados)}")
    print(f"  Classificadas como carro: {carros}")
    print(f"  Classificadas como não-carro: {nao_carros}")
    
    return resultados

def main():
    """
    Função principal para execução via linha de comando.
    """
    # Configura o parser de argumentos
    parser = argparse.ArgumentParser(description='Classificador de Imagens - Carros vs. Não-Carros')
    
    # Argumentos obrigatórios
    parser.add_argument('--modelo', type=str, required=True,
                        help='Caminho para o arquivo do modelo (.npz)')
    
    # Grupo mutuamente exclusivo: imagem ou diretório
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument('--imagem', type=str,
                       help='Caminho para a imagem a ser classificada')
    grupo.add_argument('--diretorio', type=str,
                       help='Caminho para o diretório com imagens a serem classificadas')
    
    # Argumentos opcionais
    parser.add_argument('--limite', type=int, default=None,
                        help='Limite de imagens a processar no diretório')
    parser.add_argument('--sem-visualizacao', action='store_true',
                        help='Desativa a visualização dos resultados')
    
    # Analisa os argumentos
    args = parser.parse_args()
    
    # Verifica se o arquivo do modelo existe
    if not os.path.isfile(args.modelo):
        print(f"Erro: Arquivo de modelo não encontrado: {args.modelo}")
        return 1
    
    # Processa uma única imagem ou um diretório
    if args.imagem:
        processar_imagem(args.imagem, args.modelo, not args.sem_visualizacao)
    else:
        processar_diretorio(args.diretorio, args.modelo, args.limite)
    
    return 0

if __name__ == "__main__":
    # Configura o backend do matplotlib para Agg (não interativo)
    import matplotlib
    matplotlib.use('Agg')
    
    # Executa a função principal
    sys.exit(main())
