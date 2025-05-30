# -*- coding: utf-8 -*-
"""
Reconhecimento de Imagens com Regressão Logística
Otimizado para Mac M1 com processamento paralelo

Este script implementa um sistema completo de reconhecimento de imagens usando
regressão logística implementada manualmente, sem bibliotecas de alto nível.
Otimizado para Apple Silicon (M1) com paralelização e vetorização.

Autor: Flávio Augusto Marques Adão
Data: Maio 2025
"""

import os
import numpy as np
from PIL import Image
import random
import multiprocessing
from functools import partial
import time
import matplotlib
matplotlib.use('Agg')  # Backend não interativo
import matplotlib.pyplot as plt

# =====================================================================
# PARTE 1: CARREGAMENTO E PRÉ-PROCESSAMENTO DE IMAGENS
# =====================================================================

def carregar_e_preprocessar_imagem(caminho_imagem, tamanho_alvo=(64, 64)):
    """
    Carrega uma imagem, converte para escala de cinza, redimensiona, achata e normaliza.
    
    Parâmetros:
        caminho_imagem (str): Caminho absoluto para o arquivo de imagem
        tamanho_alvo (tuple): Dimensões para redimensionar a imagem (altura, largura)
    
    Retorna:
        numpy.ndarray: Vetor unidimensional normalizado representando a imagem,
                      ou None se ocorrer um erro
    
    Processo:
        1. Abre a imagem usando PIL (Pillow)
        2. Converte para escala de cinza (1 canal)
        3. Redimensiona para o tamanho padrão
        4. Converte para array NumPy
        5. Achata o array 2D para 1D
        6. Normaliza os valores de pixel para [0,1]
    """
    try:
        # Carrega a imagem e converte para escala de cinza (L = luminância)
        imagem = Image.open(caminho_imagem).convert("L")
        
        # Redimensiona para o tamanho padrão para garantir consistência
        imagem = imagem.resize(tamanho_alvo)
        
        # Converte para array NumPy com precisão de ponto flutuante
        matriz_imagem = np.array(imagem, dtype=np.float64)
        
        # Achata a matriz 2D para um vetor 1D (necessário para regressão logística)
        # Ex: Uma imagem 64x64 se torna um vetor de 4096 elementos
        imagem_achatada = matriz_imagem.flatten()
        
        # Normaliza os valores de pixel para o intervalo [0,1]
        # Divisão por 255 porque os valores originais estão no intervalo [0,255]
        imagem_normalizada = imagem_achatada / 255.0
        
        return imagem_normalizada
    
    except Exception as e:
        print(f"Erro ao processar a imagem {caminho_imagem}: {e}")
        return None

def carregar_conjunto_dados_paralelo(dir_veiculos, dir_nao_veiculos, tamanho_alvo=(64, 64), max_imagens_por_classe=None):
    """
    Carrega o conjunto de dados dos diretórios especificados usando processamento paralelo.
    Otimizado para Mac M1 com multiprocessamento.
    
    Parâmetros:
        dir_veiculos (str): Diretório contendo imagens de veículos (classe positiva)
        dir_nao_veiculos (str): Diretório contendo imagens de não-veículos (classe negativa)
        tamanho_alvo (tuple): Dimensões para redimensionar as imagens
        max_imagens_por_classe (int): Número máximo de imagens a carregar por classe
                                     (None para carregar todas)
    
    Retorna:
        tuple: (matriz_caracteristicas, matriz_rotulos)
            - matriz_caracteristicas: Array NumPy com formato (num_amostras, num_caracteristicas)
            - matriz_rotulos: Array NumPy com formato (num_amostras,)
    """
    inicio = time.time()
    
    # Obtém lista de arquivos de imagem válidos (ignorando arquivos ocultos e não-imagens)
    arquivos_veiculos = [f for f in os.listdir(dir_veiculos) 
                        if f.lower().endswith((".png", ".jpg", ".jpeg")) 
                        and not f.startswith(".")]
    
    arquivos_nao_veiculos = [f for f in os.listdir(dir_nao_veiculos) 
                            if f.lower().endswith((".png", ".jpg", ".jpeg")) 
                            and not f.startswith(".")]
    
    # Limita o número de imagens se especificado
    if max_imagens_por_classe is not None:
        arquivos_veiculos = arquivos_veiculos[:max_imagens_por_classe]
        arquivos_nao_veiculos = arquivos_nao_veiculos[:max_imagens_por_classe]
    
    # Determina o número de processos com base nos núcleos disponíveis
    # O Mac M1 tem 8 núcleos (4 de performance + 4 de eficiência)
    num_processos = multiprocessing.cpu_count()
    print(f"Utilizando {num_processos} processos para carregamento paralelo")
    
    # Cria caminhos completos para as imagens
    caminhos_veiculos = [os.path.join(dir_veiculos, f) for f in arquivos_veiculos]
    caminhos_nao_veiculos = [os.path.join(dir_nao_veiculos, f) for f in arquivos_nao_veiculos]
    
    # Função parcial para passar o tamanho_alvo
    func_carregar = partial(carregar_e_preprocessar_imagem, tamanho_alvo=tamanho_alvo)
    
    # Processamento paralelo para veículos
    print(f"Carregando {len(caminhos_veiculos)} imagens de veículos de {dir_veiculos}...")
    with multiprocessing.Pool(processes=num_processos) as pool:
        resultados_veiculos = pool.map(func_carregar, caminhos_veiculos)
    
    # Processamento paralelo para não-veículos
    print(f"Carregando {len(caminhos_nao_veiculos)} imagens de não-veículos de {dir_nao_veiculos}...")
    with multiprocessing.Pool(processes=num_processos) as pool:
        resultados_nao_veiculos = pool.map(func_carregar, caminhos_nao_veiculos)
    
    # Filtra resultados None e prepara características e rótulos
    caracteristicas = []
    rotulos = []
    
    # Adiciona veículos (classe 1)
    for resultado in resultados_veiculos:
        if resultado is not None:
            caracteristicas.append(resultado)
            rotulos.append(1)
    
    # Adiciona não-veículos (classe 0)
    for resultado in resultados_nao_veiculos:
        if resultado is not None:
            caracteristicas.append(resultado)
            rotulos.append(0)
    
    # Combina características e rótulos e embaralha-os juntos
    combinado = list(zip(caracteristicas, rotulos))
    random.shuffle(combinado)
    
    if combinado:  # Verifica se a lista não está vazia
        caracteristicas[:], rotulos[:] = zip(*combinado)
    else:
        print("Aviso: Nenhum dado carregado para embaralhar.")
        return np.array([]), np.array([])  # Retorna arrays vazios
    
    # Converte listas para arrays NumPy
    matriz_caracteristicas = np.array(caracteristicas)
    matriz_rotulos = np.array(rotulos)
    
    fim = time.time()
    tempo_total = fim - inicio
    
    print(f"Conjunto de dados carregado: {len(matriz_caracteristicas)} amostras em {tempo_total:.2f} segundos.")
    print(f"Formato das características: {matriz_caracteristicas.shape}")
    
    return matriz_caracteristicas, matriz_rotulos

def normalizar_dados(X_treino, X_teste):
    """
    Normaliza os dados usando padronização (z-score): (x - média) / desvio_padrão
    Esta normalização é mais robusta que a simples divisão por 255.
    
    Parâmetros:
        X_treino (numpy.ndarray): Conjunto de treinamento
        X_teste (numpy.ndarray): Conjunto de teste
    
    Retorna:
        tuple: (X_treino_norm, X_teste_norm) - Dados normalizados
    
    Nota: A normalização é calculada apenas no conjunto de treino e 
          aplicada tanto no treino quanto no teste para evitar vazamento de dados.
    """
    # Calcula média e desvio padrão por característica no conjunto de treino
    media = np.mean(X_treino, axis=1, keepdims=True)
    desvio_padrao = np.std(X_treino, axis=1, keepdims=True)
    
    # Evita divisão por zero
    desvio_padrao[desvio_padrao == 0] = 1.0
    
    # Aplica a normalização
    X_treino_norm = (X_treino - media) / desvio_padrao
    X_teste_norm = (X_teste - media) / desvio_padrao
    
    return X_treino_norm, X_teste_norm

def dividir_dados(caracteristicas, rotulos, tamanho_teste=0.2):
    """
    Divide o conjunto de dados em conjuntos de treinamento e teste.
    
    Parâmetros:
        caracteristicas (numpy.ndarray): Matriz de características, formato (num_amostras, num_caracteristicas)
        rotulos (numpy.ndarray): Vetor de rótulos, formato (num_amostras,)
        tamanho_teste (float): Proporção do conjunto de teste (0.0 a 1.0)
    
    Retorna:
        tuple: (X_treino, y_treino, X_teste, y_teste)
            - X_treino: Características de treinamento, formato (num_caracteristicas, num_amostras_treino)
            - y_treino: Rótulos de treinamento, formato (1, num_amostras_treino)
            - X_teste: Características de teste, formato (num_caracteristicas, num_amostras_teste)
            - y_teste: Rótulos de teste, formato (1, num_amostras_teste)
    
    Nota: Os dados são transpostos para o formato esperado pela regressão logística:
          - Características: (num_caracteristicas, num_amostras) em vez de (num_amostras, num_caracteristicas)
          - Rótulos: (1, num_amostras) em vez de (num_amostras,)
    """
    num_amostras = len(caracteristicas)
    if num_amostras == 0:
        print("Erro: Não há amostras para dividir.")
        # Retorna arrays vazios com as dimensões corretas esperadas
        num_features = caracteristicas.shape[1] if caracteristicas.ndim > 1 else 0
        return np.empty((num_features, 0)), np.empty((1, 0)), np.empty((num_features, 0)), np.empty((1, 0))

    # Calcula o número de amostras para teste
    num_amostras_teste = int(num_amostras * tamanho_teste)
    num_amostras_treino = num_amostras - num_amostras_teste

    # Divide os dados
    X_treino = caracteristicas[:num_amostras_treino]
    y_treino = rotulos[:num_amostras_treino]
    X_teste = caracteristicas[num_amostras_treino:]
    y_teste = rotulos[num_amostras_treino:]

    print(f"Dados divididos: {len(X_treino)} amostras de treino, {len(X_teste)} amostras de teste.")

    # Remodela rótulos para serem vetores linha (1, m)
    y_treino = y_treino.reshape(1, -1)
    y_teste = y_teste.reshape(1, -1)

    # Transpõe características para que o formato seja (num_caracteristicas, num_amostras)
    # Isso é necessário para a implementação eficiente da regressão logística
    X_treino = X_treino.T
    X_teste = X_teste.T

    print(f"Formatos finais - X_treino: {X_treino.shape}, y_treino: {y_treino.shape}, X_teste: {X_teste.shape}, y_teste: {y_teste.shape}")

    return X_treino, y_treino, X_teste, y_teste

# =====================================================================
# PARTE 2: IMPLEMENTAÇÃO DA REGRESSÃO LOGÍSTICA
# =====================================================================

def inicializar_parametros(dimensao):
    """
    Inicializa os parâmetros do modelo de regressão logística.
    
    Parâmetros:
        dimensao (int): Número de características (dimensão do espaço de entrada)
    
    Retorna:
        tuple: (pesos, bias)
            - pesos: Vetor de pesos, formato (dimensao, 1)
            - bias: Termo de viés (escalar)
    
    Nota: A inicialização com zeros é comum para regressão logística,
          diferente de redes neurais onde inicialização aleatória é preferida.
    """
    # Inicializa pesos como vetor de zeros com formato (dimensao, 1)
    pesos = np.zeros((dimensao, 1), dtype=np.float64)
    
    # Inicializa bias como escalar zero
    bias = 0.0
    
    print(f"Inicializados pesos w com formato: {pesos.shape} e bias b: {bias}")
    return pesos, bias

def sigmoide(z):
    """
    Calcula a função sigmoide (ou função logística): σ(z) = 1 / (1 + e^(-z))
    
    Parâmetros:
        z (numpy.ndarray): Entrada da função sigmoide
    
    Retorna:
        numpy.ndarray: Saída da função sigmoide, valores entre 0 e 1
    
    Fórmula matemática:
        σ(z) = 1 / (1 + e^(-z))
    
    Propriedades:
        - Mapeia qualquer número real para o intervalo (0,1)
        - É diferenciável em todo seu domínio
        - Derivada: σ'(z) = σ(z) * (1 - σ(z))
    """
    # Limita z para evitar overflow/underflow em exp
    # Valores muito negativos ou positivos podem causar problemas numéricos
    z_limitado = np.clip(z, -500, 500)
    
    # Calcula a função sigmoide
    s = 1 / (1 + np.exp(-z_limitado))
    
    return s

def propagar(pesos, bias, X, Y, lambda_reg=0.1):
    """
    Implementa a propagação para frente e para trás para regressão logística.
    Calcula o custo e os gradientes usando entropia cruzada binária com regularização L2.
    
    Parâmetros:
        pesos (numpy.ndarray): Vetor de pesos, formato (num_caracteristicas, 1)
        bias (float): Termo de viés
        X (numpy.ndarray): Matriz de características, formato (num_caracteristicas, num_amostras)
        Y (numpy.ndarray): Vetor de rótulos, formato (1, num_amostras)
        lambda_reg (float): Parâmetro de regularização L2 (0 = sem regularização)
    
    Retorna:
        tuple: (gradientes, custo)
            - gradientes: Dicionário com gradientes dw e db
            - custo: Valor da função de custo (entropia cruzada binária)
    
    Fórmulas matemáticas:
        1. Propagação para frente:
           z = w^T * X + b
           A = σ(z)  (onde σ é a função sigmoide)
        
        2. Função de custo (entropia cruzada binária com regularização L2):
           J = (-1/m) * Σ[Y * log(A) + (1-Y) * log(1-A)] + (λ/2m) * ||w||²
        
        3. Gradientes:
           dw = (1/m) * X * (A-Y)^T + (λ/m) * w
           db = (1/m) * Σ(A-Y)
    """
    num_amostras = X.shape[1]  # Número de exemplos de treinamento

    # PROPAGAÇÃO PARA FRENTE (DE X PARA O CUSTO)
    # Calcula a ativação A = sigmoide(pesos^T * X + bias)
    Z = np.dot(pesos.T, X) + bias  # Z = w^T * X + b
    A = sigmoide(Z)                # A = σ(Z)

    # Calcula o custo J usando entropia cruzada binária com regularização L2
    # Adiciona um pequeno epsilon aos argumentos do log para evitar log(0)
    epsilon = 1e-9
    
    # Termo de entropia cruzada
    termo_entropia = -1/num_amostras * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))
    
    # Termo de regularização L2 (não inclui o bias)
    termo_regularizacao = lambda_reg / (2 * num_amostras) * np.sum(np.square(pesos))
    
    # Custo total
    custo = termo_entropia + termo_regularizacao

    # PROPAGAÇÃO PARA TRÁS (PARA ENCONTRAR OS GRADIENTES)
    # Calcula o gradiente da perda em relação a Z
    dZ = A - Y  # dL/dZ = A - Y (derivada da entropia cruzada em relação a Z)

    # Calcula o gradiente da perda em relação a w (com regularização L2)
    # Implementação vetorizada: dw = (1/m) * X * dZ^T + (λ/m) * w
    dw = (1/num_amostras) * np.dot(X, dZ.T) + (lambda_reg / num_amostras) * pesos

    # Calcula o gradiente da perda em relação a b
    # Implementação vetorizada: db = (1/m) * Σ(dZ)
    db = (1/num_amostras) * np.sum(dZ)

    # Garante que o custo seja um escalar
    custo = np.squeeze(custo)

    # Armazena gradientes em um dicionário
    gradientes = {
        "dw": dw,
        "db": db
    }

    return gradientes, custo

def otimizar_em_lotes(pesos, bias, X, Y, num_iteracoes, taxa_aprendizado, 
                     tamanho_lote=64, lambda_reg=0.1, imprimir_custo=True):
    """
    Otimiza os parâmetros do modelo usando gradiente descendente em mini-lotes.
    Implementação otimizada para Mac M1 com processamento eficiente de lotes.
    
    Parâmetros:
        pesos (numpy.ndarray): Vetor de pesos inicial, formato (num_caracteristicas, 1)
        bias (float): Termo de viés inicial
        X (numpy.ndarray): Matriz de características, formato (num_caracteristicas, num_amostras)
        Y (numpy.ndarray): Vetor de rótulos, formato (1, num_amostras)
        num_iteracoes (int): Número de iterações do gradiente descendente
        taxa_aprendizado (float): Taxa de aprendizado (passo do gradiente descendente)
        tamanho_lote (int): Tamanho do mini-lote para processamento
        lambda_reg (float): Parâmetro de regularização L2
        imprimir_custo (bool): Se True, imprime o custo a cada 100 iterações
    
    Retorna:
        tuple: (parametros, gradientes, custos)
            - parametros: Dicionário com pesos e bias otimizados
            - gradientes: Dicionário com gradientes finais
            - custos: Lista de custos ao longo das iterações
    
    Algoritmo:
        1. Para cada iteração:
           a. Divide os dados em mini-lotes
           b. Para cada mini-lote:
              i. Calcula gradientes e custo
              ii. Atualiza parâmetros: w := w - α * dw, b := b - α * db
           c. Registra o custo periodicamente
    """
    custos = []
    num_amostras = X.shape[1]
    
    # Garantir que pesos tenha o formato correto (num_caracteristicas, 1)
    if pesos.shape != (X.shape[0], 1):
        print(f"Aviso: Ajustando formato dos pesos iniciais de {pesos.shape} para ({X.shape[0]}, 1)")
        pesos = pesos.reshape(X.shape[0], 1)

    # Implementação de decaimento da taxa de aprendizado
    taxa_aprendizado_original = taxa_aprendizado
    
    for i in range(num_iteracoes):
        custo_epoca = 0
        num_lotes = int(np.ceil(num_amostras / tamanho_lote))
        
        # Decaimento da taxa de aprendizado
        taxa_aprendizado = taxa_aprendizado_original / (1 + 0.01 * i)
        
        # Embaralha os índices para cada época
        indices = np.random.permutation(num_amostras)
        X_embaralhado = X[:, indices]
        Y_embaralhado = Y[:, indices]
        
        # Processa cada mini-lote
        for j in range(num_lotes):
            inicio = j * tamanho_lote
            fim = min(inicio + tamanho_lote, num_amostras)
            
            # Extrai o mini-lote atual
            X_lote = X_embaralhado[:, inicio:fim]
            Y_lote = Y_embaralhado[:, inicio:fim]
            
            # Calcula gradientes e custo para este mini-lote
            gradientes, custo_lote = propagar(pesos, bias, X_lote, Y_lote, lambda_reg)
            
            # Atualiza parâmetros usando os gradientes deste mini-lote
            pesos = pesos - taxa_aprendizado * gradientes["dw"]
            bias = bias - taxa_aprendizado * gradientes["db"]
            
            # Acumula o custo ponderado pelo tamanho do lote
            custo_epoca += custo_lote * (fim - inicio) / num_amostras
        
        # Registra os custos a cada 100 iterações
        if i % 100 == 0:
            custos.append(custo_epoca)
            if imprimir_custo:
                # Verifica se custo é NaN antes de imprimir
                if np.isnan(custo_epoca):
                    print(f"Custo após iteração {i}: NaN - Verifique a taxa de aprendizado ou os dados.")
                else:
                    print(f"Custo após iteração {i}: {custo_epoca:.6f} (taxa={taxa_aprendizado:.6f})")
        
        # Verificação periódica do formato dos pesos
        if i % 100 == 0 and pesos.shape != (X.shape[0], 1):
            print(f"Aviso: Corrigindo formato dos pesos na iteração {i}")
            pesos = pesos.reshape(X.shape[0], 1)

    # Verificação final do formato dos pesos
    if pesos.shape != (X.shape[0], 1):
        print(f"Aviso: Ajustando formato final dos pesos")
        pesos = pesos.reshape(X.shape[0], 1)

    # Calcula gradientes finais para todo o conjunto de dados
    gradientes, _ = propagar(pesos, bias, X, Y, lambda_reg)
    
    parametros = {
        "pesos": pesos,
        "bias": bias
    }

    return parametros, gradientes, custos

def prever(pesos, bias, X):
    """
    Faz previsões usando o modelo de regressão logística treinado.
    
    Parâmetros:
        pesos (numpy.ndarray): Vetor de pesos treinado, formato (num_caracteristicas, 1)
        bias (float): Termo de viés treinado
        X (numpy.ndarray): Matriz de características, formato (num_caracteristicas, num_amostras)
    
    Retorna:
        numpy.ndarray: Previsões binárias (0 ou 1), formato (1, num_amostras)
    
    Algoritmo:
        1. Calcula Z = w^T * X + b
        2. Calcula A = σ(Z) (probabilidades)
        3. Converte probabilidades em classes: 1 se A > 0.5, 0 caso contrário
    """
    num_amostras = X.shape[1]
    Y_previsao = np.zeros((1, num_amostras))
    
    # Verifica se os pesos já estão no formato correto (num_caracteristicas, 1)
    if pesos.shape != (X.shape[0], 1):
        # Se não estiver, tenta ajustar o formato
        try:
            pesos = pesos.reshape(X.shape[0], 1)
        except ValueError as e:
            print(f"Erro ao redimensionar pesos: {e}")
            print(f"Formato dos pesos: {pesos.shape}, Formato esperado: ({X.shape[0]}, 1)")
            # Se não for possível redimensionar, tenta transpor
            if len(pesos.shape) == 2 and pesos.shape[1] == X.shape[0]:
                pesos = pesos.T
            else:
                raise ValueError(f"Não foi possível ajustar os pesos para o formato correto. Formato atual: {pesos.shape}")

    # Calcula o vetor A prevendo as probabilidades
    # A = σ(w^T * X + b)
    Z = np.dot(pesos.T, X) + bias
    A = sigmoide(Z)

    # Converte as probabilidades A em previsões binárias (0 ou 1)
    # Limiar de decisão: 0.5
    Y_previsao = (A > 0.5).astype(int)

    return Y_previsao

def avaliar_modelo(Y_previsao, Y_verdadeiro):
    """
    Calcula métricas de avaliação para o modelo de classificação.
    
    Parâmetros:
        Y_previsao (numpy.ndarray): Previsões do modelo, formato (1, num_amostras)
        Y_verdadeiro (numpy.ndarray): Rótulos verdadeiros, formato (1, num_amostras)
    
    Retorna:
        dict: Dicionário com métricas de avaliação
    
    Métricas calculadas:
        - Acurácia: (VP + VN) / (VP + VN + FP + FN)
        - Precisão: VP / (VP + FP)
        - Revocação (Recall): VP / (VP + FN)
        - Pontuação F1: 2 * (Precisão * Revocação) / (Precisão + Revocação)
        - Matriz de confusão: VP, VN, FP, FN
    
    Onde:
        - VP: Verdadeiros Positivos (previsão=1, verdadeiro=1)
        - VN: Verdadeiros Negativos (previsão=0, verdadeiro=0)
        - FP: Falsos Positivos (previsão=1, verdadeiro=0)
        - FN: Falsos Negativos (previsão=0, verdadeiro=1)
    """
    # Garante que Y_verdadeiro também seja um vetor linha para comparação
    if Y_verdadeiro.shape[0] != 1:
        Y_verdadeiro = Y_verdadeiro.T  # Transpõe se for um vetor coluna

    # Acurácia: proporção de previsões corretas
    acuracia = np.mean(Y_previsao == Y_verdadeiro) * 100
    print(f"Acurácia: {acuracia:.2f}%")

    # Calcula elementos da matriz de confusão
    VP = np.sum((Y_previsao == 1) & (Y_verdadeiro == 1))  # Verdadeiros Positivos
    VN = np.sum((Y_previsao == 0) & (Y_verdadeiro == 0))  # Verdadeiros Negativos
    FP = np.sum((Y_previsao == 1) & (Y_verdadeiro == 0))  # Falsos Positivos
    FN = np.sum((Y_previsao == 0) & (Y_verdadeiro == 1))  # Falsos Negativos

    print(f"Verdadeiros Positivos (VP): {VP}")
    print(f"Verdadeiros Negativos (VN): {VN}")
    print(f"Falsos Positivos (FP): {FP}")
    print(f"Falsos Negativos (FN): {FN}")

    # Precisão: quão precisas são as previsões positivas
    # Precisão = VP / (VP + FP)
    precisao = VP / (VP + FP) if (VP + FP) > 0 else 0
    
    # Revocação (Recall): capacidade de encontrar todos os positivos
    # Revocação = VP / (VP + FN)
    revocacao = VP / (VP + FN) if (VP + FN) > 0 else 0
    
    # Pontuação F1: média harmônica entre precisão e revocação
    # F1 = 2 * (Precisão * Revocação) / (Precisão + Revocação)
    pontuacao_f1 = 2 * (precisao * revocacao) / (precisao + revocacao) if (precisao + revocacao) > 0 else 0

    print(f"Precisão: {precisao:.4f}")
    print(f"Revocação (Recall): {revocacao:.4f}")
    print(f"Pontuação F1: {pontuacao_f1:.4f}")

    # Retorna todas as métricas em um dicionário
    metricas = {
        "acuracia": acuracia,
        "precisao": precisao,
        "revocacao": revocacao,
        "pontuacao_f1": pontuacao_f1,
        "VP": VP,
        "VN": VN,
        "FP": FP,
        "FN": FN
    }
    return metricas

def plotar_custo(custos, taxa_aprendizado, caminho_saida="grafico_custo.png"):
    """
    Plota a evolução do custo ao longo das iterações e salva o gráfico.
    
    Parâmetros:
        custos (list): Lista de valores de custo
        taxa_aprendizado (float): Taxa de aprendizado usada no treinamento
        caminho_saida (str): Caminho para salvar o gráfico
    """
    plt.figure(figsize=(10, 6))
    
    # Plota os custos a cada 100 iterações
    iteracoes = np.arange(len(custos)) * 100
    plt.plot(iteracoes, custos, 'b-', linewidth=2)
    
    plt.title(f'Evolução do Custo Durante o Treinamento (Taxa de Aprendizado = {taxa_aprendizado})', fontsize=14)
    plt.xlabel('Iterações', fontsize=12)
    plt.ylabel('Custo (Entropia Cruzada Binária)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adiciona anotações para valores inicial e final
    if len(custos) > 0:
        plt.annotate(f'Custo inicial: {custos[0]:.2f}', 
                    xy=(0, custos[0]), 
                    xytext=(50, 20),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        plt.annotate(f'Custo final: {custos[-1]:.2f}', 
                    xy=(iteracoes[-1], custos[-1]), 
                    xytext=(-50, -20),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300)
    print(f"Gráfico da função de custo salvo em {caminho_saida}")
    return caminho_saida

# =====================================================================
# PARTE 3: FUNÇÃO PRINCIPAL E EXECUÇÃO
# =====================================================================

def principal(dir_veiculos, dir_nao_veiculos):
    """
    Função principal que orquestra todo o fluxo de reconhecimento de imagens.
    
    Parâmetros:
        dir_veiculos (str): Diretório contendo imagens de veículos
        dir_nao_veiculos (str): Diretório contendo imagens de não-veículos
    
    Retorna:
        dict: Resultados do modelo (métricas, caminhos de arquivos, etc.)
    """
    # --- Configurações ---
    # Ajuste estes parâmetros para otimizar o desempenho
    
    # Configurações de processamento de imagem
    LARGURA_IMG = 64
    ALTURA_IMG = 64
    TAMANHO_ALVO = (LARGURA_IMG, ALTURA_IMG)
    
    # Configurações de treinamento do modelo
    TAMANHO_TESTE = 0.20  # 20% para teste
    TAXA_APRENDIZADO = 0.003  # Taxa de aprendizado otimizada
    NUM_ITERACOES = 5000  # Número de iterações para convergência
    TAMANHO_LOTE = 128  # Tamanho do mini-lote para processamento
    LAMBDA_REG = 0.05  # Parâmetro de regularização L2
    IMPRIMIR_CUSTO = True
    
    print("Iniciando o processo de reconhecimento de imagens com regressão logística...")
    inicio_total = time.time()

    # --- 1. Carregar e Preparar Dados ---
    print("\n--- Passo 1: Carregando e Preparando Dados ---")
    
    # Verifica se os diretórios existem
    if not os.path.isdir(dir_veiculos) or not os.path.isdir(dir_nao_veiculos):
        print("Erro: Diretório de veículos ou não-veículos não encontrado.")
        print(f"Diretórios procurados:")
        print(f"  Veículos: {dir_veiculos}")
        print(f"  Não-Veículos: {dir_nao_veiculos}")
        return {"erro": "Diretórios não encontrados"}

    # Carrega todas as imagens (None = sem limite)
    caracteristicas, rotulos = carregar_conjunto_dados_paralelo(
        dir_veiculos, dir_nao_veiculos, 
        tamanho_alvo=TAMANHO_ALVO, 
        max_imagens_por_classe=None  # Usar todas as imagens disponíveis
    )

    if caracteristicas.shape[0] == 0:
        print("Erro: Nenhuma característica foi carregada.")
        return {"erro": "Falha ao carregar imagens"}

    # Divide os dados em conjuntos de treinamento e teste
    X_treino, y_treino, X_teste, y_teste = dividir_dados(
        caracteristicas, rotulos, tamanho_teste=TAMANHO_TESTE
    )
    
    # Normalização avançada (z-score)
    print("Aplicando normalização z-score aos dados...")
    X_treino, X_teste = normalizar_dados(X_treino, X_teste)

    # --- 2. Inicializar e Treinar Modelo ---
    print("\n--- Passo 2: Inicializando e Treinando o Modelo de Regressão Logística ---")
    
    # Obtém o número de características (pixels na imagem achatada)
    num_caracteristicas = X_treino.shape[0]
    pesos, bias = inicializar_parametros(num_caracteristicas)

    # Treina o modelo usando gradiente descendente em mini-lotes
    print(f"Iniciando treinamento com {NUM_ITERACOES} iterações, tamanho de lote {TAMANHO_LOTE}, regularização L2 {LAMBDA_REG}...")
    inicio_treino = time.time()
    
    parametros, gradientes, custos = otimizar_em_lotes(
        pesos, bias, X_treino, y_treino,
        num_iteracoes=NUM_ITERACOES,
        taxa_aprendizado=TAXA_APRENDIZADO,
        tamanho_lote=TAMANHO_LOTE,
        lambda_reg=LAMBDA_REG,
        imprimir_custo=IMPRIMIR_CUSTO
    )
    
    fim_treino = time.time()
    tempo_treino = fim_treino - inicio_treino
    print(f"Treinamento concluído em {tempo_treino:.2f} segundos.")

    # Recupera parâmetros aprendidos
    pesos_aprendidos = parametros["pesos"]
    bias_aprendido = parametros["bias"]

    # --- 3. Fazer Previsões ---
    print("\n--- Passo 3: Fazendo Previsões ---")
    
    # Prever no conjunto de treinamento
    Y_previsao_treino = prever(pesos_aprendidos, bias_aprendido, X_treino)
    
    # Prever no conjunto de teste
    Y_previsao_teste = prever(pesos_aprendidos, bias_aprendido, X_teste)

    # --- 4. Avaliar Modelo ---
    print("\n--- Passo 4: Avaliando o Desempenho do Modelo ---")
    
    print("Avaliação no Conjunto de Treinamento:")
    metricas_treino = avaliar_modelo(Y_previsao_treino, y_treino)

    print("\nAvaliação no Conjunto de Teste:")
    metricas_teste = avaliar_modelo(Y_previsao_teste, y_teste)

    # --- 5. Plotar Função de Custo ---
    print("\n--- Passo 5: Plotando a Função de Custo ---")
    
    if custos:
        caminho_grafico = plotar_custo(
            custos, 
            TAXA_APRENDIZADO, 
            caminho_saida="grafico_custo.png"
        )
    else:
        print("Sem dados de custo para plotar.")
        caminho_grafico = None

    # --- 6. Salvar Modelo ---
    print("\n--- Passo 6: Salvando o Modelo ---")
    
    # Salva os parâmetros do modelo em um arquivo NumPy
    np.savez(
        'modelo_regressao_logistica.npz',
        pesos=pesos_aprendidos,
        bias=bias_aprendido
    )
    print("Modelo salvo em 'modelo_regressao_logistica.npz'")
    
    fim_total = time.time()
    tempo_total = fim_total - inicio_total
    print(f"\nProcesso finalizado em {tempo_total:.2f} segundos.")
    
    # Retorna resultados
    resultados = {
        "metricas_treino": metricas_treino,
        "metricas_teste": metricas_teste,
        "caminho_grafico": caminho_grafico,
        "caminho_modelo": "modelo_regressao_logistica.npz",
        "tempo_total": tempo_total,
        "tempo_treino": tempo_treino
    }
    
    return resultados

# =====================================================================
# EXECUÇÃO PRINCIPAL
# =====================================================================

if __name__ == "__main__":
    """
    Ponto de entrada principal do script.
    
    Para executar:
    python reconhecimento_imagens_otimizado.py
    
    Certifique-se de ajustar os caminhos dos diretórios abaixo para corresponder
    à localização das suas imagens de veículos e não-veículos.
    """
    # Ajuste estes caminhos para corresponder à localização das suas imagens
    DIRETORIO_VEICULOS = "data/vehicles"
    DIRETORIO_NAO_VEICULOS = "data/non-vehicles"
    
    # Verifica se as bibliotecas necessárias estão instaladas
    try:
        import PIL
        import numpy
        import matplotlib
        import multiprocessing
    except ImportError as e:
        print(f"Erro ao importar bibliotecas necessárias: {e}")
        print("Tentando instalar bibliotecas...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", 
                                  "Pillow", "numpy", "matplotlib"])
            print("Bibliotecas instaladas com sucesso. Por favor, execute o script novamente.")
        except Exception as e:
            print(f"Falha ao instalar bibliotecas: {e}")
            print("Por favor, instale-as manualmente: pip install Pillow numpy matplotlib")
        sys.exit(1)
    
    # Define o backend do matplotlib para Agg (não interativo)
    matplotlib.use("Agg")
    
    # Executa o fluxo principal
    resultados = principal(DIRETORIO_VEICULOS, DIRETORIO_NAO_VEICULOS)
    
    # Exibe um resumo dos resultados
    if "erro" not in resultados:
        print("\n=== RESUMO DOS RESULTADOS ===")
        print(f"Acurácia no conjunto de teste: {resultados['metricas_teste']['acuracia']:.2f}%")
        print(f"Pontuação F1 no conjunto de teste: {resultados['metricas_teste']['pontuacao_f1']:.4f}")
        print(f"Tempo total de execução: {resultados['tempo_total']:.2f} segundos")
        print(f"Gráfico de custo: {resultados['caminho_grafico']}")
        print(f"Modelo salvo em: {resultados['caminho_modelo']}")
    else:
        print(f"\nErro durante a execução: {resultados['erro']}")
