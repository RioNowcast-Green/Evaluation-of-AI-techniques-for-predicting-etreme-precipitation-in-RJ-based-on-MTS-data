
#####################################################################
import tensorflow as tf
import numpy as np
from pathlib import Path
import random
import torch
import torch.nn as nn

from zipfile import ZipFile
import pandas as pd

import matplotlib.pyplot as plt

import h5py
#####################################################################

#####################################################################
# Feature subsets (colocar na seção 4?)
#####################################################################

Features = {'SC1': ['s_dia', 'c_dia', 's_hora', 'c_hora', 'Temp_Amb', 'Pres_Atm', 'Umidade', 'Rad', 'Precip', 'POv_Calc', 'Tpov_dif', 'Vento_x', 'Vento_y', 'Lat', 'Long', 'Alt'], 
                'SC2': ['s_dia', 'c_dia', 's_hora', 'c_hora', 'Temp_Amb', 'Pres_Atm', 'Umidade', 'Rad', 'Precip', 'POv_Calc', 'Tpov_dif', 'Vento_x', 'Vento_y', 'Lat', 'Long', 'Alt', 'Vento_dv_01h', 'Vento_dv_03h', 'Vento_dv_06h', 'Vento_dv_12h', 'Vento_dv_18h', 'Vento_dv_24h', 'Vento_ddir_01h', 'Vento_ddir_03h', 'Vento_ddir_06h', 'Vento_ddir_12h', 'Vento_ddir_18h', 'Vento_ddir_24h', 'DP_01h', 'DP_03h', 'DP_06h', 'DP_12h', 'DP_18h', 'DP_24h', 'DTemp_01h', 'DTemp_03h', 'DTemp_06h', 'DTemp_12h', 'DTemp_18h', 'DTemp_24h'],
                'SC3': ['s_dia', 'c_dia', 's_hora', 'c_hora', 'Temp_Amb', 'Pres_Atm', 'Umidade', 'Rad', 'Precip', 'POv_Calc', 'Tpov_dif', 'Vento_x', 'Vento_y', 'Lat', 'Long', 'Alt', 'TSM', 'Dist_TSM'], 
                'SC4': ['s_dia', 'c_dia', 's_hora', 'c_hora', 'Temp_Amb', 'Pres_Atm', 'Umidade', 'Rad', 'Precip', 'POv_Calc', 'Tpov_dif', 'Vento_x', 'Vento_y', 'Lat', 'Long', 'Alt', 'TEMP_100.0', 'TEMP_150.0', 'TEMP_200.0', 'TEMP_250.0', 'TEMP_300.0', 'TEMP_400.0', 'TEMP_500.0', 'TEMP_700.0', 'TEMP_850.0', 'TEMP_925.0', 'TEMP_1000.0', 'POv_dep_100.0', 'POv_dep_150.0', 'POv_dep_200.0', 'POv_dep_250.0', 'POv_dep_300.0', 'POv_dep_400.0', 'POv_dep_500.0', 'POv_dep_700.0', 'POv_dep_850.0', 'POv_dep_925.0', 'POv_dep_1000.0', 'Vento_dir_100.0', 'Vento_dir_150.0', 'Vento_dir_200.0', 'Vento_dir_250.0', 'Vento_dir_300.0', 'Vento_dir_400.0', 'Vento_dir_500.0', 'Vento_dir_700.0', 'Vento_dir_850.0', 'Vento_dir_925.0', 'Vento_dir_1000.0', 'Vento_vel_100.0', 'Vento_vel_150.0', 'Vento_vel_200.0', 'Vento_vel_250.0', 'Vento_vel_300.0', 'Vento_vel_400.0', 'Vento_vel_500.0', 'Vento_vel_700.0', 'Vento_vel_850.0', 'Vento_vel_925.0', 'Vento_vel_1000.0', 'Vento_x_100.0', 'Vento_x_150.0', 'Vento_x_200.0', 'Vento_x_250.0', 'Vento_x_300.0', 'Vento_x_400.0', 'Vento_x_500.0', 'Vento_x_700.0', 'Vento_x_850.0', 'Vento_x_925.0', 'Vento_x_1000.0', 'Vento_y_100.0', 'Vento_y_150.0', 'Vento_y_200.0', 'Vento_y_250.0', 'Vento_y_300.0', 'Vento_y_400.0', 'Vento_y_500.0', 'Vento_y_700.0', 'Vento_y_850.0', 'Vento_y_925.0', 'Vento_y_1000.0', 'dt_sond'], 
                'SC5': ['TSM', 'Tpov_dif', 'DTemp_03h', 'DP_01h', 's_dia', 'Vento_dv_01h', 'Vento_ddir_01h', 'Vento_y', 'Vento_dv_03h', 'Rad', 'POv_dep_700.0', 'Vento_x', 'c_dia', 'Vento_y_925.0', 'Vento_ddir_12h', 'Precip']}

  
#####################################################################
# Export/Import tensors (h5) (avaliar importar de Tensror methods)
#####################################################################

def exportar_tensores_h5(arquivo, tensores, nomes=None, opt=4):
    '''
    Exporta um arquivo h5 como datasets nomeados
    Compressão com GZIP

    '''
    arq = arquivo + '.h5'
    if nomes is None: nomes = [f'T{i}' for i in range(len(tensores))]
    with h5py.File(arq, 'w') as f:
        for i, t in enumerate(tensores):
            print(f'Processando tensor {i},{nomes[i]}, {t.shape}')
            f.create_dataset(nomes[i], data=t, compression='gzip', compression_opts=opt)  
            print('-----')


def ler_h5(arquivo):
    '''
    Interpreta o arquivo h5, retronando um dicionario de tensores
    '''
    L = {}
    with h5py.File(arquivo, 'r') as file:
        for k in file.keys():
            print(k)
            data = np.array(file[k])
            print(data.shape)
            try: L[k] = tf.constant(data)
            except: print(f'Erro ao tentar criar tensor {k}')
    return L

# Importar arquivo parquet
def import_parquet(Dir, Arq_parq):
    
    dataset = pd.read_parquet(Dir+Arq_parq)

    return dataset

#####################################################################
# Scaling functions
#####################################################################

# Standardization (X - \mu)/\sigma $

def padr(x, media, desvio): return (x - media) / desvio
def despadr(x, media, desvio): return desvio*x + media 
def mae_padr(x, desvio): return desvio/x
def mse_padr(x, desvio): return x/(desvio**2)    
def mae_despadr(x, desvio): return desvio*x
def mse_despadr(x, desvio): return (desvio**2)*x        

# ### Normalization $(x - X_{min})/(X_{max}-X_{min})$

def norm(x, max_, min_, li=0, ls=1): 
    return li + (ls-li)*((x - min_)/(max_-min_))   
def desnorm(x, max_, min_, li=0, ls=1): 
    return min_ + (max_-min_)*((x-li)/(ls-li))
def mae_norm(x, max_, min_): return (x/(max_-min_))
def mse_norm(x, max_, min_): return x/((max_-min_)**2)
def mae_desnorm(x, max_, min_): return (x*(max_-min_))
def mse_desnorm(x, max_, min_): return (x*(max_-min_)**2)

#####################################################################
# CodeCarbon: start tracker
#####################################################################

from codecarbon import EmissionsTracker
from codecarbon import OfflineEmissionsTracker

def init_tracker(proj, file, country="BRA"):

    T = OfflineEmissionsTracker(
        country_iso_code=country,
        project_name=proj,
        output_file=file,
        allow_multiple_runs=True, 
        log_level="error"
    )
    return T

#####################################################################
# Timeseries plots
#####################################################################

# Função para Plotar exemplos
def plotar_exemplos(X, Yr, Yp, j_shape, ind_alv, grafs, seq2seq=True, ResNet=False,escala=None, est=None, chanels_first=False, fy =[0.9,1.1], li=0, ls=1):

    """
    Plotar exemplos de amostra, com valores deentrada e preditos

    Parâmetros:
    
    - X: tensor com as entradas
    - Yr e Yp: Devem conter somente o atributo alvo a ser plotado, se houver mais de um.
    - j_shape: tupla; formado da janela (entrada, saída, horizonte)
    - ind_alv: int; índice do atributo alvo em X
    - grafs: int; número de gexemplos a ser plotado
    - seq2seq: bool; arquitetura do modelo (True para seq-to-seq)
    - escala: str/None; 'p' para padronizado, 'n' para normalizado; default: None
    - est: lista/tupla; estatísticas (min, max, media, desvio) para o atributo alvo; usado se escala for fornecida
    - fy: lista/tupla; fatores de escala (mínimo e máximo) para o eixo y
    - li, ls: limites inferior e suerior de escala usados na normalização 'Max/Min'
    
    """
    # escala: 'p' (padronizado), 'n', normalizado, 'None': dados reais
    # est: estatisticas do alvo, conforme escala

    if chanels_first: # reordenar pata amostras X tempo x atributos
        X = tf.transpose(X, perm=[0,2,1])
        Yr = tf.transpose(Yr, perm=[0,2,1])
        Yp = tf.transpose(Yp, perm=[0,2,1])
    
    if est is not None: min_, max_, med, desv = est      
    
    lex = len(Yr)
    grafs = min(grafs, lex)
    Pinds = np.sort(random.sample(range(0,lex), grafs))
    e, s, h = j_shape

    
    fig, axes = plt.subplots(nrows=grafs, ncols=1, figsize=(12, 2*grafs))

    for g in range(grafs):
        plt.sca(axes[g])
    
        pos = Pinds[g]
        if seq2seq: 
            yr = Yr[pos,-1,:,0]
            yp = Yp[pos,-1,:,0]
        elif ResNet: 
            yr = Yr[pos, -h:]
            yp = Yp[pos, -h:]
        else: 
            yr = Yr[pos]   
            yp =Yp[pos]
        
        if escala=='p': yr = despadr(yr, med, desv)
        if escala=='n': yr = desnorm(yr, max_, min_, li, ls)
            
        if escala=='p': yp = despadr(yp,med, desv)
        if escala=='n': yp = desnorm(yp, max_, min_, li, ls)
            
        if seq2seq:
            yp_seq =Yp[pos,:-1,0,0] # predição 1h a frente para cada tempo
            if escala=='p': yp_seq = despadr(yp_seq,med, desv)
            if escala=='n': yp_seq = desnorm(yp_seq,max_, min_, li, ls)

        if ResNet:
            yp_seq =Yp[pos,:-h,0] # predição um horizonte a frente para cada tempo
            if escala=='p': yp_seq = despadr(yp_seq,med, desv)
            if escala=='n': yp_seq = desnorm(yp_seq,max_, min_, li, ls)
        
        x = X[pos,:,ind_alv]
        if escala=='p': x = despadr(x,med, desv)
        if escala=='n': x = desnorm(x, max_, min_, li, ls)

            
        tx = np.arange(e)
        ty = np.arange(e,e+h)
        
        plt.plot(tx, x, 'k.-', label='Entrada') 
        plt.plot(ty[-h:], yr[-h:], 'b.', label='Real') 
        plt.plot(ty, yp, 'rx', label='Predito')
        if seq2seq: plt.plot((tx[1:]), yp_seq, 'gx', label='Predito')
        if ResNet: plt.plot((tx[h:]), yp_seq, 'gx', label='Predito')    
        if escala is not None: plt.ylim((min_*fy[0],max_*fy[1]))
        
    plt.show()

# Plotar valores real vs predito
def plot_real_vs_pred(Yr, Yp, alvo = 0, chanels_first=False, per_h=False, seq2seq=False, est= None, escala=None, li=0, ls=1, base=None):
    """
    Plot de valores reais vs. preditos (por instante de tempo no horinonte de predição ou para todo o horizonte)

    Yr, Yp: tesnsores para valores reais e preditos
    alvo: int; índice do atributoalvo em Y

    chanels_first: bool; verdadeiro para forma 'amostra x atributos x tempo'
    """
    
    if chanels_first:
        Yr = tf.transpose(Yr, perm=[0,2,1])
        Yp = tf.transpose(Yp, perm=[0,2,1])

    h=Yr.shape[-2]

    if base is not None: 
        bs = ' - '+base
    else: bs=''    
    
    if est is not None: min_, max_, med, desv = est
    # else: min_, max_, med, desv = 4*[None]   

    if escala == 'p' and est is not None:
        Yr = despadr(Yr, med, desv) 
        Yp = despadr(Yp, med, desv) # stats[var][2], stats[var][3])
    
    if escala == 'n' and est is not None:
        Yr = desnorm(Yr, max_, min_, li, ls)
        Yp = desnorm(Yp, max_, min_, li, ls)
    
    if per_h:
        for t in range(h):
            plt.figure(figsize=(5,5))
            if seq2seq:
                Yrt = Yr[:,-1,t,alvo]
                Ypt = Yp[:,-1,t,alvo]
            else:
                Yrt = Yr[:,t,alvo]
                Ypt = Yp[:,t,alvo]
            
            plt.plot(Yrt, Ypt, "o", markersize=1)
            plt.plot(Yrt, Yrt, "k")
            plt.xlabel("Real") #, fontsize=16)
            plt.ylabel("Previsto")
            plt.title(f"Real vs. Previsto - h: {t+1}"+bs)
            plt.show()
    
    else:
        plt.figure(figsize=(5,5))
        
        if seq2seq:
            Yr_ = np.ravel(Yr[:,-1,:,alvo])
            Yp_ = np.ravel(Yp[:,-1,:,alvo]) 
        else:
            Yr_ = np.ravel(Yr[...,alvo]) 
            Yp_ = np.ravel(Yp[...,alvo])
    
        
        plt.plot(Yr_, Yp_, "o", markersize=1)
        plt.plot(Yr_, Yr_, "k")
        plt.xlabel("Real") #, fontsize=16)
        plt.ylabel("Previsto")
        plt.title(f"Real vs. Previsto"+bs)
        plt.show()    

#####################################################################
# Flaten tensors (for XGboost)
#####################################################################

def tensor_flat(T, tempo_1o=True):
    """
    Retorna um tensor 2D com formato 'amostras x (tempo * atributos)'
    A segunda dimensão é ordenada primeiro por atributo depois por tempo.
    Ex:
    Um tensor no formato 'amostras x tempo x attributos', com atributos 'a' e 'b'
    T = [[['a1', 'b1'],
          ['a2', 'b2'],
          ['a3', 'b3'],
          ['a4', 'b4'],
          ['a5', 'b5']]]
    será transformado em:
    [['a1', 'a2', 'a3', 'a4', 'a5', 'b1', 'b2', 'b3', 'b4', 'b5']]

    Parâmetros:
    T: tensor com formato 'amostras x tempo x attributos' ou 'amostras x attributos x tempo'
    tempo_1o: bool; True (deafult), se T.shape = 'amostras x tempo x attributos'
    """
    
    if tempo_1o: T = tf.transpose(T, perm=[0,2,1]) # transposição para 'amostras x attributos x tempo'
    sh = T.shape
    T = tf.reshape(T, (sh[0],sh[1]*sh[2]))
    return T