#####################################################################
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path

from zipfile import ZipFile
import pandas as pd

from datetime import date, datetime, timedelta
import pytz
import h5py
#####################################################################


#####################################################################
# Split dataset by date blocks
#####################################################################

tz = pytz.utc

def ind_blocos(df, mes_i=7, dia_i=1, mes_f=None, dia_f=None, verb=False):
    """
    Returns indexes to divide blocks
    """ 

    dti, dtf = df.index.get_level_values(1).min(), df.index.get_level_values(1).max()
    ymin, ymax = dti.year, dtf.year
    # m_min, m_max = dti.month, dtf.month
    
    if mes_f is None: mes_f, dia_f = mes_i, dia_i
    
    d0 = max(dti, datetime(ymin, mes_i, dia_i, tzinfo=tz))
    
    if verb: print(f'Dataset date range: {dti} - {dtf} \nFirst block initial date: {d0}')

    Data_blocos = []     
    
    for ano in range(ymin, ymax+2):
        d1 = datetime(ano, mes_f, dia_f, tzinfo=tz)  
        if d1>d0 and d0<=dtf: Data_blocos.append([d0,d1])    
        d0 = max(dti, datetime(ano, mes_i, dia_i, tzinfo=tz))
        if d0 < d1: d0 = datetime(ano+1, mes_i, dia_i, tzinfo=tz)

    return Data_blocos

def div_blocos(df, nome_fonte = 'Fonte', estacao = None, **kw):
    """
    Splits dataset (df) in blocks, by informed dates.
    
    Parameters:
    df: Pandas DataFrame
    estacao: station list index (['01_GR', '02_SC', '09_MB', '10_VM', '11_FC', '12_JP'], None for all stations)
    kw: 'mes' (month) and 'dia' (day) to split data (default: mes=7, dia=1)
    """     

    if estacao is not None: df = df.xs(estacao, level=nome_fonte, drop_level=False)
    Blocos = []
    datas = ind_blocos(df, **kw)
    for dt in datas:
        d0, d1 = dt
        c1 = df.index.get_level_values(1)>=d0
        c2 = df.index.get_level_values(1)<d1
        bl = df[c1&c2]
        Blocos.append(bl)
    return Blocos    

#####################################################################
# Scaling functions
#####################################################################

# Standardization (x-men/std_dev)

def padr(x, media, desvio): return (x - media) / desvio  

# Normalization (max/min)

def norm(x, max_, min_, li=0, ls=1):  return li + (ls-li)*((x - min_)/(max_-min_))
   

#####################################################################
# Sliding windows
#####################################################################

def janelas(df, entrada, saida, horizonte, passo=1, batch=None, cols_alvo=None):
    
    '''
    Returns tf.datasets for input (X) and output (Y) sequences (sliding windows)

    X: DataFrame 
    entrada: input time length 
    saida: output time length 
    horizonte: output horizon
    passo: sliding stride
    cols_alvo: list - target attributes names
    '''
    
    janela = entrada + horizonte
    
    data = np.array(df, dtype=np.float32)
    
    J = keras.utils.timeseries_dataset_from_array(data, targets=None, 
                                              sequence_length=janela, 
                                              sequence_stride=passo,
                                              batch_size=batch)

    X = J.map(lambda x: x[...,:entrada,:])
    X = tf.data.Dataset.get_single_element(X.batch(len(X)))
    
    desloc = (janela-saida)

    if cols_alvo is None:
        Y = J.map(lambda y: y[...,desloc:,:])
    else:
        cols = list(df.columns) # feature names
        alvo_ind = [cols.index(x) for x in cols_alvo] # target feature indexes
        Y = J.map (lambda y: tf.stack([y[...,desloc:,ind] for ind in alvo_ind], axis=-1))

    Y = tf.data.Dataset.get_single_element(Y.batch(len(Y)))
    
    return X, Y

def janelas_seq2seq(df, entrada, horizonte, passo=1, cols_alvo=None, batch=None):
    
    '''
    Returns tf.datasets for input (X) and output (Y) sequences (sliding windows), seq-to-seq version

    X: DataFrame 
    entrada: input time length 
    horizonte: output horizon
    passo: sliding stride
    cols_alvo: list - target attributes names
    '''
    
    janela = entrada + horizonte
    
    data = np.array(df, dtype=np.float32)
    atributos = 1
    
    if len(data.shape)>1: atributos = data.shape[-1]
    
    J = keras.utils.timeseries_dataset_from_array(data, targets=None, 
                                              sequence_length=janela, 
                                              sequence_stride=passo,
                                              batch_size=batch)
    X = J.map(lambda x: x[...,:entrada,:])
    X = tf.data.experimental.get_single_element(X.batch(len(X)))

    Jy = tf.data.experimental.get_single_element(J.batch(len(J)))
    
    if cols_alvo is not None:
        cols = list(df.columns) # Feature names
        alvo_ind = [cols.index(x) for x in cols_alvo] # target feature(s) index(es)
        Jy = tf.gather( Jy, alvo_ind, axis=-1)
    
    Ind = tf.range(1,entrada+1)[:, tf.newaxis] + tf.range(horizonte)[tf.newaxis, :]
    Y = tf.gather(Jy, Ind, axis=1, batch_dims=0)
    print(X.shape, Y.shape)
 
    return X, Y

def criar_janela_fontes(df, entrada, horizonte, saida=None, passo=1, alvos=None, nome_fonte = 'Fonte', seq2seq =False):
    
    '''
    Returns tf tensors for input (X) and output (Y) sequences (sliding windows), regarding station
    
    nome_fonte: string - nome do índice com as fontes
    alvos: lista - nomes do(s) atributo(s) alvo(s)
    
    df: DataFrame, with meteorological station as one of the indexes 
    entrada: int - input time length 
    saida: int output time length 
    horizonte: int - output horizon
    passo: int -sliding stride
    alvos: list or None - target features
    nome_fonte: string - name of meteorological station index
    seq2seq: bool - ttpe of sequence: False (deafult) for seq-to-vetor; True for seq-to-seq
    '''
    Lx, Ly = [], []

    fi = df.index.names.index(nome_fonte)
    fontes = df.index.levels[fi]

    T_amos = 0
    atributos = list(df.columns)
        
    for f in fontes:
        try:
            filtro = df.xs(f, level=nome_fonte)
            
            tam = len(filtro)
            janela = entrada + horizonte
            n_amos = 1+ (tam-janela)//passo
            T_amos+=n_amos                      
            
            print(f'Fonte: {f}; timestamps: {tam}')
            print(f'Número de amostras: {n_amos}')
            print('------------------------------ \n')
            
            if seq2seq: x, y = janelas_seq2seq(filtro[atributos], entrada, horizonte, passo, cols_alvo = alvos)
            else: x, y = janelas(filtro[atributos], entrada, saida, horizonte, passo, cols_alvo = alvos)
    
            Lx.append(x)
            Ly.append(y)
        except: pass

    print(f'Número total de amostras: {T_amos}')

    Tx= tf.concat(Lx, axis=0)
    Ty= tf.concat(Ly, axis=0)

    print(f'Entradas: {Tx.shape}')
    print(f'Saídas: {Ty.shape}')

    return Tx, Ty

#####################################################################
# Balancear tensores X, Y
#####################################################################
 
def dividir_lotes(X, Y, lim=0, alvo=-1, batch_size=1024):
    """
    Divide a set of samples X,Y acording to the limit informed in Y tensor for the target variable
    Perform procedure in batches.
    """
    def div_lote(X_batch, Y_batch, lim, alvo):
    
        if len(Y_batch.shape) == 4:
            corte = Y_batch[:, -1, :, alvo] > lim  # for seq2seq, dimension 'time' is restriteda to the last sequence
        else:
            corte = Y_batch[..., alvo] > lim
    
        corte = tf.reduce_any(corte, axis=1)
    
        # Samples with values above limit
        Ysup_batch = tf.boolean_mask(Y_batch, corte, axis=0)
        Xsup_batch = tf.boolean_mask(X_batch, corte, axis=0)
    
        # Samples without values above limit
        Yinf_batch = tf.boolean_mask(Y_batch, ~corte, axis=0)
        Xinf_batch = tf.boolean_mask(X_batch, ~corte, axis=0)

        return Xsup_batch, Ysup_batch, Xinf_batch, Yinf_batch

    dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size)
    Xsup_list, Ysup_list, Xinf_list, Yinf_list = [], [], [], []

    print("Processando lotes")
    c=1
    d = len(dataset)
    for X_batch, Y_batch in dataset:
        p = round(100*c/d,2)
        Xsup, Ysup, Xinf, Yinf = div_lote(X_batch, Y_batch, lim, alvo)
        Xsup_list.append(Xsup)
        Ysup_list.append(Ysup)
        Xinf_list.append(Xinf)
        Yinf_list.append(Yinf)
        print(f'{p}%', end='\r')
        c+=1
    
    Xsup = tf.concat(Xsup_list, axis=0)
    Ysup = tf.concat(Ysup_list, axis=0)
    Xinf = tf.concat(Xinf_list, axis=0)
    Yinf = tf.concat(Yinf_list, axis=0)

    return Xsup, Ysup, Xinf, Yinf

def balancear_lotes(X, Y, lim=0, alvo=-1, bs=1024, fat=1):

    '''
    Procedes balancing in batches.
    '''

    # Divide 
    Xsup, Ysup, Xinf, Yinf = dividir_lotes(X, Y, lim=lim, alvo=alvo, batch_size=bs)
    
    # Subsample
    
    print("Balanceando dataset")
    
    if len(Ysup) < len(Yinf):
        indices = np.random.choice(len(Yinf), size=fat*len(Ysup), replace=False)
        Xinf = tf.gather(Xinf, indices)
        Yinf = tf.gather(Yinf, indices)
    else:
        indices = np.random.choice(len(Ysup), size=fat*len(Yinf), replace=False)
        Xsup = tf.gather(Xsup, indices)
        Ysup = tf.gather(Ysup, indices)

    # Concatenação
    Xf = tf.concat((Xinf, Xsup), 0)
    Yf = tf.concat((Yinf, Ysup), 0)

    return Xf, Yf

#####################################################################
# Export/Import tensors (h5)
#####################################################################

def exportar_tensores_h5(arquivo, tensores, nomes=None, opt=4):
    '''
    Export .h5 file with tensors (GZIP compression)
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
    Interprets an .h5 file and retrona a dicionary of tensors
    '''
    L = {}
    with h5py.File(arquivo, 'r') as file:
        for k in file.keys():
            print(k)
            L[k] = tf.constant(np.array(file[k]))
    return L

def importar_tensores_h5(arquivo):
    '''
    Interprets an .h5 file and retrona a dicionary of tensors.
    Unzip file if it is in .zip format
    '''

    if '.zip' in arquivo:
        with ZipFile(arquivo, 'r') as zipf:
            nome = zipf.namelist()[0]
            print(f'Extaindo {nome}')
            zipf.extract(nome)
            print('Lendo tensores para lista')
            tensores = ler_h5(nome)
        print(f'Excluindo {nome}')
        Path(nome).unlink() 
        
    else: tensores = ler_h5(arquivo)
    return tensores    


