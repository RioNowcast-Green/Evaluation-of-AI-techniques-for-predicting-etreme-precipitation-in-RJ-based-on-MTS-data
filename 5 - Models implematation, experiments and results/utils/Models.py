###################################

import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tsai.basics import *
from tsai.models.PatchTST import PatchTST 
import DLinear

###################################

#### FC-ANN (seq-to-vector) 

def FC_ANN(loss, i_shape, o_shape, cam = 2, unids=[32], act=['relu'], lr = 0.001, ub=True):
    # i_shape/o_shape, time x features (sem incluir batch)
    np.random.seed(42)
    tf.random.set_seed(42)

    saidas, alvos = o_shape
    
    li = keras.Input(shape=i_shape)
    l0 = keras.layers.Flatten()

    if len(act)==1: act = (cam+1)*act
    if len(unids)==1: unids = cam*unids

    L1 = []
    for c in range(cam):
        L1.append(keras.layers.Dense(units=unids[c], activation=act[c]))
                            
    l2 = keras.layers.Dense(saidas*alvos)
    lf = keras.layers.Reshape([saidas, alvos])

    camadas = [li, l0] + L1+ [l2, lf]
    model = keras.models.Sequential(camadas)
    
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    metrics = ['mae', 'mse']
    # if loss in metrics: metrics.remove(loss)
    
    model.compile(loss=loss, optimizer=optimizer,  metrics=metrics)
    return model

#### LSTM

def LSTM_vec(loss, i_shape, o_shape, cam = 2, unids=[32, 32], act=['tanh'], ld_act =None, ld_ki = 'glorot_uniform', metrics = ['mae', 'mse'], lr = 0.001, **kwargs):    
    np.random.seed(42)
    tf.random.set_seed(42)

    entradas, atributos = i_shape
    saidas, alvos = o_shape

    if len(act)==1: act = cam*act
    if len(unids)==1: unids = cam*unids
    
    li = keras.Input(shape=i_shape)

    LSTM = []
    rs=True
    for c in range(cam):
        if c == cam-1: rs = False
        LSTM.append(keras.layers.LSTM(units=unids[c], activation=act[c], return_sequences=rs, **kwargs))
    
    ld = keras.layers.Dense(saidas*alvos, activation=ld_act,  kernel_initializer = ld_ki)
    lf = keras.layers.Reshape([saidas, alvos])
   
    camadas = [li] + LSTM + [ld, lf]
    model = keras.models.Sequential(camadas)
    
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    if loss in metrics: metrics.remove(loss)
    
    model.compile(loss=loss, optimizer=optimizer,  metrics=metrics)
    return model
    
def LSTM_seq(loss, i_shape, o_shape, cam = 2, unids=[32, 32], act=['tanh'], metrics = ['mae', 'mse'],  ld_act =None, ld_ki = 'glorot_uniform', lr = 0.001, **kwargs):
    """
    
    Parâmeterss:
    loss: loss function (string ou class)
    i_shape: time (input) x features 
    o_shape: time(output) x features 
    cam: number of layers
    act: activation function
    ld_act/ld_ki: activation and initializer for dense layer    
    """
       
    np.random.seed(2025)
    tf.random.set_seed(2025)

    entradas, atributos = i_shape
    saidas, alvos = o_shape

    if len(act)==1: act = cam*act
    if len(unids)==1: unids = cam*unids
    
    li = keras.Input(shape=i_shape)

    LSTM = []
    for c in range(cam):
        LSTM.append(keras.layers.LSTM(units=unids[c], activation=act[c], return_sequences=True, **kwargs))
    
    ltd = keras.layers.TimeDistributed(keras.layers.Dense(saidas*alvos, activation=ld_act,  kernel_initializer = ld_ki))
    lf = keras.layers.Reshape([entradas, saidas, alvos])

    camadas = [li] + LSTM + [ltd, lf]
    model = keras.models.Sequential(camadas)
    
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    if loss in metrics: metrics.remove(loss)
    
    model.compile(loss=loss, optimizer=optimizer,  metrics=metrics)
    return model

#### 1D-CNN

def m1DCNN(loss, i_shape, o_shape,  cam = 2, filtros=[32], metrics = ['mae', 'mse'], act=['relu'], ks=3, lr=0.001, pd='same', dlr=None, 
           comp=True, seq2seq=True):
    
    np.random.seed(42)
    tf.random.set_seed(42)

    entradas, atributos = i_shape
    saidas, alvos = o_shape

    if dlr is None: dlr=cam*[1]
    if len(act)==1: act = (cam+1)*act
    if len(filtros)==1: filtros = cam*filtros
    
    li = keras.Input(shape=i_shape) # => (batch, time, features)
    CNN = []
    for c in range(cam):
        CNN.append(keras.layers.Conv1D(filtros[c], activation=act[c], kernel_size=ks, padding=pd, dilation_rate=dlr[c])) # => (batch, time, filters)
    # ld = (keras.layers.Dense(alvos*saidas, activation=act[-1]))
    
    if seq2seq:
        ld = (keras.layers.Dense(alvos*saidas, activation=act[-1]))
        lf = keras.layers.Reshape([entradas, saidas, alvos])
        camadas = [li] + CNN + [ld, lf]
    else: 
        ld = keras.layers.Dense(alvos, activation=act[-1])
        camadas = [li] + CNN + [ld]

    model = keras.models.Sequential(camadas)
    
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    if loss in metrics: metrics.remove(loss)
    
    if comp: model.compile(loss=loss, optimizer=optimizer,  metrics=metrics)
    return model
    
#### CNN-LSTM
    
def CNN_LSTM(loss, i_shape, o_shape, cam_cnn = 1, filtros = [32],  act_cnn=['relu'], cam_lstm = 1, unids=[32], act_lstm=['tanh'], ks=3, lr=0.001, pd='same',
             ki = 'glorot_uniform', **kwargs):
    np.random.seed(42)
    tf.random.set_seed(42)

    entradas, atributos = i_shape
    saidas, alvos = o_shape

    if len(act_cnn)==1: act_cnn = cam_cnn*act_cnn
    if len(filtros)==1: filtros = cam_cnn*filtros

    if len(act_lstm)==1: act_lstm = cam_lstm*act_lstm
    if len(unids)==1: unids = cam_lstm*unids    

    
    li = keras.Input(shape=i_shape)

    CNN = []
    for c in range(cam_cnn):
        CNN.append(keras.layers.Conv1D(filtros[c], kernel_size=ks, activation=act_cnn[c], padding=pd))  
    
    LSTM = []
    for c in range(cam_lstm):
        LSTM.append(keras.layers.LSTM(unids[c], return_sequences=True, activation=act_lstm[c], **kwargs))
    
    dense = keras.layers.TimeDistributed(keras.layers.Dense(saidas*alvos, kernel_initializer=ki))
    lf = keras.layers.Reshape([entradas, saidas, alvos])

    camadas = [li] + CNN + LSTM + [dense, lf]
    model = keras.models.Sequential(camadas)
    
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    metrics = ['mae', 'mse']

    if loss in metrics: metrics.remove(loss)
    
    model.compile(loss=loss, optimizer=optimizer,  metrics=metrics)
    # model.summary()
    return model

#### PatchTST

# Transform model to single attribute output
class Forecaster_1target(nn.Module):
    def __init__(self, modelo_multi, c_in, c_out=1, ch_first=True):
        super(Forecaster_1target, self).__init__()
        self.input = modelo_multi
        self.dense = nn.Linear(in_features=c_in, out_features=c_out) 
        self.ch_first = ch_first # Formato 'Channels First'

    def forward(self, x):
        # Passa a entrada pelo modelo existente
        x = self.input(x)

        if self.ch_first:
            x = x.permute(0, 2, 1) # Transforma a saída para a forma (amostras x tempo x atributos)
            x = self.dense(x) # Passa pela camada densa
            x = x.permute(0, 2, 1) # Transforma de volta para a forma (amostras x 1 x tempo)
        
        else: x = self.dense(x)
        
        return x

def PatchTST_1targ(**configs):
    
    model_in = PatchTST(**configs)
    model_out = Forecaster_1target(model_in, configs['c_in'], c_out=1)
    return model_out

class Configs:
    def __init__(self, seq_len, pred_len, enc_in, individual, kernel_size):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual
        self.kernel_size = kernel_size

#### DLinear

def DLinear_1targ(c_out = 1, **configs):
   
    Conf = Configs(**configs)
        
    model_in = DLinear.Model(Conf)
    model_out = Forecaster_1target(model_in, configs['enc_in'], c_out=c_out, ch_first=False)

    return model_out

#### Auto-Encoder

def CNN_encoder(i_shape, filtros, ks=3, ps=4,act='relu'):
    
    tempo, atributos = i_shape
    CNN_enc = []
    li = keras.Input(i_shape)
    L = len(filtros)
    for l in range(L):
        CNN_enc.append(keras.layers.Conv1D(filtros[l], activation=act, kernel_size=ks, padding='same')) # => (batch, time, filters))
        CNN_enc.append(keras.layers.MaxPool1D(ps, padding = 'same')) # => (batch, time/ps, filters))
    # lf = keras.layers.Flatten()
        
    return tf.keras.Sequential([li]+CNN_enc)

def CNN_decoder(o_shape, filtros, ks=3, ps=4, act='relu'):
    tempo, atributos = o_shape
    L = len(filtros)
    t0 = int(tempo/ps**L)
    
    input_dec = keras.Input([t0,filtros[0]])
    print(input_dec.shape)
    
    CNN_dec = []
    for l in range(len(filtros)): 
        CNN_dec.append(keras.layers.UpSampling1D(ps)) # => (batch, time*ps, filters))
        CNN_dec.append(keras.layers.Conv1D(filtros[l], activation=act, kernel_size=ks, padding='causal')) # => (batch, time, filters))

    ld = keras.layers.Dense(atributos, activation=act)
    

    return tf.keras.Sequential([input_dec]+CNN_dec+[ld])

class AutoEncoder(tf.keras.models.Model):
    def __init__(self, model_enc, model_dec, act='tanh'):
        super(AutoEncoder, self).__init__()
                         
        self.encoder = model_enc
        self.decoder = model_dec
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def perda_reconst(self, X, loss_f, bs=1024):
        '''
        Calcula a perda de reconstrução para uma função de perda pasada como argumento
        A função de perda (argumento) deve retornar um array na forma amostra X atributos
        Rwealiza o procedimento em lotes
        '''
        dataset = tf.data.Dataset.from_tensor_slices(X).batch(bs)
        c=1
        d = len(dataset)
        p = round(100*c/d,2)
        Loss_list = []
        for X_batch in dataset:
            p = round(100*c/d,2)
            Xpred = self.call(X_batch)
            Loss_list.append(loss_f(X_batch, Xpred))
            print(f'{p}%', end='\r')
            c+=1
        Loss = tf.concat(Loss_list, axis=0)
        return tf.reduce_mean(Loss, axis=1) 

    def classificar(X, loss_f, limiar):
        '''
        Calcula a perda de reconterução de acordo com a função pasaada pelo argumento 'loss'
        Classifica instâncias de X de acordo com a perda de reconstrução e o limiar
        '''
        Loss = self.perda_reconst(X, loss_f)
        return tf.math.greater(Loss, limiar)
    
    def reconst_MAE(self, X, bs=1024):
        '''
        Calcula a perda de reconstrução para uma função de perda MAE
        Rwealiza o procedimento em lotes
        '''
        dataset = tf.data.Dataset.from_tensor_slices(X).batch(bs)
        c=1
        d = len(dataset)
        p = round(100*c/d,2)
        Loss_list = []
        for X_batch in dataset:
            p = round(100*c/d,2)
            Xpred = self.call(X_batch)
            loss_ = tf.keras.losses.mae(X_batch, Xpred) 
            Loss_list.append(loss_)
            print(f'{p}%', end='\r')
            c+=1
        Loss = tf.concat(Loss_list, axis=0)
        return tf.reduce_mean(Loss, axis=1) 
    
    def reconst_MSE(self, X, bs = 1024):
        '''
        Calcula a perda de reconstrução para uma função de perda MSE
        Rwealiza o procedimento em lotes
        '''
        dataset = tf.data.Dataset.from_tensor_slices(X).batch(bs)
        c=1
        d = len(dataset)
        p = round(100*c/d,2)
        Loss_list = []
        for X_batch in dataset:
            p = round(100*c/d,2)
            Xpred = self.call(X_batch)
            loss_ = tf.keras.losses.mse(X_batch, Xpred) 
            Loss_list.append(loss_)
            print(f'{p}%', end='\r')
            c+=1
        Loss = tf.concat(Loss_list, axis=0)
        return tf.reduce_mean(Loss, axis=1)  


#####################################################################
# Métricas e Funções de Perda personalizadas
#####################################################################

# MSE personalizado (para  o último intervalo de tempo)

class wmse(tf.keras.losses.Loss):
    def __init__(self, Pesos={0:1.0}, name="wmse"):
        super().__init__(name=name)
        self.Pesos = Pesos

    def call(self, y_true, y_pred):
        mse = tf.square(y_true - y_pred)
        loss = mse

        for lim, p in self.Pesos.items():
            loss = tf.where((y_true>=lim)&(y_true>y_pred),p*mse,loss)

        return tf.reduce_mean(loss)
    
# Para uso no pacote XGBoost: 
def wmse_xgb(Pesos={0: 1.0}):
    def loss(preds, dtrain):
        labels = dtrain.get_label().reshape(preds.shape)
        grad = -2 * (labels - preds)
        hess = 2 * np.ones_like(preds)

        # Aplicar penalizações conforme os limiares
        for lim, p in Pesos.items():
            mask = (labels >= lim) & (labels > preds)
            grad[mask] *= p
            hess[mask] *= p

        return grad, hess
    return loss

def mse_xgb(preds, dtrain):
    labels = dtrain.get_label().reshape(preds.shape)
    grad = -2 * (labels - preds)
    hess = 2 * np.ones_like(preds)
    return grad, hess

def wmse_eval(Pesos):
    def feval(preds, dtrain):
        labels = dtrain.get_label().reshape(preds.shape)
        mse = (labels - preds) ** 2
        loss = mse.copy()

        for lim, p in Pesos.items():
            mask = (labels >= lim) & (labels > preds)
            loss[mask] = p * mse[mask]

        return 'wmse', float(np.mean(loss))
    return feval

def mse_eval(preds, dtrain):
    labels = dtrain.get_label().reshape(preds.shape)
    mse = (labels - preds) ** 2
    loss = mse.copy()
    return 'mse', float(np.mean(loss))
    
class torch_wmse(nn.Module):
    '''
    wmse para Pythorch
    y_true e y_pred com apenas um atributo
    Pesos: dict(limiar, peso)
    '''
    def __init__(self, Pesos={0:1.0}, tempo_1o=False):
        super(torch_wmse, self).__init__()
        
        self.Pesos = Pesos
        self.tempo_1o = tempo_1o
        self.name='wmse'

    def forward(self, y_true, y_pred):
        y_true = torch.tensor(y_true, dtype=torch.float32, requires_grad=True)
        y_pred = torch.tensor(y_pred, dtype=torch.float32, requires_grad=True)

        if self.tempo_1o:
            y_true = torch.transpose(y_true, 1,2)
            y_pred = torch.transpose(y_pred, 1,2)
        
        dif = y_true - y_pred
        mse = torch.square(dif)
        loss = mse

        # loss = torch.zeros(y_true.shape[1])
        
        for lim, p in self.Pesos.items():
            loss =  torch.where((y_true>=lim)&(y_true>y_pred),p*mse,loss)
        
        return torch.mean(loss)

#####################################################################

def build_model(Arch, loss_name, X_shape, Y_shape, ativ='relu', W=None):
    '''
    Build model according to architecture and parameters informed.
    
    Parameters: 
    Arch: str - chosen architecture ('FC-ANN', 'XGBoost', '1D-CNN', 'CNN-LSTM', 'LSTM');
    loss_name: str - loss function name;
    X_shape: list - shape of input tensor; 
    Y_shape: : list - shape of output tensor;
    ativ: str - activation function (default:'relu'), 
    W: list - weights for wMSE loss function.

    '''
    
    # Assert architecture
    if Arch not in ['FC-ANN', 'XGBoost', '1D-CNN', 'CNN-LSTM', 'LSTM']: 
        print ("Invalid model architecture.")
        return (None, None)

    # Loss Function
    if Arch == 'XGBoost': 
        if W is not None: Wmse = wmse_xgb(W)
        else: Wmse = None
        Loss = {'mse':mse_xgb, 'wmse': Wmse} 
        
    else:
        if W is not None: Wmse = wmse(W)
        else: Wmse = None
        Loss = {'mse': tf.keras.losses.MeanSquaredError(), 'wmse': wmse}  
        
    loss = Loss[loss_name] 

    # Shape
    if Arch in ['FC-ANN', 'XGBoost', 'DLinear', 'PatchTST']: # Models seq-to-vector
        ish = X_shape[1:]
        osh = Y_shape[1:]

    elif Arch in ['1D-CNN', 'CNN-LSTM', 'LSTM']: # Models seq-to-seq
        ish = X_shape[1:]
        osh = Y_shape[2:]

    # learning ratio
    lr = float(input("Enter learning ratio (default 0.001): ") or 0.001)

    if Arch == 'FC-ANN':
        cam = int(input("Enter number of layers (default 2): ") or 2)
        #  number of units per layer (as a list)
        unids = [int(x) for x in input("Enter number of units per layer (split by spaces) or a single number to repeat it for all layers (default: 1024): ").split()] or [1024],    
        
        params = {'layers': cam, 'units': unids, 'loss': loss_name}
        model = mdl.FC_ANN(loss, ish, osh, cam=cam, unids=unids, act=[ativ], lr=lr)

    elif Arch == 'XGBoost':
        Eval = {'mse': mse_eval, 'wmse': wmse_eval}
    
        params = {
            "max_depth": int(input("Enter tree max depth (default - 3): ") or 3),
            "learning_rate": lr,
            'eval_metric': ['mae'],

            'epoc': int(input("Enter number of epochs (default - 100): ") or 100),
            'pac': int(input("Enter patience (default - 20): ") or 20),
            'eval': Eval[loss_name]
            }
        model = None

    elif Arch == '1D-CNN':
        cam = int(input("Enter number of layers (default 8): ") or 8)
        filtros = int(input("Enter number of filters (default 32): ") or 32)
        filt= cam*[filtros]
        kernel = int(input("Enter kernel size (default 3): ") or 3) 
        pad= 'causal'

        params = {'layers': cam, 'filters': filt, 'kernel size': kernel, 'loss': loss_name}
        model = mdl.m1DCNN(loss, ish, osh, cam=cam,filtros=filt,act=[ativ], lr=lr,ks=kernel, pd=pad)

    elif Arch == 'LSTM':
        cam = int(input("Enter number of layers (default 2): ") or 2)
        unids = [int(x) for x in input("Enter number of units per layer (split by spaces) or a single number to repeat it for all layers (default: 32): ").split()] or [32]

        params = {'layers': cam, 'units': unids, 'loss': loss_name}
        model = mdl.LSTM_seq(loss, ish, osh, act=[ativ], cam=cam, unids=unids,  lr=lr, **configs)

    elif Arch == 'CNN-LSTM':
        cam_cnn = int(input("Enter number of 1D-CNN layers (default 2): ") or 2)
        filtros = int(input("Enter number of 1D-CNN filters (default 32): ") or 32)
        filt= cam_cnn*[filtros]
        kernel = int(input("Enter 1D-CNN kernel size (default 3): ") or 3) 
        pad= 'causal'

        cam_lstm = int(input("Enter number of LSTM layers (default 2): ") or 2)
        unids = [int(x) for x in input("Enter number of units per layer (split by spaces) or a single number to repeat it for all layers (default: 32): ").split()] or [32]    

        params = {'CNN layers': cam_cnn, 'CNN filters': filt, 'CNN kernel size': kernel, 'LSTM layers': cam,  'LSTM units': unids, 'loss': loss_name}
        model = CNN_LSTM(loss, ish, osh, cam_cnn=cam_cnn, filtros=filt, act_cnn=[ativ], cam_lstm=cam_lstm, unids=unids, act_lstm=[ativ], 
                      lr=lr,ks=kernel, pd=pad)

    model.summary()
    return model, params

def build_model_TSF(X, Y, Arch, loss_name, splits, ativ='relu', W=None, bs=64):
    '''
    Build model according to architecture and parameters informed.
    
    Parameters: 
    X: input tensor;
    Y: target tensor;
    Arch: str - chosen architecture ('DLinear', 'PatchTST');
    loss_name: str - loss function name;
    X_shape: list - shape of input tensor; 
    Y_shape: : list - shape of output tensor;
    ativ: str - activation function (default:'relu'), 
    W: list - weights for wMSE loss function.
    '''

    # from tsai.tslearner import TSForecaster
    
    # Assert architecture
    if Arch not in ['DLinear', 'PatchTST']: 
        print ("Invalid model architecture.")
        return (None, None)

    # Loss Function   
    if W is not None: Wmse = torch_wmse(W)
    else: Wmse = None 
    Loss = {'mse':nn.MSELoss(), 'wmse': Wmse}    
        
    loss = Loss[loss_name] 

    # Shape
    # Models seq-to-vector
    ish = X.shape[1:]
    osh = Y.shape[1:]   

    if Arch == 'DLinear':
        larg_mm = input("Enter moving avaeage width (default: 9): ") or 9 # used to obtaind trend componet of timeseries
        configs = dict(
            # Window
            seq_len=ish[0], # input window size
            pred_len=osh[0], # prediction horizon
            enc_in = ish[1], # number of channels (attributes)
            
            # Model
            individual=False, # Forecast the series individually or togheter (see DLinear documentation for detrails).
            
            # Decomposion
            kernel_size = larg_mm,
        )

        model_ = DLinear_1targ(**configs) 

        model = TSForecaster(X, Y, loss_func=loss, splits=splits, batch_size=bs, arch=model_, metrics=[mse, mae])

    if Arch == 'PatchTST':
        configs = dict(
            # Window
            c_in = ish[0],
            c_out = None,
            seq_len = ish[1],
            pred_dim = osh[1],

            # Model
            activation = ativ,
            patch_len= int(input("Enter patch length (deafult: 6) ")) or 6,  # length of the patch applied to the time series to create patches
            n_layers= int(input("Enter number of encoder layers (deafult: 3) ")) or 3, 
            stride= int(input("Enter stride (dafault: 1): ")) or 1,  # stride used when creating patches
            n_heads= int(input("Enter number of heads (default: 8): ")) or 8,  
            d_model= int(input("Enter dimension of model(default: 32): ")) or 32,  
            d_ff=int(input("Enter dimension of fully connected network (default: 254)")) or 254, 
        )   

        model_ = PatchTST_1targ(**configs) 

        model = TSForecaster(X, Y, loss_func=loss, splits=splits, batch_size=bs, arch=model_, metrics=[mse, mae]) 

    model.summary()
    return model, configs
