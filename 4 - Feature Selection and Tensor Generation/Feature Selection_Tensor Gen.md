# Preparing the data for model training -- Building Tensors 

The dataset generated through the previous procedures has the dimension  $time \times features$.   

To be processed by time series prediction models, the data must be presented as a set of samples in the format $samples \times time \times features$, in which each sample is a sequence 'X' with an input length 'N' for predicting a sequence 'Y' on the prediction horizon 'H' (where 'N' and 'H' are the 'time' dimension of the input and output sequences 'X' and 'Y').

In this file, the procedures for this purpose is presented.  
From the previous dataset, tensors will be generated for the steps of training, validation and test of the models, following the sequence below:

- Dataset split 
- Data standardization
- Feature selection
- Sequence formation: sliding window procedure
- Data balancing

**Note:** building or processing large tensors is memory consuming. The use of GPU is recommended, if it is possible.   
However, efforts have been made to enable CPU processing when a GPU is not available.
The sliding window tensor creation and data balancing procedures are performed in batches, the size of which can be adjusted by the user to suit the available hardware.

In the example below, tensors are created for a reduced portion of the dataset, considering only the station of Jacareagua with a subset of features (called SC3, with 18 of the 113 available features -- see part 4 - feature selection).   
In this case, all the process can be done using CPU.

# Settings

```python
import tensorflow as tf
import os
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) # config. for dynamic GPU allocation

import numpy as np
import pandas as pd
from datetime import datetime

# Be sure 'utils' folder is correctly placed in your project
from .utils import Tensor_Methods as mt
```

# 1 Import Data

```python
# dataset
    
Arq_parq = 'Dados Tratados.parquet'
dataset = pd.read_parquet(Arq_parq)
display(dataset)
```
<div style="overflow: scroll;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>Dia_ano</th>
      <th>s_dia</th>
      <th>c_dia</th>
      <th>Hora</th>
      <th>s_hora</th>
      <th>c_hora</th>
      <th>Lat</th>
      <th>Long</th>
      <th>Alt</th>
      <th>Vento_vel</th>
      <th>...</th>
      <th>Vento_y_200.0</th>
      <th>Vento_y_250.0</th>
      <th>Vento_y_300.0</th>
      <th>Vento_y_400.0</th>
      <th>Vento_y_500.0</th>
      <th>Vento_y_700.0</th>
      <th>Vento_y_850.0</th>
      <th>Vento_y_925.0</th>
      <th>Vento_y_1000.0</th>
      <th>dt_sond</th>
    </tr>
    <tr>
      <th>timestamp</th>
      <th>Dt_Hr</th>
      <th>Fonte</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.274803e+09</th>
      <th>2010-05-25 16:00:00+00:00</th>
      <th>01_GR</th>
      <td>145</td>
      <td>0.615285</td>
      <td>-0.788305</td>
      <td>13</td>
      <td>-0.258819</td>
      <td>-9.659258e-01</td>
      <td>-23.05028</td>
      <td>-43.594720</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1.274807e+09</th>
      <th>2010-05-25 17:00:00+00:00</th>
      <th>01_GR</th>
      <td>145</td>
      <td>0.615285</td>
      <td>-0.788305</td>
      <td>14</td>
      <td>-0.500000</td>
      <td>-8.660254e-01</td>
      <td>-23.05028</td>
      <td>-43.594720</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1.274810e+09</th>
      <th>2010-05-25 18:00:00+00:00</th>
      <th>01_GR</th>
      <td>145</td>
      <td>0.615285</td>
      <td>-0.788305</td>
      <td>15</td>
      <td>-0.707107</td>
      <td>-7.071068e-01</td>
      <td>-23.05028</td>
      <td>-43.594720</td>
      <td>0.0</td>
      <td>1.5</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1.274814e+09</th>
      <th>2010-05-25 19:00:00+00:00</th>
      <th>01_GR</th>
      <td>145</td>
      <td>0.615285</td>
      <td>-0.788305</td>
      <td>16</td>
      <td>-0.866025</td>
      <td>-5.000000e-01</td>
      <td>-23.05028</td>
      <td>-43.594720</td>
      <td>0.0</td>
      <td>1.9</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1.274818e+09</th>
      <th>2010-05-25 20:00:00+00:00</th>
      <th>01_GR</th>
      <td>145</td>
      <td>0.615285</td>
      <td>-0.788305</td>
      <td>17</td>
      <td>-0.965926</td>
      <td>-2.588190e-01</td>
      <td>-23.05028</td>
      <td>-43.594720</td>
      <td>0.0</td>
      <td>2.3</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1.732993e+09</th>
      <th>2024-11-30 19:00:00+00:00</th>
      <th>12_JP</th>
      <td>335</td>
      <td>-0.508671</td>
      <td>0.860961</td>
      <td>16</td>
      <td>-0.866025</td>
      <td>-5.000000e-01</td>
      <td>-22.94000</td>
      <td>-43.402778</td>
      <td>20.0</td>
      <td>1.6</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1.732997e+09</th>
      <th>2024-11-30 20:00:00+00:00</th>
      <th>12_JP</th>
      <td>335</td>
      <td>-0.508671</td>
      <td>0.860961</td>
      <td>17</td>
      <td>-0.965926</td>
      <td>-2.588190e-01</td>
      <td>-22.94000</td>
      <td>-43.402778</td>
      <td>20.0</td>
      <td>1.3</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1.733000e+09</th>
      <th>2024-11-30 21:00:00+00:00</th>
      <th>12_JP</th>
      <td>335</td>
      <td>-0.508671</td>
      <td>0.860961</td>
      <td>18</td>
      <td>-1.000000</td>
      <td>-1.836970e-16</td>
      <td>-22.94000</td>
      <td>-43.402778</td>
      <td>20.0</td>
      <td>0.5</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1.733004e+09</th>
      <th>2024-11-30 22:00:00+00:00</th>
      <th>12_JP</th>
      <td>335</td>
      <td>-0.508671</td>
      <td>0.860961</td>
      <td>19</td>
      <td>-0.965926</td>
      <td>2.588190e-01</td>
      <td>-22.94000</td>
      <td>-43.402778</td>
      <td>20.0</td>
      <td>0.7</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1.733008e+09</th>
      <th>2024-11-30 23:00:00+00:00</th>
      <th>12_JP</th>
      <td>335</td>
      <td>-0.508671</td>
      <td>0.860961</td>
      <td>20</td>
      <td>-0.866025</td>
      <td>5.000000e-01</td>
      <td>-22.94000</td>
      <td>-43.402778</td>
      <td>20.0</td>
      <td>0.3</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>896488 rows × 113 columns</p>
</div>

<br>

```python
# Date range per station
fontes = dataset.index.levels[2]
fontes
for f in fontes:
    ds_f = dataset.xs(f, level='Fonte', drop_level=False)
    print(f, ds_f.index.get_level_values(1).min(), ds_f.index.get_level_values(1).max())
```
> **Output:**   

    01_GR 2010-05-25 16:00:00+00:00 2024-12-23 05:00:00+00:00
    02_SC 2000-08-19 03:00:00+00:00 2024-12-23 05:00:00+00:00
    09_MB 2002-11-08 00:00:00+00:00 2024-11-30 23:00:00+00:00
    10_VM 2007-04-13 00:00:00+00:00 2024-11-30 23:00:00+00:00
    11_FC 2007-05-18 00:00:00+00:00 2024-11-30 23:00:00+00:00
    12_JP 2017-08-10 00:00:00+00:00 2024-11-30 23:00:00+00:00
    

```python
# Ranking FI

arq_rank = 'Feature_Importance_rank.xlsx'

FI_rank = pd.read_excel(arq_rank, index_col=0)
FI_rank
```

<div style="height: 500px; overflow-y: scroll;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank_FI</th>
      <th>Rank_Niv</th>
    </tr>
    <tr>
      <th>rank</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TSM</td>
      <td>Tpov_dif</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tpov_dif</td>
      <td>DTemp_03h</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DTemp_03h</td>
      <td>POv_dep_700.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DP_01h</td>
      <td>TSM</td>
    </tr>
    <tr>
      <th>4</th>
      <td>s_dia</td>
      <td>Vento_dv_03h</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Vento_dv_01h</td>
      <td>Vento_dv_01h</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Vento_ddir_01h</td>
      <td>Lat</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Vento_y</td>
      <td>DP_01h</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Vento_dv_03h</td>
      <td>DTemp_01h</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Rad</td>
      <td>DTemp_12h</td>
    </tr>
    <tr>
      <th>10</th>
      <td>POv_dep_700.0</td>
      <td>s_dia</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Vento_x</td>
      <td>Rad</td>
    </tr>
    <tr>
      <th>12</th>
      <td>c_dia</td>
      <td>s_hora</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Vento_y_925.0</td>
      <td>c_hora</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Vento_ddir_12h</td>
      <td>Temp_Amb</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Temp_Amb</td>
      <td>Vento_x_850.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Lat</td>
      <td>Vento_dv_06h</td>
    </tr>
    <tr>
      <th>17</th>
      <td>DP_18h</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>DTemp_06h</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>DTemp_24h</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Vento_dv_24h</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Vento_ddir_03h</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Vento_ddir_18h</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Vento_x_700.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Vento_ddir_06h</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>c_hora</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Vento_dv_18h</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>DP_24h</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Pres_Atm</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>TEMP_1000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30</th>
      <td>DTemp_01h</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31</th>
      <td>s_hora</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>32</th>
      <td>DP_03h</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Vento_y_850.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>34</th>
      <td>TEMP_100.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>35</th>
      <td>DTemp_12h</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Vento_dv_12h</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Vento_ddir_24h</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Vento_y_150.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Vento_dv_06h</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>40</th>
      <td>POv_Calc</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>41</th>
      <td>dt_sond</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

<br>

# 2 Split dataset

## 2.1 Set parameters for tensors generation

```python
# Stations: 0: '01_GR', 1: '02_SC', 2: '09_MB', 3: '10_VM', 4: '11_FC', 5: '12_JP' (None, to include all)
est = fontes[5]

# Scale type
esc= 'p' #  'n' for normalization (max/min) or 'p' for standardization (x-mean/std-dev)

## Data balancing options (0,1 or 2- see Part 6): 
n_bal = 1

## Feature subset (see part 4)
at_k = 'SC3'
alvo = ['Precip']

## Sliding windows parameters (hours):
  # e: input time length 
  # S: output time length 
  # h: output horizon
  # p: sliding stride

e, s, h, p = 48, 12, 12, 1 

# Sequence Type:
  # '0' dor seq-to-vector (ANN, XGBoost, PatchTST, DLinear)
  # '1' for 'seq-to-seq (LSTM, CNN-LSTM)
seq = 1 

# Batch size
bs=64
```

```python
# Display parameters:

p_est = [est if est is not None else 'All'][0]
print(f'Station: {p_est}')

print(f'Scale: {esc}, target: {alvo}, feature subset: {at_k}')
p_seq = ['Seq-to-seq' if seq else 'Seq-to-Vector'][0]
print(f'Window: {e} x {s} x {h} x {p} - {p_seq}')
```

    Station: 12_JP
    Scale: p, alvo: ['Precip'], feature subset: SC3
    Window: 48 x 12 x 12 x 1 - Seq-to-seq
    

## 2.2 Split Dataset in blocks

Generally, dataset with structured data are split based on percentage of length.   
However, the problem under study present some particularities.
Rainfall values has a strong seasonality, with major occurrences during summer.  
Splitting dataset without considering that could lead to sequences where occurrences of heavy rainfall would be interrupted.

Instead, the strategy used consists of first dividing the data into blocks, in which the split dates will always fall within the least rainy period (june/july).
So, when building the datasets for train, validation and test steps, each one will contain a number of entire blocks.   
Then, for each dataset, the tensors are built using the sliding window procedure.


```python

# Splittin data in Blocks

Blocos = mt.div_blocos(dataset, estacao = est)
    
print(f'Number of blocks: {len(Blocos)}')
```
> **Output:**   
>   
>`    Number of blocks: 8`
> 

```python
# Dates per block:
for i, b in enumerate(Blocos):
    V = b.index.get_level_values(1)
    print(i, V.min(), V.max())
```
> **Output:**   
```output
    0 2017-08-10 00:00:00+00:00 2018-06-30 23:00:00+00:00
    1 2018-07-01 00:00:00+00:00 2019-06-30 23:00:00+00:00
    2 2019-07-01 00:00:00+00:00 2020-06-30 23:00:00+00:00
    3 2020-07-01 00:00:00+00:00 2021-06-30 23:00:00+00:00
    4 2021-07-01 00:00:00+00:00 2022-06-30 23:00:00+00:00
    5 2022-07-01 00:00:00+00:00 2023-06-30 23:00:00+00:00
    6 2023-07-01 00:00:00+00:00 2024-06-30 23:00:00+00:00
    7 2024-07-01 00:00:00+00:00 2024-11-30 23:00:00+00:00
```
>

```python
# Feature names
atributos = list(dataset.columns)

# Max rainfall value per block
Precip_max = {}

for b in range(len(Blocos)):
    Precip_max[b] = Blocos[b].Precip.max()

Precip_max    
```
> **Output:**   
>
> `   {0: 80.2, 1: 52.2, 2: 43.4, 3: 25.4, 4: 53.2, 5: 45.6, 6: 32.0, 7: 24.6}`
>

```python
# Counting target extreme values (above a threshold)

lim = 25
Cont_ext = {}
for b in range(len(Blocos)):
    cond = Blocos[b].Precip>lim
    Cont_ext[b] = cond.sum()

df_Pmax= pd.DataFrame([Precip_max, Cont_ext], index=['Precip_max', 'Cont_ext']).T

df_Pmax
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precip_max</th>
      <th>Cont_ext</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>80.2</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>52.2</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>43.4</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25.4</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53.2</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>45.6</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>32.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>24.6</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>



## 2.3 Organize train, validation and test bases

Since buoys dataset starts in 2016, when TSM data is used (subsets SC3 or SC5, see part 4), only blocks from 2016 are taken.


```python
boias = 0
if at_k in ['SC3', 'SC5']: 
    boias = 1

    # Blocks from 2016 per station (indexes adjusted)
    if est == fontes[0]: sel_ind = [6, 7, 8, 9, 10, 11, 12, 13, 14] # GR
    elif est == fontes[2]: sel_ind = [14, 15, 16, 17, 18, 19, 20, 21, 22] # MB
    elif est in fontes[3:5]: sel_ind = [9, 10, 11, 12, 13, 14, 15, 16, 17] # VM, FC
    elif est == fontes[5]: sel_ind = list(range(len(Blocos))) #JP: Filter is not necessary
    else: sel_ind = [16,17,18,19,20,21,22,23,24] # All stations, SC3, SC5    

else:
  # When other feature subsets are chosen and no station is filtred, blocks with less than 5 precipitation instances above 25mm/h are discarded
    if est is None and alvo[0] == 'Precip': sel_ind= list(df_Pmax[df_Pmax.Cont_ext>=5].index)
    else: sel_ind = list(range(len(Blocos)))   
    
sel_ind
```
> **Outputs:**   
>`    [0, 1, 2, 3, 4, 5, 6, 7]`
>

**Selecting blocks per base**

```python
# Nº of blocks per base

l = len(sel_ind)
bts = round(l*.2) # test
bvl = round((l-bts)*.2) # validation
btr = l-bvl-bts # train

btr, bvl, bts
```
> **Outputs:**   
>`    (5, 1, 2)`
>


```python
# Blocks for train, validation and test bases

bl_tr = sel_ind[:btr]
bl_vl = sel_ind[btr:btr+bvl]
bl_ts = sel_ind[-bts:]
sel_bl = bl_tr, bl_vl, bl_ts
sel_bl
```
> **Outputs:**   
>`    ([0, 1, 2, 3, 4], [5], [6, 7])`
>


```python
# DataFrames per base
df_tr = pd.concat([Blocos[b] for b in bl_tr])
df_vl = pd.concat([Blocos[b] for b in bl_vl])
df_ts = pd.concat([Blocos[b] for b in bl_ts])
```

# 3 Data scaling

```python
# train base statistics

tr_data = pd.concat([Blocos[b] for b in bl_tr])
tr_min, tr_max = tr_data.min(), tr_data.max()
tr_med, tr_desv = tr_data.mean(), tr_data.std()
Li, Ls = 0,1

Tr_stats = pd.DataFrame([tr_min, tr_max, tr_med, tr_desv], index=['min','max','med', 'desv'])
Tr_stats
```

<div style = 'overflow: scroll;'>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dia_ano</th>
      <th>s_dia</th>
      <th>c_dia</th>
      <th>Hora</th>
      <th>s_hora</th>
      <th>c_hora</th>
      <th>Lat</th>
      <th>Long</th>
      <th>Alt</th>
      <th>Vento_vel</th>
      <th>...</th>
      <th>Vento_y_200.0</th>
      <th>Vento_y_250.0</th>
      <th>Vento_y_300.0</th>
      <th>Vento_y_400.0</th>
      <th>Vento_y_500.0</th>
      <th>Vento_y_700.0</th>
      <th>Vento_y_850.0</th>
      <th>Vento_y_925.0</th>
      <th>Vento_y_1000.0</th>
      <th>dt_sond</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>-0.999991</td>
      <td>-0.999963</td>
      <td>0.000000</td>
      <td>-1.000000e+00</td>
      <td>-1.000000e+00</td>
      <td>-22.940000</td>
      <td>-43.402897</td>
      <td>20.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-67.143523</td>
      <td>-65.586364</td>
      <td>-57.380815</td>
      <td>-48.682662</td>
      <td>-45.104195</td>
      <td>-27.551757</td>
      <td>-16.766694</td>
      <td>-20.070364</td>
      <td>-12.122035</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>366.00000</td>
      <td>0.999991</td>
      <td>1.000000</td>
      <td>23.000000</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>-22.939722</td>
      <td>-43.402778</td>
      <td>20.0</td>
      <td>7.700000</td>
      <td>...</td>
      <td>29.839906</td>
      <td>37.339881</td>
      <td>26.123455</td>
      <td>22.514000</td>
      <td>19.139161</td>
      <td>15.358221</td>
      <td>13.688828</td>
      <td>14.345204</td>
      <td>11.397925</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>med</th>
      <td>182.68813</td>
      <td>0.006695</td>
      <td>0.021471</td>
      <td>11.500000</td>
      <td>-1.276404e-17</td>
      <td>-5.549043e-17</td>
      <td>-22.939873</td>
      <td>-43.402827</td>
      <td>20.0</td>
      <td>0.790709</td>
      <td>...</td>
      <td>-20.145425</td>
      <td>-17.690864</td>
      <td>-15.134401</td>
      <td>-10.947588</td>
      <td>-7.435628</td>
      <td>-3.527190</td>
      <td>-0.759963</td>
      <td>0.529316</td>
      <td>0.781351</td>
      <td>6.379479</td>
    </tr>
    <tr>
      <th>desv</th>
      <td>106.54904</td>
      <td>0.712811</td>
      <td>0.701012</td>
      <td>6.922267</td>
      <td>7.071150e-01</td>
      <td>7.071150e-01</td>
      <td>0.000107</td>
      <td>0.000059</td>
      <td>0.0</td>
      <td>0.785425</td>
      <td>...</td>
      <td>16.343911</td>
      <td>14.942516</td>
      <td>13.070571</td>
      <td>10.053695</td>
      <td>7.925125</td>
      <td>5.704853</td>
      <td>3.968574</td>
      <td>3.690446</td>
      <td>2.974979</td>
      <td>5.230790</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 113 columns</p>

</div>

```python
# Save train stats for future use (experiment plots)

se = input('Salvar estatésticas (S/N)?')
Dir_exp = os.path.join(os.path.curdir, [est if est is not None else 'Todas'][0]+'/')
os.makedirs(Dir_exp, exist_ok=True)

if se.upper() == 'S':
    print(Dir_exp)

    Arquivo = 'Tr_stats_' +  [est[-2:]if est is not None else 'T'][0] + ['_boias' if boias else '_geral'][0] + '.parquet'
    print(Arquivo)
    Tr_stats.to_parquet(Dir_exp+Arquivo)
```
>**Outputs:**
>   
>`    .\12_JP/` <br>
>`    Tr_stats_JP_boias.parquet`
>   

```python
# Scaling blocks

Blocos_p = {k: v for k, v in zip(sel_ind, [Blocos[x] for x in sel_ind])}

if esc == 'p':
    for k in Blocos_p.keys():
        Blocos_p[k] = (Blocos_p[k]-tr_med)/tr_desv

if esc == 'n':
    for k in Blocos_p.keys():
        Blocos_p[k] = Li + (Ls-Li)*((Blocos_p[k]-tr_min)/(tr_max-tr_min))
```

# 4 Feature Selection

```python
# Feature names
fonte = ['Fonte']  
Data_Hora = ['Dt_Hr']

var_data = [ 'Dia_ano',  's_dia', 'c_dia', 'Hora', 's_hora', 'c_hora']
var_data_s = [ 's_dia', 'c_dia', 's_hora', 'c_hora']

var_loc = ['Lat', 'Long', 'Alt']
var_orig = ['Vento_vel', 'Vento_dir', 'Temp_Amb', 'Pres_Atm', 'Umidade',  'Rad', 'Precip']

var_boias = ['TSM', 'Dist_TSM']

var_calc  =  ['POv_Calc',  'Tpov_dif', 'Vento_x', 'Vento_y', 'Vento_dv_01h',  'Vento_dv_03h',  'Vento_dv_06h',  'Vento_dv_12h',  'Vento_dv_18h', 
             'Vento_dv_24h',  'Vento_ddir_01h',  'Vento_ddir_03h', 'Vento_ddir_06h', 'Vento_ddir_12h', 'Vento_ddir_18h', 'Vento_ddir_24h',
             'DP_01h',  'DP_03h', 'DP_06h', 'DP_12h', 'DP_18h', 'DP_24h', 'DTemp_01h', 'DTemp_03h', 'DTemp_06h', 'DTemp_12h', 'DTemp_18h', 'DTemp_24h', 'Precip_niv']

var_sonda = ['TEMP_100.0', 'TEMP_150.0', 'TEMP_200.0', 'TEMP_250.0', 'TEMP_300.0', 'TEMP_400.0', 'TEMP_500.0', 'TEMP_700.0', 'TEMP_850.0', 'TEMP_925.0',
             'TEMP_1000.0', 'POv_dep_100.0', 'POv_dep_150.0', 'POv_dep_200.0', 'POv_dep_250.0', 'POv_dep_300.0', 'POv_dep_400.0', 'POv_dep_500.0',
             'POv_dep_700.0', 'POv_dep_850.0', 'POv_dep_925.0', 'POv_dep_1000.0', 'Vento_dir_100.0', 'Vento_dir_150.0', 'Vento_dir_200.0', 'Vento_dir_250.0',
             'Vento_dir_300.0', 'Vento_dir_400.0', 'Vento_dir_500.0', 'Vento_dir_700.0', 'Vento_dir_850.0', 'Vento_dir_925.0', 'Vento_dir_1000.0', 'Vento_vel_100.0',
             'Vento_vel_150.0', 'Vento_vel_200.0', 'Vento_vel_250.0', 'Vento_vel_300.0', 'Vento_vel_400.0', 'Vento_vel_500.0', 'Vento_vel_700.0', 'Vento_vel_850.0',
             'Vento_vel_925.0', 'Vento_vel_1000.0', 'Vento_x_100.0', 'Vento_x_150.0', 'Vento_x_200.0', 'Vento_x_250.0', 'Vento_x_300.0', 'Vento_x_400.0', 'Vento_x_500.0',
             'Vento_x_700.0', 'Vento_x_850.0', 'Vento_x_925.0', 'Vento_x_1000.0', 'Vento_y_100.0', 'Vento_y_150.0', 'Vento_y_200.0', 'Vento_y_250.0',
             'Vento_y_300.0', 'Vento_y_400.0', 'Vento_y_500.0', 'Vento_y_700.0', 'Vento_y_850.0', 'Vento_y_925.0', 'Vento_y_1000.0', 'dt_sond']
```


```python
# Combined features
var_est_mod = ['s_dia', 'c_dia', 's_hora', 'c_hora','Temp_Amb', 'Pres_Atm', 'Umidade', 'Rad', 'Precip', 'POv_Calc','Tpov_dif', 'Vento_x', 'Vento_y']

var_vdv = ['Vento_dv_01h',  'Vento_dv_03h',  'Vento_dv_06h',  'Vento_dv_12h',  'Vento_dv_18h', 'Vento_dv_24h']
var_vdd = ['Vento_ddir_01h',  'Vento_ddir_03h', 'Vento_ddir_06h', 'Vento_ddir_12h', 'Vento_ddir_18h', 'Vento_ddir_24h']
var_dp = ['DP_01h',  'DP_03h', 'DP_06h', 'DP_12h', 'DP_18h', 'DP_24h']
var_dtemp = ['DTemp_01h', 'DTemp_03h', 'DTemp_06h', 'DTemp_12h', 'DTemp_18h', 'DTemp_24h']
```


```python
# Radiosounds features
sd_Temp = [x for x in var_sonda if 'TEMP' in x]
sd_POvD = [x for x in var_sonda if 'POv' in x]
sd_Vd = [x for x in var_sonda if 'Vento_dir' in x]
sd_Vv = [x for x in var_sonda if 'Vento_vel' in x]

sd_Vx = [x for x in var_sonda if 'Vento_x' in x]
sd_Vy = [x for x in var_sonda if 'Vento_y' in x]
```

**Note:**   
In this study, tensors were built considering 5 scenarious. In each one, models were trained using the followuing features:

- SC1: attributes from meteorological stations (total: 16 features)
- SC2: attributes from meteorological stations, adding engineered features (variations of features on dfferent lags) (total: 40 features)
- SC3: attributes from meteorological stations and meteorological buoys (total: 18 features)
- SC4: attributes from meteorological stations and radiosounds (total: 83 features)
- SC5: 15 best ranked attributes by Feature Importance (from Section 3 of this repository) plus target attribute.

```python
SC1 = var_est_mod + var_loc 
SC2 = SC1 + var_vdv + var_vdd + var_dp + var_dtemp 
SC3 = SC1 + var_boias 
SC4 = SC1 + var_sonda 
SC5 = list(FI_rank.Rank_FI[:15])+ alvo
Subsets={'SC1':SC1, 'SC2':SC2, 'SC3':SC3, 'SC4':SC4, 'SC5':SC5}
```

```python
# Feature selection

atrib = Subsets[at_k]

# When a station is filtered, location features (contants) are removed
if est is not None: 
    for x in var_loc+['Dist_TSM']:
        try: atrib.remove(x)
        except:pass

if alvo is not None: 
    alvo_i = [atrib.index(a) for a in alvo]
    print(alvo_i) #target column index

len(atrib), atrib, 
```
>**Outputs:**   
>
```output
    [8]

    (14,
     ['s_dia',
      'c_dia',
      's_hora',
      'c_hora',
      'Temp_Amb',
      'Pres_Atm',
      'Umidade',
      'Rad',
      'Precip',
      'POv_Calc',
      'Tpov_dif',
      'Vento_x',
      'Vento_y',
      'TSM'])
```
>

```python
# Subset statistics (scaled values) (means is nearly zero; std. dev. is nearly one)
Blocos_p[sel_ind[0]][atrib].describe().T
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>s_dia</th>
      <td>7800.0</td>
      <td>4.222310e-02</td>
      <td>1.035626</td>
      <td>-1.412275</td>
      <td>-1.081049</td>
      <td>2.309139e-01</td>
      <td>1.062264</td>
      <td>1.393490</td>
    </tr>
    <tr>
      <th>c_dia</th>
      <td>7800.0</td>
      <td>1.332988e-01</td>
      <td>0.946905</td>
      <td>-1.455817</td>
      <td>-0.750960</td>
      <td>2.197686e-01</td>
      <td>1.055094</td>
      <td>1.395881</td>
    </tr>
    <tr>
      <th>s_hora</th>
      <td>7800.0</td>
      <td>-1.281027e-18</td>
      <td>1.000052</td>
      <td>-1.414197</td>
      <td>-0.999988</td>
      <td>1.046455e-16</td>
      <td>0.999988</td>
      <td>1.414197</td>
    </tr>
    <tr>
      <th>c_hora</th>
      <td>7800.0</td>
      <td>-4.213154e-17</td>
      <td>1.000052</td>
      <td>-1.414197</td>
      <td>-0.999988</td>
      <td>-8.120192e-18</td>
      <td>0.999988</td>
      <td>1.414197</td>
    </tr>
    <tr>
      <th>Temp_Amb</th>
      <td>7800.0</td>
      <td>5.580425e-02</td>
      <td>0.974459</td>
      <td>-2.802039</td>
      <td>-0.607465</td>
      <td>2.602001e-02</td>
      <td>0.704754</td>
      <td>3.306568</td>
    </tr>
    <tr>
      <th>Pres_Atm</th>
      <td>7800.0</td>
      <td>-1.722031e-01</td>
      <td>0.965282</td>
      <td>-2.983343</td>
      <td>-0.875747</td>
      <td>-3.166701e-01</td>
      <td>0.500642</td>
      <td>2.694262</td>
    </tr>
    <tr>
      <th>Umidade</th>
      <td>7800.0</td>
      <td>-2.266322e-01</td>
      <td>1.017658</td>
      <td>-3.928429</td>
      <td>-0.929091</td>
      <td>1.557761e-01</td>
      <td>0.666302</td>
      <td>1.036433</td>
    </tr>
    <tr>
      <th>Rad</th>
      <td>7800.0</td>
      <td>1.612578e-02</td>
      <td>1.014738</td>
      <td>-0.706732</td>
      <td>-0.682683</td>
      <td>-5.954348e-01</td>
      <td>0.493419</td>
      <td>3.246176</td>
    </tr>
    <tr>
      <th>Precip</th>
      <td>7800.0</td>
      <td>-1.425168e-02</td>
      <td>1.090235</td>
      <td>-0.128707</td>
      <td>-0.128707</td>
      <td>-1.287071e-01</td>
      <td>-0.128707</td>
      <td>61.594319</td>
    </tr>
    <tr>
      <th>POv_Calc</th>
      <td>7800.0</td>
      <td>-2.022816e-01</td>
      <td>0.988946</td>
      <td>-4.987141</td>
      <td>-0.855851</td>
      <td>-1.086322e-01</td>
      <td>0.577899</td>
      <td>2.133875</td>
    </tr>
    <tr>
      <th>Tpov_dif</th>
      <td>7800.0</td>
      <td>2.031019e-01</td>
      <td>1.061929</td>
      <td>-0.904847</td>
      <td>-0.626096</td>
      <td>-2.209632e-01</td>
      <td>0.756321</td>
      <td>6.513424</td>
    </tr>
    <tr>
      <th>Vento_x</th>
      <td>7800.0</td>
      <td>9.426180e-03</td>
      <td>1.021832</td>
      <td>-6.372223</td>
      <td>-0.340567</td>
      <td>1.582082e-01</td>
      <td>0.311799</td>
      <td>4.224009</td>
    </tr>
    <tr>
      <th>Vento_y</th>
      <td>7800.0</td>
      <td>1.730047e-02</td>
      <td>1.064423</td>
      <td>-4.604466</td>
      <td>-0.314397</td>
      <td>1.018478e-01</td>
      <td>0.234917</td>
      <td>5.439866</td>
    </tr>
    <tr>
      <th>TSM</th>
      <td>7800.0</td>
      <td>-3.660972e-01</td>
      <td>1.291922</td>
      <td>-3.234472</td>
      <td>-1.332836</td>
      <td>-5.687087e-01</td>
      <td>0.742465</td>
      <td>2.934990</td>
    </tr>
  </tbody>
</table>


# 5 Sliding windows


```python
%%time
# Sliding windows per block
try: del Jan
except: pass

strategy = tf.distribute.MirroredStrategy() # usefull for GPU processing

with strategy.scope():

    Jan={}
    t0 = datetime.now()
    for b in sel_ind:

        print(f"Criando janejas para o bloco {b}:")
        print('')
        Jan[b] = mt.criar_janela_fontes(Blocos_p[b][atrib], e, h, s, p, alvos=alvo, seq2seq=seq)
        print('')
    t1 = datetime.now()    
```

>**Outputs: **
<div style='overflow-y:scroll; height: 500px'>

>
```output

    Criando janejas para o bloco 0:
    
    Fonte: 12_JP; timestamps: 7800
    Número de amostras: 7741
    ------------------------------ 
    
      (7741, 48, 14) (7741, 48, 12, 1)
    Número total de amostras: 7741
    Entradas: (7741, 48, 14)
    Saídas: (7741, 48, 12, 1)
    
    Criando janejas para o bloco 1:
    
    Fonte: 12_JP; timestamps: 8760
    Número de amostras: 8701
    ------------------------------ 
    
    (8701, 48, 14) (8701, 48, 12, 1)
    Número total de amostras: 8701
    Entradas: (8701, 48, 14)
    Saídas: (8701, 48, 12, 1)
    
    Criando janejas para o bloco 2:
    
    Fonte: 12_JP; timestamps: 8784
    Número de amostras: 8725
    ------------------------------ 
    
    (8725, 48, 14) (8725, 48, 12, 1)
    Número total de amostras: 8725
    Entradas: (8725, 48, 14)
    Saídas: (8725, 48, 12, 1)
    
    Criando janejas para o bloco 3:
    
    Fonte: 12_JP; timestamps: 8760
    Número de amostras: 8701
    ------------------------------ 
    
    (8701, 48, 14) (8701, 48, 12, 1)
    Número total de amostras: 8701
    Entradas: (8701, 48, 14)
    Saídas: (8701, 48, 12, 1)
    
    Criando janejas para o bloco 4:
    
    Fonte: 12_JP; timestamps: 8760
    Número de amostras: 8701
    ------------------------------ 
    
    (8701, 48, 14) (8701, 48, 12, 1)
    Número total de amostras: 8701
    Entradas: (8701, 48, 14)
    Saídas: (8701, 48, 12, 1)
    
    Criando janejas para o bloco 5:
    
    Fonte: 12_JP; timestamps: 8760
    Número de amostras: 8701
    ------------------------------ 
    
    (8701, 48, 14) (8701, 48, 12, 1)
    Número total de amostras: 8701
    Entradas: (8701, 48, 14)
    Saídas: (8701, 48, 12, 1)
    
    Criando janejas para o bloco 6:
    
    Fonte: 12_JP; timestamps: 8784
    Número de amostras: 8725
    ------------------------------ 
    
    (8725, 48, 14) (8725, 48, 12, 1)
    Número total de amostras: 8725
    Entradas: (8725, 48, 14)
    Saídas: (8725, 48, 12, 1)
    
    Criando janejas para o bloco 7:
    
    Fonte: 12_JP; timestamps: 3672
    Número de amostras: 3613
    ------------------------------ 
    
    (3613, 48, 14) (3613, 48, 12, 1)
    Número total de amostras: 3613
    Entradas: (3613, 48, 14)
    Saídas: (3613, 48, 12, 1)
    
    CPU times: total: 15.5 s
    Wall time: 3.53 s
```
>
</div>

```python
# Tensors per base
try:
    del Xtr, Ytr, Xvl, Yvl, Xts, Yts
except: pass

# Train
with strategy.scope():
    Xtr = tf.concat([Jan[x][0] for x in bl_tr],0)  
    Ytr = tf.concat([Jan[x][1] for x in bl_tr],0) 

Xtr.shape, Ytr.shape
```
>**Outputs:**   
>`    (TensorShape([42569, 48, 14]), TensorShape([42569, 48, 12, 1]))`
>


```python
# Vaidation
with strategy.scope():
    Xvl = tf.concat([Jan[x][0] for x in bl_vl],0)  
    Yvl = tf.concat([Jan[x][1] for x in bl_vl],0) 

Xvl.shape, Yvl.shape
```
>**Outputs:**   
>`    (TensorShape([8701, 48, 14]), TensorShape([8701, 48, 12, 1]))`
>
 

```python
# Test 
with strategy.scope():
    Xts = tf.concat([Jan[x][0] for x in bl_ts],0)  
    Yts = tf.concat([Jan[x][1] for x in bl_ts],0) 

Xts.shape, Yts.shape
```

>**Outputs:**   
>`    (TensorShape([12338, 48, 14]), TensorShape([12338, 48, 12, 1]))`
>    
 
# 6 Data Balancing

Data balancing procedure is executed after sliding windows formation (to avoid loss of temporal information in sequences).

The parameter `n_bal` defined in part 2.1 may be set as:

0 - no balacing is performed;

1 - data is subsampled to produce pairs of tensors X, Y (input/output) in wicht the Y tensor (prediction horizon) contains an equal number of samples with and without rainfall occurrences in their sequences.   
The procedure is performad in batches for train and validation bases. 

2 - data is subsampled to produce pairs of tensors X, Y (input/output) in wicht the Y tensor (prediction horizon) contains only samples with rainfall occurrences in their sequences (all sequences with no rainfall is discarded).   

```python
%%time
# target
variavel = 'Precip' 
a = alvo.index(variavel)

# Limit:
if n_bal >0:
  lim_bl = 0
  if esc== 'p': lim_blp = mt.padr(lim_bl, tr_med[variavel] ,tr_desv[variavel]) 
  if esc== 'n': lim_blp = mt.norm(lim_bl, tr_max[variavel] ,tr_min[variavel], Li, Ls) 

with strategy.scope():

    t2 = datetime.now()
    if alvo is not None and n_bal != 0: 

      if n_bal == 1: 

        # Train base
        print("Balanceamento da base de treino")
        Xtrb, Ytrb = mt.balancear_lotes(Xtr, Ytr, lim_blp, a) # balance data  
        print(f'X_tr: {Xtrb.shape}, Y_tr: {Ytrb.shape}')
        print("------")
        print("  ")
        del Xtr, Ytr

        # Validation base
        print("Balanceamento da base de validação")
        Xvb, Yvb  = mt.balancear_lotes(Xvl, Yvl, lim_blp, a)
        print(f'X_vl: {Xvb.shape}, Y_vl: {Yvb.shape}')
        del Xvl, Yvl    

      elif n_bal ==2:

        # Train base
        print("Balanceamento da base de treino")
        Xtr, Ytr, *_ = mt.dividir_lotes(Xtr, Ytr, lim_blp, a) # split data (only portion above 'lim' is taken)
        print(f'X_tr: {Xtr.shape}, Y_tr: {Ytr.shape}')
        print("------")
        print("  ")
                
        # Validation base
        print("Balanceamento da base de validação")
        Xvl, Yvl, *_  = mt.dividir_lotes(Xvl, Yvl, lim_blp, a)
        print(f'X_vl: {Xvl.shape}, Y_vl: {Yvl.shape}')   


    t3 = datetime.now()    
```
>**Outputs:**   
```output
    Balanceamento da base de treino
    Processando lotes
    Balanceando dataset
    X_tr: (25012, 48, 14), Y_tr: (25012, 48, 12, 1)
    ------
      
    Balanceamento da base de validação
    Processando lotes
    Balanceando dataset
    X_vl: (6448, 48, 14), Y_vl: (6448, 48, 12, 1)
    CPU times: total: 1.59 s
    Wall time: 569 ms
```    
>

```python
len(Xtrb), len(Xvb), len(Xts)    
```
>**Outputs:**   
>`    (25012, 6448, 12338)`
  

# 7 Export Tensors


```python

tensores = [Xtrb, Ytrb, Xvb, Yvb, Xts, Yts]

nomes_t = ['X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test']

nome = at_k + [est[-2:]if est is not None else 'T'][0] + ['_seq' if seq else '_vec'][0]

arquivo = Dir_exp + nome

with strategy.scope():

    t4 = datetime.now()
    mt.exportar_tensores_h5(arquivo, tensores, nomes=nomes_t,opt=9)
    t5= datetime.now()

print('\n')

print(t5)

```
>**Outputs:**   
```output

    Processando tensor 0,X_train, (25012, 48, 14)
    -----
    Processando tensor 1,Y_train, (25012, 48, 12, 1)
    -----
    Processando tensor 2,X_val, (6448, 48, 14)
    -----
    Processando tensor 3,Y_val, (6448, 48, 12, 1)
    -----
    Processando tensor 4,X_test, (12338, 48, 14)
    -----
    Processando tensor 5,Y_test, (12338, 48, 12, 1)
    -----
    
    
    2026-06-08 17:52:52.032475
 ```

