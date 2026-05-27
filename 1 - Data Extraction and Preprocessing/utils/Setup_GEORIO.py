
#### bibliotecas
import pytz
import pandas as pd
import numpy as np
from datetime import date, datetime, time, timedelta, timezone
from matplotlib import pyplot as plt
import seaborn as sns
import math
import warnings
from dateutil import parser
import chardet
#########################



#########################
# Correction of Brazilian Daylight Saving Time (HBV) dates found in GeoRio data

# HBV start/end dates by year (from 1996 to 2019)
# Note: HBV stopped after 2018/2019 summer.

HBV_datas_str = [
['06/10/1996',	'16/02/1997'],
['06/10/1997',	'01/03/1998'],
['11/10/1998',	'21/02/1999'],
['03/10/1999',	'27/02/2000'],
['08/10/2000',	'18/02/2001'],
['14/10/2001',	'17/02/2002'],
['03/11/2002',	'16/02/2003'],
['19/10/2003',	'15/02/2004'],
['02/11/2004',	'20/02/2005'],
['16/10/2005',	'19/02/2006'],
['05/11/2006',	'25/02/2007'],
['14/10/2007',	'17/02/2008'],
['19/10/2008',	'15/02/2009'],
['18/10/2009',	'21/02/2010'],
['17/10/2010',	'20/02/2011'],
['16/10/2011',	'26/02/2012'],
['21/10/2012',	'17/02/2013'],
['20/10/2013',	'16/02/2014'],
['19/10/2014',	'22/02/2015'],
['18/10/2015',	'21/02/2016'],
['16/10/2016',	'19/02/2017'],
['15/10/2017',	'18/02/2018'],
['04/11/2018',	'17/02/2019']
]

### Verificar uso
def date_par(x, f='%Y/%m/%d %H%M %Z', tz='UTC', day_first=False): 
    if ':' in x:
        dt = parser.parse(x, dayfirst=day_first)
    else:
        dt = datetime.strptime(x, f)      
    if tz is not None:
        tzinfo = pytz.timezone(tz)
        dt = tzinfo.localize(dt)
    return dt  

HBV_df = pd.DataFrame(HBV_datas_str)
tz=None
HBV_df['inicio']=HBV_df.apply(lambda lin: date_par(lin[0], f='%d/%m/%Y', tz=tz), axis=1)
HBV_df['fim']=HBV_df.apply(lambda lin: date_par(lin[1], f='%d/%m/%Y',tz=tz), axis=1)
HBV_df=HBV_df.drop(columns=[0,1])
HBV_df['ano'] = HBV_df.apply(lambda lin: lin.inicio.year, axis=1)

def HBV_corrig(data_col):
    '''
    Corrects information of Brazilian Dayligth Saving Time (HBV), based on official HBV stard/end dates.
    Period considered: from 1996 to 2019.

    Input dates (data_col) must be tz-aware
    '''
    data_col = pd.Series(data_col) 
    
    HBV = []

    for d in range(len(data_col)):
        data = data_col[d]
        y=data.year
        rept = data_col[d] in data_col[:d].values

        # Dates after 2018/19 summer
        if data>= HBV_df.iloc[-1,1]: HBV.append(False) 

        # Dates before HBV ending per year
        elif data<HBV_df[HBV_df.ano==y-1]['fim'].item():
            if rept: HBV.append(False) # date repetitions occurs after summertime ends (clock returns  1 hour)
            else: HBV.append(True)
        
        elif y<2019: 
            #Data depois do começo do HBV (fi do ano y)
            if data>=HBV_df[HBV_df.ano==y]['inicio'].item(): HBV.append(True)
            else: HBV.append(False)
        else: HBV.append(False)
    return HBV

#########################

def concat_dfs_GeoRio(lista_arq, skip=0, sep='\s+', dec='.', verb=False, colunas=None, dados_loc=None):
    '''
    Concatenates GeoRio files ia a single DataFrame with standard variables names
    - Column names (variables) definition 
    - Date/Hour formatting
    - Numerical values formatting (to decimal '.')
    - Include coordinates metadata (lat./long./alt.)
    '''
    warnings.filterwarnings('ignore')
    
    df = pd.DataFrame()
    
    for arquivo, arq_ind in zip(lista_arq, range(len(lista_arq))):
                
        w = [12, 10, 6, 6, 10, 10, 13, 9, 10] # column widths
        data = pd.read_fwf(arquivo, widths=w, skiprows=skip, delim_whitespace=False).drop(0)
        
        # Concatenates date and hour
        
        Dt_HBV = list(map(lambda d, h: date_par(d + " " + h, day_first=True, tz=None), data.Dia, data.Hora)) #dates w/ brazilian summertime format
        data.insert(loc=0, column='Dt_Orig', value=Dt_HBV)
        
        #Verify HBV
        HBV_corrigido = HBV_corrig (Dt_HBV)
        data.insert(loc=1, column='HBV_cor', value=HBV_corrigido)
               
        data.drop(columns=['Dia', 'Hora'], inplace=True) 
        
        # Nomes de coluna padronizados
        if colunas is not None: data.columns = colunas
        
        # Metadados de localização
        if dados_loc is not None:
            lat, long, alt = dados_loc
            data.insert(loc=3, column='Lat', value=lat)  
            data.insert(loc=4, column='Long', value=long)
            data.insert(loc=5, column='Alt', value=alt)
        
        if verb:
            print(arq_ind, ': ', arquivo, 'linhas: ', len(data), 'colunas: ', len(data.columns)) 
            print(data.columns)

        df = pd.concat([df, data], ignore_index=True)

    return df

#########################

def percentile_inv(data, val):
    '''
    Returns the corresponding percentile of a given value ('val') in a set ('data').
    '''
    sort = np.sort(np.ravel(data))
    return np.interp(val, sort, np.linspace(0, 100, len(sort)))

def nanpercentile_inv(data, val):
    '''
    Returns the q-th percentile of  'val' in 'data' (disregarding NaN in 'data').
    '''
    sort = pd.Series(np.sort(np.ravel(data)))
    sort = sort.dropna()   
    return np.interp(val, sort, np.linspace(0, 100, len(sort)))

#########################

def difer(x, lag=1):
    ''' 
    differentiation (1st order)
    '''
    X = list(x)
    dif_x=[]
    for t in range(len(X)):
        if t < lag: dif_x.append(0)
        else: dif_x.append(X[t]-X[t-lag])    
    return dif_x    

def difer_d(x, d, lag=1):
    '''
    differentiation ('d' order)
    '''    
    X = list(x)
    dif_d = X
    for i in range(d):      
        dif_d = difer(dif_d, lag)   
    return dif_d  

def ret1(x, lag=1, lag_max=None):
    '''
    Return type 1 (relative differentiation)
    '''
    if lag_max==None: lag_max=len(x)
    dif_x=[]
    for t in range(len(x)):
        L=lag
        if t < L: dif_x.append(0)    
        else: 
            while pd.isna(x[t-L]) and lag<t and L<=lag_max: L+=1          
            dif_x.append((x[t]-x[t-L])/x[t-L])    
    return dif_x 

def ret2(x, lag=1):
    '''
     Return type 2 (logaritmical differentiation)
    '''
    log_x = list(map(lambda a: math.log(a), x))
    return difer(log_x, lag)

def difer_time(x, l=1):
    '''
    Differentiation of timestamp values. Output in seconds.
    x: a serires with series with continuous integer indices;
    l: lag index used in the difference operator
    '''
    dif_x=[]
    for t in range(len(x)):
        if t < l: dif_x.append(0)
        else: dif_x.append((x[t]-x[t-l]).total_seconds())    
    return np.array(dif_x)  


#########################

def Unicos (df):
    '''
    Return unique values of 'df' columns.
    '''
    Unicos = {}
    for col in df.columns:
        Unicos[col] = np.sort(list(df[col].unique()))
    return Unicos

def cont_NA(df):
    '''
    Counts occurences of NaN vales in 'df'.
    '''
    for col in df.columns:
        print(col, df[col].isna().sum())

#########################


def plot_periodo(df, col_dado, col_tempo, inicio=None, fim=None):
    '''
    Plot an specif attribute for a period of time
    '''
    plt.plot(df[col_tempo][inicio:fim], df[col_dado][inicio:fim], '-')
    plt.xticks(rotation=45)        

def plot_por_hora_fitro_dias(df, c_dado, c_dia, c_hora, **kwargs):
    '''
    Plot hourly variation, filterd by a period  of time
    '''
    c1 = df[c_dia]>=dia_ini
    c2 = df[c_dia]<=dia_fim
    
    if dia_ini<=dia_fim: df_filtrado = df[c1 & c2]
    else: df_filtrado = df[c1 | c2]    
    
    plt.plot(df_filtrado[c_hora], df_filtrado[c_dado], '.', **kwargs)
    
    stats = df_filtrado[c_dado].describe()
    med = stats['mean']
    dev = stats['std']
    p75 = stats['75%']
    
    plt.plot([0,24], [med, med])
    plt.plot([0,24], [dev, dev])
    plt.plot([0,24], [p75, p75])
    plt.title("Radiação por hora do dia - perído: dias "+ str(dia_ini) + " a "+ str(dia_fim) )

def graf_fltro_data (dado, c_Y, c_Data, di, df, c_Hora=None, hi=None, hf=None, form=':', endplot=True, Titulo=None, **kwargs):
    '''
    Plot an attribute by date and period of day
    '''

    c1 = dado[c_Data]>datetime.fromisoformat(di)
    c2 = dado[c_Data]<datetime.fromisoformat(df)
    c3 = c4 = None
    mod_titulo=False
    if Titulo is None:
        Titulo = "Dadas: " + di + " a " + df
        mod_titulo=True
    if c_Hora is not None:
        c3 = dado[c_Hora]>=hi
        c4 = dado[c_Hora]<hf
        filtro = dado[c1 & c2][c3 & c4]
        if mod_titulo: Titulo = Titulo + "Hora: " + str(hi) + "h a " + str(hf) + "h"  
    else: filtro = dado[c1 & c2]    
    plt.plot(filtro[c_Data], filtro[c_Y], form, **kwargs)
    plt.xticks(rotation=30)
    plt.title(Titulo)
    if endplot: plt.show()

def dist_hora(df, col_h, col_dado, step=2, p=98, max_=True):
    '''
    Plot PDFs for periods of hours of the day, highlighting a specified percentile, and min/max values
    '''
    grafs = int(24/step)
    fig, axes = plt.subplots(grafs,1, figsize=(10,30))
    
    #P98 = {}
    Perc = {}
    Filtros = {}
    nomes = []
    for h in range(0,24,step): 
        
        c1, c2 = df[col_h]>=h, df[col_h]<h+step
        filtro_hora = df[c1&c2]
        nome = str(h)+'-'+str(h+step)
        Filtros[nome]=filtro_hora
        nomes.append(nome)
        med = np.median(filtro_hora[col_dado].dropna())
        perc = np.percentile(filtro_hora[col_dado].fillna(med), p)
        if max_: m, m_lab = filtro_hora[col_dado].max(), 'max: '
        else:  m, m_lab = filtro_hora[col_dado].min(), 'min: '

        
        #P98[nome] = p98
        Perc[nome] = perc
            
        plt.sca(axes[int(h/step)])
        #plt.hist(filtro_hora.Rad)
        sns.kdeplot(filtro_hora[col_dado])
        #sns.kdeplot(filtro_hora.Rad.dropna())
        y_max = plt.gca().get_ylim()[1]
        #plt.plot([p98, p98],[0,y_max], 'r-', label = "P98: "+str(p98))
        plt.plot([perc, perc],[0,y_max], 'r--', label = "P"+str(p)+": "+str(perc))
        plt.plot(m, 0, 'bo', label =m_lab+str(m))
        plt.legend()
        plt.title(nome)
        plt.subplots_adjust(hspace=0.5)
    plt.show()
    return Perc, Filtros

def Trend_percent_periodo(Percentis, Filtros, cDado='Temp_Amb', cData='Data', zoom=False, dz=None):
    '''
    Plot trends by periods (hours of the day)
    '''
    for nome in Percentis.keys():
        plt.plot(Filtros[nome][cData], Filtros[nome][cDado], linewidth=0.5)
        plt.plot(plt.gca().get_xlim(),2*[Percentis[nome]])
        plt.ylabel(cDado)
        plt.title(nome)
        plt.xticks(rotation=30)
        if ~zoom and dz is not None:
            plt.gca().set_xlim(dz)
        plt.show()

def coluna_percentis(d_Percentis, d_Hora, step=2, nome='Perc'):
    '''
    Uses the percentile dict (return by 'dist_hora') to generete the corresponding percentiles (related to the timestamps in d_Hora).
    Returns a pd.Series to be used in plots.
    '''
    Perc= []                     
    for h in d_Hora:
        k=int(step*(h//step))
        key = str(k)+'-'+str(k+step)
        Perc.append(d_Percentis[key])
    return pd.Series(Perc,name=nome)  

#########################
