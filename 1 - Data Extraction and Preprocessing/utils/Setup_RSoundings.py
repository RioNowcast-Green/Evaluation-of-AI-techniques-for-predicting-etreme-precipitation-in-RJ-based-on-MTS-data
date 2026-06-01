import numpy as np
import pandas as pd

##### Radiosondes data handling and processing ######

# Note: Find the hole description of IGRA radiosondes data format in the following link: https://www1.ncdc.noaa.gov/pub/data/igra/data/igra2-data-format.txt

# Varible names (customized) and their respective column positions in the .txt file
Var_names = np.array([
['LVLTYP1', 1, 1], # major level type indicator
['LVLTYP2', 2, 2], # minor level type indicator
['dt', 4, 8], # elapsed time (MMMSS)
['Press', 10, 15], # Pressure (Pa)
['PFLAG', 16, 16], # pressure processing flag
['Alt', 17, 21], # Altitude (m)
['ZFLAG', 22, 22], # Alt. processing flag
['TEMP', 23, 27], # Temperature (°C*10)
['TFLAG', 28, 28], # Temp. processing flag
['Umid', 29, 33], # Relative Humidity (%*10)
['POv_dep', 35, 39], # Dew point depression (°C*10)
['Vento_dir', 41, 45], # wind direction (degrees)
['Vento_vel', 47, 51], # Wind speed (10*m/s) 
])

Header = np.array([
# ['HEADERC', 1, 1], # not used (header line)
# ['ID', 2, 12], # note used (ID line)
['Ano', 14, 17], # Year
['Mes', 19, 20], # Month
['Dia', 22, 23], # Day
['Hora', 25, 26], # Hour
['Hora_lanc', 28, 31], # Launch time (HHMM)
# ['Num_Elev', 33, 36], # not used (number of levels)
# ['P_SRC', 38, 45], # not used (pressure source)
# ['NP_SRC', 47, 54], # not used (source for number of pressure levels)
['Lat', 56, 62], # Latitude (degrees)
['Long', 64, 71], # Longitude (degrees)
])

# Headers keys
kHead = Header[:,0] 

# .txt columns start/end indexes in the header section of each sounding record. 
# -2 to adjust indexes, discounting the first columns (header rec indicator) and return 0-based indexes.
vHead = (Header[:,1:3]).astype(int)-2   

d_Head =  {k:v for k, v in zip(kHead, vHead)}

# Variables keys
kVar = Var_names[:,0]

# .txt columns start/end indexes in the data record section of each sounding record. 
# -1 to return 0-based indexes.
vVar = (Var_names[:,1:3]).astype(int)-1 # 

d_Var =  {k:v for k, v in zip(kVar, vVar)}

def read_sounding(file, ID):
    '''
    Reads sounding records from .txt file and returns a pandas DataFrame with the variables and launch timestamps as columns.
    '''
    with open(file, 'r', encoding="utf8") as f:
        data = f.read()
    
    entries = data.split("#") 
    columns = list(d_Head.keys()) +list(d_Var.keys())
    output = []
    for ent in entries: 
    
        rows = ent.split("\n")
        for r in rows:
            if r =='':
                pass
            elif r.startswith(ID): 
                Cols_Data = {}
                for k in d_Head.keys():
                    Cols_Data[k]=r[d_Head[k][0]:d_Head[k][1]+1]
            else: 
                Cols_Var = {}
                for k in d_Var.keys():
                    Cols_Var[k]=(r[d_Var[k][0]:d_Var[k][1]+1]).strip()
                output.append( list(Cols_Data.values()) +list (Cols_Var.values()))

    return pd.DataFrame(output, columns=columns)
