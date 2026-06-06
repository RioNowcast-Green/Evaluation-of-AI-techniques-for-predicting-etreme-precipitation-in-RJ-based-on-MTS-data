import math

# Angular differences

seno, cos = math.sin, math.cos

def ang_dif(a, b, graus=True):
    # a, b, a-b in degrees (default) or radians (graus=False)
    # Retorna o menor ângula
    if graus: #converte para rad para o cálculo
        a = math.radians(a)
        b = math.radians(b)
    dif_cos = cos(a)*cos(b)+seno(a)*seno(b)
    dif_ang = math.acos(round(dif_cos,15))

    if graus: dif_ang = math.degrees(dif_ang)
    
    return dif_ang


def difer_ang(x, lag=1, graus=True, verb=False):
    X = list(x)
    dif_ang=[]
    for t in range(len(X)):
        if t < lag: dif_ang.append(np.nan)               
        else: 
            a, b = X[t], X[t-lag]
            if verb: print(a, b)
            dif_ab = ang_dif(a,b,graus)
            dif_ang.append(dif_ab)
            if verb: print(dif_ab)
    return dif_ang 

# Dew Point calculation (Magnus Formula)
def ln(x): return math.log(x)
    
def Pto_orv(temp, umid):
    b, c = 17.67, 243.5
    ghama = ln(umid*0.01) + b*temp/(c+temp)
    return c*ghama/(b-ghama)