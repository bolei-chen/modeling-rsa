import numpy as np 
import warnings
warnings.simplefilter('ignore') 
  
columns = ["x","y","u_mean","v_mean","w_mean","p_mean","dissipation_mean","vorticity_mean", "uu","vv","ww","uv","uw","vw","pp"]
i2c = dict(zip(range(0, len(columns)), columns)) 
c2i = dict(zip(columns, range(0, len(columns)))) 
 
def load(file_path): 
    data = np.loadtxt(file_path, skiprows=20)
    return data
     
def select(corpus, c, x):
    if x == -1:
        return range(0, len(corpus))
    m = c2i[c] 
    x_nearest = 0 
    diff = np.inf
    for sample in corpus: 
        curr_diff = np.abs(x - sample[m]) 
        if (curr_diff < diff):
            x_nearest = sample[m] 
            diff = curr_diff 
    selected_is = [i for i in range(0, len(corpus)) if corpus[i, m] == x_nearest] 
    return selected_is 
 
def to_rst(corpus, i):
    delta = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]) 
     
    meanings = np.array([['uu', 'uv', 'uw'],
                        ['uv', 'vv', 'vw'],
                        ['uw', 'vw', 'ww']])
    s = corpus[i]
     
    t = np.array([[s[c2i[meanings[i][j]]] for j in range(0, 3)] for i in range(0, 3)])
    k = np.sum([t[i][i] for i in range(0, 3)]) / 2
     
    nt = np.zeros((3, 3))
    nt = t / (2 * k) - delta / 3 
     
    return nt 

def to_cs(lambdas): 
    c1 = lambdas[0] - lambdas[1]
    c2 = 2 * (lambdas[1] - lambdas[2]) 
    c3 = 3 * lambdas[2] + 1 
    return [c1, c2, c3] 

def to_lambdas(t):
    return np.sort(np.linalg.eig(np.nan_to_num(t))[0])[::-1]
    
def to_rsa(t):
    k = np.sum([t[i][i] for i in range(0, 3)]) / 2
    for i in range(0, 3):
        t[i][i] -= 2 * k / 3 
    return t 

def to_barcoords(cs):
    coords = np.array([[1, 0], [0, 0], [0.5, np.sqrt(3) / 2]])
    coords = np.array([coords[i] * cs[i] for i in range(0, len(coords))])
    return coords[0] + coords[1] + coords[2]

def to_cs_wrapper(corpus, i):
    return to_cs(to_lambdas(to_rsa(to_rst(corpus, i))))


