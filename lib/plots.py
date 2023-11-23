import numpy as np
from matplotlib import pyplot as plt 
from utils import * 
from qvalue_machine import * 
 
def plot_barcentric(corpus, x, marker='s', c='pink', edgecolors='green', s=5):
      
    corners = np.array([[1, 0], [0, 0], [0.5, np.sqrt(3) / 2]])
    triangle = plt.Polygon(corners, fill=False, color='green')
    plt.gca().add_patch(triangle) 
     
    if x != -1: 
        selected_is = select(corpus, 'x', x) 
        cvalues = np.array([get_cs(corpus, i) for i in range(len(corpus)) if i in selected_is])
    else:
        cvalues = np.array([get_cs(corpus, i) for i in range(len(corpus))])
         
    coords = cs2coords(cvalues) 
     
    xs, ys = zip(*coords)
    plt.scatter(xs, ys, marker=marker, c=c, edgecolors=edgecolors, s=s) 
    plt.title("barycentric triangle map") 
    plt.show() 
    return 0  

def evaluate_ffnn_tflow(corpus, x, model, ms, cs, edgecolors, s=5): 
    corners = np.array([[1, 0], [0, 0], [0.5, np.sqrt(3) / 2]])
    triangle = plt.Polygon(corners, fill=False, color='green')
    plt.gca().add_patch(triangle) 
     
    qvm = Qvalue_Machine(corpus) 
    selected_is = select(corpus, 'x', x) 
    qvalues = dict((i, torch.FloatTensor([qvm.q1(i), qvm.q4(i), qvm.q6(i), qvm.q7(i), qvm.q10(i)]).nan_to_num(0)) for i in selected_is) 
     
    cvalues_real = np.array([get_cs(corpus, i) for i in selected_is])  
    coords_real = cs2coords(cvalues_real) 
    xs_real, ys_real = zip(*coords_real) 
     
    cvalues_pred = [model.predict(qvalue) for _, qvalue in qvalues.items()] 
    coords_pred = cs2coords(cvalues_pred) 
    xs_pred, ys_pred = zip(*coords_pred) 

    for i in range(0, 10):
        print(cvalues_real[i])
        print(cvalues_pred[i]) 
        print('----------------') 

    plt.scatter(xs_real, ys_real, marker=ms[0], c=cs[0], edgecolors=edgecolors[0], s=s)
    plt.scatter(xs_pred, ys_pred, marker=ms[1], c=cs[1], edgecolors=edgecolors[1], s=s)
    plt.title('barycentric evaludation map') 
    plt.show() 
    return 0 


def evaluate_rf(corpus, x, model, ms, cs, edgecolors, s=5):
    corners = np.array([[1, 0], [0, 0], [0.5, np.sqrt(3) / 2]])
    triangle = plt.Polygon(corners, fill=False, color='green')
    plt.gca().add_patch(triangle) 
 
    qvm = Qvalue_Machine(corpus) 
    selected_is = select(corpus, 'x', x) 
    qvalues = dict((i, np.nan_to_num([qvm.q1(i), qvm.q4(i), qvm.q6(i), qvm.q7(i), qvm.q10(i)])) for i in selected_is) 

    cvalues_real = np.array([get_cs(corpus, i) for i in selected_is])  
    coords_real = cs2coords(cvalues_real) 
    xs_real, ys_real = zip(*coords_real) 
      
    cvalues_pred = [model.predict([qvalue])[0] for _, qvalue in qvalues.items()] 
    coords_pred = cs2coords(cvalues_pred) 
    xs_pred, ys_pred = zip(*coords_pred) 

    plt.scatter(xs_real, ys_real, marker=ms[0], c=cs[0], edgecolors=edgecolors[0], s=s)
    plt.scatter(xs_pred, ys_pred, marker=ms[1], c=cs[1], edgecolors=edgecolors[1], s=s)
    plt.title('barycentric evaludation map') 
    plt.show() 
    return 0 

     