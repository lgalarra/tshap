import numpy as np
import matplotlib.pyplot as plt
#from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from numpy import dot
from numpy.linalg import norm

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d


# evaluation

def cos_sim(attribs_1,attribs_2):
    dot_product = attribs_1 * attribs_2
    return (attribs_1 * attribs_2).sum(axis=2) / (norm(attribs_1,axis=2) * norm(attribs_2,axis=2))

def cos_sim_avg(attribs_1,attribs_2):
    cs = cos_sim(attribs_1,attribs_2)
    cs[np.isnan(cs)] = 0
    return np.mean(cs)

def avg_precision(pred, truth):
    TP = np.sum(pred * truth > 0,axis = 2)
    TPFP = np.sum(pred != 0,axis = 2)
    TPFP[TPFP == 0] += 1
    return np.mean(TP / TPFP)

def avg_recall(pred, truth):
    TP = np.sum(pred * truth > 0,axis = 2)    
    TPFN = np.sum(truth != 0,axis = 2)
    TPFN[TPFN == 0] += 1
    return np.mean(TP / TPFN)

def avg_f1(pred, truth):
    TP = np.sum(pred * truth > 0,axis = 2)
    TPFP = np.sum(pred != 0,axis = 2)
    TPFN = np.sum(truth != 0,axis = 2)
    return np.mean(2 * TP / (TPFP + TPFN))

def wrap_predict(predict_fnc,X):
    if predict_fnc.__name__ == 'predict_proba':
        return predict_fnc(X)[:,1]
    else:
        return predict_fnc(X)

def perturb_data(X, X_attribs, r = 0.1, perturbation='invert'):
    noise_sigma = 0.2*(np.max(X) - np.min(X))    
    X_copy = X.copy()
    topk = int((1.0 - r) * X_copy.shape[-1])
    rank_attrib = np.argsort(np.argsort(X_attribs))
    relevant_mask = rank_attrib >= topk  
    relevant_mask[X_attribs == 0] = False  # only perturb if the attribution is positive to avoid rewarding wrong attribution
    
    for i in range(X_copy.shape[0]):
        if perturbation == 'invert':                    
            X_copy[i,relevant_mask[i,...]] = np.max(X_copy[i,...]) - X_copy[i,relevant_mask[i,...]]
        elif perturbation == 'zero':            
            X_copy[i,relevant_mask[i,...]] = 0
        elif perturbation == 'noise':
            X_copy[i,relevant_mask[i,...]] += np.random.normal(0,noise_sigma,relevant_mask[i,...].sum())
    
    return X_copy

def predict_pos_pert(model_fnc, X, X_attribs, r = 0.1, perturbation='invert'):
    #y_pred = wrap_predict(model_fnc,X)
    X_attribs_pos = X_attribs.copy()
    X_attribs_pos[X_attribs_pos < 0] = 0
    X_perturbed = perturb_data(X, X_attribs_pos, r=r, perturbation=perturbation)
    y_pred_prtb = wrap_predict(model_fnc,X_perturbed)
    return y_pred_prtb

def predict_neg_pert(model_fnc, X, X_attribs, r = 0.1, perturbation='invert'):
    #y_pred = wrap_predict(model_fnc,X)
    X_attribs_neg = X_attribs.copy()
    X_attribs_neg[X_attribs_neg > 0] = 0
    X_perturbed = perturb_data(X, np.abs(X_attribs_neg), r=r, perturbation=perturbation)
    y_pred_prtb = wrap_predict(model_fnc,X_perturbed)
    return y_pred_prtb

def faithful_eval(model_fnc, X, X_attribs, r = 0.1, perturbation='zero'):
    fe_score = np.mean(predict_neg_pert(model_fnc,X,X_attribs, r=r, perturbation=perturbation) - 
                       predict_pos_pert(model_fnc,X,X_attribs, r=r, perturbation=perturbation))
    return fe_score

# visualisation

def plot_saliency_map_and_attributions(sample, attributions, attribs_names ,title = 'Saliency map'):

	# if len(sample.shape) == 1: # if the univariate input is a 1D array
	# 	sample = np.expand_dims(sample, axis=0)
	# 	attribution = np.expand_dims(attribution, axis=0)

    n_attribs = len(attributions)

    x = np.array([ii for ii in range(sample.shape[-1])])

	


    fig, axs = plt.subplots(n_attribs, 2, sharex=True, figsize=(12, 1.5*n_attribs),constrained_layout=True)

    for p in range(n_attribs):
        y = sample
        sy = attributions[p]

        cap = max(abs(sy.min()), abs(sy.max()),0.0001)
        cvals = [-cap, 0, cap]
        # if saliency.min() < 0:
        #     cvals  = [saliency.min(), 0, saliency.max()]
        # else:
        #     cvals  = [0,0, saliency.max()]
        colors = ["blue","gray","red"]
        norm=plt.Normalize(min(cvals),max(cvals))
        tuples = list(zip(map(norm,cvals), colors))
        cmap = LinearSegmentedColormap.from_list("", tuples)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(sy)
        lc.set_linewidth(2)


        current_ax = axs[p][0]

        line = current_ax.add_collection(lc)
        current_ax.set_xlim(x.min(), x.max())
        current_ax.set_ylim(y.min() - 1, y.max()+1)
        if len(attribs_names) >= n_attribs:
            current_ax.set_ylabel(attribs_names[p])

        axs[p][1].plot(x,sy)
        axs[p][1].axhline(0.0, linestyle='dotted', color='red')



    

    fig.align_ylabels(axs)



    plt.show()

