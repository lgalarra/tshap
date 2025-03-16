import numpy as np
from scipy.signal import find_peaks

def get_np_window_mask(series_length, window_start, window_length):
    mask = np.zeros((1, series_length))
    if window_start >= 0:
        mask[...,window_start:(window_start + window_length)] = 1
    else:
        mask[...,:(window_length + window_start)] = 1
    return mask

def tshap_window_explanation_pi(fnc, input_sample, baselines, window_length = 20, stride = 5, return_window_scores = False):    
    
    all_samples = np.array([input_sample])
    sl = input_sample.shape[-1]
    fs_exp = np.zeros(input_sample.shape)
    wd_pos = [ws for ws in range(0, sl-window_length + stride,stride)]
    wd_scores = np.zeros(len(wd_pos))
    #wd_scores = np.zeros(input_sample.shape)
    bsize = len(baselines)
    
    all_samples = np.vstack((all_samples, baselines))

    for ws in wd_pos:
        feature_mask = get_np_window_mask(sl, ws, window_length)
        for baseline in baselines:            
            fused_samples = np.array([        
            input_sample * (1 - feature_mask) + baseline * feature_mask,
            input_sample * feature_mask + baseline * (1 - feature_mask),                    
            ])
            all_samples = np.vstack((all_samples, fused_samples))

    #payout = clf.predict_proba(all_samples)[:,0]
    payout = fnc(all_samples)

    wd_scores = np.zeros(sl)

    for i in range(sl):
        if i%stride == 0 and i <= wd_pos[-1]:
            cur_pos = bsize + (i//stride)*bsize*2
            val = 0
            for ii in range(bsize):
                val += ((payout[0] - payout[cur_pos + ii*2 +1]) + (payout[cur_pos +ii*2+2] - payout[1 + ii]))/2
            val = val/bsize 
            if i > 0:
                wd_scores[i-stride:i+1] = np.linspace(wd_scores[i-stride],val,stride + 1)
            else:
                wd_scores[i] = val

    

    interpolated_exp = np.zeros(len(wd_scores))
    for i in range(len(wd_scores)):
        interpolated_exp[i: i + window_length] += wd_scores[i] / (window_length * min(i+1,stride))
    
    
    if return_window_scores:
        return interpolated_exp, np.arange(0, sl), wd_scores
    else:
        return interpolated_exp
    


            
        
def indi_window_shap_attrib(fnc, input_sample, baselines, window_start, window_end):
    all_samples = np.array([input_sample])
    sl = input_sample.shape[-1]    
    bsize = len(baselines)   
    all_samples = np.vstack((all_samples, baselines))

    
    feature_mask = get_np_window_mask(sl, window_start, window_end - window_start)
    for baseline in baselines:            
        fused_samples = np.array([        
        input_sample * (1 - feature_mask) + baseline * feature_mask,
        input_sample * feature_mask + baseline * (1 - feature_mask),                    
        ])
        all_samples = np.vstack((all_samples, fused_samples))

    payout = fnc(all_samples)
    val = 0
    for i in range(bsize):
        val += ((payout[0] - payout[bsize + i*2 +1]) + (payout[bsize +i*2+2] - payout[1 + i]))/2
    val = val/bsize        
    
    return val

def find_rois(wd_pos, wd_scores, window_len):
    abs_wd_scores = np.abs(wd_scores)
    rel_threshold = min(0.05*np.max(abs_wd_scores), np.percentile(abs_wd_scores,q=50))
    # print(0.1*np.max(abs_wd_scores))
    # print(np.percentile(abs_wd_scores,q=20))
    # rel_threshold = 0.05*np.max(abs_wd_scores)
    wd_count = len(wd_pos)
    zones = []
    i = 0

    while i < wd_count:
        if abs_wd_scores[i] > rel_threshold:        
            candidate_start = i
            candidate_end = i
            for ci in range(i + 1, wd_count):
                if abs_wd_scores[ci] > rel_threshold and wd_scores[ci] * wd_scores[candidate_start] > 0:
                    candidate_end = ci
                else:
                    break
            
            zones.append([candidate_start, candidate_end, 1 if wd_scores[candidate_start] > 0 else -1])       

        else:
            candidate_start = i
            candidate_end = i
            for ci in range(i+1, wd_count):
                if abs_wd_scores[ci] <= rel_threshold:
                    candidate_end = ci
                else:
                    break
            if candidate_end - candidate_start > 3:
                zones.append([candidate_start, candidate_end, 0])
        i = candidate_end + 1
        
    #print(zones)
    rois = []

    for i in range(len(zones)):
        if zones[i][2] != 0:
            if i == 0:
                last_zone = 0
            else:
                last_zone = zones[i-1][2]
            
            if i == len(zones) - 1:
                next_zone = 0
            else:
                next_zone = zones[i+1][2]
            
            if last_zone * zones[i][2] >= 0:
                roi_start= wd_pos[zones[i][0]] + window_len
            else:
                roi_start= wd_pos[zones[i][0]] + window_len//2
            
            if next_zone * zones[i][2] >= 0:
                roi_end = wd_pos[zones[i][1]]
            else:
                roi_end = wd_pos[zones[i][1]] + window_len//2
            
            if roi_end > roi_start:
                rois.append([roi_start, roi_end])
    return rois

def tshap_explanation_single_instance(fnc, input_sample, baselines, window_length = 20, stride = 5, roi = True):
    sm, window_pos, window_scores =tshap_window_explanation_pi(fnc, input_sample, baselines, window_length = window_length, stride = stride, return_window_scores=True)
    roi_sm = np.zeros(sm.shape)
    if roi:
        rois = find_rois(window_pos,window_scores,window_length)
        
        for r in rois:
            attrib = indi_window_shap_attrib(fnc, input_sample, baselines, r[0], r[1])
            roi_sm[...,r[0]:r[1] + 1] = attrib / (r[1] - r[0] + 1) 
    
    return sm, roi_sm

def tshap_explanation(fnc, X, baselines, window_length = 20, stride = 5, roi = True):
    if fnc.__name__ == 'predict_proba':
        final_fnc = lambda X: fnc(X)[:,0]
    else:
        final_fnc = fnc
        
    X_attribs = np.zeros(X.shape)
    X_roi_attribs = np.zeros(X.shape)
    for i in range(X.shape[0]):
        X_attribs[[i]], X_roi_attribs[[i]] = tshap_explanation_single_instance(final_fnc, X[i], baselines, window_length = window_length, stride = stride, roi = roi)
        

    
    return X_attribs, X_roi_attribs