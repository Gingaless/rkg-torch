
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
import numpy as np
from utils import list_map, arr_map
import warnings

#warnings.filterwarnings('ignore', category=RuntimeWarning)


def fit_KM(data, **KM_kwargs):
    KM = KMeans(**KM_kwargs)
    KM.fit(data)
    return KM


def find_KM_elbow_by_thr(data, ini_thr_cost_ratio, factor_function, data_mapping_function=None, **KM_kwargs):
    assert len(data) > 0
    _data = list_map(data_mapping_function, data) if data_mapping_function != None else data
    min_of_factor = np.min(list_map(factor_function, data))
    for i in range(1, len(data)+1):
        KM = fit_KM(_data, n_clusters=i, **KM_kwargs)
        if KM.inertia_/min_of_factor < ini_thr_cost_ratio:
            return KM
    raise Exception('There exists no k value of which cost ratio is lower than initial threshold cost raito.')


def find_KM_elbow_by_log_decay_argmin(data, ini_thr_cost_ratio, factor_function, data_mapping_function=None, **KM_kwargs):
    
    if len(data) < 3:
        return find_KM_elbow_by_thr(data, ini_thr_cost_ratio, factor_function, data_mapping_function=data_mapping_function, **KM_kwargs)
    else:
        _data = list_map(data_mapping_function, data) if data_mapping_function != None else data
        KMs = [fit_KM(_data, n_clusters=i, **KM_kwargs) for i in range(1, len(_data)+1)]
        log_costs = np.log(arr_map(lambda KM : KM.inertia_, KMs))
        log_costs = [x for x in log_costs if x != -np.inf]
        log_decay = [log_costs[i] - log_costs[i-1] for i in range(1, len(log_costs))]
        if np.nan in log_decay:
            log_decay = [x for x in log_decay if x!=np.nan]
        if -np.inf in log_decay:
            log_decay = [x for x in log_decay if x!=-np.inf]
        return KMs[np.argmin(log_decay)]

