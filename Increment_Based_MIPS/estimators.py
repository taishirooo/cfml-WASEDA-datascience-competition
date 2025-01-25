from copy import copy

import numpy as np
from sklearn.linear_model import LogisticRegression


def calc_avg(dataset: dict) -> np.ndarray:
    """AVG推定量を実行する."""
    return dataset["r"].mean()


def calc_dm(dataset: dict, pi: np.ndarray, q_hat: np.ndarray) -> float:
    """DM推定量を実行する."""
    return (q_hat * pi).sum(1).mean()


def calc_ips(dataset: dict, pi: np.ndarray, max_value: float = 100) -> float:
    """IPS推定量を実行する."""
    num_data = dataset["num_data"]
    a, r, pi_0 = dataset["a"], dataset["r"], dataset["pi_0"]

    idx = np.arange(num_data)
    w = pi[idx, a] / pi_0[idx, a]  # importance weights

    return np.minimum((w * r).mean(), max_value)


def calc_dr(
    dataset: dict, pi: np.ndarray, q_hat: np.ndarray, max_value: float = 100
) -> float:
    """DR推定量を実行する."""
    num_data = dataset["num_data"]
    a, r, pi_0 = dataset["a"], dataset["r"], dataset["pi_0"]

    idx = np.arange(num_data)
    w = pi[idx, a] / pi_0[idx, a]  # importance weights

    dr = (q_hat * pi).sum(1)  # direct method
    dr += w * (r - q_hat[idx, a])  # correction term

    return np.minimum(dr.mean(), max_value)


def calc_mips(
    dataset: dict,
    pi: np.ndarray,
    replace_c: int = 0,
    is_estimate_w: bool = False,
) -> float:
    """MIPS推定量を実行する."""
    num_data = dataset["num_data"]
    num_actions, num_clusters = dataset["num_actions"], dataset["num_clusters"]
    x, a, c, r = dataset["x"], dataset["a"], copy(dataset["c"]), dataset["r"]
    pi_0, phi_a = dataset["pi_0"], copy(dataset["phi_a"])
    min_value, max_value = r.min(), r.max()

    if replace_c > 0:
        c[c >= num_clusters - replace_c] = num_clusters - replace_c - 1
        phi_a[phi_a >= num_clusters - replace_c] = num_clusters - replace_c - 1

    if is_estimate_w:
        x_c = np.c_[x, np.eye(num_clusters)[c]]
        pi_a_x_c_model = LogisticRegression(C=5, random_state=12345)
        pi_a_x_c_model.fit(x_c, a)

        w_x_a_full = pi / pi_0
        pi_a_x_c_hat = np.zeros((num_data, num_actions))
        pi_a_x_c_hat[:, np.unique(a)] = pi_a_x_c_model.predict_proba(x_c)
        w_x_c_hat = (pi_a_x_c_hat * w_x_a_full).sum(1)

        return np.clip((w_x_c_hat * r).mean(), min_value, max_value)

    else:
        pi_0_c = np.zeros((num_data, num_clusters - replace_c))
        pi_c = np.zeros((num_data, num_clusters - replace_c))
        for c_ in range(num_clusters - replace_c):
            pi_0_c[:, c_] = pi_0[:, phi_a == c_].sum(1)
            pi_c[:, c_] = pi[:, phi_a == c_].sum(1)

        # 周辺重要度重み
        w_x_c = pi_c[np.arange(num_data), c] / pi_0_c[np.arange(num_data), c]

        return np.clip((w_x_c * r).mean(), min_value, max_value)
    
def calc_new_dr(dataset:dict, pi:np.ndarray, q1_hat:np.ndarray, q0_hat:np.ndarray) ->float:
    """
    IPS推定量

    Args:
        dataset (dict): _description_
        pi (np.ndarray): _description_

    Returns:
        float: _description_
    """
    
    pi_0 = dataset["pi_0"]
    
    ones = np.ones((dataset["num_data"],dataset["num_actions"]))
    a = dataset["a_mat"]
    not_a = ones-dataset["a_mat"]
    
    #r_a_1にかかるウエイトと，r_a_0にかかるウエイトを作成
    
    
    w_ips_1 = (pi*a) / (pi_0*a)  # n×|A|
    w_ips_0 =  (pi*not_a) / ((ones*not_a)-(pi_0*not_a)) # n×|A|
    
    w_ips_1 = np.nan_to_num(w_ips_1,nan=0)
    w_ips_0 = np.nan_to_num(w_ips_0, nan=0)
    
    dr =(dataset["r_mat"]-q1_hat*a)*w_ips_1 - (dataset["r_mat"]-q0_hat*not_a)*w_ips_0
    
    dr += (pi * (q1_hat - q0_hat))
    
    return dr.sum(1).mean()
    
    
    
    
def calc_new_ips_compe(dataset:dict, pi:np.ndarray) ->float:
    """
    IPS推定量

    Args:
        dataset (dict): _description_
        pi (np.ndarray): _description_

    Returns:
        float: _description_
    """
    
    pi_0 = dataset["pi_0"]
    
    ones = np.ones((dataset["num_data"],dataset["num_actions"]))
    a = dataset["a_mat"]
    not_a = ones-dataset["a_mat"]
    
    
    #r_a_1にかかるウエイトと，r_a_0にかかるウエイトを作成
    w_1 = pi*a / pi_0*a  # n×|A|
    w_0 = ((pi*not_a)) / ((ones*not_a)-(pi_0*not_a)) # n×|A|
    
    w_1 = np.nan_to_num(w_1,nan=0)
    w_0 = np.nan_to_num(w_0, nan=0)
    
    return (dataset["r_mat"]*w_1 - dataset["r_mat"]*w_0).sum(1).mean()
    
def calc_new_ips(dataset:dict, pi:np.ndarray) ->float:
    """
    IPS推定量

    Args:
        dataset (dict): _description_
        pi (np.ndarray): _description_

    Returns:
        float: _description_
    """
    
    pi_0 = dataset["pi_0"]
    
    r_mat = dataset["r_mat"]
    a_mat = dataset["a_mat"]
    not_a_mat = 1 - a_mat
    
    
    #r_a_1にかかるウエイトと，r_a_0にかかるウエイトを作成
    w_1 = (pi*a_mat) / pi_0  # n×|A|
    w_0 = (pi*not_a_mat)/ (1-pi_0) # n×|A|
    
    temp1 = (r_mat * w_1).sum(1)
    temp2 = (r_mat * w_0).sum(1)
    
    return (temp1 - temp2).mean()

def calc_new_mips(
    dataset: dict,
    pi: np.ndarray,
    replace_c: int = 0,
    is_estimate_w: bool = False,
) -> float:
    """MIPS推定量を実行する."""
    num_data = dataset["num_data"]
    num_actions, num_clusters = dataset["num_actions"], dataset["num_clusters"]
    x, a, c, r = dataset["x"], dataset["a"], copy(dataset["c"]), dataset["r"]
    pi_0, phi_a = dataset["pi_0"], copy(dataset["phi_a"])
    min_value, max_value = r.min(), r.max()
    a_mat, r_mat = dataset["a_mat"], dataset["r_mat"]
    c_mat = dataset["c_mat"]
    not_c_mat = 1 - c_mat
    a_not_mat = 1 - a_mat

    if replace_c > 0:
        c[c >= num_clusters - replace_c] = num_clusters - replace_c - 1
        phi_a[phi_a >= num_clusters - replace_c] = num_clusters - replace_c - 1

    if is_estimate_w:
        x_c = np.c_[x, np.eye(num_clusters)[c]]
        pi_a_x_c_model = LogisticRegression(C=5, random_state=12345)
        pi_a_x_c_model.fit(x_c, a)

        w_x_a_full = pi / pi_0
        pi_a_x_c_hat = np.zeros((num_data, num_actions))
        pi_a_x_c_hat[:, np.unique(a)] = pi_a_x_c_model.predict_proba(x_c)
        w_x_c_hat = (pi_a_x_c_hat * w_x_a_full).sum(1)

        return np.clip((w_x_c_hat * r).mean(), min_value, max_value)

    else:
        pi_0_c = np.zeros((num_data, num_clusters - replace_c))
        pi_c = np.zeros((num_data, num_clusters - replace_c))
        r_c_mat = np.zeros((num_data, num_clusters ))
        pi_0_c_2 = np.zeros((num_data, num_clusters - replace_c))
        for c_ in range(num_clusters - replace_c):
            pi_0_c[:, c_] = pi_0[:, phi_a == c_].sum(1)
            pi_0_c_2[:, c_] = (1-pi_0)[:, phi_a == c_].sum(1)
            pi_c[:, c_] = pi[:, phi_a == c_].sum(1)
            r_c_mat[:,c_] = r_mat[:,phi_a == c_].sum(1)

        # 周辺重要度重み
        w_x_c_1 = (pi_c*c_mat)/pi_0_c
        #w_x_c_0 = ( pi_c* not_c_mat)/ (1 - pi_0_c)
        w_x_c_0 = pi_c / pi_0_c_2
        
        #w_x_c = pi_c[np.arange(num_data), c] / pi_0_c[np.arange(num_data), c]
        
        
        temp1 = w_x_c_1.sum(1) * r
        #temp2 = (w_x_c_0*r_c_mat).sum(1)
        temp2 = np.zeros(num_data)
        r_not_mat = r_mat * a_not_mat
        for i in range(num_data):
            for a in range(num_actions):
                reward = r_not_mat[i,a] 
                temp2[i] += w_x_c_0[i,phi_a[a]] * reward
        
        return (temp1-temp2).mean()
    

def calc_new_offcem(
    dataset: dict,
    pi: np.ndarray,
    q_1_hat: np.ndarray,
    q_0_hat: np.ndarray,
    replace_c: int = 0,
    max_value: float = 100,
) -> float:
    """OffCEM推定量を実行する."""
    num_data = dataset["num_data"]
    num_actions, num_clusters = dataset["num_actions"], dataset["num_clusters"]
    x, a, c, r = dataset["x"], dataset["a"], copy(dataset["c"]), dataset["r"]
    pi_0, phi_a = dataset["pi_0"], copy(dataset["phi_a"])
    min_value, max_value = r.min(), r.max()
    a_mat, r_mat = dataset["a_mat"], dataset["r_mat"]
    c_mat = dataset["c_mat"]
    not_c_mat = 1 - c_mat
    a_not_mat = 1 - a_mat

    if replace_c > 0:
        c[c >= num_clusters - replace_c] = num_clusters - replace_c - 1
        phi_a[phi_a >= num_clusters - replace_c] = num_clusters - replace_c - 1

    pi_0_c = np.zeros((num_data, num_clusters - replace_c))
    pi_c = np.zeros((num_data, num_clusters - replace_c))
    r_c_mat = np.zeros((num_data, num_clusters ))
    pi_0_c_2 = np.zeros((num_data, num_clusters - replace_c))
    for c_ in range(num_clusters - replace_c):
        pi_0_c[:, c_] = pi_0[:, phi_a == c_].sum(1)
        pi_0_c_2[:, c_] = (1-pi_0)[:, phi_a == c_].sum(1)
        pi_c[:, c_] = pi[:, phi_a == c_].sum(1)
        r_c_mat[:,c_] = r_mat[:,phi_a == c_].sum(1)

    idx = np.arange(num_data)
    # 周辺重要度重み
    w_x_c_1 = (pi_c*c_mat)/pi_0_c
    w_x_c_0 = pi_c / pi_0_c_2
    
    temp1 = w_x_c_1.sum(1) * (r - q_1_hat[idx, a]) + (q_1_hat* pi).sum(1)
    
    temp2 = np.zeros(num_data)
    r_not_mat = r_mat * a_not_mat
    for i in range(num_data):
        for a in range(num_actions):
            reward = r_not_mat[i,a] 
            if reward == 0:
                continue
            temp2[i] += w_x_c_0[i,phi_a[a]] * (reward - q_0_hat[i,a]) + (q_0_hat[i,a] * pi[i,a])

    return (temp1 - temp2).mean()
