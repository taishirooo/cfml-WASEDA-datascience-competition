import numpy as np
from sklearn.utils import check_random_state

from utils import sample_action_fast, softmax, eps_greedy_policy, sigmoid


def generate_synthetic_data(
    num_data: int,
    theta_g: np.ndarray,
    M_g: np.ndarray,
    b_g: np.ndarray,
    theta_h: np.ndarray,
    M_h: np.ndarray,
    b_h: np.ndarray,
    phi_a: np.ndarray,
    lambda_: float = 0.5,
    dim_context: int = 5,
    num_actions: int = 50,
    num_def_actions: int = 0,
    num_clusters: int = 3,
    beta: float = -3.0,
    random_state: int = 12345,
) -> dict:
    """行動特徴量を活用したオフ方策評価におけるログデータを生成する."""
    random_ = check_random_state(random_state)
    x = random_.normal(size=(num_data, dim_context))
    one_hot_a, one_hot_c = np.eye(num_actions), np.eye(num_clusters)

    # 期待報酬関数を定義する
    g_x_c = (
        (x - x ** 2) @ theta_g + (x ** 3 + x ** 2 - x) @ M_g @ one_hot_c + b_g
    ) / 10
    h_x_a = (
        (x ** 3 + x ** 2 - x) @ theta_h + (x - x ** 2) @ M_h @ one_hot_a + b_h
    ) / 10
    q_x_a = (1 - lambda_) * g_x_c[:, phi_a] + lambda_ * h_x_a

    # データ収集方策を定義する
    pi_0 = softmax(beta * q_x_a)
    pi_0[:, :num_def_actions] = 0
    pi_0 = pi_0 / pi_0.sum(1)[:, np.newaxis]

    # 行動や報酬を抽出する
    a = sample_action_fast(pi_0, random_state=random_state)
    q_x_a_factual = q_x_a[np.arange(num_data), a]
    r = random_.normal(q_x_a_factual)

    return dict(
        num_data=num_data,
        num_actions=num_actions,
        num_clusters=num_clusters,
        x=x,
        a=a,
        c=phi_a[a],
        r=r,
        phi_a=phi_a,
        pi_0=pi_0,
        g_x_c=(1 - lambda_) * g_x_c,
        h_x_a=lambda_ * h_x_a,
        q_x_a=q_x_a,
    )


def calc_true_value(
    dim_context: int,
    num_actions: int,
    num_clusters: int,
    theta_g: np.ndarray,
    M_g: np.ndarray,
    b_g: np.ndarray,
    phi_a: np.ndarray,
    theta_h: np.ndarray,
    M_h: np.ndarray,
    b_h: np.ndarray,
    lambda_: float,
) -> float:
    """評価方策の真の性能を近似する."""
    test_bandit_data = generate_synthetic_data(
        num_data=10000,
        dim_context=dim_context,
        num_actions=num_actions,
        num_clusters=num_clusters,
        theta_g=theta_g,
        M_g=M_g,
        b_g=b_g,
        theta_h=theta_h,
        M_h=M_h,
        b_h=b_h,
        lambda_=lambda_,
        phi_a=phi_a,
    )

    q_x_a = test_bandit_data["q_x_a"]
    pi = eps_greedy_policy(q_x_a)

    return (q_x_a * pi).sum(1).mean()



def generate_synthetic_data2(
    num_data: int,
    phi_a: np.ndarray,
    theta_g_0: np.ndarray,  # 以下 q0 の値
    M_g_0: np.ndarray,
    b_g_0: np.ndarray,
    theta_h_0: np.ndarray,
    M_h_0: np.ndarray,
    b_h_0: np.ndarray,
    theta_g_1: np.ndarray,  # 以下 q1 の値
    M_g_1: np.ndarray,
    b_g_1: np.ndarray,
    theta_h_1: np.ndarray,
    M_h_1: np.ndarray,
    b_h_1: np.ndarray,
    lambda1_: float,
    lambda0_: float,
    dim_context: int,
    num_actions: int,
    num_clusters: int,
    beta: float,
    num_def_actions: int = 0,
    random_state: int = 12345,
) -> dict:
    """行動特徴量を活用したオフ方策評価におけるログデータを生成する."""
    random_ = check_random_state(random_state)
    x = random_.normal(size=(num_data, dim_context))
    one_hot_a, one_hot_c = np.eye(num_actions), np.eye(num_clusters)
    
    # 期待報酬関数1を定義する
    g_x_c_1 = np.clip(
        ((x - x**2 - x**3) @ theta_g_1 + (x**3 + x**2) @ M_g_1 @ one_hot_c + b_g_1) / 10,
        0,
        None,
    )
    h_x_a_1 = np.clip(
        ((x**3 + x**2) @ theta_h_1 + (x - x**2 - x**3) @ M_h_1 @ one_hot_a + b_h_1) / 10,
        0,
        None,
    )
    q_x_a_1 = np.clip((1 - lambda1_) * g_x_c_1[:, phi_a] + lambda1_ * h_x_a_1, 0, None)

    # 期待報酬関数0を定義する
    g_x_c_0 = np.clip(
        ((x - x**2) @ theta_g_0 + (x**3 + x**2 - x) @ M_g_0 @ one_hot_c + b_g_0) / 10,
        0,
        None,
    )
    h_x_a_0 = np.clip(
        ((x**3 + x**2 - x) @ theta_h_0 + (x - x**2) @ M_h_0 @ one_hot_a + b_h_0) / 10,
        0,
        None,
    )
    q_x_a_0 = np.clip((1 - lambda0_) * g_x_c_0[:, phi_a] + lambda0_ * h_x_a_0, 0, None)

    # 期待報酬関数の差を CATE として定義
    cate_x_a = q_x_a_1 - q_x_a_0

    # データ収集方策を定義する
    pi_0 = softmax(beta * cate_x_a)
    pi_0[:, :num_def_actions] = 0
    pi_0 = pi_0 / pi_0.sum(1)[:, np.newaxis]

    # 行動や報酬を抽出する
    a = sample_action_fast(pi_0, random_state=random_state)
    a_mat = np.zeros((num_data, num_actions), dtype=int)
    a_mat[np.arange(num_data), a] = 1
    pscore_mat = a_mat * pi_0 + (1 - a_mat) * (1 - pi_0)

    q_x_a_factual = np.clip(a_mat * q_x_a_1 + (1 - a_mat) * q_x_a_0, 0, None)
    r_mat = np.clip(random_.normal(q_x_a_factual, scale=0.1), 0, None)  # 報酬を正値に制約

    c = phi_a[a]
    c_mat = np.zeros((num_data, num_clusters))
    cluster_num = np.zeros(num_data, int)
        
    for i in range(num_data):
        c_mat[i, c[i]] = 1
        cluster_num[i] = np.sum(phi_a == c[i])

    return dict(
        num_data=num_data,  # データの数
        num_actions=num_actions,  # アクション数
        num_clusters=num_clusters,
        x=x,  # 特徴量
        a=a,  # 行動
        c=c,
        c_mat=c_mat,
        r=r_mat[np.arange(num_data), a],
        a_mat=a_mat,
        r_mat=r_mat,
        phi_a=phi_a,
        pi_0=pi_0,
        pscore=pi_0[np.arange(num_data), a],
        pscore_mat=pscore_mat,
        g_x_c_1=(1 - lambda1_) * g_x_c_1,
        h_x_a_1=lambda1_ * h_x_a_1,
        g_x_c_0=(1 - lambda0_) * g_x_c_0,
        h_x_a_0=lambda0_ * h_x_a_0,
        q_x_a_1=q_x_a_1,
        q_x_a_0=q_x_a_0,
        cate_x_a=cate_x_a,
        cluster_num= cluster_num
    )
    
    
def generate_synthetic_data3(
    num_data: int,
    phi_a: np.ndarray,
    theta_g_0: np.ndarray,#以下q0の値
    M_g_0: np.ndarray,
    b_g_0: np.ndarray,
    theta_h_0: np.ndarray,
    M_h_0: np.ndarray,
    b_h_0: np.ndarray,
    theta_g_1: np.ndarray,  # 以下 q1 の値
    M_g_1: np.ndarray,
    b_g_1: np.ndarray,
    theta_h_1: np.ndarray,
    M_h_1: np.ndarray,
    b_h_1: np.ndarray,
    lambda1_: float ,
    lambda0_: float ,
    dim_context: int ,
    num_actions: int ,
    num_clusters: int ,
    beta: float ,
    num_def_actions: int = 0 ,
    random_state: int = 12345,
) -> dict:
    """行動特徴量を活用したオフ方策評価におけるログデータを生成する."""
    random_ = check_random_state(random_state)
    x = random_.normal(size=(num_data, dim_context))
    one_hot_a, one_hot_c = np.eye(num_actions), np.eye(num_clusters)
    
    # 期待報酬関数1を定義する
    g_x_c_1 = (
            (x - x ** 2 -x**3) @ theta_g_1 + (x ** 3 + x ** 2 ) @ M_g_1 @ one_hot_c + b_g_1
        ) / 10
    h_x_a_1 = (
            (x ** 3 + x ** 2 ) @ theta_h_1 + (x - x ** 2 - x**3) @ M_h_1 @ one_hot_a + b_h_1
        ) / 10
    q_x_a_1 = (1 - lambda1_) * g_x_c_1[:, phi_a] + lambda1_ * h_x_a_1

    #期待報酬関数０を定義する
    g_x_c_0 = (
            (x - x ** 2) @ theta_g_0 + (x ** 3 + x ** 2 - x) @ M_g_0 @ one_hot_c + b_g_0
        ) / 10
    h_x_a_0 = (
            (x ** 3 + x ** 2 - x) @ theta_h_0 + (x - x ** 2) @ M_h_0 @ one_hot_a + b_h_0
        ) / 10
    q_x_a_0 = (1 - lambda0_) * g_x_c_0[:, phi_a] + lambda0_ * h_x_a_0

    #期待報酬関数の差をCATEとして定義
    cate_x_a = q_x_a_1 - q_x_a_0

    #データ収集方策を定義する
    pi_0 = softmax(beta*cate_x_a)
    pi_0[:, :num_def_actions] = 0
    pi_0 = pi_0 / pi_0.sum(1)[:, np.newaxis]

    # 行動や報酬を抽出する
    a = sample_action_fast(pi_0, random_state=random_state)
    a_mat = np.zeros((num_data, num_actions), dtype=int)
    a_mat[np.arange(num_data), a] = 1
    pscore_mat = a_mat * pi_0 + (1 - a_mat) * (1 - pi_0)

    q_x_a_factual = a_mat * q_x_a_1 + (1 - a_mat) * q_x_a_0
    r_mat = random_.normal(q_x_a_factual)
    
    c = phi_a[a]
    c_mat = np.zeros((num_data,num_clusters))
    for i in range(num_data):
        c_mat[i,c[i]] = 1
    

    return dict(
        num_data=num_data, # データの数
        num_actions=num_actions, #アクション数
        num_clusters = num_clusters,
        x=x,  #特徴量
        a=a,   #行動
        c = c, 
        c_mat = c_mat,     
        r=r_mat[np.arange(num_data), a],
        a_mat=a_mat,
        r_mat=r_mat,
        phi_a = phi_a,
        pi_0=pi_0,
        pscore=pi_0[np.arange(num_data), a],
        pscore_mat=pscore_mat,
        g_x_c_1=(1 - lambda1_) * g_x_c_1,
        h_x_a_1=lambda1_ * h_x_a_1,
        g_x_c_0=(1 - lambda0_) * g_x_c_0,
        h_x_a_0=lambda0_ * h_x_a_0,
        q_x_a_1=q_x_a_1,
        q_x_a_0=q_x_a_0,
        cate_x_a=cate_x_a,
    )
    
def generate_synthetic_data4(
    num_data: int,
    phi_a: np.ndarray,
    theta_g_0: np.ndarray,#以下q0の値
    M_g_0: np.ndarray,
    b_g_0: np.ndarray,
    theta_h_0: np.ndarray,
    M_h_0: np.ndarray,
    b_h_0: np.ndarray,
    theta_g_1: np.ndarray,  # 以下 q1 の値
    M_g_1: np.ndarray,
    b_g_1: np.ndarray,
    theta_h_1: np.ndarray,
    M_h_1: np.ndarray,
    b_h_1: np.ndarray,
    lambda1_: float = 0.5,
    lambda0_: float = 0.5,
    dim_context: int = 5,
    num_actions: int = 50,
    num_def_actions: int = 0,
    num_clusters: int = 3,
    beta: float = -3.0,
    random_state: int = 12345,
) -> dict:
    """行動特徴量を活用したオフ方策評価におけるログデータを生成する."""
    random_ = check_random_state(random_state)
    x = random_.normal(size=(num_data, dim_context))
    one_hot_a, one_hot_c = np.eye(num_actions), np.eye(num_clusters)
    
    # 期待報酬関数1を定義する
    g_x_c_1 = (
            sigmoid((x - x ** 2) @ theta_g_1 + (x ** 3 + x ** 2 - x) @ M_g_1 @ one_hot_c + b_g_1
        ) )
    h_x_a_1 = (
            sigmoid((x ** 3 + x ** 2 - x) @ theta_h_1 + (x - x ** 2) @ M_h_1 @ one_hot_a + b_h_1
        ) )
    q_x_a_1 = (1 - lambda1_) * g_x_c_1[:, phi_a] + lambda1_ * h_x_a_1

    #期待報酬関数０を定義する
    g_x_c_0 = (
            sigmoid((x - x ** 2) @ theta_g_0 + (x ** 3 + x ** 2 - x) @ M_g_0 @ one_hot_c + b_g_0
        ) )
    h_x_a_0 = (
           sigmoid( (x ** 3 + x ** 2 - x) @ theta_h_0 + (x - x ** 2) @ M_h_0 @ one_hot_a + b_h_0
        ) )
    q_x_a_0 = (1 - lambda0_) * g_x_c_0[:, phi_a] + lambda0_ * h_x_a_0

    #期待報酬関数の差をCATEとして定義
    cate_x_a = q_x_a_1 - q_x_a_0
    
    
    #データ収集方策を定義する
    pi_0 = softmax(beta*cate_x_a)
    pi_0[:, :num_def_actions] = 0
    pi_0 = pi_0 / pi_0.sum(1)[:, np.newaxis]

    # 行動や報酬を抽出する
    a = sample_action_fast(pi_0, random_state=random_state)
    a_mat = np.zeros((num_data, num_actions), dtype=int)
    a_mat[np.arange(num_data), a] = 1
    pscore_mat = a_mat * pi_0 + (1 - a_mat) * (1 - pi_0)

    q_x_a_factual = a_mat * q_x_a_1 + (1 - a_mat) * q_x_a_0
    r_mat = random_.normal(q_x_a_factual)
    
    c = phi_a[a]
    c_mat = np.zeros((num_data,num_clusters))
    c_mat[np.arange(num_data), c] = 1
    

    return dict(
        num_data=num_data, # データの数
        num_actions=num_actions, #アクション数
        num_clusters = num_clusters,
        x=x,  #特徴量
        a=a,   #行動
        c = c, 
        c_mat = c_mat,     
        r=r_mat[np.arange(num_data), a],
        a_mat=a_mat,
        r_mat=r_mat,
        phi_a = phi_a,
        pi_0=pi_0,
        pscore=pi_0[np.arange(num_data), a],
        pscore_mat=pscore_mat,
        g_x_c_1=(1 - lambda1_) * g_x_c_1,
        h_x_a_1=lambda1_ * h_x_a_1,
        g_x_c_0=(1 - lambda0_) * g_x_c_0,
        h_x_a_0=lambda0_ * h_x_a_0,
        q_x_a_1=q_x_a_1,
        q_x_a_0=q_x_a_0,
        cate_x_a=cate_x_a,
    )   
    
def calc_true_value2(
    dim_context: int,
    num_actions: int,
    num_clusters: int,
    phi_a: np.ndarray,
    theta_g_0,
    M_g_0,
    b_g_0,
    theta_h_0,
    M_h_0,
    b_h_0,
    theta_g_1,
    M_g_1,
    b_g_1,
    theta_h_1,
    M_h_1,
    b_h_1,
    lambda0_,
    lambda1_,
    beta
):
    """評価方策の真の性能を近似する."""
    test_bandit_data = generate_synthetic_data4(
        num_data=10000,
        dim_context=dim_context,
        num_actions=num_actions,
        num_clusters=num_clusters,
        phi_a=phi_a,
        theta_g_0=theta_g_0,
        M_g_0=M_g_0,
        b_g_0=b_g_0,
        theta_h_0=theta_h_0,
        M_h_0=M_h_0,
        b_h_0=b_h_0,
        theta_g_1=theta_g_1,
        M_g_1=M_g_1,
        b_g_1=b_g_1,
        theta_h_1=theta_h_1,
        M_h_1=M_h_1,
        b_h_1=b_h_1,
        lambda0_=lambda0_,
        lambda1_=lambda1_,
        beta=beta,
    )

    cate_x_a = test_bandit_data["cate_x_a"]
    pi =eps_greedy_policy(cate_x_a)
    
    q_x_a_1 =test_bandit_data["q_x_a_1"]
    q_x_a_0 = test_bandit_data["q_x_a_0"]

    return ((q_x_a_1-q_x_a_0)*pi).sum(1).mean()
    #return (q_x_a_0*pi).sum(1).mean()
    #return (q_x_a_1*pi).sum(1).mean()

