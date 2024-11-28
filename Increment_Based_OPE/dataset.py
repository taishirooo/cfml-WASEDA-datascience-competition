import numpy as np
from sklearn.utils import check_random_state
from scipy.stats import rankdata

from utils import sample_action_fast, sigmoid, softmax, eps_greedy_policy


def generate_synthetic_data(
    num_data: int, #生成するデータ数
    dim_context: int, #コンテキストの次元数
    num_actions: int,# 使用できるアクション数
    T: int,  #各データにおけるタイムステップ数
    theta: np.ndarray,  # d x |A|  #コンテキストとアクションに対する線形結合の重みパラメータ
    M: np.ndarray,  # d x |A|    コンテキストに基づく期待報酬関数に関わるパラメータ
    b: np.ndarray,  # |A| x 1    アクション固有のバイアス項で、num_actions x 1 の形状を持つ。
    W: np.ndarray,  # T x T     時間ステップ間の関係をモデル化するための行列で、T x T の形状を持ちます。
    eps: float = 0.0,       #ε-greedy方策における探索率を指定します。ランダムにアクションを選択する確率を決めます。
    beta: float = 1.0,     #ソフトマックス方策で使われる温度パラメータ。beta が大きいとより決定的な行動選択になります。
    reward_noise: float = 0.5,     #報酬に対するノイズの大きさを指定します。報酬の生成にランダムなばらつきを与える要因です。
    p: list = [0.0, 1.0, 0.0],     #ユーザーの行動タイプを選択する確率を指定します。[独立, カスケード, 全アクション] という形で、各行動タイプの選択確率を指定します。
    p_rand: float = 0.2,        #ユーザーのランダム行動が選択される確率を指定します。
    is_online: bool = False,     #オンライン環境かどうかを指定します。オンラインであれば、時刻ごとに異なる行動方針が選ばれます。
    random_state: int = 12345,
) -> dict:
    """オフ方策評価におけるログデータを生成する."""
    random_ = check_random_state(random_state)
    #x は正規分布に従うコンテキスト（特徴量ベクトル）のデータセットを生成, e_a はアクションの単位行列
    x, e_a = random_.normal(size=(num_data, dim_context)), np.eye(num_actions)

    # 期待報酬関数を定義する
    #シグモイド関数により、期待報酬は0から1の範囲に変換され、確率的な評価が行われます。
    base_q_func = (
        sigmoid((x ** 3 + x ** 2 - x) @ theta + (x - x ** 2) @ M @ e_a + b) / T
    )
    
    #ユーザーの行動タイプを決定し、その行動に基づいてC_ 行列を作成
    user_behavior_matrix = np.r_[
        np.eye(T),  # independent
        np.tril(np.ones((T, T))),  # cascade　下三角行列を生成　カスケード型の行動を表しており、前のタイムステップが次のタイムステップに影響を与えること
        np.ones((T, T)),  # all　全ての要素が1の行列　すべてのタイムステップが相互に関連している（すべてのアクションを行う）ことを示しています
    ].reshape((3, T, T))
    
    #3つの行動パターン（independent, cascade, all）のいずれかを、確率 p に基づいてランダムに選択します。
    # p は p=[0.0, 1.0, 0.0] のように、各行動パターンが選択される確率です。ここでは、カスケード行動 (cascade) が選択される可能性
    # が最も高い設定です。
    user_behavior_idx = random_.choice(3, p=p, size=num_data)
    C_ = user_behavior_matrix[user_behavior_idx]
    
    #ランダム行動を表す行列を生成
    #この行列は (7, T, T) の形状に変換され、7つの異なるランダム行動パターンが生成されます
    user_behavior_matrix_rand = random_.choice(
        [-1, 0, 1], p=[0.2, 0.4, 0.4], size=7 * T * T
    ).reshape((7, T, T))
    user_behavior_rand_idx = random_.choice(7, size=num_data)
    C_rand = user_behavior_matrix_rand[user_behavior_rand_idx]

    is_rand = random_.binomial(2, p=p_rand, size=num_data).reshape(num_data, 1, 1)
    C = np.clip(C_ + is_rand * C_rand, 0, 1)
    
    #w = 1 の場合は、ソフトマックス方策が選ばれます（決定的な行動選択が行われます）。
    #w = 0 の場合は、ε-greedy方策が選ばれます（ランダム性が導入され、探索的な行動が行われます）
    if is_online:
        w = random_.binomial(1, p=0.5, size=num_data)[:, np.newaxis]
    else:
        w = random_.binomial(1, p=0.0, size=num_data)[:, np.newaxis]

    # データ収集方策を定義する
    #w が 1 であれば、ソフトマックス方策が適用され、各アクションが確率的に選ばれます。
    #w が 0 であれば、ε-greedy方策が適用され、ある程度の確率でランダムに探索が行われます。
    pi_0 = w * softmax(beta * base_q_func)
    pi_0 += (1 - w) * eps_greedy_policy(base_q_func, eps=eps)

    # 行動や報酬を抽出する
    #行動（アクション）と報酬をタイムステップごとにシミュレーションして抽出するための操作
    # 具体的には、各タイムステップで行動を選び、その行動に対する期待報酬と実際の報酬を計算
    a_t = np.zeros((num_data, T), dtype=int)
    r_t = np.zeros((num_data, T), dtype=float)
    q_t = np.zeros((num_data, T), dtype=float)
    
    #pi_0 に基づいて、各タイムステップでどのアクションを選択するかをランダムに決定します
    for t in range(T):
        a_t_ = sample_action_fast(pi_0, random_state=random_state + t)
        a_t[:, t] = a_t_
    idx = np.arange(num_data)
    for t in range(T):
        q_func_factual = base_q_func[idx, a_t[:, t]]
        for t_ in range(T):
            if t_ != t:
                q_func_factual += (
                    C[:, t, t_]
                    * W[t, t_]
                    * base_q_func[idx, a_t[:, t]]
                    / np.abs(t - t_)
                )
        q_t[:, t] = q_func_factual
        r_t[:, t] = random_.normal(q_func_factual, scale=reward_noise)

    return dict(
        num_data=num_data, # データの数
        T=T,  # タイムステップ数
        num_actions=num_actions,  # アクション数
        x=x,   # コンテキスト（特徴量ベクトル）
        w=w.flatten(),  # 行動方針（収集方策かランダム方策か）
        a_t=a_t, # 各タイムステップのアクション
        r_t=r_t,  # 各タイムステップの報酬
        pi_0=pi_0,  # 収集方策
        q_t=q_t,  # 各タイムステップの期待報酬
        base_q_func=base_q_func,   # 基本的な期待報酬関数
    )


def calc_true_value(
    dim_context: int,
    num_actions: int,
    T: int,
    theta: np.ndarray,
    M: np.ndarray,
    b: np.ndarray,
    W: np.ndarray,
    eps: float = 0.0,
    beta: float = 1.0,
    num_data: int = 100000,
) -> float:
    """評価方策の真の性能を近似する."""
    bandit_data = generate_synthetic_data(
        num_data=num_data,
        dim_context=dim_context,
        num_actions=num_actions,
        T=T,
        theta=theta,
        M=M,
        b=b,
        W=W,
        eps=eps,
        beta=beta,
        is_online=True,
        random_state=12345,
    )
    w, q_t = bandit_data["w"], bandit_data["q_t"]
    value_of_pi = (w * q_t.mean(1)).sum() / w.sum()
    value_of_pi_0 = ((1 - w) * q_t.mean(1)).sum() / (1 - w).sum()

    return value_of_pi_0, value_of_pi


def generate_synthetic_data2(
    num_data: int,
    dim_context: int,
    num_actions: int,
    beta: float,
    theta_1: np.ndarray,
    M_1: np.ndarray,
    b_1: np.ndarray,
    theta_0: np.ndarray,
    M_0: np.ndarray,
    b_0: np.ndarray,
    random_state: int = 12345,
) -> dict:
    """オフ方策学習におけるログデータを生成する."""
    random_ = check_random_state(random_state)
    x = random_.normal(size=(num_data, dim_context))
    one_hot_a = np.eye(num_actions)

    # 期待報酬関数を定義する
    q_x_a_0 = (
        (x ** 3 + x ** 2 - x) @ theta_0 + (x - x ** 2) @ M_0 @ one_hot_a + b_0
    ) / num_actions
    q_x_a_1 = (
        (x - x ** 2) @ theta_1 + (x ** 3 + x ** 2 - x) @ M_1 @ one_hot_a + b_1
    ) / num_actions
    cate_x_a = q_x_a_1 - q_x_a_0
    q_x_a_1 += rankdata(cate_x_a, axis=1, method="dense") <= num_actions * 0.5
    q_x_a_0 += rankdata(cate_x_a, axis=1, method="dense") <= num_actions * 0.5
    q_x_a_1 -= rankdata(cate_x_a, axis=1, method="dense") >= num_actions * 0.8
    q_x_a_0 -= rankdata(cate_x_a, axis=1, method="dense") >= num_actions * 0.8

    # データ収集方策を定義する
    pi_0 = softmax(beta * cate_x_a)

    # 行動や報酬を抽出する
    a = sample_action_fast(pi_0, random_state=random_state)
    a_mat = np.zeros((num_data, num_actions), dtype=int)
    a_mat[np.arange(num_data), a] = 1
    pscore_mat = a_mat * pi_0 + (1 - a_mat) * (1 - pi_0)

    q_x_a_factual = a_mat * q_x_a_1 + (1 - a_mat) * q_x_a_0
    r_mat = random_.normal(q_x_a_factual)   #ここを変更する

    return dict(
        num_data=num_data, # データの数
        num_actions=num_actions, 
        x=x,  
        a=a,   
        r=r_mat[np.arange(num_data), a],
        a_mat=a_mat,
        r_mat=r_mat,
        pi_0=pi_0,
        pscore=pi_0[np.arange(num_data), a],
        pscore_mat=pscore_mat,
        q_x_a_1=q_x_a_1,
        q_x_a_0=q_x_a_0,
        cate_x_a=cate_x_a,
    )


def generate_waseda_student_data2(
    num_data: int,
    dim_context: int,
    num_actions: int,
    beta: float,
    theta_1: np.ndarray,
    M_1: np.ndarray,
    b_1: np.ndarray,
    theta_0: np.ndarray,
    M_0: np.ndarray,
    b_0: np.ndarray,
    random_state: int = 12345,
    random_policy =False,
) -> dict:
    """オフ方策学習におけるログデータを生成する."""
    random_ = check_random_state(random_state)
    x = random_.normal(size=(num_data, dim_context))
    one_hot_a = np.eye(num_actions)

    # 期待報酬関数を定義する
    q_x_a_0 = (
        sigmoid((x ** 3 + x ** 2 - x) @ theta_0 + (x - x ** 2) @ M_0 @ one_hot_a + b_0)
    ) 
    q_x_a_1 = (
        sigmoid((x - x ** 2) @ theta_1 + (x ** 3 + x ** 2 - x) @ M_1 @ one_hot_a + b_1)
    ) 
    cate_x_a = q_x_a_1 - q_x_a_0
    """
    q_x_a_1 += (rankdata(cate_x_a, axis=1, method="dense") <= num_actions * 0.5)*0.05
    q_x_a_0 += (rankdata(cate_x_a, axis=1, method="dense") <= num_actions * 0.5)*0.05
    q_x_a_1 -= (rankdata(cate_x_a, axis=1, method="dense") >= num_actions * 0.8)*0.05
    q_x_a_0 -= (rankdata(cate_x_a, axis=1, method="dense") >= num_actions * 0.8)*0.05
    """
    
    #データ収集方策を定義
    if random_policy:
        #全ユーザーに対して全アクションを推奨するデータ収集方策を考える．
        pi_0 = np.ones((num_data, num_actions)) / num_actions  # 全ユーザーに全アクションを等確率で推薦する方策に変更
    else:
        pi_0 = softmax(beta * q_x_a_1)
    
    
    

    # 行動や報酬を抽出する
    a = sample_action_fast(pi_0, random_state=random_state)
    a_mat = np.zeros((num_data, num_actions), dtype=int)
    a_mat[np.arange(num_data), a] = 1
    pscore_mat = a_mat * pi_0 + (1 - a_mat) * (1 - pi_0)

    q_x_a_factual = a_mat * q_x_a_1 + (1 - a_mat) * q_x_a_0
    #r_mat = random_.normal(q_x_a_factual)   #ここを変更する
    
    
    r_mat = random_.binomial(n=1,p=q_x_a_factual)

    return dict(
        num_data=num_data, # データの数
        num_actions=num_actions, #アクション数
        x=x,  #特徴量
        a=a,   #行動
        r=r_mat[np.arange(num_data), a],
        a_mat=a_mat,
        r_mat=r_mat,
        pi_0=pi_0,
        pscore=pi_0[np.arange(num_data), a],
        pscore_mat=pscore_mat,
        q_x_a_1=q_x_a_1,
        q_x_a_0=q_x_a_0,
        cate_x_a=cate_x_a,
    )
    


def calc_true_value2(
    num_data:int,
    dim_context: int,
    num_actions: int,
    theta_1: np.ndarray,
    theta_0:np.ndarray,
    M_1: np.ndarray,
    M_0: np.ndarray,
    b_1: np.ndarray,
    b_0: np.ndarray,
    beta: float = 1.0,
) -> float:
    """評価方策の真の性能を近似する."""
    bandit_data = generate_waseda_student_data2(
        num_data=num_data,
        dim_context=dim_context,
        num_actions=num_actions,
        beta=beta,
        theta_1=theta_1,
        M_1=M_1,
        b_1=b_1,
        theta_0=theta_0,
        M_0=M_0,
        b_0=b_0
    )
    
    cate_x_a = bandit_data["cate_x_a"]
    pi =eps_greedy_policy(cate_x_a)
    
    q_x_a_1 =bandit_data["q_x_a_1"]
    q_x_a_0 = bandit_data["q_x_a_0"]

    return (pi*q_x_a_1+(1-pi)*q_x_a_0).sum(1).mean()