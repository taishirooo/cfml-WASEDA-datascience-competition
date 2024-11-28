from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_random_state
import torch
import torch.nn as nn
import torch.optim as optim

from utils import GradientBasedPolicyDataset,GradientBasedPolicyDataset_dr


@dataclass
class IPSBasedGradientPolicyLearner:
    """勾配ベースのアプローチに基づく、メール配信経由の視聴時間を最大化するオフ方策学習.
    cateを学習するのではなく，p_x_aに基づき学習している．
    
    
    """
    dim_x: int
    num_actions: int
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 16
    learning_rate_init: float = 0.01
    alpha: float = 1e-6
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    solver: str = "adagrad"
    max_iter: int = 30
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""
        layer_list = []
        input_size = self.dim_x
        torch.manual_seed(self.random_state)

        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))
        layer_list.append(("softmax", nn.Softmax(dim=1)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random_ = check_random_state(self.random_state)
        self.naive_value = []
        self.cate_value = []

    def fit(self, dataset: dict, dataset_test: dict) -> None:
        x, a, r = dataset["x"], dataset["a"], dataset["r"]
        pscore, pi_0 = dataset["pscore"], dataset["pi_0"]

        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        training_data_loader = self._create_train_data_for_opl(
            x,
            a,
            r,
            pscore,
            pi_0, #違くない？
            pi_0, #違くない？
        )

        # start policy training
        q_x_a_1 = dataset_test["q_x_a_1"]
        q_x_a_0 = dataset_test["q_x_a_0"]
        for _ in range(self.max_iter):
            self.nn_model.train()
            for x, a, r, p, _, _ in training_data_loader:
                optimizer.zero_grad()
                pi = self.nn_model(x)
                loss = -self._estimate_policy_gradient(
                    a=a,
                    r=r,
                    pscore=p,
                    pi=pi,
                ).mean()
                loss.backward()
                optimizer.step()
            pi = self.predict(dataset_test)
            self.naive_value.append((pi * q_x_a_1).sum(1).mean())
            self.cate_value.append((pi * q_x_a_1 + (1.0 - pi) * q_x_a_0).sum(1).mean())

    def _create_train_data_for_opl(
        self,
        x: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        pscore: np.ndarray,
        q_hat: np.ndarray,
        pi_0: np.ndarray,
    ) -> tuple:
        dataset = GradientBasedPolicyDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(q_hat).float(),
            torch.from_numpy(pi_0).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader

    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        r: torch.Tensor,
        pscore: torch.Tensor,
        pi: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob = torch.log(pi + self.log_eps)
        idx = torch.arange(a.shape[0], dtype=torch.long)

        iw = current_pi[idx, a] / pscore
        estimated_policy_grad_arr = iw * r * log_prob[idx, a]

        return estimated_policy_grad_arr

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:

        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["x"]).float()
        return self.nn_model(x).detach().numpy()


@dataclass
class CateBasedGradientPolicyLearner_ips:
    """勾配ベースのアプローチに基づく、プラットフォーム全体の視聴時間を最大化するオフ方策学習."""
    dim_x: int
    num_actions: int
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 16
    learning_rate_init: float = 0.01
    alpha: float = 1e-6
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    solver: str = "adagrad"
    max_iter: int = 30
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""
        layer_list = []
        input_size = self.dim_x
        torch.manual_seed(self.random_state)

        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))
        layer_list.append(("softmax", nn.Softmax(dim=1)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random_ = check_random_state(self.random_state)
        self.naive_value = []
        self.cate_value = []

    def fit(self, dataset: dict, dataset_test: dict) -> None:
        x, a, r = dataset["x"], dataset["a"], dataset["r"]
        a_mat, r_mat, pscore_mat = (
            dataset["a_mat"],
            dataset["r_mat"],
            dataset["pscore_mat"],
        )

        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        training_data_loader = self._create_train_data_for_opl(
            x,
            a,
            r,
            pscore_mat,
            a_mat, ######
            r_mat, ########
        )

        # start policy training
        q_x_a_1 = dataset_test["q_x_a_1"]
        q_x_a_0 = dataset_test["q_x_a_0"]
        for _ in range(self.max_iter):
            self.nn_model.train()
            for x_, a_, r_, pscore_mat_, a_mat_, r_mat_ in training_data_loader:
                optimizer.zero_grad()
                pi = self.nn_model(x_)
                loss = -self._estimate_policy_gradient(
                    a=a_,
                    a_mat=a_mat_,
                    r=r_,
                    r_mat=r_mat_,
                    pscore_mat=pscore_mat_,
                    pi=pi,
                ).mean()
                loss.backward()
                optimizer.step()
            pi = self.predict(dataset_test)
            self.naive_value.append((pi * q_x_a_1).sum(1).mean())
            self.cate_value.append((pi * q_x_a_1 + (1.0 - pi) * q_x_a_0).sum(1).mean())

    def _create_train_data_for_opl(
        self,
        x: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        pscore: np.ndarray,
        q_hat: np.ndarray,
        pi_0: np.ndarray,
    ) -> tuple:
        dataset = GradientBasedPolicyDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(q_hat).float(),
            torch.from_numpy(pi_0).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader

    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        a_mat: torch.Tensor,
        r: torch.Tensor,
        r_mat: torch.Tensor,
        pscore_mat: torch.Tensor,
        pi: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob1 = torch.log(pi + self.log_eps)
        log_prob2 = torch.log((1.0 - pi) + self.log_eps)
        idx = torch.arange(a.shape[0], dtype=torch.long)

        estimated_policy_grad_arr = (
            current_pi[idx, a] * r / pscore_mat[idx, a]
        ) * log_prob1[idx, a]
        estimated_policy_grad_arr += (
            (1 - a_mat) * (1.0 - current_pi) * (r_mat / pscore_mat) * log_prob2
        ).sum(1)

        return estimated_policy_grad_arr

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:

        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["x"]).float()
        return self.nn_model(x).detach().numpy()
    
    
@dataclass
class CateBasedGradientPolicyLearner_dr:
    """勾配ベースのアプローチに基づく，履修率を最大化するオフ方策学習"""
    dim_x: int
    num_actions: int
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 16
    learning_rate_init: float = 0.01
    alpha: float = 1e-6
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    solver: str = "adagrad"
    max_iter: int = 30
    random_state: int = 12345
    
    def __post_init__(self) -> None:
        "Initialize class"
        layer_list = []
        input_size = self.dim_x
        torch.manual_seed(self.random_state)
        
        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer == nn.ReLU
        elif self.activation == "elu":
            activation_layer  = nn.ELU
            
        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
            
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))
        layer_list.append(("softmax", nn.Softmax(dim=1)))
               
        self.nn_model = nn.Sequential(OrderedDict(layer_list))
        
        self.random_ = check_random_state(self.random_state)
        self.naive_value = []
        self.cate_value = []
        
        
    def fit(self, dataset:dict, dataset_test:dict,q1_hat, q0_hat ) -> None: #q1,a0を追加
        x, a, r = dataset["x"], dataset["a"], dataset["r"]
        a_mat, r_mat, pscore_mat = dataset["a_mat"], dataset["r_mat"], dataset["pscore_mat"]
        
        #print("fitの",pscore_mat)
        
        #pi_0を追加
        pi_0 = dataset["pi_0"]
        
        ## q1,q0の条件文を追加
        if q0_hat is None:
            q0_hat= np.zeros(r.shape[0], self.num_actions)
         
        if q1_hat is None:
            q1_hat= np.zeros(r.shape[0], self.num_actions)   
         
        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")
        
        training_data_loader = self._create_train_data_for_opl(
             x,
             a,
                r,
                a_mat,
                r_mat,           
                pscore_mat,
                q1_hat,
                q0_hat
            #pi_0,           
        )
        
        q_x_a_1 = dataset_test["q_x_a_1"]
        q_x_a_0 = dataset_test["q_x_a_0"]
        
        for _ in range(self.max_iter):
            self.nn_model.train()
            for x_, a_, r_,  a_mat_, r_mat_, pscore_mat_, q1_hat_, q0_hat_ in training_data_loader:
                optimizer.zero_grad()
                pi = self.nn_model(x_)
                loss = -self._estimate_policy_gradient(
                    a=a_,
                    a_mat=a_mat_,
                    r=r_,
                    r_mat=r_mat_,
                    pscore_mat=pscore_mat_,
                    pi=pi,
                    q1_hat=q1_hat_,
                    q0_hat=q0_hat_,
                ).mean()
                loss.backward()
                optimizer.step()
            pi = self.predict(dataset_test)
            self.naive_value.append((pi * q_x_a_1).sum(1).mean())
            self.cate_value.append((pi * q_x_a_1 + (1.0 - pi) * q_x_a_0).sum(1).mean())
         
    def _create_train_data_for_opl(
        self,
        x: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        a_mat:np.ndarray,
        r_mat: np.ndarray,           
        pscore: np.ndarray,
        q1_hat: np.ndarray,
        q0_hat:np.ndarray,
        #pi_0: np.ndarray,
    ) -> tuple:
        dataset = GradientBasedPolicyDataset_dr(
            torch.from_numpy(x).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
            torch.from_numpy(a_mat).float(),
            torch.from_numpy(r_mat).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(q1_hat).float(),
            torch.from_numpy(q0_hat).float(),
            #torch.from_numpy(pi_0).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader 
    
    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        a_mat: torch.Tensor,
        r: torch.Tensor,
        r_mat: torch.Tensor,
        pscore_mat: torch.Tensor,
        pi: torch.Tensor,
        ##追加##
        q1_hat: torch.Tensor,
        q0_hat: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob1 = torch.log(pi + self.log_eps)
        log_prob2 = torch.log((1.0 - pi) + self.log_eps)
        idx = torch.arange(a.shape[0], dtype=torch.long)
        
        
        """
        print("current_pi", current_pi[idx,a])
        print("r",r)
        print("pscore", pscore_mat[idx, a])
        
        #print("r - q1_hat[idx, a]", r - q1_hat[idx, a])
        print( "計算1ips",current_pi[idx, a] * r / pscore_mat[idx, a])
        print( "計算１dr",current_pi[idx, a] * (r - q1_hat[idx, a]) / pscore_mat[idx, a])
        
        print("計算2",(1 - a_mat) * (1.0 - current_pi) * ((r_mat - q0_hat) / pscore_mat) * log_prob2)
        
        print("計算3",current_pi* (q1_hat + q0_hat) )
        """
        

        estimated_policy_grad_arr = (
            current_pi[idx, a] * ((r - q1_hat[idx,a])  / pscore_mat[idx, a])
        ) * log_prob1[idx, a]
        
        #print("1",estimated_policy_grad_arr)
        
        estimated_policy_grad_arr += (
            (1 - a_mat) * (1.0 - current_pi) * ((r_mat - q0_hat ) / pscore_mat) * log_prob2
        ).sum(1)
        
        #print("2",estimated_policy_grad_arr)
        
        estimated_policy_grad_arr += ((current_pi* q1_hat)*log_prob1).sum(1)
        estimated_policy_grad_arr += ((current_pi*q0_hat)*log_prob2).sum(1)
        
       # print("3",estimated_policy_grad_arr)

        return estimated_policy_grad_arr
    
    def predict(self, dataset_test: np.ndarray) -> np.ndarray:

        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["x"]).float()
        return self.nn_model(x).detach().numpy()           
           
  
 
 
  
        
@dataclass
class Keisan:
    a : float
    b : float
        
    def add(self):
        x = self.a+self.b
        return x
    
    def mainasu(self):
        y = self.a - self.b
        return y