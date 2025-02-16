import numpy as np
import scipy

class call_back_class:
    def __init__(self,print_flag):
        import gc
        import time
        self.time_ = time
        self.iteration = 0
        self.params = []
        self.time = time.time()
        self.print_flag=print_flag
        pass

    def __call__(self, *args, **kwargs):
        #gc.collect()
        self.iteration += 1
        self.params.append(args[0])
        now = self.time_.time()
        if self.print_flag:print('Iteration:' + str(self.iteration) + ', Xk=' + str(args[0]) + ', Time:' + str(now - self.time) + 's')
        self.time = now
        return True

class Optimizer:
    def __init__(self, code, env):
        self.code = code
        self.env = env

    def maximize(self, func,init_params,print_flag=True):
        _func=lambda x: -1*func(x)
        call_back = call_back_class(print_flag)
        mini_result=scipy.optimize.minimize(_func, init_params, callback=call_back,method='l-bfgs-b')
        return -1 * mini_result.fun, mini_result.x

    def call_optimal_fidelity(self,para):
        self.code.set_parameters(para)
        return self.env.optimal_fidelity(self.code)

    def call_channel_fidelity(self,para):
        num_para = self.code.num_para
        num_para_rec = self.code.num_para_rec
        self.code.set_parameters(para[:num_para])
        self.code.set_parameters_rec(para[-num_para_rec:])
        return self.env.channel_fidelity(self.code)

    def call_state_fidelity(self,para,ave=False):
        num_para = self.code.num_para
        num_para_rec = self.code.num_para_rec
        self.code.set_parameters(para[:num_para])
        self.code.set_parameters_rec(para[-num_para_rec:])
        return self.env.state_fidelity(self.code,ave)

    def maximize_optimal_fidelity(self,seed=42,init_params=None,print_flag=True):
        if init_params is None:
            np.random.seed(seed)
            init_params=np.random.uniform(low=0.0, high=2*np.pi,size=self.code.num_para)
        return self.maximize(self.call_optimal_fidelity,init_params,print_flag)
    def maximize_channel_fidelity(self,seed=42,init_params=None,print_flag=True):
        if init_params is None:
            np.random.seed(seed)
            init_params=np.random.uniform(low=0.0, high=2*np.pi,size=self.code.num_para+self.code.num_para_rec)
        return self.maximize(self.call_channel_fidelity,init_params,print_flag)
    def maximize_state_fidelity(self,seed=42,init_params=None,print_flag=True,ave=False):
        if init_params is None:
            np.random.seed(seed)
            init_params=np.random.uniform(low=0.0, high=2*np.pi,size=self.code.num_para+self.code.num_para_rec)
        return self.maximize(self.call_state_fidelity,init_params,print_flag)
