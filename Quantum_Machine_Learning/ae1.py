#!pip install pennylane
#!pip install ipyparallel

import pennylane as qml
from pennylane import numpy as np
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt

from ipyparallel import Client

rc = Client()
dview = rc[:]

@dview.parallel(block=True)
def run_session(args):
    import pennylane as qml
    from pennylane import numpy as np
    import time
    from datetime import datetime
    import os
    cur_dir = os.getcwd()
    
    Nwires = 5
    dev = qml.device("default.qubit", wires=Nwires)
    
    @qml.qnode(dev)
    def circuit(params, data):
        qml.AngleEmbedding(data, wires=range(Nwires))
        qml.StronglyEntanglingLayers(params, wires=range(Nwires))
        return qml.expval(qml.PauliZ(Nwires-1))
        #return qml.expval(qml.PauliZ(0))
    
    def cost(params, single_sample):
         return (1 - circuit(params, single_sample)) ** 2 
        
    def cost_batch(params, data):
         return np.sum((1 - circuit(params, single_sample)) ** 2 for single_sample in data) / Nbatch


    OptimizerClass, caption, Niter, Ninit,Nbatch, lrs, seed, dir4save = args
    
    np.random.seed(seed)
    date_string = datetime.now().strftime("%Y.%m.%d")

    print(f"Start for {OptimizerClass.__name__}  date = {date_string}")

    all_costs = []
    all_params = []
    all_costs_mean = []
    all_costs_stdev = []
    cur_lrs = []
    
    use_metric = False
    if "QNG" in OptimizerClass.__name__:
        use_metric = True

    times = []
    for lr in lrs:
        lr_costs = []
        lr_accuracies = []
        lr_params = []
        timer = time.time()
        for i in range(Ninit):
            data = [np.random.random([Nwires], requires_grad=False) for _ in range(Nbatch)]
            init_params = np.random.random(qml.StronglyEntanglingLayers.shape(Nwires, Nwires), requires_grad=True)        
            opt = OptimizerClass(lr)
            params = init_params
            costs = []
            for it in range(Niter):
                for sample in data:
                    cost_fn = lambda p: cost(p, sample)
                    #cost_fn = lambda p: cost_batch(p, data)
                    if( use_metric):
                        metric_fn = lambda p: qml.metric_tensor(circuit, approx="block-diag")(p, sample)
                        #metric_fn = lambda p: np.sum((qml.metric_tensor(circuit, approx="block-diag")(p, sample) for sample in data), axis = 0)/Nbatch
                        params, loss = opt.step_and_cost(cost_fn, params,  metric_tensor_fn=metric_fn)
                    else:
                        params, loss = opt.step_and_cost(cost_fn, params)
                costs.append(loss)
            lr_costs.append(costs)
            lr_params.append(params)
        timer = time.time() - timer
        times.append(timer)
        print(f"{OptimizerClass.__name__}:  lr = {lr:.5f}   time = {timer:.4} sec  loss = {np.mean(lr_costs, axis=0)[-1]}")
        all_costs.append(lr_costs)
        all_params.append(lr_params)
        all_costs_mean.append(np.mean(lr_costs, axis=0))
        all_costs_stdev.append(np.std(lr_costs, axis=0))
        cur_lrs.append(lr)
        prefix = f"{dir4save}/{caption}_{OptimizerClass.__name__}_{date_string}"
        np.save(f"{prefix}_lrs.npy",np.array(cur_lrs))
        np.save(f"{prefix}_all.npy",np.array(all_costs, dtype=object), allow_pickle=True)
        np.save(f"{prefix}_params.npy",np.array(all_params, dtype=object), allow_pickle=True)
        np.save(f"{prefix}_moments.npy",np.array([all_costs_mean,all_costs_stdev], dtype=object), allow_pickle=True)
    return cur_lrs, times
	
	
def show(caption, OptimazerClass, date):
    if isinstance(caption, tuple):
        costs_mean = None
        costs_stdev = None
        llrs = None
        for cap in caption:
            prefix = f"{cap}_{OptimazerClass.__name__}_{date}"
            [costs_mean_,costs_stdev_] = np.load(f"{prefix}_moments.npy", allow_pickle=True)
            llrs_ = np.load(f"{prefix}_lrs.npy")
            (Nlr_,Niter) = np.array(costs_mean_).shape
            if llrs is None:
                llrs = llrs_
                costs_mean = costs_mean_
                costs_stdev = costs_stdev_
            else:
                llrs = np.concatenate((llrs, llrs_), axis=0)
                costs_mean = np.concatenate((costs_mean, costs_mean_), axis=0)
                costs_stdev = np.concatenate((costs_stdev, costs_stdev_), axis=0)
    else:
        prefix = f"{caption}_{OptimazerClass.__name__}_{date}"
        [costs_mean,costs_stdev] = np.load(f"{prefix}_moments.npy", allow_pickle=True)
        llrs = np.load(f"{prefix}_lrs.npy")
        (Nlr,Niter) = np.array(costs_mean).shape
    
    fig, ax = plt.subplots(figsize=(8, 5))
    iters = np.linspace(1,Niter,Niter)
    for index,lr in enumerate(llrs):
        arr_y = costs_mean[index]
        arr_y_plus_err  = [costs_mean[index][i] - costs_stdev[index][i] for i in range(Niter)]
        arr_y_minus_err = [costs_mean[index][i] + costs_stdev[index][i] for i in range(Niter)]
        ax.plot(iters, arr_y, label=f"lr = {lr:.5f}")
        ax.fill_between(iters, arr_y_minus_err, arr_y_plus_err, alpha=0.1)
    
    ax.set_title(f"{caption}: Costs for {OptimazerClass.__name__}")
    #ax.set_xlim(0, max(xpos)); ax.set_ylim(-1, 1)
    ax.set_xlabel("Iteration");
    ax.set_ylabel("Cost");
    ax.legend()
    ax.grid()
    plt.yscale('log')
    fig.tight_layout()
    
    plt.savefig(f"{caption}_{OptimazerClass.__name__}.png", dpi=300, bbox_inches='tight')
    plt.show() 	
	
	

cur_dir = os.getcwd()

#change parameters as needed
####
Nbatch = 10
Niter = 1
Ninit = 1
lrs = np.logspace(-3, 0, num=10, base=10)
seed = 3141592
####

task_name = "ea1"
date_str = datetime.now().strftime("%Y.%m.%d")
param_list = [
              (qml.QNGOptimizer, f"{task_name}_1", Niter, Ninit,Nbatch, lrs[1:3], seed, cur_dir),
              (qml.QNGOptimizer, f"{task_name}_2", Niter, Ninit,Nbatch, lrs[3:5], seed, cur_dir),
              (qml.QNGOptimizer, f"{task_name}_3", Niter, Ninit,Nbatch, lrs[5:7], seed, cur_dir), 
              (qml.QNGOptimizer, f"{task_name}_4", Niter, Ninit,Nbatch, lrs[7:9], seed, cur_dir), 
              (qml.MomentumQNGOptimizer, f"{task_name}_1", Niter, Ninit,Nbatch, lrs[0:2], seed, cur_dir),
              (qml.MomentumQNGOptimizer, f"{task_name}_2", Niter, Ninit,Nbatch, lrs[2:4], seed, cur_dir),
              (qml.MomentumQNGOptimizer, f"{task_name}_3", Niter, Ninit,Nbatch, lrs[4:6], seed, cur_dir), 
              (qml.MomentumQNGOptimizer, f"{task_name}_4", Niter, Ninit,Nbatch, lrs[6:8], seed, cur_dir), 
              (qml.AdamOptimizer, f"{task_name}", Niter, Ninit,Nbatch, lrs, seed, cur_dir),
              (qml.MomentumOptimizer, f"{task_name}", Niter, Ninit,Nbatch, lrs, seed, cur_dir)
             ]
timer = time.time()
results = run_session.map(param_list)
timer = time.time() - timer
print(f"total time {timer/3600:.4} hr")
for index,(l,t) in enumerate(results):
    print(f"task{index+1}: {np.sum(t)} sec")

show((f"{task_name}_1",f"{task_name}_2",f"{task_name}_3",f"{task_name}_4"),qml.MomentumQNGOptimizer,date_str)
show((f"{task_name}_1",f"{task_name}_2",f"{task_name}_3",f"{task_name}_4"),qml.QNGOptimizer,date_str)
show(f"{task_name}",qml.MomentumOptimizer,date_str)
show(f"{task_name}",qml.AdamOptimizer,date_str)