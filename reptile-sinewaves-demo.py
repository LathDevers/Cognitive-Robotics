# Source: https://openai.com/blog/reptile/
# Source: https://gist.github.com/joschu/f503500cda64f2ce87c8288906b09e2d#file-reptile-sinewaves-demo-py
import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy

seed = 0
plot = True
innerStepSize = 0.02 # stepsize in inner SGD
innerEpochs = 1 # number of epochs of each inner SGD
outerStepSize0 = 0.1 # stepsize of outer optimization, i.e., meta-optimization
n = 30000 # number of outer updates; each iteration we sample one task and update on it

rng = np.random.RandomState(seed)
torch.manual_seed(seed)

# Define task distribution
x_all = np.linspace(-5, 5, 50)[:,None] # All of the x points
ntrain = 10 # Size of training minibatches

# Define model. Reptile paper uses ReLU, but Tanh gives slightly better results
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1),
)

def gen_task():
    "Generate a random sine function"
    phase = rng.uniform(low=0, high=2*np.pi)           # ф = [0,2π[
    ampl = rng.uniform(0.1, 5)                         # A = [0.1,5[
    f_randomsine = lambda x : np.sin(x + phase) * ampl # A ⋅ sin(x + ф)
    return f_randomsine

def totorch(x):
    return ag.Variable(torch.Tensor(x))

def train_on_batch(x, y):
    x = totorch(x)
    y = totorch(y)
    model.zero_grad()
    ypred = model(x)
    loss = (ypred - y).pow(2).mean()
    loss.backward()
    for param in model.parameters():
        param.data -= innerStepSize * param.grad.data

def predict(x):
    x = totorch(x)
    return model(x).data.numpy()

# Choose a fixed task and minibatch for visualization
f_plot = gen_task()
xtrain_plot: np.ndarray = x_all[rng.choice(len(x_all), size=ntrain)] # training points

# Reptile training loop
for i in range(n):
    weights_before = deepcopy(model.state_dict())
    # Generate task
    f = gen_task() # in each iteration: generate a random sine wave
    y_all = f(x_all) # calculate values on the whole domain
    # Do SGD on this task
    inds = rng.permutation(len(x_all)) # randomly shuffled indeces i. e. [0,50[ --- for example {2 17 28 5 ... 25 38}
    #print(inds)
    for _ in range(innerEpochs):
        for start in range(0, len(x_all), ntrain): # (start, stop, step) --- (0, 50, 10) --> {0 10 20 30 40}
            mbinds = inds[start:start+ntrain] # minibatch indeces
            train_on_batch(x_all[mbinds], y_all[mbinds]) # train on 10 set of x and y pairs
    # Interpolate between current weights and trained weights from this task
    # I.e. (weights_before - weights_after) is the meta-gradient
    weights_after = model.state_dict()
    outerStepSize = outerStepSize0 * (1 - i / n) # linear schedule
    model.load_state_dict({name : 
        weights_before[name] + (weights_after[name] - weights_before[name]) * outerStepSize 
        for name in weights_before})

    # Periodically plot the results on a particular task and minibatch
    #if plot and i==0 or (i+1) % 1000 == 0:
    if plot and i==n-1:
        plt.cla()
        f = f_plot
        weights_before = deepcopy(model.state_dict()) # save snapshot before evaluation
        plt.plot(x_all, predict(x_all), label="pred after 0", color=(0,0,1))
        for inneriter in range(32):
            train_on_batch(xtrain_plot, f(xtrain_plot))
            if (inneriter+1) % 8 == 0:
                frac = (inneriter+1) / 32
                plt.plot(x_all, predict(x_all), label="pred after %i"%(inneriter+1), color=(frac, 0, 1-frac))
        plt.plot(x_all, f(x_all), label="true", color=(0,1,0))
        lossval = np.square(predict(x_all) - f(x_all)).mean()
        plt.plot(xtrain_plot, f(xtrain_plot), "x", label="train", color="k")
        plt.ylim(-4,4)
        plt.legend(loc="lower right")
        plt.pause(0.01)
        model.load_state_dict(weights_before) # restore from snapshot
        print(f"-----------------------------")
        print(f"iteration               {i+1}")
        print(f"loss on plotted curve   {lossval:.3f}") # would be better to average loss over a set of examples, but this is optimized for brevity