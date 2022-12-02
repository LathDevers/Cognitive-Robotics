# Source: https://openai.com/blog/reptile/ & https://gist.github.com/joschu/f503500cda64f2ce87c8288906b09e2d#file-reptile-sinewaves-demo-py
# consider the cases:
    # (i)  of plain learning from a random weight initialization and
    # (ii) learning after weight optimization according to the Reptile algorithm of the example implementation
# visualize in each case the results before and after training and compare the outcomes of (i) and (ii) above
import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy

seed = 0
innerStepSize = 0.02 # stepsize in inner SGD
innerEpochs = 1 # number of epochs of each inner SGD
outerStepSize0 = 0.1 # stepsize of outer optimization, i.e., meta-optimization
n = 30000 # number of outer updates; each iteration we sample one task and update on it
useReptile = True

rng = np.random.RandomState()
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
    "Generate a random `sine function` with random `phase` and random `amplitude`."
    phase = rng.uniform(low=0, high=2*np.pi)           # 𝛟 ∈ [0,2π[
    ampl = rng.uniform(0.1, 5)                         # A ∈ [0.1,5[
    f_randomsine = lambda x : ampl * np.sin(x + phase) # A ⋅ sin(x + 𝛟)
    return f_randomsine

def totorch(x):
    """Creates a `Variable` object from a `list` or `array`. `Variables` can be used to compute gradients with the `backward()` function."""
    return ag.Variable(torch.Tensor(x))

def train_on_batch(x, y):
    """Computes loss between `y` and `model(x)`. Then calculates gradients and changes model parameters accordingly."""
    x = totorch(x)
    y = totorch(y)
    model.zero_grad() # Sets gradients of all model parameters to zero. This is necessary before running the backward() function, as gradients are accumulated over multiple backward passes.
    # Get model prediction for x (should be: yₚᵣₑ = y = A⋅sin(x+𝛟))
    ypred = model(x)
    loss = (ypred - y).pow(2).mean() # mean squared error
    loss.backward() # compute gradients + future calls will accumulate gradients into `param.grad`
    for param in model.parameters(): # Iterator over module parameters.
        param.data -= innerStepSize * param.grad.data # `param.grad` attribute contains the gradients computed 

def predict(x):
    """Runs the neural network on the input `x` and returns the output as a numpy array."""
    x = totorch(x)
    return model(x).data.numpy()

# Choose a fixed task and minibatch for visualization
f_plot = gen_task()
xtrain_plot = x_all[rng.choice(len(x_all), size=ntrain)] # training points

# Reptile training loop
for i in range(n):
    weights_before = deepcopy(model.state_dict())
    # Generate task
    f = gen_task() # in each iteration: generate a random sine wave
    y_all = f(x_all) # calculate values on the whole domain
    # Do SGD on this task
    inds = rng.permutation(len(x_all)) # randomly shuffled indeces i. e. [0,50[ --- for example {2 17 28 5 ... 25 38}
    for _ in range(innerEpochs):
        for start in range(0, len(x_all), ntrain): # (start, stop, step) --- (0, 50, 10) --> {0 10 20 30 40}
            mbinds = inds[start:start+ntrain] # minibatch indeces
            train_on_batch(x_all[mbinds], y_all[mbinds]) # train on 10 set of x and y pairs
    # Interpolate between current weights (weights_before) and trained weights (weights_after) from this task
    # I.e. (weights_before - weights_after) is the meta-gradient
    weights_after = model.state_dict()
    outerStepSize = outerStepSize0 * (1 - i / n) # linear schedule, i.e. (1 - i/n) is ~1 if i=0, and ~0 if i=n
    model.load_state_dict({name : 
        weights_before[name] + (weights_after[name] - weights_before[name]) * outerStepSize
        for name in weights_before})

    # Plot the results on a particular task and minibatch
    if i==0 or (i+1) % 10000 == 0:
        plt.cla()
        f = f_plot
        weights_before = deepcopy(model.state_dict()) # save snapshot before evaluation
        plt.plot(x_all, predict(x_all), label="pred after 0", color=(0,0,1))
        for j in range(32):
            train_on_batch(xtrain_plot, f(xtrain_plot))
            if (j + 1) % 8 == 0:
                frac = (j + 1) / 32
                plt.plot(x_all, predict(x_all), label="pred after %i"%(j + 1), color=(frac, 0, 1 - frac))
        plt.plot(x_all, f(x_all), label="true", color=(0,1,0))
        lossval = np.square(predict(x_all) - f(x_all)).mean() # would be better to average loss over a set of examples, but this is optimized for brevity
        plt.plot(xtrain_plot, f(xtrain_plot), "x", label="train", color="k")
        plt.ylim(-4,4)
        plt.legend(loc="lower right")
        if i != n-1:
            plt.pause(0.01)
        else:
            plt.pause(5)
        model.load_state_dict(weights_before) # restore from snapshot
        print(f"-----------------------------")
        print(f"iteration               {i+1}")
        print(f"loss on plotted curve   {lossval:.3f}")