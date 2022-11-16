# replace the sine function by the forward kinematics of a two-link robot arm
import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy

seed = 0
innerStepSize = 0.02
innerEpochs = 1
outerStepSize0 = 0.1
n = 30000

rng = np.random.RandomState() #seed)
#torch.manual_seed(seed)

no_of_cells = 10 # 40
# joint angles x₁ ∈ [−π/2, π/2] and x₂ ∈ [−π/2, π/2]
x_all = np.array([
    np.linspace(-np.pi/2, np.pi/2, no_of_cells*5)[:,None],
    np.linspace(-np.pi/2, np.pi/2, no_of_cells*5)[:,None]
])
#
y1_min = -1
y1_max = 3
y2_min = -3
y2_max = 3
y1_all_size = (y1_max-y1_min)*no_of_cells
y2_all_size = (y2_max-y2_min)*no_of_cells
#
y1_all = np.linspace(y1_min, y1_max, y1_all_size)
y2_all = np.linspace(y2_min, y2_max, y2_all_size)
error_plane = np.zeros((y2_all_size,y1_all_size)) - .1
#
ntrain = 10 # Size of training minibatches

# Define model. Reptile paper uses ReLU, but Tanh gives slightly better results
model = nn.Sequential(
    nn.Linear(2, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 2),
)

def gen_task():
    "Generate a random forward kinematics problem of a two-link robot arm."
    A = rng.uniform(1, 2)                   # segment A ∈ [1, 2]
    B = rng.uniform(0.5, 1)                 # segment B ∈ [0.5, 1]
    f = lambda x: np.array([
            A * np.cos(x[0]) + B * np.cos(x[0] + x[1]), # y₁ = A⋅cos(x₁) + B⋅cos(x₁ + x₂)
            A * np.sin(x[0]) + B * np.sin(x[0] + x[1]), # y₂ = A⋅sin(x₁) + B⋅sin(x₁ + x₂)
    ])
    return f

def totorch(x):
    """Creates a `Variable` object from a `list` or `array`. `Variables` can be used to compute gradients with the `backward()` function."""
    return ag.Variable(torch.Tensor(x))

def train_on_batch(x, y):
    """Computes loss between `y` and `model(x)`. Then calculates gradients and changes model parameters accordingly."""
    x = np.transpose(x)
    y = np.transpose(y)
    x = totorch(x)
    y = totorch(y)
    model.zero_grad() # Sets gradients of all model parameters to zero. This is necessary before running the backward() function, as gradients are accumulated over multiple backward passes.
    y_pred = model(x)
    #print(f"y_pred shape {y_pred.shape}")
    #print(f"y_true shape {y.shape}")
    #
    loss = torch.sqrt(((y - y_pred)**2).sum()) # √(Σ(y-yₚ)²)
    loss.backward()
    #
    #loss = torch.cdist(y,y_pred)
    #loss.sum().backward()
    #
    for param in model.parameters(): # Iterator over module parameters.
        param.data -= innerStepSize * param.grad.data # `param.grad` attribute contains the gradients computed 

def predict(x):
    """Runs the neural network on the input `x` and returns the output as a numpy array."""
    x = totorch(x)
    return model(x).data.numpy()

def update_error_plane():
    for x1 in x_all[0]:
        for x2 in x_all[1]:
            y_true = f(conv(x1,x2))
            y_pred = predict(conv(x1,x2).reshape(2))
            error = np.sqrt(np.square(y_pred[0] - y_true[0]) + np.square(y_pred[1] - y_true[1]))
            put_into_plane(y_true[0], y_true[1], error)

def put_into_plane(y1, y2, value):
    i = int((y1+abs(y1_all[0]))*no_of_cells)
    j = int((y2+abs(y2_all[0]))*no_of_cells)
    error_plane[j,i] = value

def conv(x1, x2):
    return np.array([x1, x2])

# Choose a fixed task and minibatch for visualization
f_plot = gen_task()
x_train_plot = np.array([
    x_all[0,rng.choice(x_all.shape[1], size=ntrain)],
    x_all[1,rng.choice(x_all.shape[1], size=ntrain)]
]) # training points

# Reptile training loop
for i in range(n):
    weights_before = deepcopy(model.state_dict())
    # Generate task
    f = gen_task()
    # calculate values on the whole domain
    y_all = f(x_all)
    # Do SGD on this task
    inds1 = rng.permutation(x_all.shape[1]) # shuffled indeces
    inds2 = rng.permutation(x_all.shape[1])
    for _ in range(innerEpochs):
        for start in range(0, x_all.shape[1], ntrain):
            mbinds1 = inds1[start : start+ntrain] # minibatch indeces
            mbinds2 = inds2[start : start+ntrain]
            train_on_batch(conv(x_all[0,mbinds1], x_all[1,mbinds2]), conv(y_all[0,mbinds1], y_all[1,mbinds2]))
    # Interpolate between current weights (weights_before) and trained weights (weights_after) from this task
    # I.e. (weights_before - weights_after) is the meta-gradient
    weights_after = model.state_dict()
    outerStepSize = outerStepSize0 * (1 - i / n) # linear schedule
    model.load_state_dict({name : 
        weights_before[name] + (weights_after[name] - weights_before[name]) * outerStepSize
        for name in weights_before})
    # Plot the results on a particular task and minibatch
    if (i==0 or (i+1) % 10000 == 0 or i==n-1):
    #if i==n-1:
        print(f"-------------------------")
        plt.cla()
        plt.gca().set_aspect('equal')
        f = f_plot
        weights_before = deepcopy(model.state_dict()) # save snapshot before evaluation
        plt.pcolor(y1_all, y2_all, error_plane)
        plt.pause(0.01)
        for j in range(32):
            train_on_batch(x_train_plot, f(x_train_plot))
            if (j + 1) % 8 == 0:
                frac = (j + 1) / 32
                update_error_plane()
                print(f"pred after {j + 1} : {np.average(error_plane, weights=(error_plane >= 0))}")
                plt.pcolor(y1_all, y2_all, error_plane)
        plt.plot(f(x_train_plot)[0], f(x_train_plot)[1], "x", label="train", color="k")
        if i != n-1:
            plt.pause(0.01)
        else:
            plt.pause(5)
        model.load_state_dict(weights_before) # restore from snapshot