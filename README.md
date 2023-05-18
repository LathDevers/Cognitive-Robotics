This repository contains data for **Miniproject 10: Meta-Learning with Reptile** in the **Cognitive Robotics** lecture taught by *Prof. Dr. Helge Ritter* (AG Neuroinformatik, Universität Bielefeld). The miniproject focuses on exploring a variant of the Model-Agnostic Meta-Learning (MAML) method called **Reptile**. For more information on Reptile and its scalability as a meta-learning algorithm, refer to the web article "Reptile - a scalable Meta-Learning Algorithm" published by OpenAI [here](https://openai.com/blog/reptile/). Additionally, you may find insights on first-order meta-learning algorithms in the paper titled [On First-Order Meta-Learning Algorithms](/src/metalearning-Nichol2018.pdf) by A. Nichol, J. Achiam, and J. Schulman.

### Tasks:

1. read the web-article, the comment paper and inspect the code (take details from the original article where missing)
2. reproduce the learning of the example function (parameters as in the original paper, i.e. 1-64-64-1-network, 32 gradient steps, amplitude and phase ranges as above)
3. consider the cases 
   1. of plain learning from a random weight initialization and
   2. learning after weight optimization according to the Reptile algorithm of the example implementation
4. visualize in each case the results before and after training and compare the outcomes of (i) and (ii) above
5. finally, replace the sine function by the forward kinematics of a two-link robot arm with segment lengths $A ∈ [1, 2]$ and $B ∈ [0.5, 1]$, and joint angles $x_1, x_2 ∈ [−\frac{π}{2}, \frac{π}{2}]$, end effector coordinates $(y_1, y_2)$ (and no phase):

    $y_1 = A\cdot\cos(x_1) + B\cdot\cos(x_1 + x_2)$\
    $y_2 = A\cdot\sin(x_1) + B\cdot\sin(x_1 + x_2)$
    
    (this requires to use a network that can transform a pair $(x_1, x_2)$ of joint angles to a pair of end effector coordinates $(y_1, y_2)$. E.g. experiment with a 2-64-64-2-shaped network and again use 10 randomly sampled training points)
1. repeat the above experiments for this case, now visualizing the error as an error surface above the $y_1, y_2$ space (e.g., visualizing error as color)
2. create an interactive result report about your exploration results on the Reptile algorithm
