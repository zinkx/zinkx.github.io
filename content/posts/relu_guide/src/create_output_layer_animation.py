import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation



def train_loop(train_steps):
    for _ in range(train_steps):
        #sample random train data
        batch = np.random.rand(batchsize,1) * (value_high-value_low) + value_low
        targets = function(batch)
        batch = torch.tensor(batch, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        predictions = network(batch)
        
        loss_value = loss(targets, predictions)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

#define animation loop
def animate(i):    
    train_loop(10)
    y = network(x_torch).detach().numpy().squeeze()
    line.set_ydata(y)
        
    return line,

filename = '../images/relu_net_20neurons_out.gif'
# setting
function = lambda x : x*x
value_low = -5
value_high = 5
x = np.linspace(-5,5, 1000)
batchsize = 256

#setup your network
network = torch.nn.Sequential(torch.nn.Linear(1,20),torch.nn.ReLU(),torch.nn.Linear(20,1)) 
optimizer = torch.optim.SGD(network.parameters(), lr = 0.01)
loss = torch.nn.MSELoss()

#load data to torch
x_torch = torch.tensor(x.reshape(-1,1), dtype=torch.float32)

#create plot
fig, axs = plt.subplots(1, figsize=(7,5))
#plot true function
axs.plot(x, function(x), linestyle="dashed")

#first aproximation random weights
y = network(x_torch).detach().numpy().squeeze()
line, = axs.plot(x,y)

#run animation
ani = animation.FuncAnimation(fig, animate, repeat=False, frames = 100,
                                        interval=10, blit=True, save_count=3)
plt.tight_layout()
#save to gif
writergif = animation.PillowWriter(fps=30)
ani.save(filename,writer=writergif)
