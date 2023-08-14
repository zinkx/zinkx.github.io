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


def data_gen():
    cnt = 0
    while cnt <= total_num_steps:
        temp_tensor = x_torch
        y_arrays = []
        #iterate over network layers
        for i, module in enumerate(list(network.modules())[0]):
            temp_tensor = module(temp_tensor)
            temp_numpy = temp_tensor.detach().numpy()
            #iterate over neurons
            for hidden_dim in range(temp_numpy.shape[1]):
                    y_arrays.append(temp_numpy[:,hidden_dim])
        
        train_loop(steps_per_loop)
        cnt+=steps_per_loop
        # return updated lines
        yield y_arrays

def run(data):
    # update the data
    y_arrays = data

    #update subplots
    for ax in axs:
            ax.figure.canvas.draw()
    
    for i in range(len(lines)):
        lines[i].set_data(x,y_arrays[i])
    
    return lines

filename = 'relu_multi_layer.gif'
# setting
function = lambda x : x*x
value_low = -5
value_high = 5
x = np.linspace(-5,5, 1000)
batchsize = 256

#setup your network
network = torch.nn.Sequential(torch.nn.Linear(1,2),torch.nn.ReLU(),torch.nn.Linear(2,1)) 
optimizer = torch.optim.SGD(network.parameters(), lr = 0.01)
loss = torch.nn.MSELoss()

#setup animation speed and length 
steps_per_loop = 10
total_num_steps = 5000

#load data to torch
x_torch = torch.tensor(x.reshape(-1,1), dtype=torch.float32)

# create figure with one subplot for each layer
fig, axs = plt.subplots(len(list(network.modules())[0]),1,figsize = (10,10))

# intialize line objects (one in each axes)
lines = []
x_temp = x_torch
#iterate over layers
for  i, module in enumerate(list(network.modules())[0]):
    x_temp = module(x_temp)
    y = x_temp.detach().numpy()
    #iterate over neurons
    for hidden_dim in range(y.shape[1]): 
        line, = axs[i].plot(x,y[:,hidden_dim])
        lines.append(line)
    axs[i].set_title(str(module))

#draw goal function in final layer
axs[-1].plot(x,function(x), linestyle = 'dashed')

#define axis range
for ax in axs:
    ax.set_xlim(value_low, value_high)    
    ax.set_ylim(-30, 30)
    ax.grid()

ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=1,
    repeat=True)
#save to file
plt.tight_layout()
writergif = animation.PillowWriter(fps=30)
ani.save(filename,writer=writergif)
