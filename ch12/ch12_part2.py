# coding: utf-8


import sys
from python_environment_check import check_packages
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from itertools import islice
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from scipy.special import expit

# # Machine Learning with PyTorch and Scikit-Learn  
# # -- Code Examples

# ## Package version checks

# Add folder to path in order to load from the check_packages.py script:



sys.path.insert(0, '..')


# Check recommended package versions:





d = {
    'numpy': '1.21.2',
    'scipy': '1.7.0',
    'sklearn': '1.0.0',
    'matplotlib': '3.4.3',
    'torch': '1.9.0',
}
check_packages(d)


# # Chapter 12: Parallelizing Neural Network Training with PyTorch  (Part 2/2)
# 

# - [Building an NN model in PyTorch](#Building-an-NN-model-in-PyTorch)
#   - [The PyTorch neural network module (torch.nn)](#The-PyTorch-neural-network-module-(torch.nn))
#   - [Building a linear regression model](#Building-a-linear-regression-model)
#   - [Model training via the torch.nn and torch.optim modules](#Model-training-via-the-torch.nn-and-torch.optim-modules)
#   - [Building a multilayer perceptron for classifying flowers in the Iris dataset](#Building-a-multilayer-perceptron-for-classifying-flowers-in-the-Iris-dataset)
#   - [Evaluating the trained model on the test dataset](#Evaluating-the-trained-model-on-the-test-dataset)
#   - [Saving and reloading the trained model](#Saving-and-reloading-the-trained-model)
# - [Choosing activation functions for multilayer neural
# networks](#Choosing-activation-functions-for-multilayer-neural-networks)
#   - [Logistic function recap](#Logistic-function-recap)
#   - [Estimating class probabilities in multiclass classification via the softmax function](#Estimating-class-probabilities-in-multiclass-classification-via-the-softmax-function)
#   - [Broadening the output spectrum using a hyperbolic tangent](#Broadening-the-output-spectrum-using-a-hyperbolic-tangent)
#   - [Rectified linear unit activation](#Rectified-linear-unit-activation)
# - [Summary](#Summary)

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).





# ## Building a neural network model in PyTorch

# ### The PyTorch neural network module (torch.nn)

# ### Building a linear regression model







X_train = np.arange(10, dtype='float32').reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 
                    7.4, 8.0, 9.0], dtype='float32')

plt.plot(X_train, y_train, 'o', markersize=10)
plt.xlabel('x')
plt.ylabel('y')

#plt.savefig('figures/12_07.pdf')
plt.show()





X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm)

# On some computers the explicit cast to .float() is
# necessary
y_train = torch.from_numpy(y_train).float()

train_ds = TensorDataset(X_train_norm, y_train)
for feature, label in train_ds:
    print("feaure =", feature, "    label =", label)

print("_"*40)
batch_size = 2
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
for i, (feature, label) in enumerate(train_dl):
    print(f"batch {i+1}:")
    print("    feaure =\n", feature, "    label =", label)


# Because the batch size is $2$, we have 5 different batches. And notice after shuffling, (feature, label) pairs are preserved.



torch.manual_seed(1)

# Weight terms (all scalar for this example, because feature x is a scalar)
weight = torch.randn(1, requires_grad=True)  # requires_grad=True means the gradient of Loss w.r.t. this tensor will be tracked through the network.
bias = torch.zeros(1, requires_grad=True)  # requires_grad=True means the gradient of Loss w.r.t. this tensor will be tracked through the network.

# Loss function
def loss_fn(input, target):
    return (input - target).pow(2).mean()  # Mean squared error

# Our model to generate yhat values
def model(xb):
    return xb @ weight + bias

learning_rate = 0.001
num_epochs = 200
log_epochs = 10
 


# ![image-3.png](attachment:image-3.png)
# 
# ![image-2.png](attachment:image-2.png)



for epoch in range(num_epochs):
    # Because batch size is 2, this is a mini-batch stochastic gradient descent algorithm. (Looping over each 2 data points separately)
    for x_batch, y_batch in train_dl:  # train_dl is a torch.DataLoader class initiated above
        pred = model(x_batch)  # yhat, prediction vector of our model, here it has shape (2,) because batch size is 2.
        loss = loss_fn(pred, y_batch)  # loss, MSE of the model (scalar) 
        loss.backward()  # starts the backpropagation process such that 

        with torch.no_grad():
            weight -= weight.grad * learning_rate  # gradient descent update
            bias -= bias.grad * learning_rate  # gradient descent update
            weight.grad.zero_()  # reset the gradient, because we are manually looping.
            bias.grad.zero_()  # reset the gradient, because we are manually looping.
 
    if epoch % log_epochs==0:
        print(f'Epoch {epoch}  Loss {loss.item():.4f}')


# ![image.png](attachment:image.png)
# 
# ![image-2.png](attachment:image-2.png)



print('Final Parameters:', weight.item(), bias.item())
 
X_test = np.linspace(0, 9, num=100, dtype='float32').reshape(-1, 1)
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
X_test_norm = torch.from_numpy(X_test_norm)
y_pred = model(X_test_norm).detach().numpy()


fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(1, 2, 1)
plt.plot(X_train_norm, y_train, 'o', markersize=10)
plt.plot(X_test_norm, y_pred, '--', lw=3)
plt.legend(['Training examples', 'Linear Reg.'], fontsize=15)
ax.set_xlabel('x', size=15)
ax.set_ylabel('y', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
 
#plt.savefig('figures/12_08.pdf')

plt.show()


# ### My try:



np.set_printoptions(precision=3, suppress=True)

n = 1000
k = n//2
full_X = np.random.normal(20, 5, (n, 3))  # X is a (200, 3) shape array, 200 observations, 3 dimensions
X = full_X[:k]
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_test = full_X[k:]
d = X.shape[1]
X[:5, :]  # first few rows of X




true_params = [np.array([2, 1, 3]), 10]  # true parameters to generate Y, where beta = (2, 1, 3), beta0 = 10
full_Y = full_X @ true_params[0] + true_params[1] + np.random.standard_normal(n)  # Generate Y with additional gaussian noise ~N(0, 9)
Y = full_Y[:k]

# mean and std of the train Y labels
Y_train_mean = Y.mean()
Y_train_std = Y.std()
print("Y_train_std =", Y_train_std)
Y = (Y - Y_train_mean) / Y_train_std  # normalize labels as well.

Y_test = full_Y[k:]
Y_test_normalized = (Y_test - Y_test.mean()) / Y_test.std()
Y[:5]  # first few rows of Y




# Initialize TensorDataset object to be used in a DataLoader object later
XY_ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
for x, y in islice(XY_ds, 5):  # First 5 (feature, label) pairs
    print("x =\n", x, "    y =", y)




# Initialize DataLoader objcet
XY_dl = DataLoader(XY_ds, batch_size=5, shuffle=True)
for i, (feature, label) in enumerate(XY_dl):
    print("_"*40)
    print(f"batch {i+1}:")
    print("feature tensor:\n", feature, "    label =", label)




num_epochs = 1000  # run 1000 epochs
eta = 1e-4  # learning rate

# Initialize random weights with requires_grad=True
beta = torch.randn(d, requires_grad=True)
beta0 = torch.randn(1, requires_grad=True)

def model_linreg(x):
    yhat = x @ beta + beta0
    return yhat

def lossfunc(y, yhat):
    """Calculates the MSE for the batch.

    Args:
        y (torch.Tensor): shape (batch_size,) rank 1 tensor, true labels
        yhat (torch.Tensor): shape (batch_size,) rank 1 tensor, predicted labels
    """
    return ((yhat - y)**2).mean()

loss_arr = []
for epoch in range(num_epochs):
    for x_batch, y_batch in XY_dl:
        # print("beta0 =", beta0)
        yhat = model_linreg(x_batch)
        loss = lossfunc(y_batch, yhat)
        loss.backward()
        # print("beta.grad =", beta.grad)
        # print("beta0.grad =", beta0.grad)
        
        with torch.no_grad():
            beta -= eta*beta.grad
            beta0 -= eta*beta0.grad
            beta.grad.zero_()
            beta0.grad.zero_()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch} loss =", loss)
    loss_arr.append(loss.detach().numpy())

plt.plot(loss_arr)
plt.title("Loss")

print("Final predictions:")
print("beta =", beta)
print("beta0 =", beta0)
print("beta dtype =", beta.dtype)
print("beta0 dtype =", beta0.dtype)


# Now let's check the final weights against the test data



X_test_normalized = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
X_test_normalized = torch.from_numpy(X_test_normalized).to(torch.float32)  # convert from np.array to torch.Tensor object
yhat_test_normalized = model_linreg(X_test_normalized)  # we get normalized labels
yhat_test = yhat_test_normalized * Y_train_std + Y_train_mean  # we de-normalize the predicted labels

# Calculate normalized loss
loss_test_normalized = lossfunc(Y_test_normalized, yhat_test_normalized.detach())
print("MSE on the normalized test dataset =", loss_test_normalized)

# Calculate de-normalized loss
loss_test_denormalized = lossfunc(Y_test, yhat_test.detach())
print("MSE on the de-normalized test dataset =", loss_test_denormalized)

test_y_diff = yhat_test.detach() - Y_test
print("yhat_test - y_test =\n", test_y_diff)
plt.scatter(Y_test, test_y_diff)
plt.xlabel("True labels $y$")
plt.ylabel("Residuals $\hat{y} - y$")


# ### Model training via the torch.nn and torch.optim modules

# ### *Single Layer Linear NN architecture*
# 
# ![image.png](attachment:image.png)




# Decide the number of input and output neurons in the layer
input_size = 1
output_size = 1
model = nn.Linear(input_size, output_size)  # This is just a single linear layer.

# Choose loss function (e.g. Mean Squared Error, Hinge Loss etc.)
loss_fn = nn.MSELoss(reduction='mean')

# Choose optimizer (e.g. Stochastic Gradient Descent)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# ![image.png](attachment:image.png)



for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        # 1. Generate predictions
        pred = model(x_batch)[:, 0] 

        # 2. Calculate loss
        loss = loss_fn(pred, y_batch)

        # 3. Compute gradients
        loss.backward()

        # 4. Update parameters using gradients
        optimizer.step()  # one step forward via SGD

        # 5. Reset the gradients to zero
        optimizer.zero_grad()
        
    if epoch % 50==0:
        print(f'Epoch {epoch}  Loss {loss.item():.4f}')




print('Final Parameters:', model.weight.item(), model.bias.item())
 
X_test = np.linspace(0, 9, num=100, dtype='float32').reshape(-1, 1)
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
X_test_norm = torch.from_numpy(X_test_norm)
y_pred = model(X_test_norm).detach().numpy()


fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(1, 2, 1)
plt.plot(X_train_norm.detach().numpy(), y_train.detach().numpy(), 'o', markersize=10)
print("X_test_norm type:", type(X_test_norm))
plt.plot(X_test_norm, y_pred, '--', lw=3)
plt.legend(['Training examples', 'Linear reg.'], fontsize=15)
ax.set_xlabel('x', size=15)
ax.set_ylabel('y', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
 
#plt.savefig('ch12-linreg-2.pdf')

plt.show()


# ## Building a multilayer perceptron for classifying flowers in the Iris dataset




iris = load_iris()
X = iris['data']
y = iris['target']
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1./3, random_state=1)

print("X_train type:", type(X_train))
print("X_train[:5] =", X_train[:5])
print("y_train type:", type(y_train))
print("y_train[:5] =", y_train[:5])





# Standardize datasets and convert to torch.Tensor objects
X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm).float()
y_train = torch.from_numpy(y_train)  # labels does NOT need standardization as they are (integer) CLASS labels.

train_ds = TensorDataset(X_train_norm, y_train)

torch.manual_seed(1)
batch_size = 2
train_dl = DataLoader(train_ds, batch_size, shuffle=True)




for i, (feature, label) in enumerate(train_dl):
    print("_"*40)
    print(f"batch{i}:")
    print("features =\n", feature)
    print("    label =\n", label)


# ## NN Architecture of this model:
# 
# ![image-2.png](attachment:image-2.png)



class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)  
        self.layer2 = nn.Linear(hidden_size, output_size)  

    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        x = nn.Softmax(dim=1)(x)
        return x

input_size = X_train_norm.shape[1]
hidden_size = 16
output_size = 3

model = Model(input_size, hidden_size, output_size)

learning_rate = 0.001

loss_fn = nn.CrossEntropyLoss()
 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




num_epochs = 100
loss_hist = [0] * num_epochs
accuracy_hist = [0] * num_epochs

for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch.long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_hist[epoch] += loss.item()*y_batch.size(0)
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        accuracy_hist[epoch] += is_correct.sum()

    loss_hist[epoch] /= len(train_dl.dataset)
    accuracy_hist[epoch] /= len(train_dl.dataset)




fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(loss_hist, lw=3)
ax.set_title('Training loss', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)

ax = fig.add_subplot(1, 2, 2)
ax.plot(accuracy_hist, lw=3)
ax.set_title('Training accuracy', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()

#plt.savefig('figures/12_09.pdf')
 
plt.show()


# My try:



class MyNNModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = nn.Tanh()(x)
        x = self.layer2(x)
        x = nn.Softmax(dim=1)(x)
        return x


iris = load_iris()
X = iris['data']
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1./3, random_state=1)

X_train = torch.from_numpy((X_train - X_train.mean()) / X_train.std(ddof=1)).float()
y_train = torch.from_numpy(y_train).long()  # Use long instead of float for classification

# Move data to the selected device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")
X_train = X_train.to(device)
y_train = y_train.to(device)

# Construct DatSet object
train_ds = TensorDataset(X_train, y_train)

# Construct DataLoader object
train_dl = DataLoader(train_ds, batch_size=20, shuffle=True)

# Initialize my Module subclass
# feature vector size is extracted from X_train array
# Hidden size is kind of arbitrary ?
# There are 3 classes of flowers so output_size = 3
model = MyNNModule(input_size=X_train.shape[1], hidden_size=16, output_size=3)
model.to(device)  # convert model to cuda device

eta = 0.001  # learning rate
num_epochs = 10000  # number of epochs
loss_fn = nn.CrossEntropyLoss()  # Cross entropy loss
optimizer = torch.optim.Adam(model.parameters(), eta)  # Use SGD for updating, learning rate is eta

loss_arr = [0] * num_epochs
# Iterate over epochs
for epoch in range(num_epochs):
    for (batch_x, batch_y) in train_dl:
        loss = 0
        # print("batch_x =\n", batch_x)
        # print("batch_y =\n", batch_y)

        # 1. Generate predictions
        pred = model(batch_x)

        # 2. Calculate and print loss
        loss += loss_fn(pred, batch_y)
        loss.backward()  # back-propagate the gradients

        # 3. Step forward
        optimizer.step()
        optimizer.zero_grad()  # reset gradient

    loss_arr[epoch] = (loss.detach().cpu().numpy())

print(loss_arr)
plt.plot(loss_arr)


# ### Evaluating the trained model on the test dataset



X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
X_test_norm = torch.from_numpy(X_test_norm).float()
y_test = torch.from_numpy(y_test) 
pred_test = model(X_test_norm)

correct = (torch.argmax(pred_test, dim=1) == y_test).float()
accuracy = correct.mean()
 
print(f'Test Acc.: {accuracy:.4f}')


# ### Saving and reloading the trained model



path = 'iris_classifier.pt'
torch.save(model, path)




model_new = torch.load(path)
model_new.eval()




pred_test = model_new(X_test_norm)

correct = (torch.argmax(pred_test, dim=1) == y_test).float()
accuracy = correct.mean()
 
print(f'Test Acc.: {accuracy:.4f}')




path = 'iris_classifier_state.pt'
torch.save(model.state_dict(), path)




model_new = Model(input_size, hidden_size, output_size)
model_new.load_state_dict(torch.load(path))


# ## Choosing activation functions for multilayer neural networks
# 

# ### Logistic function recap




X = np.array([1, 1.4, 2.5]) ## first value must be 1
w = np.array([0.4, 0.3, 0.5])

def net_input(X, w):
    return np.dot(X, w)

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)

print(f'P(y=1|x) = {logistic_activation(X, w):.3f}') 




# W : array with shape = (n_output_units, n_hidden_units+1)
# note that the first column are the bias units

W = np.array([[1.1, 1.2, 0.8, 0.4],
              [0.2, 0.4, 1.0, 0.2],
              [0.6, 1.5, 1.2, 0.7]])

# A : data array with shape = (n_hidden_units + 1, n_samples)
# note that the first column of this array must be 1

A = np.array([[1, 0.1, 0.4, 0.6]])
Z = np.dot(W, A[0])
y_probas = logistic(Z)
print('Net Input: \n', Z)

print('Output Units:\n', y_probas) 




y_class = np.argmax(Z, axis=0)
print('Predicted class label:', y_class) 


# ### Estimating class probabilities in multiclass classification via the softmax function



def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

y_probas = softmax(Z)
print('Probabilities:\n', y_probas)

np.sum(y_probas)




torch.softmax(torch.from_numpy(Z), dim=0)


# ### Broadening the output spectrum using a hyperbolic tangent




def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)

z = np.arange(-5, 5, 0.005)
log_act = logistic(z)
tanh_act = tanh(z)
plt.ylim([-1.5, 1.5])
plt.xlabel('Net input $z$')
plt.ylabel('Activation $\phi(z)$')
plt.axhline(1, color='black', linestyle=':')
plt.axhline(0.5, color='black', linestyle=':')
plt.axhline(0, color='black', linestyle=':')
plt.axhline(-0.5, color='black', linestyle=':')
plt.axhline(-1, color='black', linestyle=':')
plt.plot(z, tanh_act,
    linewidth=3, linestyle='--',
    label='Tanh')
plt.plot(z, log_act,
    linewidth=3,
    label='Logistic')
plt.legend(loc='lower right')
plt.tight_layout()

#plt.savefig('figures/12_10.pdf')
plt.show()




np.tanh(z)




torch.tanh(torch.from_numpy(z))
 





expit(z)




torch.sigmoid(torch.from_numpy(z))


# ### Rectified linear unit activation



torch.relu(torch.from_numpy(z))




IPythonImage(filename='figures/12_11.png', width=500)


# ## Summary

# ---
# 
# Readers may ignore the next cell.




