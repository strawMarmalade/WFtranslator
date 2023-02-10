import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0"
import numpy as np
import random
# from PIL import Image
# from types import SimpleNamespace
from tqdm import tqdm
## Imports for plotting
# import matplotlib.pyplot as plt
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('svg', 'pdf') # For export
# import matplotlib
# matplotlib.rcParams['lines.linewidth'] = 2.0
# import seaborn as sns
# sns.reset_orig()

import shapes as sh

## PyTorch
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
# Torchvision
# import torchvision
# from torchvision import transforms

# # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
# DATASET_PATH = "../data"
# # Path to the folder where the pretrained models are saved
# CHECKPOINT_PATH = "../saved_models/tutorial5"

# Function for setting the seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("device is ", device)

def genData(amount, N=201):
    WFDataSinos = []
    WFData = []
    for counter in range(amount):
        #print(counter)
        # if counter > amount//2:
        #     randSize = np.random.randint(2, 4)
        #     shape = generatePolygon(randSize)
        #     WFSetList = polygonToWFsetList(shape, gridSize=N, angleAccuracy=360)
        # else:
        shape = sh.genEll()
        WFSetList = sh.ellipseToWFsetList(shape, gridSize=N, angleAccuracy=360)
        perm = np.random.permutation(N+1)
        WF = torch.tensor(dim3WFListGridNoDouble(WFSetList, N=N)[perm,:,:].reshape(-1)) #torch.nonzero?
        SinoWF = torch.tensor(sh.dim3getSinoWFFromListAsGrid(WFSetList, N=N)[perm,:,:].reshape(-1))
        #arr = [np.array([SinoWF[j][0], SinoWF[j][1], WF[j][0], WF[j][1], SinoWF[j][2], WF[j][2]]) for j in range(len(WF))]
        WFData.append(WF)
        WFDataSinos.append(SinoWF)
    return (WFDataSinos,WFData)

def dim3WFListGridNoDouble(WFList, N=201):
    WF = np.zeros((N+1,N+1,180),dtype="float16")#change this back to float32s
    for val in WFList:
        pointGrid = val[0]
        x = pointGrid[0]
        y = pointGrid[1]
        angles = [ang%180 for ang in val[1]]
        for angle in angles:
            WF[x,y,angle] = 1
    return WF


class WFDataset(data.Dataset):

    def __init__(self, size):
        """
        Inputs:
            size - Number of data points we want to generate
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        self.size = size
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        # Each data point in the XOR dataset has two variables, x and y, that can be either 0 or 1
        # The label is their XOR combination, i.e. 1 if only x or only y is 1 while the other is 0.
        # If x=y, the label is 0.
        data, label = genData(self.size)

        self.data = data
        self.label = label

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label

set_seed(43)
train_dataset = WFDataset(1000)
#val_dataset = WFDataset(100)
#test_set = WFDataset(10)

set_seed(42)
train_set, _ = torch.utils.data.random_split(train_dataset, [900, 100])
# set_seed(42)
# _, val_set = torch.utils.data.random_split(val_dataset, [90, 10])

train_loader = data.DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
# val_loader = data.DataLoader(val_set, batch_size=4, shuffle=False, drop_last=False, num_workers=4)
# test_loader = data.DataLoader(test_set, batch_size=4, shuffle=False, drop_last=False, num_workers=4)


use_amp = True

def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
    # Set model to train mode
    model.train()

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:
            with torch.cuda.amp.autocast(enabled=use_amp, device_type='cuda', dtype=torch.float16):
                ## Step 1: Move input data to device (only strictly necessary if we use GPU)
                data_inputs = data_inputs.to(device)
                data_labels = data_labels.to(device)

                ## Step 2: Run the model on the input data
                preds = model(data_inputs)
                preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]

                ## Step 3: Calculate the loss
                loss = loss_module(preds, data_labels.float())

            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            ## Step 5: Update the parameters
            optimizer.step()

class SimpleClassifier(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        # Initialize the modules we need to build the network
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x

model = SimpleClassifier(num_inputs=202*180*180, num_hidden=200, num_outputs=202*202*180)
model.to(device, memory_format=torch.channels_last)
loss_module = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
print("starting training")
train_model(model, optimizer, train_loader, loss_module)

state_dict = model.state_dict()
torch.save(state_dict, "/home/lukasb/model2.tar")

def eval_model(model, data_loader):
    model.eval() # Set model to eval mode
    true_preds, num_preds = 0., 0.

    with torch.no_grad(): # Deactivate gradients for the following code
        for data_inputs, data_labels in data_loader:

            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            #preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds) # Sigmoid to map predictions between 0 and 1
            pred_labels = (preds >= 0.5).long() # Binarize predictions to 0 and 1
            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_labels == data_labels).sum()/len(data_labels[0])
            num_preds += data_labels.shape[0]

    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")

test_set = WFDataset(100)
test_loader = data.DataLoader(test_set, batch_size=4, shuffle=False, drop_last=False, num_workers=4)
eval_model(model, test_loader)
