{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_8863/3982182672.py:12: DeprecationWarning: `set_matplotlib_formats` is deprecated since IPython 7.23, directly use `matplotlib_inline.backend_inline.set_matplotlib_formats()`\n",
      "  set_matplotlib_formats('svg', 'pdf') # For export\n",
      "/home/leo/miniconda3/lib/python3.10/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: /home/leo/miniconda3/lib/python3.10/site-packages/torchvision/image.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE\n",
      "  warn(f\"Failed to load image Python extension: {e}\")\n"
     ]
    }
   ],
   "source": [
    "## Standard libraries\n",
    "import os\n",
    "import numpy as np\n",
    "import random\n",
    "from PIL import Image\n",
    "from types import SimpleNamespace\n",
    "from tqdm import tqdm\n",
    "## Imports for plotting\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "from IPython.display import set_matplotlib_formats\n",
    "set_matplotlib_formats('svg', 'pdf') # For export\n",
    "import matplotlib\n",
    "matplotlib.rcParams['lines.linewidth'] = 2.0\n",
    "import seaborn as sns\n",
    "sns.reset_orig()\n",
    "\n",
    "import shapes as sh\n",
    "\n",
    "## PyTorch\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.utils.data as data\n",
    "import torch.optim as optim\n",
    "# Torchvision\n",
    "import torchvision\n",
    "from torchvision import transforms"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)\n",
    "DATASET_PATH = \"../data\"\n",
    "# Path to the folder where the pretrained models are saved\n",
    "CHECKPOINT_PATH = \"../saved_models/tutorial5\"\n",
    "\n",
    "# Function for setting the seed\n",
    "def set_seed(seed):\n",
    "    random.seed(seed)\n",
    "    np.random.seed(seed)\n",
    "    torch.manual_seed(seed)\n",
    "    if torch.cuda.is_available():\n",
    "        torch.cuda.manual_seed(seed)\n",
    "        torch.cuda.manual_seed_all(seed)\n",
    "set_seed(42)\n",
    "\n",
    "# Ensure that all operations are deterministic on GPU (if used) for reproducibility\n",
    "torch.backends.cudnn.deterministic = True\n",
    "torch.backends.cudnn.benchmark = False\n",
    "\n",
    "device = torch.device(\"cuda:0\") if torch.cuda.is_available() else torch.device(\"cpu\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def genData(amount, N=201):\n",
    "    WFDataSinos = []\n",
    "    WFData = []\n",
    "    for counter in range(amount):\n",
    "        #print(counter)\n",
    "        # if counter > amount//2:\n",
    "        #     randSize = np.random.randint(2, 4)\n",
    "        #     shape = generatePolygon(randSize)\n",
    "        #     WFSetList = polygonToWFsetList(shape, gridSize=N, angleAccuracy=360)\n",
    "        # else:\n",
    "        shape = sh.genEll()\n",
    "        WFSetList = sh.ellipseToWFsetList(shape, gridSize=N, angleAccuracy=360)\n",
    "        perm = np.random.permutation(N+1)\n",
    "        WF = torch.tensor(dim3WFListGridNoDouble(WFSetList, N=N)[perm,:,:].reshape(-1)) #torch.nonzero?\n",
    "        SinoWF = torch.tensor(sh.dim3getSinoWFFromListAsGrid(WFSetList, N=N)[perm,:,:].reshape(-1))\n",
    "        #arr = [np.array([SinoWF[j][0], SinoWF[j][1], WF[j][0], WF[j][1], SinoWF[j][2], WF[j][2]]) for j in range(len(WF))]\n",
    "        WFData.append(WF)\n",
    "        WFDataSinos.append(SinoWF)\n",
    "    return (WFDataSinos,WFData)\n",
    "\n",
    "def dim3WFListGridNoDouble(WFList, N=201):\n",
    "    WF = np.zeros((N+1,N+1,180),dtype=\"float32\")#change this back to float32s\n",
    "    for val in WFList:\n",
    "        pointGrid = val[0]\n",
    "        x = pointGrid[0]\n",
    "        y = pointGrid[1]\n",
    "        angles = [ang%180 for ang in val[1]]\n",
    "        for angle in angles:\n",
    "            WF[x,y,angle] = 1\n",
    "    return WF\n",
    "\n",
    "\n",
    "class WFDataset(data.Dataset):\n",
    "\n",
    "    def __init__(self, size):\n",
    "        \"\"\"\n",
    "        Inputs:\n",
    "            size - Number of data points we want to generate\n",
    "            std - Standard deviation of the noise (see generate_continuous_xor function)\n",
    "        \"\"\"\n",
    "        super().__init__()\n",
    "        self.size = size\n",
    "        self.generate_continuous_xor()\n",
    "\n",
    "    def generate_continuous_xor(self):\n",
    "        # Each data point in the XOR dataset has two variables, x and y, that can be either 0 or 1\n",
    "        # The label is their XOR combination, i.e. 1 if only x or only y is 1 while the other is 0.\n",
    "        # If x=y, the label is 0.\n",
    "        data, label = genData(self.size)\n",
    "\n",
    "        self.data = data\n",
    "        self.label = label\n",
    "\n",
    "    def __len__(self):\n",
    "        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]\n",
    "        return self.size\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        # Return the idx-th data point of the dataset\n",
    "        # If we have multiple things to return (data point and label), we can return them as tuple\n",
    "        data_point = self.data[idx]\n",
    "        data_label = self.label[idx]\n",
    "        return data_point, data_label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "set_seed(43)\n",
    "train_dataset = WFDataset(100)\n",
    "val_dataset = WFDataset(100)\n",
    "test_set = WFDataset(10)\n",
    "\n",
    "set_seed(42)\n",
    "train_set, _ = torch.utils.data.random_split(train_dataset, [90, 10])\n",
    "set_seed(42)\n",
    "_, val_set = torch.utils.data.random_split(val_dataset, [90, 10]);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([6544800])"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "torch.Tensor.size(train_dataset.data[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_loader = data.DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)\n",
    "val_loader = data.DataLoader(val_set, batch_size=8, shuffle=False, drop_last=False, num_workers=4)\n",
    "test_loader = data.DataLoader(test_set, batch_size=8, shuffle=False, drop_last=False, num_workers=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):\n",
    "    # Set model to train mode\n",
    "    model.train()\n",
    "\n",
    "    # Training loop\n",
    "    for epoch in tqdm(range(num_epochs)):\n",
    "        for data_inputs, data_labels in data_loader:\n",
    "\n",
    "            ## Step 1: Move input data to device (only strictly necessary if we use GPU)\n",
    "            data_inputs = data_inputs.to(device)\n",
    "            data_labels = data_labels.to(device)\n",
    "\n",
    "            ## Step 2: Run the model on the input data\n",
    "            preds = model(data_inputs)\n",
    "            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]\n",
    "\n",
    "            ## Step 3: Calculate the loss\n",
    "            loss = loss_module(preds, data_labels.float())\n",
    "\n",
    "            ## Step 4: Perform backpropagation\n",
    "            # Before calculating the gradients, we need to ensure that they are all zero.\n",
    "            # The gradients would not be overwritten, but actually added to the existing ones.\n",
    "            optimizer.zero_grad()\n",
    "            # Perform backpropagation\n",
    "            loss.backward()\n",
    "\n",
    "            ## Step 5: Update the parameters\n",
    "            optimizer.step()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "class SimpleClassifier(nn.Module):\n",
    "\n",
    "    def __init__(self, num_inputs, num_hidden, num_outputs):\n",
    "        super().__init__()\n",
    "        # Initialize the modules we need to build the network\n",
    "        self.linear1 = nn.Linear(num_inputs, num_hidden)\n",
    "        self.act_fn = nn.Tanh()\n",
    "        self.linear2 = nn.Linear(num_hidden, num_outputs)\n",
    "\n",
    "    def forward(self, x):\n",
    "        # Perform the calculation of the model to determine the prediction\n",
    "        x = self.linear1(x)\n",
    "        x = self.act_fn(x)\n",
    "        x = self.linear2(x)\n",
    "        return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = SimpleClassifier(num_inputs=202*180*180, num_hidden=100, num_outputs=202*202*180)\n",
    "model.to(device)\n",
    "loss_module = nn.BCEWithLogitsLoss()\n",
    "optimizer = torch.optim.SGD(model.parameters(), lr=0.1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  0%|          | 0/100 [00:00<?, ?it/s]"
     ]
    }
   ],
   "source": [
    "train_model(model, optimizer, train_loader, loss_module)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "state_dict = model.state_dict()\n",
    "torch.save(state_dict, \"our_model.tar\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "def eval_model(model, data_loader):\n",
    "    model.eval() # Set model to eval mode\n",
    "    true_preds, num_preds = 0., 0.\n",
    "\n",
    "    with torch.no_grad(): # Deactivate gradients for the following code\n",
    "        for data_inputs, data_labels in data_loader:\n",
    "\n",
    "            # Determine prediction of model on dev set\n",
    "            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)\n",
    "            preds = model(data_inputs)\n",
    "            #preds = preds.squeeze(dim=1)\n",
    "            preds = torch.sigmoid(preds) # Sigmoid to map predictions between 0 and 1\n",
    "            pred_labels = (preds >= 0.5).long() # Binarize predictions to 0 and 1\n",
    "            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)\n",
    "            true_preds += (pred_labels == data_labels).sum()/len(data_labels[0])\n",
    "            num_preds += data_labels.shape[0]\n",
    "\n",
    "    acc = true_preds / num_preds\n",
    "    print(f\"Accuracy of the model: {100.0*acc:4.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy of the model: 249.98%\n"
     ]
    }
   ],
   "source": [
    "eval_model(model, test_loader)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "57164c7611d90ceeabef372623b6aac80902bfc5abfc3644ac98a064f211ff35"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}