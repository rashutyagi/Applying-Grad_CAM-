from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchsummary
from torchsummary import summary

def display_my_images(train_loader, classes):
	images, labels = next(iter(train_loader))
	fig=plt.figure(figsize=(40,16))
	for i in range(10):
		ax=fig.add_subplot(2,10, i+1)
		img=np.squeeze(images[i].numpy())
		img=img/2 +0.5
		img=np.transpose(img, (1, 2, 0))
		ax.imshow(img)
		ax.set_title(str(classes[labels[i].item()]))

