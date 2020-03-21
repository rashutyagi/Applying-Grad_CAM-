from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchsummary
from torchsummary import summary

from Albumentations_transform import TrainAlbumentation,TestAlbumentation


def get_and_transform_the_data():
	use_cuda = torch.cuda.is_available()

	cuda = torch.cuda.is_available()
	print("CUDA Available?", cuda)
	
	SEED=1  # for reproducability
	
	torch.manual_seed(SEED)

	if cuda:
		torch.cuda.manual_seed(SEED)
        
        
	transform_test = TestAlbumentation()
	transform_train = TrainAlbumentation()
    
	#transform = transforms.Compose(
	#	[transforms.RandomCrop(32, padding=4),
	#	 transforms.RandomHorizontalFlip(),
	#	 transforms.ToTensor(),
	#	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
											download=True, transform=transform_train)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
											  shuffle=True, num_workers=4)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
										   download=True, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=128,
											 shuffle=False, num_workers=4)

	classes = ('plane', 'car', 'bird', 'cat',
			   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	
	return trainset, testset, train_loader, test_loader, classes
