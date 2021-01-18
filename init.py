import torch
from torch import nn

from skimage import io
import scipy
import numpy as np

import json
import os

from Model import SegNet
from Utils import Structured_Dataset


if __name__ == "__main__":

	# Hyperparameters 
	BATCH_SIZE = 32
	INPUT_IMAGE_SIZE = 128
	EPOCS = 10

	
	# Training data setup
	img_Folder = ""
	txt_File = ""
	img_Size = 180

	OUTPUT_PATH = './'
	SAVE_MODEL_EVERY_N_EPOC = 2


	dataset_obj = Structured_Dataset(txt_file=txt_File, root_dir=img_Folder, image_size=img_Size)
	train_data_loader = torch.utils.data.DataLoader(dataset_obj, batch_size=6, shuffle=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
	model = SegNet(3,22).to(device)

	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	mse = nn.MSELoss(reduction = 'mean')

	for epoch in range(EPOCS):
			
		for i, batch in enumerate(train_data_loader):
			labels, imageLs = batch
			imageLs =  imageLs.float().cuda()
			# reset gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(imageLs)
			loss = 0
			for o,l in zip(outputs, labels):
			loss += mse(o, l.cuda())
			loss.backward()
			optimizer.step()
	
	if i % 100: print(loss)
	
	OUTPUT_path = f'{OUTPUT_PATH}Weights_{epoch}_loss_{loss}.pt'

	if n % SAVE_MODEL_EVERY_N_EPOC:
			torch.save(model, OUTPUT_path)

	print("__done__")
	print(f"Final Loss: {loss}")


