from Model import SegNet
from Utils import Structured_Dataset

import torch
from torch.utils.tensorboard import SummaryWriter

from skimage import io
import scipy
import numpy as np

import json
import os

def weighted_loss(outputs, labels):
    weighted_filter = torch.ones(outputs.size(), dtype=torch.int32, device) * 0.1
	loss = ((o - l) ** 2) * weighted_filter.add(o)
	return torch.sum(loss)   

if __name__ == "__main__":
	# Hyperparameters 
	BATCH_SIZE 		 = 32
	INPUT_IMAGE_SIZE = 128
	EPOCS 			 = 10

	# Training data setup
	T_img_Folder = ""
	T_txt_File = ""

	V_img_Folder = ""
	V_txt_File = ""
	img_Size = 180

	OUTPUT_PATH 	= "Model"
	LOG_LOSS_T_PATH = "Loss/Train"
	LOG_LOSS_V_PATH = "Loss/Validate"
	SAVE_MODEL_EVERY_N_EPOC = 2

	###########
	if not os.path.exists(OUTPUT_PATH):
    	os.mkdir(OUTPUT_PATH)   
	if not os.path.exists(LOG_LOSS_T_PATH):
       	os.mkdir(LOG_LOSS_T_PATH) 
	if not os.path.exists(LOG_LOSS_V_PATH):
        os.mkdir(LOG_LOSS_V_PATH) 

	train_dataset = Structured_Dataset(txt_file=T_txt_File, root_dir=T_img_Folder, image_size=img_Size)
	val_dataset   = Structured_Dataset(txt_file=V_txt_File, root_dir=V_img_Folder, image_size=img_Size)
	
	train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=True)
	val_data_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=6, shuffle=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
	model = SegNet(3,22).to(device)

	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	mse = torch.nn.MSELoss(reduction = 'mean')

	writer = SummaryWriter()

	for epoch in range(EPOCS):
    		
		# Train
		model.train(True)
		for i, batch in enumerate(train_data_loader):
			labels, imageLs = batch
			imageLs = imageLs.float().cuda()
			# reset gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(imageLs)
			loss = 0
			for o,l in zip(outputs, labels):
    			loss += weighted_loss(outputs=o, labels=l)

			writer.add_scalar(LOG_LOSS_T_PATH, loss, epoch)
			loss.backward()
			optimizer.step()		
			if i % 500: print(f"{i} Training Loss: {loss}")

		# Validate
		model.train(False)
		for i, batch in enumerate(val_data_loader):
    		labels, imageLs = batch
			imageLs = imageLs.float().cuda()
			outputs = model(imageLs)
			
			loss = 0
			for o,l in zip(outputs, labels):
    			
			loss += mse(o, l.cuda())

			writer.add_scalar(LOG_LOSS_V_PATH, loss, epoch)
			if i % 100: print(f"{i} Validation Loss: {loss}")

		writer.flush()
	
	OUTPUT_path = f'{OUTPUT_PATH}/Weights_{epoch}_loss_{loss}.pt'
	if n % SAVE_MODEL_EVERY_N_EPOC:
			torch.save(model, OUTPUT_path)

	print("__done__")
	writer.close()
	# tensorboard --logdir=runs		http://localhost:6006/


