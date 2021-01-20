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
  weighted_filter = torch.ones(outputs.size(), dtype=torch.int32, device="cuda:0") * 0.1
  loss = ((outputs - labels) ** 2)
  loss = loss * weighted_filter.add(labels).cuda()
  return torch.mean(loss).cuda()

if __name__ == "__main__":
	# Hyperparameters 
	BATCH_SIZE 		 = 24
	INPUT_IMAGE_SIZE = 128
	EPOCS 			 = 10

	# Training data setup
	T_img_Folder = "/content/Hands_ex/Training"
	T_txt_File = "/content/Hands_ex/Training/train_set.txt"

	V_img_Folder = "/content/Hands_ex/Validation"
	V_txt_File = "/content/Hands_ex/Validation/val_set.txt"
	img_Size = 180

	OUTPUT_PATH 	= "Model"
	LOG_LOSS_DIR 	= "Loss"
	LOG_LOSS_T_PATH = "Loss/Train"
	LOG_LOSS_V_PATH = "Loss/Validate"
	SAVE_MODEL_EVERY_N_EPOC = 2

	###########
	if not os.path.exists(OUTPUT_PATH):
		os.mkdir(OUTPUT_PATH)   
	if not os.path.exists(LOG_LOSS_DIR):
		os.mkdir(LOG_LOSS_DIR) 
		os.mkdir(LOG_LOSS_T_PATH) 
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
				loss += weighted_loss(outputs=o, labels=l.cuda())

			writer.add_scalar(LOG_LOSS_T_PATH, loss, epoch)
			loss.backward()
			optimizer.step()		
			if i % 100 == 0: print(f"{i} Training Loss: {loss}")

		# Validate
		model.train(False)
		for i, batch in enumerate(val_data_loader):
			labels, imageLs = batch
			imageLs = imageLs.float().cuda()
			outputs = model(imageLs)
			
			loss = 0
			for o,l in zip(outputs, labels):
				loss += weighted_loss(outputs=o, labels=l.cuda())
				loss += mse(o, l.cuda())

			writer.add_scalar(LOG_LOSS_V_PATH, loss, epoch)
			if i % 100 == 0: print(f"{i} Validation Loss: {loss}")

		writer.flush()
	
	if n % SAVE_MODEL_EVERY_N_EPOC == 0:
    	OUTPUT_path = f'{OUTPUT_PATH}/Weights_{epoch}_loss_{loss}.pt'
		torch.save(model, OUTPUT_path)

	print("__done__")
	writer.close()
	# tensorboard --logdir=runs		http://localhost:6006/


