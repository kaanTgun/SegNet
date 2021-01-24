from Model import SegNet
from Utils import Structured_Dataset, weighted_loss

import torch
from torch.utils.tensorboard import SummaryWriter

from skimage import io
import scipy
import numpy as np

import json
import os


def train():
	"""
	Train Model
	"""
	# Hyperparameters
	BATCH_SIZE 		 = 24
	INPUT_IMAGE_SIZE = 128
	EPOCS 			 = 6

	# Training data setup
	T_img_Folder = "/content/Hands_ex/Training"
	T_txt_File = "/content/Hands_ex/Training/train_set.txt"

	V_img_Folder = "/content/Hands_ex/Validation"
	V_txt_File = "/content/Hands_ex/Validation/val_set.txt"

	OUTPUT_PATH 	= "Model"
	LOG_LOSS_DIR 	= "Loss"
	LOG_LOSS_T_PATH = "Loss/Train"
	LOG_LOSS_V_PATH = "Loss/Validate"
	SAVE_MODEL_EVERY_N_EPOC = 1

	###########
	if not os.path.exists(OUTPUT_PATH):
		os.mkdir(OUTPUT_PATH)
	if not os.path.exists(LOG_LOSS_DIR):
		os.mkdir(LOG_LOSS_DIR)
		os.mkdir(LOG_LOSS_T_PATH)
		os.mkdir(LOG_LOSS_V_PATH)

	train_dataset = Structured_Dataset(txt_file=T_txt_File, root_dir=T_img_Folder, image_size=INPUT_IMAGE_SIZE)
	val_dataset   = Structured_Dataset(txt_file=V_txt_File, root_dir=V_img_Folder, image_size=INPUT_IMAGE_SIZE)

	train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
	val_data_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,num_workers=4, shuffle=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
	model = SegNet(3,22).to(device)
	model_location = next(model.parameters()).device

	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	writer = SummaryWriter()

	for epoch in range(EPOCS):

		# Train
		model.train(True)
		for i, batch in enumerate(train_data_loader):
			labels, imageLs = batch
			imageLs = imageLs.float().to(model_location)
			# reset gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(imageLs)
			loss = 0
			for o,l in zip(outputs, labels):
				loss += weighted_loss(model_location, outputs=o, labels=l.to(model_location))

			writer.add_scalar(LOG_LOSS_T_PATH, loss, epoch)
			loss.backward()
			optimizer.step()
			
			if (i % 100) == 0:
				print(f"{epoch}-{i} Training Loss: {loss}")
			
			if (i % 1000) == 0:
				OUTPUT_path = f'{OUTPUT_PATH}/Weights_E_{epoch}_i_{i}_loss_{loss}.pt'
				torch.save(model, OUTPUT_path)
				writer.flush()


		# Validate
		model.train(False)
		for i, batch in enumerate(val_data_loader):
			labels, imageLs = batch
			imageLs = imageLs.float().to(model_location)
			outputs = model(imageLs)

			loss = 0
			for o,l in zip(outputs, labels):
				loss += weighted_loss(model_location, outputs=o, labels=l.to(model_location))

			writer.add_scalar(LOG_LOSS_V_PATH, loss, epoch)
			if i % 100 == 0: 
				print(f"{epoch}-{i} Validation Loss: {loss}")

		writer.flush()

	if (epoch % SAVE_MODEL_EVERY_N_EPOC) == 0:
		OUTPUT_path = f'{OUTPUT_PATH}/Weights_{epoch}_loss_{loss}.pt'
		torch.save(model, OUTPUT_path)

	print("__done__")
	writer.close()
	# tensorboard --logdir=runs		http://localhost:6006/


if __name__ == "__main__":
	train()