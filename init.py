from Model import SegNet
import Utils

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import json
import os

def train():
	"""
	Train Model
	"""
	# Hyperparameters
	BATCH_SIZE 		 		= 16
	INPUT_IMAGE_SIZE 	= 128
	EPOCS 			 			= 6
	NUM_WORKERS 			= 4

	# Training data setup
	T_img_Folder = "Hands_ex/Training"
	T_txt_File = "Hands_ex/Training/train_set.txt"

	V_img_Folder = "Hands_ex/Validation"
	V_txt_File = "Hands_ex/Validation/val_set.txt"

	OUTPUT_PATH 	= "Model"
	LOG_LOSS_DIR 	= "Loss"
	LOG_LOSS_T_PATH = "Loss/Train"
	LOG_LOSS_V_PATH = "Loss/Validate"
	SAVE_MODEL_EVERY_N_EPOC = 1

	Utils.create_log_folders(OUTPUT_PATH, LOG_LOSS_DIR, LOG_LOSS_T_PATH, LOG_LOSS_V_PATH)

	train_dataset = Utils.Structured_Dataset(txt_file=T_txt_File, root_dir=T_img_Folder, image_size=INPUT_IMAGE_SIZE)
	val_dataset   = Utils.Structured_Dataset(txt_file=V_txt_File, root_dir=V_img_Folder, image_size=INPUT_IMAGE_SIZE)

	train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
	val_data_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

	device 	= torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
	model 	= SegNet(3,22, deeper=True).to(device)
	model_location = next(model.parameters()).device

	loss_function = torch.nn.MSELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
	writer = SummaryWriter()

	for epoch in range(EPOCS):
		# Train
		model.train(True)
		for i, batch in enumerate(train_data_loader):
			labels, images = batch
			images = images.float().to(model_location)
			# Reset gradients
			optimizer.zero_grad()

			# Forward -> Loss -> Backprop -> Optimize
			outputs = model(images)
			loss = 0
			for o,l in zip(outputs, labels):
				loss += loss_function(o, l.to(model_location))

			writer.add_scalar(LOG_LOSS_T_PATH, loss, epoch)
			loss.backward()
			optimizer.step()
			
			if (i % 100) == 0:
				print(f"{epoch}-{i} Training Loss: {loss}")
			
			if (i % 1000) == 0:
				modelOutput_path = f'{OUTPUT_PATH}/Weights_E_{epoch}_i_{i}_loss_{loss}.pt'
				torch.save(model, modelOutput_path)
				writer.flush() # Flushes the event file to disk


		# Validate
		model.train(False)
		optimizer.zero_grad()
		with torch.no_grad():
			for i, batch in enumerate(val_data_loader):
				optimizer.zero_grad()
				labels, images = batch
				images = images.float().to(model_location)
				outputs = model(images)

				loss = 0
				for o,l in zip(outputs, labels):
					loss += loss_function(o, l.to(model_location))

				writer.add_scalar(LOG_LOSS_V_PATH, loss, epoch)
				if i % 10 == 0: 
					print(f"{epoch}-{i} Validation Loss: {loss}")

		writer.flush()

	if (epoch % SAVE_MODEL_EVERY_N_EPOC) == 0:
		modelOutput_path = f'{OUTPUT_PATH}/Weights_{epoch}_loss_{loss}.pt'
		torch.save(model, modelOutput_path)

	print("__done__")
	writer.close()

if __name__ == "__main__":
	train()


	# tensorboard --logdir=runs		http://localhost:6006/
