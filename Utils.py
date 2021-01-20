import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from skimage import io, transform
import scipy
import numpy as np

import json
import os
import cv2


def pad_to_square(img, pad_value):
	""" Every input image passed through the network needs to be square, 
	This function adds padding around the cropped if the image is not a perfect square

	Args:
			img (np_tensor): original RGB image
			pad_value (int): the pixal value passed is the padding

	Returns:
			image, padding : return a tuple of already padded image and padding, 
												padding is needed to offset the annotations wrt padding
	"""
	c, h, w = img.shape
	dim_diff = np.abs(h - w)
	# (upper / left) padding and (lower / right) padding
	pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
	# Determine padding
	pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
	# Add padding
	img = torch.nn.functional.pad(img, pad, "constant", value=pad_value)

	return img, pad

class Structured_Dataset(Dataset):
			
	def __init__(self, txt_file, root_dir, image_size):
		with open(txt_file, "r" ) as f:
			self.file_data = f.readlines()

		self.image_dir = root_dir
		self.image_size = image_size

	def __len__(self):
		return len(self.file_data)


	def __getitem__(self, idx):

		idx_data = json.loads(self.file_data[idx])
		img_path = os.path.join(self.image_dir, idx_data['imgpath'])
		img = cv2.imread(img_path)
		h, w, _ = img.shape

		x1,y1,w,h = idx_data['bbox']                                             # Crop Hand from the image 
		joints = np.array(idx_data['joints']).reshape((-1,3))   
		img = img[int(y1):int(y1+h),int(x1):int(x1+w)] / 256
		img = np.moveaxis(img, -1, 0)                                            # Normalize the Croped Image
		
		img_tensor = torch.from_numpy(img)
		img_tensor, pad = pad_to_square(img_tensor, 0)                           # Add padding to image w==h
		_, padded_h, padded_w = img_tensor.shape

		img_sf_h = self.image_size / padded_h 
		img_sf_w = self.image_size / padded_w
		
		img_tensor = nn.functional.interpolate(img_tensor.unsqueeze(0), size=(self.image_size, self.image_size)) # scale the image to model input shape (reshape)
		img_tensor = img_tensor.squeeze(0)

		mat_s = np.full((len(joints) +1, self.image_size//4, self.image_size//4),0 , np.float32)
		mat_m = np.full((len(joints) +1, self.image_size//2, self.image_size//2),0 , np.float32)
		mat_l = np.full((len(joints) +1, self.image_size,    self.image_size   ),0 , np.float32)

		for j, joint in enumerate(joints):
			x,y,f = joint
			if f != 2: continue
			x, y = x-x1, y-y1                                                      # Joint wrt cropped Image loaction
			x, y = x+pad[0], y+pad[2]                                              # Joint wrt added padding 
			x, y = int(x*img_sf_w), int(y*img_sf_h)

			try:
				mat_s[j][int(y/4)][int(x/4)] = 1
				mat_m[j][int(y/2)][int(x/2)] = 1
				mat_l[j][y][x]               = 1
			except IndexError:
				print(f"IndexError: {img_path}, (x,y):{x},{y}")
				pass

			mat_s[j,:,:] = scipy.ndimage.gaussian_filter(np.array(mat_s[j,:,:]), sigma = 2)*30
			mat_m[j,:,:] = scipy.ndimage.gaussian_filter(np.array(mat_m[j,:,:]), sigma = 3)*80
			mat_l[j,:,:] = scipy.ndimage.gaussian_filter(np.array(mat_l[j,:,:]), sigma = 4)*100
		
		mat_s[j+1,:,:] = np.ones((self.image_size//4, self.image_size//4)) -  np.clip(np.sum(mat_s,axis=0), 0, 1) 
		mat_m[j+1,:,:] = np.ones((self.image_size//2, self.image_size//2)) -  np.clip(np.sum(mat_m,axis=0), 0, 1) 
		mat_l[j+1,:,:] = np.ones((self.image_size   , self.image_size   )) -  np.clip(np.sum(mat_l,axis=0), 0, 1) 

		mat_s = torch.from_numpy(mat_s)
		mat_m = torch.from_numpy(mat_m)
		mat_l = torch.from_numpy(mat_l)

		return (mat_s, mat_m, mat_l), img_tensor

