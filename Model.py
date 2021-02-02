import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchsummary

import numpy as np
import os

def load_resnet50(MODEL_PATH):
	if not os.path.exists(MODEL_PATH):
		os.mkdir(MODEL_PATH)
		resnet50 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
		torch.save(resnet50, f"{MODEL_PATH}/resnet50.pth")
		
	if os.path.exists(f"{MODEL_PATH}/resnet50.pth"):
		return torch.load(f"{MODEL_PATH}/resnet50.pth")

class identity_block(nn.Module):
	def __init__(self, in_channels, filters, output = False, adjust_size=False):
		super(identity_block, self).__init__()  
		
		self.output = output
		if self.output:
			filters1, filters2, filters3, filters4 = filters
			if adjust_size:
				self.conv_4 = nn.Conv2d(filters3, filters4, kernel_size=(1,1), stride=1, padding=2)
			else:
				self.conv_4 = nn.Conv2d(filters3, filters4, kernel_size=(1,1), stride=1)
			self.bn_4 = nn.BatchNorm2d(num_features=filters4 )
		else:
			filters1, filters2, filters3 = filters

		self.conv_1 = nn.Conv2d(in_channels, filters1, kernel_size=(1,1), stride=1)
		self.bn_1 = nn.BatchNorm2d(num_features= filters1)

		self.conv_2 = nn.Conv2d(filters1, filters2, kernel_size=(3,3), stride=1, padding=1)
		self.bn_2 = nn.BatchNorm2d(num_features= filters2)

		self.conv_3 = nn.Conv2d(filters2, filters3, kernel_size=(1,1), stride=1)
		self.bn_3 = nn.BatchNorm2d(num_features= filters3)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		identity = x 
		out = self.conv_1(x)
		out = self.bn_1(out)
		out = self.relu(out)
		out = self.conv_2(out)
		out = self.bn_2(out)
		out = self.relu(out)
		out = self.conv_3(out)
		out = self.bn_3(out)
		out = out+identity
		out = self.relu(out)

		if self.output:
			out_ = self.conv_4(out)
			out_ = self.bn_4(out_)
			out_ = self.relu(out_)
			return out, out_
		return out

class conv_block(nn.Module):

	def __init__(self, in_channels, filters):
		super(conv_block, self).__init__()

		filters1, filters2, filters3 = filters
		self.conv_1 = nn.Conv2d(in_channels, filters1, kernel_size=(2,2), stride=(2,2))
		self.bn_1 = nn.BatchNorm2d(num_features= filters1)

		self.conv_2 = nn.Conv2d(filters1, filters2, kernel_size=(3,3), stride=1, padding=1)
		self.bn_2 = nn.BatchNorm2d(num_features= filters2)

		self.conv_3 = nn.Conv2d(filters2, filters3, kernel_size=(1,1), stride=1)
		self.bn_3 = nn.BatchNorm2d(num_features= filters3)

		self.relu = nn.ReLU(inplace=True)
	
	def forward(self, x):
		shortcut = x 
		out = self.conv_1(x)
		out = self.bn_1(out)
		out = self.relu(out)
		out = self.conv_2(out)
		out = self.bn_2(out)
		out = self.relu(out)
		out = self.conv_3(out)
		out_long = self.bn_3(out)

		out_short = self.conv_1(shortcut)
		out_short = self.bn_1(out_short)
		out = self.relu(out_long+out_short)
		return out

class SegNet(nn.Module):
	def __init__(self, D_in, D_out, deeper=False):
		super().__init__()
		""" Segnet is a derivation of ResNet-50 and UNet.
		Network is design to output 3 prediction maps S,M,L fromthe different sections of the network
		"""

		# D_in: 3-channels
		# D_out: # of segmentations

		self.deeper = deeper

		self.header = nn.Sequential(nn.Conv2d(in_channels= D_in, out_channels=128, kernel_size=7, stride=1, padding=3),\
																nn.ReLU())
		self.identity_1_1 = identity_block(in_channels=128, filters=[64,64,128])
		self.identity_1_2 = identity_block(128, [64,64,128,	128], output=True)

		self.conv_2_1 = conv_block(128, [256,512,256])
		self.identity_2_2 = identity_block(256, [256,512,256])
		self.identity_2_3 = identity_block(256, [256,512,256,	256], output=True )

		self.conv_3_1 = conv_block(256, [256,1024,256])
		self.identity_3_2 = identity_block(256, [256,512,256])
		self.identity_3_3 = identity_block(256, [256,512,256, 256], output=True )

		if self.deeper:
			self.conv_deep_1_1 = conv_block(256, [256,512,256])
			self.identity_deep_1_2 = identity_block(256, [256,512,256] )
			self.identity_deep_1_3 = identity_block(256, [256,512,256] )

			self.conv_deep_2_1 = conv_block(256, [512,1024,512])
			self.identity_deep_2_2 = identity_block(512, [512,1024,512] )
			self.identity_deep_2_3 = identity_block(512, [512,1024,512] )

			self.deconv_deep_3_1 = nn.ConvTranspose2d(512, 256, 3,  stride=2, padding=1, output_padding=1)
			self.identity_deep_3_2 = identity_block(256, [256,512,256] )
			self.identity_deep_3_3 = identity_block(256, [256,512,256] )

			self.deconv_deep_4_1 = nn.ConvTranspose2d(256, 256, 3,  stride=2, padding=1, output_padding=1)
			self.identity_deep_4_2 = identity_block(256, [256,512,256] )
			self.identity_deep_4_3 = identity_block(256, [256,512,256] )

			self.conv_small 			= nn.Conv2d(256*2, 128, 3, stride=2, padding=1)
			self.conv_out_small 	= nn.Conv2d(128, D_out, 1, stride=1)

		self.deconv_4_1 	= nn.ConvTranspose2d(256, 256, 3,  stride=2, padding=1, output_padding=1)
		self.identity_4_2 = identity_block(256, [256,512,256] )
		self.identity_4_3 = identity_block(256, [256,512,256] )

		self.deconv_5_1 	= nn.ConvTranspose2d(256, 128, 3,  stride=2, padding=1, output_padding=1)
		self.identity_5_2 = identity_block(128, [128,256,128] )
		self.identity_5_3 = identity_block(128, [128,256,128] )

		self.conv_medium 	= nn.Conv2d(256*2, 128, 3, stride=1, padding=1)
		self.conv_large 	= nn.Conv2d(128*2, 128, 3, stride=1, padding=1)

		self.conv_out_medium 	= nn.Conv2d(128, D_out, 1, stride=1)
		self.conv_out_large 	= nn.Conv2d(128, D_out, 1, stride=1)

		self.relu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
				
		y = self.header(x)
		y = self.identity_1_1(y)
		y, out_cnt_1_1 = self.identity_1_2(y)								#128
		
		y = self.conv_2_1(y)
		y = self.identity_2_2(y)
		y, out_cnt_2_1 = self.identity_2_3(y)								#64

		y = self.conv_3_1(y)
		y = self.identity_3_2(y)
		y, out_cnt_3_1 = self.identity_3_3(y)								#32

		if self.deeper:
			y = self.conv_deep_1_1(y) 
			y = self.identity_deep_1_2(y)											#16
			y_1 = self.identity_deep_1_3(y)

			y = self.conv_deep_2_1(y_1) 
			y = self.identity_deep_2_2(y)
			y_2 = self.identity_deep_2_3(y)										#8

			y = self.deconv_deep_3_1(y_2)											#16
			y = self.identity_deep_3_2(y_1+y)
			y_3 = self.identity_deep_3_3(y)

			y = self.deconv_deep_4_1(y_3)								 			#32
			y = self.identity_deep_4_2(y)
			y = self.identity_deep_4_3(y)

			cat_s = torch.cat((out_cnt_3_1, y), 1)
			conv_small  = self.relu(self.conv_small(cat_s))
			out_small 	= self.conv_out_small(conv_small)
			out_small 	= self.sigmoid(out_small)							#32

		y = self.relu(self.deconv_4_1(y))										#64
		y = self.identity_4_2(y)
		y = self.identity_4_3(y)
		cat_m = torch.cat((out_cnt_2_1, y), 1)
		conv_medium = self.relu(self.conv_medium(cat_m))
		out_medium 	= self.conv_out_medium(conv_medium)
		out_medium 	= self.sigmoid(out_medium)					 		#64

		y = self.relu(self.deconv_5_1(y))										#128
		y = self.identity_5_2(y)
		y = self.identity_5_3(y)
		cat_l = torch.cat((out_cnt_1_1, y), 1)
		conv_large  = self.relu(self.conv_large(cat_l))
		out_large 	= self.conv_out_large(conv_large)
		out_large 	= self.sigmoid(out_large)								#128

		return (out_small, out_medium, out_large)


from torchsummary import summary

model 	= SegNet(3, 22, deeper=True)
summary(model, (3, 128, 128))