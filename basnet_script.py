import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import BASNet

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(image_name,pred,d_dir):

	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()

	im = Image.fromarray(predict_np*255).convert('RGB')
	img_name = image_name.split("/")[-1]
	image = io.imread(image_name)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

	pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	imo.save(d_dir+imidx+'.png')


if __name__ == '__main__':

	model_dir = '/media/dp/DATA/huihua_robot/BASNet-master/basnet_bsi_itr_19200_train_2.936459_tar_0.147055.pth'

	input = torch.randn(1,3,320,320)
	print("input", input.size())

	# --------- 3. model define ---------
	print("...load BASNet...")
	net = BASNet(3,1)
	net.load_state_dict(torch.load(model_dir))
	if torch.cuda.is_available():
		net.cuda()
	net.eval()
	

	inputs_test = input.type(torch.FloatTensor).cuda()
	
	# if torch.cuda.is_available():
	# 	inputs_test = Variable(inputs_test.cuda())
	# else:
	# 	inputs_test = Variable(inputs_test)
	
	d1,d2,d3,d4,d5,d6,d7,d8 = net(inputs_test)

	script_module = torch.jit.script(net)
	torch.jit.save(script_module, "model.pt")

	# pred = normPRED(d1)
	#
	# save_output(,pred,prediction_dir)
	#
	del d1,d2,d3,d4,d5,d6,d7,d8
