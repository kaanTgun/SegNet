import numpy as np
import matplotlib.pyplot as plt


def visualize2(original, s, m, l, s_pred, m_pred, l_pred):
	""" Visualize RGB image and labeled annotations with predicted annotations by the model.
	A visual aid function to determine how the model is performing 

	Args:
			original (np_tensor): original RGB Image
			s (np_tensor): Small labeled annotations
			m (np_tensor): Mid labeled annotations
			l (np_tensor): Large labeled annotations
			s_pred (np_tensor): Small predicted annotations
			m_pred (np_tensor): Mid predicted annotations
			l_pred (np_tensor): Large predicted annotations
	"""
	fig = plt.figure(figsize=(20, 10))
	plt.subplot(1,7,1)
	plt.title('Original image')
	plt.imshow(original)

	plt.subplot(1,7,2)
	plt.title('S image')
	plt.imshow(s)
	plt.subplot(1,7,3)
	plt.title('S Pred image')
	plt.imshow(s_pred)

	plt.subplot(1,7,4)
	plt.title('M image')
	plt.imshow(m)
	plt.subplot(1,7,5)
	plt.title('M Pred image')
	plt.imshow(m_pred)

	plt.subplot(1,7,6)
	plt.title('L image')
	plt.imshow(l)
	plt.subplot(1,7,7)
	plt.title('L Pred image')
	plt.imshow(l_pred)

def testModel(dataObj, index):
	""" Test the currnet model with Test data by passing an image and visually determin how the trained model is doing

	Args:
			dataObj (Dataset): Preset Structured_Dataset class object for input data 
			index (int): Batch index in the test dataset object 
	"""
	(s,m,l), img = dataObj.__getitem__(index)
	img = img.permute(1,2,0)
	img = img.float().unsqueeze(0)
	
	if next(model.parameters()).is_cuda:
		output = model(img.cuda())    
	else:
		output = model(img)

	s_pred,m_pred,l_pred = output[0].squeeze(0).cpu(), output[1].squeeze(0).cpu(), output[2].squeeze(0).cpu()
	s_pred = s_pred.detach().numpy()
	m_pred = m_pred.detach().numpy()
	l_pred = l_pred.detach().numpy()

	for j in range(22):
		visualize2(img, s[j], m[j], l[j], s_pred[j], m_pred[j], l_pred[j])
		k = np.array(s[j])