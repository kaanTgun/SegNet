import numpy as np
import matplotlib.pyplot as plt


def visualize2(original, s, m, l, s_pred, m_pred, l_pred):
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

