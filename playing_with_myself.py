import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import network
import cv2
import os
from network_data_formation import data_processing, visualize_playing_with_myself
from class_definations import write_data3
import time






use_GPU = 1


net = network.Net()


if (use_GPU):
    gpus = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    net = torch.nn.DataParallel(net).cuda()
else:
    device = torch.device('cpu')



weight_path = './weights/net.pth'
if (os.path.isfile(weight_path)):

    if (use_GPU):
        net.load_state_dict(torch.load(weight_path))
    else:
        state_dict = torch.load(weight_path)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v

        net.load_state_dict(new_state_dict)

else:
    print ("no saved weights")


    

net.eval()
torch.no_grad()


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")



current_time = time.time()
max_count_down = 5


while (1):
	_, img = cap.read()
	real_img = cv2.flip(img, 1) 

	x =  max_count_down - np.int(time.time() - current_time)
	if x > 0:
		real_img = write_data3(real_img, 'Saving bkg in ' + str(x) + ' secs ', 0.0, 0.4, 1.0, 0.15, 0.0, 0.12, 2, 2, (255,255,255))
	else:
		break;


	cv2.imshow('real_img', real_img)
	cv2.waitKey(1)



cv2.destroyAllWindows()
# background_frame = cv2.fastNlMeansDenoisingColored(background_frame,None,10,10,7,21)








current_time = time.time()
background_frame = None
background_frame_list = []

while (1):
	_, img = cap.read()
	real_img = cv2.flip(img, 1) 
	real_img2 = cv2.flip(img, 1) 

	background_frame_list.append(real_img2)
	x =  max_count_down - np.int(time.time() - current_time)
	if x > 0:
		real_img = write_data3(real_img, 'Updating ', 0.0, 0.4, 1.0, 0.15, 0.3, 0.12, 2, 2, (255,255,255))
	else:
		background_frame_array = np.array(background_frame_list)
		background_frame = np.mean(background_frame_array, 0)
		background_frame = np.array(background_frame, dtype=np.uint8)
		break;


	cv2.imshow('real_img', real_img)
	cv2.waitKey(1)



cv2.destroyAllWindows()
# background_fram



while(1):

	
    
	_, frame = cap.read()
	frame = cv2.flip(frame, 1) 


	image_list = []
	background_list = []

	frame_array, background_frame_array, _  = data_processing(frame, background_frame, frame)

	frame_array = np.transpose(frame_array, (2, 0, 1))
	image_list.append(frame_array)

	background_frame_array = np.transpose(background_frame_array, (2, 0, 1))
	background_list.append(background_frame_array)

	


	if (use_GPU):
		image_tensor = torch.from_numpy(np.array(image_list)).cuda()
		background_tensor = torch.from_numpy(np.array(background_list)).cuda()
	else:
		image_tensor = torch.from_numpy(np.array(image_list)).cpu()
		background_tensor = torch.from_numpy(np.array(background_list)).cpu()


	predicted_tensor, predicted_prob_tensor = net.forward(image_tensor, background_tensor, 0.0)

	
	predicted_prob = predicted_prob_tensor.cpu().detach().numpy()
	visualize_playing_with_myself(frame, predicted_prob, 0.95)
	





