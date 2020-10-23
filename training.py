import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import network
import cv2
import os
from network_data_formation import data_processing, visualize_predictions_and_ground_truth
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--iterations', help='Set the number of iteration', type=int, default=1000)
parser.add_argument('-b', '--batch_size', help='Set the batch size', type=int, default=2)
args = parser.parse_args()



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




loss_function = nn.CrossEntropyLoss(reduction = 'mean')
optimizer = optim.Adam(net.parameters(), lr=0.001)

 


total_iterations = args.iterations
batch_size = args.batch_size




for i in range(total_iterations):


	all_set_folder_path = './images/'
	all_set_list = os.listdir(all_set_folder_path)

	set_number = np.random.randint(1, len(all_set_list)+1)

	set_folder_path = './images/set' + str(set_number) + '/'
	all_images_list = os.listdir(set_folder_path)


	image_list = []
	background_list = []
	label_list = []
	
	for j in range(batch_size):

		max_images = (len(all_images_list) - 1) / 2

		image_num = np.random.randint(max_images)

		full_image_path = set_folder_path + str(image_num) + '.png'
		org_frame = cv2.imread(full_image_path)
		alpha = np.random.uniform(1.0, 2.0, 1) # Contrast control (1.0-3.0)
		beta = np.random.randint(0, 50) # Brightness control (0-100)
		frame = cv2.convertScaleAbs(org_frame, alpha=alpha, beta=beta)

		
		full_background_path = set_folder_path + 'bkg.png'
		background_frame = cv2.imread(full_background_path)


		full_mask_path = set_folder_path + str(image_num) + '_mask.png'
		mask = cv2.imread(full_mask_path)


		frame_array, background_frame_array, mask_array = data_processing(frame, background_frame, mask)

		frame_array = np.transpose(frame_array, (2, 0, 1))
		image_list.append(frame_array)

		background_frame_array = np.transpose(background_frame_array, (2, 0, 1))
		background_list.append(background_frame_array)

		label_list.append(mask_array)
		




	if (use_GPU):
		image_tensor = torch.from_numpy(np.array(image_list)).cuda()
		background_tensor = torch.from_numpy(np.array(background_list)).cuda()
		label_tensor = torch.from_numpy(np.array(label_list)).cuda()
	else:
		image_tensor = torch.from_numpy(np.array(image_list)).cpu()
		background_tensor = torch.from_numpy(np.array(background_list)).cpu()
		label_tensor = torch.from_numpy(np.array(label_list)).cpu()

		

	predicted_tensor, predicted_prob_tensor = net.forward(image_tensor, background_tensor, 0.5)
	
	optimizer.zero_grad()
	loss = loss_function(predicted_tensor, label_tensor)
	loss.backward()
	optimizer.step()

	print (i,'/',total_iterations, " with loss: ", loss.cpu().detach().numpy())

torch.save(net.state_dict(), weight_path)


