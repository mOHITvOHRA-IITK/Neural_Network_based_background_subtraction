import cv2
import numpy as np
import argparse
import os
from os import path



parser = argparse.ArgumentParser()
parser.add_argument('-s', '--set_num', help='Set Number for dataset', type=int, default=1)
args = parser.parse_args()


set_folder_path = './images/set' + str(args.set_num) + '/'
all_images_list = os.listdir(set_folder_path)

bkg_img = None


global person_boundary_pixels, exit_loop
person_boundary_pixels = []
exit_loop = False



def get_mouse_click(event,x,y,flags,param):
    global person_boundary_pixels, exit_loop

    if event == cv2.EVENT_LBUTTONDOWN:
        person_boundary_pixels.append((x,y))

        if len(person_boundary_pixels) > 5:
        	initial_point = person_boundary_pixels[0]
        	final_point = person_boundary_pixels[len(person_boundary_pixels)-1]

        	x = initial_point[0] - final_point[0]
        	y = initial_point[1] - final_point[1]
        	dis = np.sqrt(x*x + y*y)

        	if dis < 10:
        		exit_loop = True


    if event == cv2.EVENT_RBUTTONDOWN:
    	exit_loop = True


    
    	








for file in all_images_list:
    
	num = file.split('.')[0]
	ext = file.split('.')[1]

	if (num.isdigit()):
		complete_mask_img_name = set_folder_path + num + '_mask.' + ext
		
		if path.exists(complete_mask_img_name) == False:
			
			img = cv2.imread(set_folder_path + file)
			person_boundary_pixels = []
			exit_loop = False
			cv2.namedWindow('Human_segmentation_annotation')
			cv2.setMouseCallback('Human_segmentation_annotation',get_mouse_click)

			h,w = img.shape[0:2]
			mask = np.zeros(shape=[h,w], dtype=np.uint8)

			while(exit_loop == False):
				cv2.imshow('Human_segmentation_annotation', img)
				cv2.waitKey(10)

				for i in range(1, len(person_boundary_pixels)):
					p1 = person_boundary_pixels[i-1]
					p2 = person_boundary_pixels[i]
					cv2.line(img, p1, p2, (255,255,255), 2) 


			if len(person_boundary_pixels) > 0:
				cv2.drawContours(mask, [np.array(person_boundary_pixels)], contourIdx=-1, color=(255),thickness=-1)
			cv2.imwrite(complete_mask_img_name, mask)
			cv2.imshow('mask', mask)
			cv2.waitKey(10)


	else:
		mask_num = num.split('_')[0]
		if (mask_num.isdigit() == False):
			print (set_folder_path + file)
			bkg_img = cv2.imread(set_folder_path + file)




cv2.imshow('bkg_img', bkg_img)
cv2.waitKey(0)

	        