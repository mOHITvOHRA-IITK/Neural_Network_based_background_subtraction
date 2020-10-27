import cv2
import numpy as np
from class_definations import write_data3



def data_processing(frame, background_frame, mask):


	normalized_frame = np.array(frame, dtype=np.float32)/255.0
	normalized_background_frame = np.array(background_frame, dtype=np.float32)/255.0
	label_mask = np.array(mask[:,:,0], dtype=np.uint8)/255
	label_mask = np.array(label_mask, dtype=int)

	return normalized_frame, normalized_background_frame, label_mask





def visualize_predictions_and_ground_truth(frame, background_frame, ground_truth, predictions, threshold):

	frame_with_gt = frame.copy()
	frame_with_gt[ground_truth[0,:,:] > 0, :] = [0, 0, 255]

	frame_with_perdictions = frame.copy()
	frame_with_perdictions[predictions[0, 1,:,:] > threshold, :] = [255, 0, 0]

	cv2.imshow('frame', frame)
	cv2.imshow('background_frame', background_frame)
	cv2.imshow('ground_truth', frame_with_gt)
	cv2.imshow('predictions', frame_with_perdictions)
	cv2.waitKey(0)



def visualize_predictions(frame, background_frame, predictions, threshold, fps):


	frame_with_perdictions = frame.copy()
	frame_with_perdictions[predictions[0, 1,:,:] > threshold, :] = [255, 0, 0]
	
	frame_with_perdictions = write_data3(frame_with_perdictions, 'fps:' + str(int(fps)), 0.05, 0.74, 0.17, 0.10, 0.01, 0.07, 1, 2, (255, 0, 255))

	# cv2.imshow('frame', frame)
	cv2.imshow('background_frame', background_frame)
	cv2.imshow('predictions', frame_with_perdictions)
	cv2.waitKey(1)




def visualize_playing_with_myself(frame, predictions, threshold):


	org_frame = frame.copy()
	flip_frame = cv2.flip(frame, 1) 



	h,w = frame.shape[0:2]
	mask = np.zeros(shape=[h,w], dtype=np.uint8)
	mask[predictions[0, 1,:,:] > threshold] = 255
	flip_mask = cv2.flip(mask, 1) 

	org_frame[flip_mask > 0, :] = flip_frame[flip_mask > 0, :]	

	cv2.imshow('play_with_myself_frame', org_frame)
	cv2.waitKey(1)