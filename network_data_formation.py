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