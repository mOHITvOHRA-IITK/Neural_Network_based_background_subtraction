from class_definations import vision_demo_class
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-b', '--background_image_timer_val', help='Set the countdown for saving background image', type=int, default=5)
parser.add_argument('-s', '--image_timer_val', help='Set the countdown for saving current image', type=int, default=5)
args = parser.parse_args()


o = vision_demo_class(args.background_image_timer_val, args.image_timer_val)

while True:
	o.contact_less_new_GUI()