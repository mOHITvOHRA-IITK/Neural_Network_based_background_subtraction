# Human Segmentation

# INTRODUCTION
A sample of codes for neural network based human segmentation. The idea of this repo is to compare two images i.e. one with person and background, and other image contains background only. Standard image subtraction based methods are very sensitive to lightning conditions, thus idea is to use neural network based comaprison of images for good performance.


# INTERFACE
To up the interface, type in the terminal `cd /path/to/the/repository` and `python main.py`. The Current frame will be dispalyed with some virtual buttons as shown below.


<p align="center">
  <img src="/images/inference_image.png" />
</p>

Each Button has a specific function, for example, 
1. `Ext` exit the code.
2. `Bkg` to store the background image. It creates a new folder `</images/set{number}>` and save the current frame with name `bkg.png` in that folder after the default timer value of `5 secs`. The timer value can be changed by adding the argument `-b time` with the command `python main.py`.
3. `Sav` save the current frame in the folder created by pressing the `Bkg` button with name `0.png`, `1.png`, etc. after the default timer value of `2 secs`. The timer value can be changed by adding the argument `-s time` with the command `python main.py`.



# Button Selection
To select any button, either touch the button with your palm or put the cursor on the button and press left click. For predicting the palm locations [posenet](https://github.com/rwightman/posenet-pytorch) is used and red circles are drawn at the palm position.


# Annotation
Annotation is a manually extensive work. To annotate the images type `python label_images.py` with argument `-s {n}` where `n` is the set number folder ceated by selecting the `Bkg` button. For each image, mark the person boundary by clicking left buttons and once the boundary is complete (or close loop detected) than corresponding mask will be generated in the same folder.


# Training
For training, use the command `python training.py` with optional arguments

1. `-i `, number of iterations, default 1000.
2. `-b `, batch size, default 2.





