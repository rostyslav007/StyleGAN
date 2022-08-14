# StyleGAN
In this project I implemented all the main components od StyleGAN model architecture. Even though learning process didn't give some relevant results,
you can use this code for understanding general model building approach and trying to integrate it in your problems.

## Project structure
### **model.py** 
includes all the basic components of StyleGAN model (Pixel Norm, Equalized blocks, Mapping layer, Noice channel, AdaIn layer, Generator and Dicriminator building blocks)

### **Generator.py** 
Generator architecture class, based on model.py building blocks with ProgressiveGAN learning approach

### **Discriminator.py**
Discriminator class, based on model.py building blocks with ProgressiveGAN learning approach

### **utils.py**
helper functions for calculating gradient with WGAN-GP approach, as described in paper https://arxiv.org/pdf/1704.00028.pdf

### **train_loop.py**
model train loop implementation

## **Dataset**
For training CelebA datased was used 

more information on it and download link you may find here https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## ***archive papers explaining components of StyleGAN***
### Progressive Growing of GANs https://arxiv.org/pdf/1710.10196.pdf
### A Style-Based Generator Architecture for GANs https://arxiv.org/pdf/1812.04948.pdf
### Improved training of Wasserstein GANs https://arxiv.org/pdf/1704.00028.pdf 
