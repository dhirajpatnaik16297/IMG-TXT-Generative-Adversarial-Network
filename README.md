# IMG-TXT-Generative-Adversarial-Network
Inspired by the paper on Stack GAN https://arxiv.org/pdf/1612.03242.pdf which generates realistic images from text,
I have tried to find relation between image and text and generating both of them simultaneously using Generative Adversarial Networks.

Following is the architecture of my model

![figure](https://user-images.githubusercontent.com/24193718/53629681-29dfa780-3c34-11e9-983d-70d4cdfe182d.jpeg)
## Please click on the image for a larger view

Programming language used: python 3.5

Libraries used: tensorflow 1.4(gpu version),
                numpy, matplotlib and pickle
### I trained them using around 12 GB of GPU memory which took a few hours.

Below are some snippets of the results of images and text generated during training. Please compare both image and text generated to know the relation between them.

1)For epoch 27
![mnist_27](https://user-images.githubusercontent.com/24193718/53632009-e25c1a00-3c39-11e9-935e-4d53eccd0fe0.png)

epoch27,Generated Nos.

['8', '3', '3', '8', '9', '8', '9', '0', '3', '6',

 '0', '6', '0', '6', '1', '0', '8', '8', '8', '0',
 
 '1', '3', '2', '9', '3', '1', '8', '9', '3', '1',
 
 '8', '6', '3', '1', '8', '1', '3', '8', '1', '8',
 
 '6', '8', '3', '8', '1', '8', '3', '1', '0', '1',
 
 '3', '9', '0', '0', '1', '8', '3', '1', '9', '1',
 
 '3', '9', '8', '8', '8', '8', '6', '8', '9', '1',
 
 '8', '6', '0', '9', '0', '9', '6', '1', '6', '8',
 
 '3', '8', '3', '0', '6', '8', '8', '8', '3', '1',
 
 '9', '3', '6', '8', '6', '8', '0', '3', '8', '8']


2)For epoch 31
![mnist_31](https://user-images.githubusercontent.com/24193718/53634016-15ed7300-3c3f-11e9-96eb-c91a79bb5d0b.png)

epoch31,Generated Nos.

['8', '6', '3', '8', '8', '8', '8', '0', '3', '8', 

 '8', '6', '8', '6', '2', '0', '4', '3', '8', '0', 
 
 '8', '8', '8', '8', '3', '2', '8', '1', '6', '3', 
 
 '8', '6', '3', '1', '8', '1', '6', '8', '2', '8', 
 
 '8', '8', '8', '8', '3', '0', '3', '2', '8', '0',
 
 '2', '8', '0', '0', '8', '8', '3', '2', '8', '1',
 
 '3', '9', '8', '1', '8', '4', '6', '8', '0', '1',
 
 '8', '8', '0', '2', '0', '8', '4', '3', '8', '8',
 
 '6', '8', '8', '0', '8', '3', '8', '8', '8', '2',
 
 '8', '3', '6', '8', '4', '4', '0', '6', '8', '8']

 #### NOTE: Please check the above given instances of the results for 27th and 31st epoch. if you look closely and compare the images and text, they are same in almost all cases. This is how i found that a particular image is related to its text.
 
 #### NOTE: epoch number in image and text are different because for text i have startted from epoch 0. So please make sure while taking a look into the results folder after training img_results folder has all images and generated_txt file has the generated text(*starting from epoch 0)
