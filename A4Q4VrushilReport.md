Vrushil Patel
Comp 4190
Assignment 4
Question 4

The program is used to mainly compare the effects of a bad feature extraction vs good feature extraction

MNIST dataset contains 60,000 images of pixels 28x28, these images are digits from 0-9.
The program loads the dataset, then splits it into 80% training and 20% validation sets.
The pixel values are then normalized to values in the range of 0-1.

I create a bad feature extractor by adding gaussian noise with a mean of 0 and standard deviation of 0.1
Then i train the convolutional neural network CNN, on the noisy images and evaluate the model's performance.

I create a good feature extractor. It uses a sharpening filter using OpenCV. Used the idea implemented by the professor 
in model_development example. Finally, I evaluate the model's performance. 

The CNN model is not changed between the good and bad feature extractors. Its input is the image of 28x28. Uses Conv2d
and MaxPooling to extract features.

The program also prints test accuracy for noisy images vs sharpened images. 
