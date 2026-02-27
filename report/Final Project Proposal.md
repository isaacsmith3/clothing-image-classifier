# Final Project Proposal

## Goal

The objective of this project is to develop a deep learning model that will automatically assess the physical condition of returned items of clothing with image classification. In e-commerce, the manual inspection of returns is a significant logistical bottleneck and a major source of fraud. In this project, I aim to bridge that gap through a model that can classify clothing into discrete condition categories (new, lightly used, heavily used, damaged).

## Approach

I will use a convolutional neural network backbone with transfer learning, likely EfficientNet-B2 or ResNet-50. The photos do not seem to be very consistent in quality, so I'll attempt to make a pipeline to augment the data and cut out the noise in the images. This could be shadows, low light conditions, motion blur, or other normal camera issues that could ensure the model learns the important features better. The final architecture could involve a feature extractor that is pre-trained, followed by a deep neural network with 3-5 class outputs. I'll have to see as I go, though.

## Datasets

The primary data source that I have right now will be "Clothing Dataset for Second-Hand Fashion" (Zenodo, 2024), a high-quality dataset of about 32,000 images that's almost 30GB of data. The data is already labeled for general wear on a scale of 1-5, and there are other tags for stains or damage. I hope my machine or colab is capable of processing this in a reasonable amount of time. Batching will likely be necessary

## Measure of Success

I will measure success by using a few different metrics that I can think of right now, but I may change them as I work on it more.

- Top 1 Accuracy
- Macro-F1 Score (ensuring the model isn't just guessing the majority class)
- Categorical cross-entropy loss: monitoring convergence during training and validation
- Confusion matrix analysis (maybe)
- Grad-Cam visualizations for confirming the features rather than the background noise

## Available Resources

I have an Apple M3 Max chip with 32 GB of RAM as my local machine. This will likely handle larger batches well. I'll use PyTorch for the backend with MPS.
