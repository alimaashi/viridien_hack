# Viridien Hack: Enhancing Land Use Classification with LIDAR & ML
## Project Overview
We developed a solution to enhance low-resolution images using LIDAR data and applied a machine learning model to predict land use types. For training, we leveraged the Land Use Scene Classification dataset from Kaggle [source](https://www.kaggle.com/datasets/apollo2506/landuse-scene-classification).

## Findings on Land Use & Impact of a Third Runway
Our model found a significant presence of residential areas surrounding Heathrow Airport. Additionally, bodies of water nearby could make it hard to build on, and the many green spaces may spark environmental concerns. On the right side of the image in the presentation, we identified densely populated residential zones. The potential displacement of these communities would cause substantial costs and logistical challenges.

## Solution & Methodology
Our upscaling BFS approach is efficient in memory usage, ensuring a clean edge between buildings and the surroundings. However, aligning the LIDAR data and optimising the priority queue required time-intensive prioritisation. Despite this, our method kept a natural colour, preserving the integrity of the original picture.

For classification, we processed a 2GB dataset and trained a machine learning model using TensorFlow. While this approach is computationally demanding, it delivered accurate land use predictions, which could be used for urban planning and infraestructure development

