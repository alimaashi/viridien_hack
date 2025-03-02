# viridien_hack
Sharpening a low-res image with LIDAR data and applying ML model to predict land use types.
Using dataset from Kaggle (https://www.kaggle.com/datasets/apollo2506/landuse-scene-classification) for land use classification ML training.

*Findings on land use and likely impact of a third runway. **
Our model found that there is significant residential area in the surroundings of Heathrow Airport. There are also some bodies of water, which might potentially impact construction in the area. The large green spaces might also prove contentious to build on, as there is an increasing concern of construction in green areas. On the right side of the image on the presentation, we can see that there are residential areas, and relocation of those people would be expensive to perform. 

Our solution for the upscaling is not memory intensive, but matching the data between LiDAR and performing the prioritisation for the queue was time intensive. nevertheless, it provided the best matching of colours, which ensured that the areas were similar to the original ones. For the Classifier, we used a 2GB dataset and ML model training using Tensorflow library which is memory and time intensive.
