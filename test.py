from osgeo import gdal
import cv2
import numpy as np
import matplotlib.pyplot as plt

orig_img = cv2.imread("orig_img.jpg")
resized_img = cv2.resize(orig_img, (5000, 5000))
file_path = "20230215_SE2B_CGG_GBR_MS4_L3_BGRN.tif"

dataset = gdal.Open(file_path).ReadAsArray()

edges = cv2.Canny(dataset.astype(np.uint8), 2, 10)
edges_resized_3channels = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

blurred = cv2.GaussianBlur(resized_img, (5,5), 0)
sharpened = cv2.addWeighted(resized_img, 3, blurred, -2, 0)
sharpened_with_lidar = cv2.addWeighted(sharpened, 1, edges_resized_3channels, 0.3, 0)

# cv2.imwrite("lidar_edges.png", edges)

r = dataset.GetRasterBand(1).ReadAsArray()
g = dataset.GetRasterBand(2).ReadAsArray()
b = dataset.GetRasterBand(3).ReadAsArray()

rgb_image = np.dstack((r, g, b))

plt.imshow(rgb_image)
plt.axis("off")
plt.show()