from osgeo import gdal
import heapq
import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys

sys.setrecursionlimit(5000 * 50)

# Open the TIFF file
file_path = "DSM_TQ0075_P_12757_20230109_20230315.tif"  # lidar
# file_path = "20230215_SE2B_CGG_GBR_MS4_L3_BGRN.tif"  # sat
lidar = gdal.Open(file_path)

x_res = lidar.RasterXSize
y_res = lidar.RasterYSize

orig_img = cv2.imread("orig_img.jpg")
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
orig_img = cv2.resize(orig_img, (x_res, y_res))
lidar = lidar.ReadAsArray()


# Step 2: Apply the linear transformation to scale the pixel values
stretched_img = lidar.astype(np.uint8)
min_val = np.min(stretched_img)
median_val = np.median(stretched_img)
stretched_img = (stretched_img - min_val) / (median_val * 2 - min_val) * 255
stretched_img = np.clip(stretched_img, 0, 255)

buckets = 15

lidar_img = stretched_img // (255 / buckets)

lidar_img = lidar_img * (255 / buckets)
plt.imshow(lidar_img, cmap="gray")
plt.show()

blurred = cv2.GaussianBlur(lidar.astype(np.uint8), (11, 11), -1)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

sharpened_img = orig_img.copy()
sharpened_img = cv2.filter2D(sharpened_img, -1, kernel)
print("sharp")
plt.imshow(sharpened_img)
plt.show()

visited = np.zeros_like(lidar_img, dtype=bool)
painted = np.zeros_like(lidar_img, dtype=bool)


# Iterative BFS with contrast-based stopping
def bfs(
    img_comp,
    img_edit,
    x,
    y,
    target_color,
    visited,
    max_region_size=120000,
):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    heap = []  # Min-heap (priority queue)
    heapq.heappush(heap, (0, x, y))  # Priority queue stores (priority, x, y)
    visited[x, y] = True
    original_color = img_edit[x, y]
    region_size = 0

    while heap:
        priority, cx, cy = heapq.heappop(heap)

        if region_size >= max_region_size:
            break

        # Only continue if the target color matches
        img_edit[cx, cy] = original_color  # Replace with original color
        region_size += 1

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if (
                0 <= nx < img_edit.shape[0]
                and 0 <= ny < img_edit.shape[1]
                and img_comp[nx, ny] == target_color
                and not visited[nx, ny]
            ):
                visited[nx, ny] = True
                color_diff = np.sum(np.abs(img_edit[nx, ny] - original_color))
                heapq.heappush(heap, (color_diff, nx, ny))


# Loop through the image
for i in range(5000):
    print(i)
    for j in range(5000):
        if not visited[i][j] and not painted[i][j]:
            target_color = dataset_img[i][j]  # Color to check
            colours = []
            coords = []
            bfs(dataset_img, sharpened_img, i, j, target_color, visited)

plt.xticks(snap=False)
plt.yticks(snap=False)

# Show Result
plt.imshow(sharpened_img)
plt.title("Sharpened Image with Edge Enhancement")
plt.show()
