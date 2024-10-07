import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.Image_Preprocessing import Mask_Extraction, Get_Skeletion, Get_Split_Information,Extract_Sample
import os

image_path = 'images_Series3/scan184.jpg'
image = cv2.imread(image_path)
cleaned_image = Mask_Extraction(image)


skeleton = Get_Skeletion(cleaned_image)

# plt.imshow(fil.skeleton_longpath)
# plt.axis('off')
# plt.show()

skeleton_length = np.count_nonzero(skeleton)
split_length = skeleton_length / 4
split_points = []

neighbors = []

split_lines = []

split_points, split_length, split_lines = Get_Split_Information(skeleton,split_length)

image_path = 'images_Series1/scan184.jpg'
sample_image = cv2.imread(image_path)
sample_mask = Mask_Extraction(sample_image)
#sample = Extract_Sample(sample_mask,sample_image)

# plt.imshow(sample_mask, cmap='gray')
# plt.title('Binary Image')
# plt.axis('off')
# plt.show()

split_image1 = sample_mask.astype(np.uint8)
split_image2 = np.zeros_like(sample_mask).astype(np.uint8)
split_image3 = np.zeros_like(sample_mask).astype(np.uint8)
split_image4 = np.zeros_like(sample_mask).astype(np.uint8)

image_shape = image.shape
sample_shape = sample_mask.shape

image_col = image_shape[1]
sample_col = sample_shape[1]
times = sample_col / image_col

#print('*************************')

def directed_area(A, B, C):
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    area = (x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3)
    area /= 2
    return area


def split_images(A, B, num):
    for y in range(split_image1.shape[0]):

        for x in range(split_image1.shape[1]):

            if directed_area(A, B, (x, y)) >= 0:
                if num == 0:
                    split_image4[y][x] = split_image1[y][x]
                    split_image1[y][x] = 0
                    # split_image1[y][x][0]  = 0
                    # split_image1[y][x][1] = 0
                    # split_image1[y][x][2] = 0
                elif num == 1:
                    split_image3[y][x]  = split_image1[y][x]
                    split_image1[y][x] = 0
                    # split_image1[y][x][0]  = 0
                    # split_image1[y][x][1] = 0
                    # split_image1[y][x][2] = 0
                else:

                    # if y % 100 == 0 and x % 100 == 0 and :
                    #     print((x, y))

                    split_image2[y][x]  = split_image1[y][x]
                    split_image1[y][x] = 0
                    # split_image1[y][x][0]  = 0
                    # split_image1[y][x][1]  = 0
                    # split_image1[y][x][2]  = 0



for num, line in enumerate(split_lines):

    if 'x =' in line:
        x_line = float(line.split('=')[1].strip())
        y1 = np.random.uniform(0, 10)
        y2 = np.random.uniform(y1 + 0.01, 20)
        x1 = x2 = (x_line * times)
        A = (x1, y1)
        B = (x2, y2)
        split_images(B, A, num)

    elif 'y =' in line and 'x' not in line:
        y_line = float(line.split('=')[1].strip())
        x1 = np.random.uniform(0, 10)
        x2 = np.random.uniform(x1 + 0.01, 20)
        y1 = y2 = (y_line * times)
        A = (x1, y1)
        B = (x2, y2)
        split_images(B, A, num)

    else:
        k = float(line.split('x')[0].split('=')[1].strip())
        b = float(line.split('+')[1].strip())
        y_line = lambda x: k * x + (b * times)
        x1 = np.random.uniform(0, 10)
        x2 = np.random.uniform(x1 + 0.01, 20)
        y1 = y_line(x1)
        y2 = y_line(x2)
        A = (x1, y1)
        B = (x2, y2)
        #print('----------')
        #print(line)
        #print(num)

        if num == 2:
            print(A,B)
            print(line)
        #if k > 0:
        split_images(A, B, num)

        # else:
        #     split_images(A, B, num)


save_dir = 'mask_slides/scan184'
os.makedirs(save_dir, exist_ok=True)
base_name = os.path.basename(image_path).split('.')[0]
file_template = os.path.join(save_dir, f'{base_name}_{{}}.jpg')

# cv2.imwrite(file_template.format(1), split_image1)
# cv2.imwrite(file_template.format(2), split_image2)
# cv2.imwrite(file_template.format(3), split_image3)
# cv2.imwrite(file_template.format(4), split_image4)

plt.imsave(file_template.format(1), split_image1, cmap='gray')
plt.imsave(file_template.format(2), split_image2, cmap='gray')
plt.imsave(file_template.format(3), split_image3, cmap='gray')
plt.imsave(file_template.format(4), split_image4, cmap='gray')


print("Images have been saved successfully.")

