import cv2


get_max = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左击按下
        # 获取鼠标按下位置的hsv值
        h, s, v = hsv[y, x]

        get_max.append([h,s,v])
        print(f'H:{h}, S:{s}, V:{v}')


img = cv2.imread(r'images/scan17.ndpi - Series 3.jpg')  # 加载图片
cropImg = img[2944:3145, 1694:1981]
resized_image = cv2.resize(cropImg, (1500, 1000))
hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)  # 将图片转为hsv

img_name = 'image'
cv2.namedWindow(img_name)
cv2.setMouseCallback(img_name, mouse_callback)  # 设置鼠标回调

cv2.imshow(img_name, resized_image)  # 展示图片
cv2.waitKey(0)

h_list = [value[0] for value in get_max]
s_list = [value[1] for value in get_max]
v_list = [value[2] for value in get_max]

print('h最大值：',max(h_list))
print('s最大值：',max(s_list))
print('v最大值：',max(v_list))

print('h最小值：',min(h_list))
print('s最小值：',min(s_list))
print('v最小值：',min(v_list))

# row = [point[0]  for point in split _points]
# col = [point[1]  for point in split_points]