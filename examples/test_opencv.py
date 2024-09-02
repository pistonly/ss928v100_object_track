import cv2

img = cv2.imread("../data/dog1_1024_683.jpg")
h, w = img.shape[0:2]
img_large = cv2.resize(img, (2 * w, 2 * h))

cv2.imwrite("../out/test_opencv.jpg", img_large)
