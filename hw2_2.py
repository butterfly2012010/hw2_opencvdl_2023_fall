import cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter import filedialog
from tkinter import Tk


# opencv api function
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 讀取圖像
img_path = '/content/histoEqualGray2.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 執行直方圖均衡化
img_eq = cv2.equalizeHist(img)

# 計算兩個圖像的直方圖
hist_original = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
hist_equalized = cv2.calcHist([img_eq], [0], None, [256], [0, 256]).flatten()

# 顯示圖像和直方圖
plt.figure(figsize=(10, 8))

# 顯示原始圖像
plt.subplot(221), plt.imshow(img, 'gray'), plt.title('Original Image')

# 顯示均衡化圖像
plt.subplot(222), plt.imshow(img_eq, 'gray'), plt.title('Equalized Image')

# 顯示原始圖像的直方圖
plt.subplot(223), plt.bar(range(256), hist_original), plt.title('Histogram of Original')

# 顯示均衡化圖像的直方圖
plt.subplot(224), plt.bar(range(256), hist_equalized), plt.title('Histogram of Equalized')

# 顯示圖形
plt.tight_layout()
# plt_path = '/mnt/data/histogram_equalization_result.png'
# plt.savefig(plt_path)
plt.show()

# plt_path


# manual
# 計算直方圖並正規化以獲得 PDF
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
pdf = hist / sum(hist)

# 計算 CDF
cdf = pdf.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

# 使用 CDF 值創建查找表
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf_final = np.ma.filled(cdf_m,0).astype('uint8')

# 應用查找表
img_eq_manual = cdf_final[img]

# 計算手動均衡化後的直方圖
hist_manual_eq = cv2.calcHist([img_eq_manual], [0], None, [256], [0, 256]).flatten()

# 顯示手動均衡化的結果
plt.figure(figsize=(10, 8))

# 顯示原始圖像
plt.subplot(221), plt.imshow(img, 'gray'), plt.title('Original Image')

# 顯示手動均衡化圖像
plt.subplot(222), plt.imshow(img_eq_manual, 'gray'), plt.title('Manually Equalized Image')

# 顯示均衡化後的直方圖（使用 OpenCV）
plt.subplot(223), plt.plot(cdf_normalized, color = 'b'), plt.hist(img.flatten(),256,[0,256], color = 'r'), plt.legend(('CDF','Histogram'), loc = 'upper left')

# 顯示手動均衡化後的直方圖
plt.subplot(224), plt.bar(range(256), hist_manual_eq), plt.title('Histogram of Manually Equalized')

# 顯示圖形
plt.tight_layout()
# plt_path_manual = '/mnt/data/histogram_manual_equalization_result.png'
# plt.savefig(plt_path_manual)
plt.show()

# plt_path_manual
