import cv2


class Pre:
   def grayscale(self, img):
       return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


   def cropping(self, img):
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

       h, w, j = img.shape
       new_width = w/2
       new_height = h/2


       if h > w:
           ymin, ymax, xmin, xmax = h // 2 - new_width, h // 2 + new_width, w // 2 - new_width, w // 2 + new_width
           # ymin, ymax, xmin, xmax = 0, w, 0, w
       else:
           ymin, ymax, xmin, xmax = h // 2 - new_height, h // 2 + new_height, w // 2 - new_height, w // 2 + new_height
           # ymin, ymax, xmin, xmax = 0, h, 0, h
           print("bukan ya")


       crop = img[int(ymin):int(ymax), int(xmin):int(xmax)]

       resize = cv2.resize(crop, (100, 100))

       return resize


   def preprocessing(self, img):
       crop = self.cropping(img)
       gray = self.grayscale(crop)

       return gray


p = Pre()

img = "dataset/batik-dayak/dayak (6).png"
img = cv2.imread(img)

cv2.imshow("belum", img)
cv2.imshow("sudah", p.preprocessing(img))

cv2.waitKey(0)

