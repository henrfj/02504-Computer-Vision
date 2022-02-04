import cv2
import matplotlib.pyplot as plt

def display_image(path):
    bgr_img = cv2.imread(path)
    b,g,r = cv2.split(bgr_img)       # get b,g,r
    image = cv2.merge([r,g,b])
    plt.imshow(image)
    plt.show()




if __name__ == "__main__":
    display_image("Castle.jpg")