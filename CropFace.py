import os
import glob
import cv2

PATH_TO_TEST_IMAGES_DIR = 'D:/conference/ICCV2019/ICCV 2019/data/Test/image'
CWD_PATH = os.getcwd()
#TEST_IMAGE_PATH = os.path.join(CWD_PATH, PATH_TO_TEST_IMAGES_DIR)
TEST_IMAGE_PATH = PATH_TO_TEST_IMAGES_DIR
DETECT_RESULT_DIR = 'D:/conference/ICCV2019/ICCV 2019/data/Test/cropface'
WIDTH = 600
HEIGHT = 800

def save_faces(cascade, image):
    img = cv2.cvtColor(cv2.imread(image, -1), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (WIDTH, HEIGHT))
    name = os.path.split(image)[1]
    name = os.path.splitext(name)[0]
    # print("name : " + name)
    for i, face in enumerate(cascade.detectMultiScale(img)):
        x, y, w, h = face
        sub_face = img[y:y + h, x:x + w]
        cv2.imwrite("%s/%s.png" % (DETECT_RESULT_DIR, name), cv2.cvtColor(sub_face, cv2.COLOR_RGB2BGR))
        print("SAVE : %s/%s.png" %(DETECT_RESULT_DIR, name))

if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier('D:/opencv_source/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
    #face_cascade = face_cascade.load('haarcascade_frontalface_default.xml')
    # cascade = cv2.CascadeClassifier(face_cascade)
    # Iterate through files
    i = 0
    for test_image in glob.glob(TEST_IMAGE_PATH + '/*.png'):
        # if i == 1:
        #     break
        save_faces(face_cascade, test_image)
        print("Completed : " + str(test_image))
        i += 1