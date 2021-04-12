import cv2
import sys


imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"


faceCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
 
)

print("Found {0} faces!".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    img = image[y:y+h, x:x+w]
    dim = (96, 96)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(f"/content/Face-recog-comp-project/detected_images/image{x}.jpg", img)


cv2.imwrite('/content/Face-recog-comp-project/image.jpg', image)
cv2.waitKey(0)
