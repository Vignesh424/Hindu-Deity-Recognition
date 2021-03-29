# USAGE
# python test_network.py --model gs.model --image examples/ganpati_01.jpg

# We import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# We construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# We load the image
image = cv2.imread(args["image"])
orig = image.copy()

# We pre-process the image for classification
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# We load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# We classify the input image
(shiva, ganpati) = model.predict(image)[0]

# We build the label
labell ="Deity: "
label =  "Lord Ganpati" if ganpati > shiva else "Lord Shiva"
proba = ganpati if ganpati > shiva else shiva
new=labell+label
labels= "{} Confidence: {:.2f}%".format(new, proba * 100)
#We draw the label on the image
output = imutils.resize(orig, width=750)
cv2.putText(output, labels, (10, 25),  cv2.FONT_HERSHEY_TRIPLEX,
	0.95, (0,255,0), 2)
filename="A.jpg"
#We show the output image
cv2.imshow("Output", output)
cv2.imwrite(filename, output)
cv2.waitKey(0)
