import numpy as np
from keras.models import  load_model
from keras.preprocessing import image

# Testing data with given images
testimage = image.load_img('predict/sample1.jpg', target_size=(64, 64))

# Converts a PIL Image instance to a Numpy array. 
# https://www.tensorflow.org/versions/r1.6/api_docs/python/tf/keras/preprocessing/image/img_to_array
testimage = image.img_to_array(testimage)

# Expand the shape of an array.
# Insert a new axis that will appear at the axis position in the expanded array shape.
# https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.expand_dims.html
testimage = np.expand_dims(testimage, 0)

testimage /= 255

# Train model load 
model = load_model('nicTrainedModel.h5')

# Get the result using train model
results = model.predict(testimage)

preds = model.predict_proba(testimage)
pred_classes = np.argmax(preds)
print("pred class",pred_classes)

final_score = preds[0][0]+preds[0][1]+preds[0][2]
print("final score",final_score)

nic_front = preds[0][2]
nic_back = preds[0][1]
license  = preds[0][0]

precentagenicf = nic_front/final_score *100
precentagenicb = nic_back/final_score *100
precentagelic = license/final_score *100


if (pred_classes == 0):
    print("License front")
elif (pred_classes == 1):
    print("NIC Back")
elif (pred_classes == 2):
    print("NIC Front")
print("prediction",preds)
print("precentage nic front ",precentagenicf)
print("precentage nic back",precentagenicb)
print("precentage license",precentagelic)



