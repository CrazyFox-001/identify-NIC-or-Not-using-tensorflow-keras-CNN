import numpy as np
from keras.models import  load_model
from keras.preprocessing import image

testimage = image.load_img('predict/12.jpg',target_size=(64,64))

testimage = image.img_to_array(testimage)
testimage = np.expand_dims(testimage,0)
#testimage  = testimage.astype('float32')
testimage /= 255
model = load_model('ids.h5')
results =model.predict(testimage)

preds = model.predict_proba(testimage)
pred_classes = np.argmax(preds)
print("pred class",pred_classes)

final_score = preds[0][0]+preds[0][1]+preds[0][2]
print("final score",final_score)

nic_front = preds[0][2]
nic_back = preds[0][1]
license  = preds[0][0]

precentage = license/final_score *100

if (pred_classes == 0):
    print("License front")
elif (pred_classes == 1):
    print("NIC Back")
elif (pred_classes == 2):
    print("NIC Front")
print("prediction",preds)
print("precentage",precentage)



