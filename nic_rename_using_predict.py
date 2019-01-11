import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import random
import os


def get_filepaths(directory):
    file_paths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths

# !important full path
full_file_paths = get_filepaths("/home/lmadhuranga/PycharmProjects/crazyfox-identify-NIC-or-Not-using-tensorflow-keras-CNN/suffle")


def preds():
    for f in full_file_paths:
        if f.endswith(".jpg"):
            testimage = image.load_img(f, target_size=(64, 64))

            testimage = image.img_to_array(testimage)
            testimage = np.expand_dims(testimage, 0)
            # testimage  = testimage.astype('float64')
            testimage /= 255
            model = load_model('trainedModles/nICTrainedModelVNic700.h5')
            results = model.predict(testimage)
            preds = model.predict_proba(testimage)
            pred_classes = np.argmax(preds)
            print("pred class", pred_classes)
            final_score = preds[0][0] + preds[0][1] + preds[0][2]
            print("final score", final_score)

        yield pred_classes, f


preds()

for pred_classes, f in preds():
    randomno = random.randint(1, 90000)._str_()
    if (pred_classes == 0):
        os.rename(f, '/home/lmadhuranga/PycharmProjects/crazyfox-identify-NIC-or-Not-using-tensorflow-keras-CNN/output' + f'{randomno}' + '.jpg')
        print("License_front")
    elif (pred_classes == 1):
        os.rename(f, '/home/lmadhuranga/PycharmProjects/crazyfox-identify-NIC-or-Not-using-tensorflow-keras-CNN/output' + f'{randomno}' + '.jpg')
        print("NIC Back")
    elif (pred_classes == 2):
        os.rename(f, '/home/lmadhuranga/PycharmProjects/crazyfox-identify-NIC-or-Not-using-tensorflow-keras-CNN/output' + f'{randomno}' + '.jpg')
        print("NIC Front")

print('Process End')