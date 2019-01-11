import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import random
import os

projectPath = 'C:\machine learning\identify-NIC-or-Not-using-tensorflow-keras-CNN'
unSortedImageDir = projectPath + '/suffle'


def get_filepaths(directory):
    file_paths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths


# !important full path
full_file_paths = get_filepaths(unSortedImageDir)


def preds():
    for f in full_file_paths:
        if f.endswith(".jpg"):
            testimage = image.load_img(f, target_size=(64, 64))

            testimage = image.img_to_array(testimage)
            testimage = np.expand_dims(testimage, 0)
            testimage /= 255
            model = load_model(projectPath+ '/trainedModles/nICTrainedModelVNic700.h5')
            preds = model.predict_proba(testimage)
            # print('                   lic_front', '     nic_back', '     nic_front')
            # print('preds', preds)
            pred_classes = np.argmax(preds)
            # print("pred_classes", pred_classes)
            final_score = preds[0][0] + preds[0][1] + preds[0][2]
            # print("final_score", final_score)

        yield pred_classes, f, preds, final_score


preds()
outPutFolder = '/output/'
counterNicFront = 0
counterNicBack = 0
counterNicLicence = 0
counterOther = 0

for pred_classes, f, preds, final_score in preds():
    # print('*********************************************************************')
    randomno = random.randint(1, 90000)
    print('                  lic_front', '     nic_back', '      nic_front')
    print('preds', preds)
    print("pred_classes", pred_classes)
    print("final_score", final_score)
    if(final_score>0.6):
        if pred_classes == 0:
             counterNicLicence += 1
             img_path = projectPath + '/output/licence_front/licence_front_' + repr(counterNicLicence) + '.jpg'
             os.rename(f, img_path)
             print('img_path', img_path)
             print('file',f)
             print("License_front")

        elif pred_classes == 1:
              counterNicBack += 1
              img_path = projectPath + '/output/nic_back/nic_back_' + repr(counterNicBack) + '.jpg'
              os.rename(f, img_path)
              print('img_path', img_path)
              print('file',f)
              print("NIC Back")

        elif pred_classes == 2:
              counterNicFront += 1
              img_path = projectPath + '/output/nic_front/nic_front_' + repr(counterNicFront) + '.jpg'
              os.rename(f, img_path)
              print('img_path', img_path)
              print('file',f)
              print("NIC Front")

        else:
            counterOther += 1
            print('file',f)
            print("Other image 1")
    else:
        counterOther += 1
        print('file', f)
        print("Other image 2 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

print('---------------------- Summery --------------------')
print('counterNicFront', counterNicFront)
print('counterNicBack', counterNicBack)
print('counterNicLicence', counterNicLicence)
print('counterOther', counterOther)
print('----------------------Process End--------------------')
