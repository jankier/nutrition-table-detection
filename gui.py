import io
import os
import PySimpleGUI as sg
from PIL import Image

import numpy as np
import cv2
import os
import pandas as pd 
import tensorflow as tf
import random

file_types = [('JPEG (*.jpg)', '*.jpg'),
              ('PNG (*.png)', '*.png'),
              ('All files (*.*)', '*.*')]
def main():
    layout = [
            [sg.Frame('Wczytywanie',
                [[sg.Text('Image File'),
                sg.Input(size=(25, 1), key='-FILE-'),
                sg.FileBrowse(file_types=file_types),
                sg.Button('Load Image', key='-LOAD-')],
                [sg.Image(key='-IMAGE1-', size=(450,450))]], 
            size=(500, 550), element_justification='c', key='-FRAME1-'),

            sg.Button('Detekcja', key='-DETECTION-', size=(7,3), visible=False),
            
            sg.Frame('Obrazek po detekcji',
                [[sg.Image(key='-IMAGE2-', size=(450,450))]], 
            size=(500, 550), element_justification='c', visible=False, key='-FRAME2-')]
    ]
    window = sg.Window('Detekcja tabeli odżywczych', layout,size=(1100,600), element_justification='c', font=('Arial', 12))
    while True:
        event, values = window.read()
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        if event == '-LOAD-':
            filename = values['-FILE-']
            if os.path.exists(filename):
                image = Image.open(filename)
                image.thumbnail((450, 450))
                bio = io.BytesIO()
                image.save(bio, format='PNG')
                window['-IMAGE1-'].update(data=bio.getvalue())
                window['-DETECTION-'].update(visible=True)
        if event == '-DETECTION-':
            window['-FRAME2-'].update(visible=True)

            train_jpg_resized = []
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            img_col = cv2.imread(filename)
            h, w = img.shape[0], img.shape[1]
            x = abs(h-w)
            maxhw = max(h,w)
            vertexMove = round(x/2)
            addColRow = (vertexMove - random.randint(0, vertexMove))
            if h > w:
                for i in range(x):
                    if(i < addColRow):
                        img = np.insert(img, 0, img[:,0], axis=1)
                    else:
                        img = np.insert(img, -1, img[:,-1], axis=1)
            else:
                    for i in range(x):
                        if (i < addColRow):
                            img = np.insert(img, 0, img[0,:], axis=0)
                        else:
                            img = np.insert(img, -1, img[-1,:], axis=0)
            img_resized = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
            window_name1 = 'Resized image1'
            cv2.waitKey(0)
            train_jpg_resized.append(img_resized)
            X_test = np.array(train_jpg_resized)
            X_test = X_test/255.0
            print()
            loaded_model = tf.keras.models.load_model("object_detection_model.h5")
            result = loaded_model.predict(X_test, batch_size=None, verbose=1, steps=None, callbacks=None, 
                                    max_queue_size=10, workers=1, use_multiprocessing=False)
            xmin = (int(round(result[0][0]*maxhw)))
            ymin = (int(round(result[0][1]*maxhw)))
            xmax = (int(round(result[0][2]*maxhw)))
            ymax = (int(round(result[0][3]*maxhw)))
            if h > w:
                xmin = xmin-vertexMove
                xmax = xmax-vertexMove
            else:
                ymin = ymin-vertexMove
                ymax = ymax-vertexMove
            window_name = 'Resized image'
            cv2.rectangle(img_col, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.waitKey(0)

            image2 = cv2.cvtColor(img_col, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(image2)
            img_pil.thumbnail((450, 450))
            bio = io.BytesIO()
            img_pil.save(bio, format='PNG')
            window['-IMAGE2-'].update(data=bio.getvalue())
            
    window.close()
if __name__ == '__main__':
    main()