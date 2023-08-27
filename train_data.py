import numpy as np
import cv2
import os
import pandas as pd
import random

def main():
    # specification of folder directory with training images
    folder_dir = r'/'
    # creation of new folder for resized images
    # resized_dir = r'D:\Programowanie\Python\train_jpg_resized'
    # if not os.path.exists(resized_dir):
    #     os.mkdir(resized_dir)
    # accesing csv data
    df = pd.read_csv(r'/')
    # specificating new csv directory and data frame for modified data
    # csv_dir = r'D:\Programowanie\Python\train_labels_nt_resized.csv'
    # df_resized = pd.DataFrame(columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax'])
    # risizing of images
    idx = 0
    train_jpg_resized = []
    coords_resized = []
    for filename in os.listdir(folder_dir):
        if (filename.endswith('.jpg')):
            isResized = False
            while(not isResized):
                try:
                    row = df.iloc[idx]
                    if row['filename'] == filename:
                        img = cv2.imread(os.path.join(folder_dir, filename), cv2.IMREAD_GRAYSCALE)
                        h, w = img.shape[0], img.shape[1]
                        x = abs(h-w)
                        xmin = row['xmin']
                        ymin = row['ymin']
                        xmax = row['xmax']
                        ymax = row['ymax']
                        maxhw = max(h,w)
                        vertexMove = round(x/2)
                        addColRow = (vertexMove - random.randint(0, vertexMove))
                        if h > w:
                            xmin += vertexMove
                            xmax += vertexMove 
                            for i in range(x):
                                if(i < addColRow):
                                    img = np.insert(img, 0, img[:,0], axis=1)
                                else:
                                    img = np.insert(img, -1, img[:,-1], axis=1)
                        else:
                            ymin += vertexMove
                            ymax += vertexMove
                            for i in range(x):
                                if (i < addColRow):
                                    img = np.insert(img, 0, img[0,:], axis=0)
                                else:
                                    img = np.insert(img, -1, img[-1,:], axis=0)
                        img_resized = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
                        # turning training data images into arrays 
                        train_jpg_resized.append(img_resized)
                        coords_resized.append((xmin/maxhw, ymin/maxhw, xmax/maxhw, ymax/maxhw))
                        # saving of images
                        # os.chdir(r'D:\Programowanie\Python\train_jpg_resized')
                        # filename_resized = os.path.splitext(filename)[0] + '_resized' + os.path.splitext(filename)[1]
                        # if not os.path.isfile(filename_resized):
                        #     cv2.imwrite(filename_resized, img_resized)
                        isResized = True
                        # writing data into csv
                        # df_resized.loc[idx] = (filename_resized, str(xmin/maxhw), str(ymin/maxhw), str(xmax/maxhw), str(ymax/maxhw))    
                except Exception:
                    pass                  
                idx += 1
    # saving csv with modified data
    # df_resized.to_csv(csv_dir, index=False)
    # saving training data images as arrays
    X_train = np.array(train_jpg_resized)
    y_train = np.array(coords_resized)
    X_train = X_train/255.0
    np.save('X_train.npy',X_train, allow_pickle=True)
    np.save('y_train.npy',y_train, allow_pickle=True)
    print('end')
main()