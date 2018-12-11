from PIL import Image
import numpy as np
import sys
import os
import csv


#Useful function
def createFileList(myDir, format='.png'):
   fileList = []
   print(myDir)
   for root, dirs, files in os.walk(myDir, topdown=False):
      for name in files:
         if name.endswith(format):
            fullName = os.path.join(root, name)
            fileList.append(fullName)
   return fileList

# load the original image
myFileList = createFileList('C:\img')

for file in myFileList:
    print(type(file))
    img_file = Image.open(file)
    # img_file.show()

    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    img_grey = img_file.convert('L')
    #img_grey.save('result.png')
    #img_grey.show()

    # Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
   
    for i,x in enumerate(value):
        for j,y in enumerate(x):

            if y != 0 :
                value[i][j] = 0
            else:
                value[i][j] = 1

    value = np.insert(value,0,file[9])
    #value = np.insert(value,1)
    print(value)
    value = value.flatten()
    print(type(value))
    with open("image.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)