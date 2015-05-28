from PIL import Image
import numpy as np
import random


def IoUofTwoSameImages(region1, region2):
    #clock-wise
    m1 = ((region1[0], region1[1]),  (region1[2], region1[1]), (region1[2],region1[3]),(region1[0], region1[3]))
    m2 = ((region2[0], region2[1]), (region2[2], region2[1]), (region2[2],region2[3]), (region2[0], region2[3]))
    result = []
    intersection = 0.0
    area = 2*abs((m1[1][0] - m1[0][0]) * (m1[3][1] - m1[0][1]))
    for p in m1:
        if(p[0] >= m2[0][0] and p[0] <= m2[1][0] and p[1] >= m2[0][1] and p[1] <= m2[3][1]):
            result.append(p)
    for p in m2:
        if(p[0] >= m1[0][0] and p[0] <= m1[1][0] and p[1] >= m1[0][1] and p[1] <= m1[3][1]):
            result.append(p)
    #print "len:", len(result)
    if(len(result) == 2):
        intersection = abs((result[1][0] - result[0][0]) * (result[1][1] - result[0][1]))
    elif(len(result) == 4):
        #find the duijiao
        if(result[0][0] != result[1][0] and result[0][1] != result[1][1]):
            intersection = abs((result[1][0] - result[0][0]) * (result[1][1] - result[0][1]))
        elif(result[0][0] != result[2][0] and result[0][1] != result[2][1]):
            intersection = abs((result[2][0] - result[0][0]) * (result[2][1] - result[0][1]))
        elif(result[0][0] != result[3][0] and result[0][1] != result[3][1]):
            intersection = abs((result[3][0] - result[0][0]) * (result[3][1] - result[0][1]))
    #print region1, region2, intersection/float(area - intersection)
    return intersection/float(area - intersection)

if __name__ == "__main__":
    #write to file
    output = open('aflw.list', 'w')
    #read faces rect from file
    faces_file = open('face_rect.txt', 'r')
    imageFaces = {}
    for line in faces_file.readlines():
        if(not line.startswith('#')):
            imagePath = line.split('\t')[1].strip()
            if(imagePath in imageFaces):
                imageFaces[imagePath].append(line)
            else:
                imageFaces[imagePath] = [line]
    faces_file.close()
    #del the proprocess image.
########################################### 
   #has_pro = open('aflw.list5', 'r')
   # for line in has_pro.readlines():
#	has_image = line.split(' ')[0].split('/')[-1].split('_')[0] + '.jpg'
 #       if('flickr/3/' + has_image in imageFaces):
 	   # print has_image
#	    imageFaces.pop('flickr/3/' + has_image, None)
#	elif('flickr/0/' + has_image in imageFaces): 
	   # print has_image
 #           imageFaces.pop('flickr/0/' + has_image, None)
       # elif('flickr/2/' + has_image in imageFaces): 
           # print has_image
         #   imageFaces.pop('flickr/2/' + has_image, None)
#####################################
    count = 0
    for imagePath, faces in imageFaces.iteritems():
        imagePath = 'aflw/data/' + imagePath
        try:
            im = Image.open(imagePath)
            face_regions = []
            for item in faces:
                face_id = item.split('\t')[0].strip()
                imageName = item.split("\t")[1].split("/")[-1].replace(".jpg", "").strip()
                face_x = int(item.split('\t')[2].strip())
                face_y = int(item.split('\t')[3].strip())
                face_w = int(item.split('\t')[4].strip())
                face_h = int(item.split('\t')[5].strip())
                #crop the face.
                face_region_x = face_x + face_w
                face_region_y = face_y + face_h
                if(face_region_x > im.size[0]):
                    face_region_x = im.size[0]
                if(face_region_y > im.size[1]):
                    face_region_y = im.size[1]
                face_region = (face_x, face_y, face_region_x, face_region_y)
                face_regions.append(face_region)

            #use sliding windows of the same size with face, select the IoU >= 0.5 as face, IoU <=0.3 as non-face.
            face_width = face_regions[0][2]-face_regions[0][0]
            face_height = face_regions[0][3]-face_regions[0][1]
            step =  face_width/ 5
            for i in range(0, im.size[0]- face_width + 1, step):
                for j in range(0, im.size[1] - face_height + 1, step):
                    #crop the image.
                   # crop = im.crop((i,j, i + face_width, j+face_height))
                    iou_face = False
                    for face_region in face_regions:
                        IoU = IoUofTwoSameImages((i,j, i + face_width, j+face_height), (face_region[0], face_region[1], face_region[2], face_region[3]))
                        if(IoU >= 0.5):
                            crop = im.crop((i,j, i + face_width, j+face_height))
			    crop.save("crop_images/face/" + imageName + "_" + str(count) + ".jpg")
                            output.write("crop_images/face/" + imageName + "_" + str(count) + ".jpg" + " " + str(IoU) + "\n")
                            count += 1
                            iou_face = True
                    if(iou_face):
                        continue
                    IoUCount = 0
                    for face_region in face_regions:
                        IoU = IoUofTwoSameImages((i,j, i + face_width, j+face_height), (face_region[0], face_region[1], face_region[2], face_region[3]))
                        if(IoU <= 0.2):
                            IoUCount += 1
                            if(IoUCount == len(face_regions)):
                                pixels = list(crop.getdata())
                                array = np.array(pixels)
                                remove = 0
                                if(type(array.std(axis=0)) is np.float64):
                                    if(array.std(axis=0) < 8 and random.random() < 0.05):
					crop = im.crop((i,j, i + face_width, j+face_height))
                                        crop.save("crop_images/non-face/" + imageName + "_" + str(count) + ".jpg")
                                        output.write("crop_images/non-face/" + imageName + "_" + str(count) + ".jpg" + " " + str(IoU) + "\n")
                                        count += 1
                                        continue
                                elif(type(array.std(axis=0)) is not np.float64):
                                    #print array.std(axis=0)
                                    for k in array.std(axis=0):
                                        if(k < 8):
                                            remove += 1
                                    if(remove == 3 and random.random() < 0.05):
                                        crop = im.crop((i,j, i + face_width, j+face_height))
				        crop.save("crop_images/non-face/" + imageName + "_" + str(count) + ".jpg")
                                        output.write("crop_images/non-face/" + imageName + "_" + str(count) + ".jpg" + " " + str(IoU) + "\n")
                                        count += 1
                                        continue
                                    if(random.random() < 0.1):
                                        crop = im.crop((i,j, i + face_width, j+face_height))
				        crop.save("crop_images/non-face/" + imageName + "_" + str(count) + ".jpg")
                                        output.write("crop_images/non-face/" + imageName + "_" + str(count) + ".jpg" + " " + str(IoU) + "\n")
                                        count += 1
            print count
        except IOError:
                print "No such file", imagePath








