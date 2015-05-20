# FaceDetection_CNN
Implement Yahoo Paper: Multi-view Face Detection Using Deep Convolutional Neural Networks\n
1. Fine-tune Alex-Net using AFLW dataset. <p/>
2. Convert fully connected layers into convolutional layers by reshaping layer parameters, 
   see http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/net_surgery.ipynb
3. Get heat map for each scale of image. 
4. Process heat map by using non-maximal suppression to accurately localize the faces.

