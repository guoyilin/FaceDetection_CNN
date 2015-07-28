# FaceDetection_CNN
Implement Yahoo Paper: Multi-view Face Detection Using Deep Convolutional Neural Networks<p/>
1. Image Preprocess aflw dataset[1]. Use iou>=0.5 as positive, iou<=0.3 as negative. <p/>
2. Fine-tune Alex-Net using AFLW dataset. The model is in Baidu Yun: http://pan.baidu.com/s/1i38IAIp <p/>
3. Convert fully connected layers into convolutional layers by reshaping layer parameters, see [2]<p/>
4. Get heat map for each scale of image. <p/>
5. Process heat map by using non-maximal suppression to accurately localize the faces.<p/>

==========
Reference:
[1]https://lrs.icg.tugraz.at/research/aflw/<p/>
[2] http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/net_surgery.ipynb<p/>

