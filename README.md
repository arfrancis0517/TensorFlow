# TensorFlow

## Object Detection and Localization
    use convolutions to extract features and classify based on the features found within the image
    
    localization with box
    
    Objects detection models classify all the object present in the image with confidence scores and bounding box as well.
    
    Popular Algorithums are R-CNN (Region), Faster-RCNN, YOLO, SSD (single shot detector) 

## Object Segmentation

    It figures out the pixels that makes up that object.
    
    We colored the image to denote each of the detected objects.
    
    We can subdivide the image into segments, these segments identify individual objetcs within the image.
    
    Two types of image segmentation: semantic segmentation (语义分割) and instance segmentation (实例分割).
    
    Semantic: all of the object with the same type form a single classification: all objects with same meaning are one colored item, grouped into the same segment.
    
    Algorithm for semantic: Fully convolutional neural networks, U-Net, DeepLab
    
    Instance: same type objects are treated as different objects
    
    Algorithm for instance: Mask-R-CNN

## Transfer learning

    Solve time-consuming and expensive questions


    without the transfer learning, we need to train the model with a lot labeled image and computation time for the model weight to classify the images
    
    the original weights are initialized randomly.
    
    like model 1 for cats and dogs, model 2 for cows and horses: we can reuse the features like ears and eyes for model 2 by using transfer learning, layers and pretrained weights. Pre-training task and downstream task
    
    Containing similar low level features.
    
    using pre-learned network we can take advantages of that, improved performance with small datasets, the pre-trained networks might have been trained on millions of data items already, so not only we can save time and effort, our architecture can take advantage of the features learned across such a huge datasets, few people have time and resources to train at that kind of scale, using pre-trained weights. 
    
    1. Freeze the Weights
        We could take the pre-exsting weights from a number of layers from the original model, typically the feature extractors, they are generally the most expensive to learn.
    
        we only reuse the earlier layers in my model (except later dense layers....)
    
        we only train my classification and dense layers with randomly initialized weights.
    
        It is effective when we have only little dataset.
    
    2. we can retain the whole CNN, using transferred weights as the starting points.
        we want to tweak the weights even further to tailor them to our specific dataset and task.
    
        when we have a lot of own data.


​    

## CNN Summery    
    CNNs earlier layers extract low level features such as edges. Later layers use the those lower level features to extract higher level features, such as shapes.
    
    Each of Convolutional layers contains multiple filters, each filter can extract features in the image, when those features are matched to labels, we can have the model to classify a picture.
    
    If there are 64 filter in each layer and pass through 2 layers, we have 64*64 effectively filtered copies of image will pass throught the next layer, so we need pooling layers to reduce the number of computations, to reduce the number of the pixcel in the image and still mantain the features of the image and enhance those features.
    
    2D image is first flatted into 1D using a flatten layer and then fed into the dense layers for a typical classification.
    
    The output layer having one neuron for each output class representing the probability that the input matches that class(define how many classes here) N-way-softmax (N is the number of classes --> neurons)   (softmax is the activation function to scale layer's output value so that they add up to one and can be treated as probability that the image is dog, cat or something else)



## Object detection

#### *Sliding Window* +*Selective search*

Sliding windows make use of rectangles that pass across the image.

**Non-maximum suppression(NMS)** to selecte only one box for the target based on **Intersection of Union(IoU)**

Suppressing or ignoring the slide windows that don't have the max IoU and keeping the max



#### Two steps:

1. Region proposal: propose a region.
2. Object detection and classification: identify the object within the region.



#### Examples:

1. **R-CNN: Regions with CNN features**:

Input image --> Extract region proposals (~2k regions): find many interesting regions with boxes and find objects based on the color similarity, texture, sizing, shapes... --> Compute CNN features (Alex net: ConvNet + SVM(scalable vector machine) to get labels for the object, and Regression to get the boxes) --> Classify the regions   2013

Transfer Learning for R-CNN: to pre-train the CNN section of the R-CNN model (Large auxiliary Dataset) and fine tune the model

2. **Fast R-CNN:**

with the removal of the expensive **selective search algorithm** 

Expects region proposals as inputs Using ConvNet to extract --> Extracted the interest from the feature map - small areas (**region of interest projection**) --> down-sampling the feature map (pooling layer) --> Softmax(multi outputs) + bbox regression

directly went from ConvNet to dense(feature vector) to softmax and bbox regression.

3. **Faster R-CNN:**

with **region proposal network (RPN)** , which is fully CNN, predict at same time the object boxes and object scores.

employing anchors or priors.

center of the anchor box come from the coordinates of the sliding window and boundries of the box come from the RPN giving the score the boundry fitting the objects.

4. pre-trained model: **RetinaNet**:

Focal Loss for Dense Object Detection.

Class subnet: retrain this net

Box subnet



## Image Segmentation

detemine the shape of the object in the image: the outline of the shape will be determined. --> detemine how to partition an image into multiple segments and have each associated with an object.-->find every pixel in the image and classify them into different class. 



Encoder: feature extraction (CNN without fully conneted layers) and down sampling, pooling the aggregate low level features to high level features --> Decoder(CNN): Up-samples feature map  and generates the pixel wise label map(to original pixel size and assign the class labels to the pixels and upsample the image), **pixel mask**.



### **Example:**

**Fully CNN (FCN)**: ConvNet to ConvNet: 

Encoder: Feature extractors like the feature extracting layer used in object detection

so we can use the CNN part of the model: **VGG16, ResNet50, MobileNet, acronym FCM** 

Decoder: **FCN-32s, FCN-16s, FCN-8s, Ground-truth** 

- SegNet: encoder are symmetric with decoder: (like Fourier transfer and inverse Fourier transfer with pooling layer and ReLU..)
- U-Net: symmetric. for Biomedical Image
- PSPNet
- Mask-RCNN: for instance seg. from Faster R-CNN, after feature extraction there will be up sampling to produce pixel-wise segmentation mosques of the image.



### **Upsampling:**

**`tf.keras.layers.UpSampling2D()` layer or `tf.keras.layers.Conv2DTranspose` (deconvolution)**

Two type of scaling(interpolation): **Nearest** (copy the value from the nearest)  and **Bilinear**(linear interpolation from nearby pixel)



```python
tf.keras.layers.UpSampling2D(
    size=(2, 2), data_format=None, interpolation="nearest", **kwargs
)
```



```python
tf.keras.layers.Conv2DTranspose(
    filters=32, kernel_size=(3,3), 
    
    strides=(1, 1), padding='valid',
    output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None,
    use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)
```



## Interpretation

#### Class Activation Maps





## G-DNN: Generative Deep Learning

