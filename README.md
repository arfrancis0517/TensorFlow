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

## CNN Summery    
    CNNs earlier layers extract low level features such as edges. Later layers use the those lower level features to extract higher level features, such as shapes.

    Each of Convolutional layers contains multiple filters, each filter can extract features in the image, when those features are matched to labels, we can have the model to classify a picture.

    If there are 64 filter in each layer and pass through 2 layers, we have 64*64 effectively filtered copies of image will pass throught the next layer, so we need pooling layers to reduce the number of computations, to reduce the number of the pixcel in the image and still mantain the features of the image and enhance those features.

    2D image is first flatted into 1D using a flatten layer and then fed into the dense layers for a typical classification.

    The output layer having one neuron for each output class representing the probability that the input matches that class(define how many classes here) N-way-softmax (N is the number of classes --> neurons)   (softmax is the activation function to scale layer's output value so that they add up to one and can be treated as probability that the image is dog, cat or something else)

