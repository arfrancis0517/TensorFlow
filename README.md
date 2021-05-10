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



------



## Generative Adversarial Networks (GANs)



### Style transfer (Creating artistic images)

There are three ways:



1. **Supervised Learning**: building a network that has **pairs** of images that should learn how to style one and match the others. 

* **Pairs:** original & desired stylized images (manually create the images), we need a lot



2. **Neural Style Transfer**: we don't need to train a network to understand how one image should map to another. Like transfer learning, but we dont have to train the model and do it with just single pair of images.

* start with pre-trained model
* Inputs: **a single pair of images** (two images): using model to extract <u>style</u> from the first image and extract <u>content</u> from the second image.
* Create an image with elements that matches the style and content, we will do it literatively and loop by **minimizing the loss** of the generated image with style of the first image and the second image.



3. **Fast Neural Style Transfer**



#### Neural Style Transfer

![style transfer](/Users/francis/Desktop/style transfer.png)



* Pre-trained CNN (VGG-19) to extract the content features from the content image, the feature only from last block
* Pre-trained CNN (VGG-19) to extract the style features from the style image, the features from every layer of networks
* Intialize our generated image from the content image
* Compare with content image with content loss, on every iteration it checks how much of the original content is presented in the generated image.
* Compare with style image using a metric called the style loss, on each layer, then average the loss.
* Two loss (content loss and style loss) added together to get the total loss.
* Use optimizer to update the generated image to reduce the total loss.
* We only have one network (the graph is easy to understand)



Advantage: only need one pair image

Drawback: need time to optimize



**Steps:**

1. Load and preprocess content & style images into tensor
2. Load pretrained model (VGG-19), define loss functions for both losses

* VGG-19 was pre-trained on ImageNet
* Not need to normalized to (0,1) 
* Do need to center them that the average pixel value is zero, shifting the distribution that is centered around zero, `tf.keras.applications.vgg19.preprocess_input` to process image

3. Define the loop:

- Optimizer to the total loss and generate a new image
- Visulize the outputs



````python
tf.keras.applications.VGG19(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)
````

[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (ICLR 2015)

The default input size for this model is 224x224.

Note: each Keras Application expects a specific kind of input preprocessing. For VGG19, call `tf.keras.applications.vgg19.preprocess_input` on your inputs before passing them to the model. Preprocessed `numpy.array` or a `tf.Tensor`with type `float32`. The images are converted from RGB to BGR, then <u>each color channel is zero-centered with respect to the ImageNet dataset, without scaling.</u>

- **include_top**: whether to include **the 3 fully-connected layers** (the classification layers) at the top of the network.
- **weights**: one of `None` (random initialization), **`imagenet` (pre-training on ImageNet)**, or the path to the weights file to be loaded.
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
- **input_shape**: optional shape tuple, only to be specified if `include_top` is False (otherwise the input shape has to be `(224, 224, 3)` (with `channels_last` data format) or `(3, 224, 224)` (with `channels_first` data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 32. E.g. `(200, 200, 3)` would be one valid value.
- **pooling**: Optional pooling mode for feature extraction when `include_top`  is `False`.
  - `None` means that the output of the model will be the 4D tensor output of the last convolutional block.
  - `avg` means that global average pooling will be applied to the output of the last convolutional block, and thus the output of the model will be a 2D tensor.
  - `max` means that global max pooling will be applied.
- **classes**: optional number of classes to classify images into, only to be specified if `include_top` is True, and if no `weights` argument is specified.
- **classifier_activation**: A `str` or callable. The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set `classifier_activation=None` to return the logits of the "top" layer.

------

**Build Model**

* Taking first convolutional layer for each block, Conv1_1, Conv2_1,.., using pre-trained filters to extract the styles from the style image 
* Taking last convolutional layer for last block, sometimes a little before last layer (second layer for last block) to extract the content from the content image 

````python
# style layers of interest
style_layers = ['block1_conv1', 
                'block2_conv1', 
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'] 

# choose the content layer and put in a list
content_layers = ['block5_conv2'] 

# combine the two lists into one list (put the style layers before the content layers)
output_layers = style_layers + content_layers  # layer_names

# Define the model to take the same input as the standard VGG-19 model, and output just the selected content and style layers.

def vgg_model(layer_names):
  """ Creates a vgg model that outputs the style and content layer activations.
      Top layer refers to the last output layer that performs classification, include_top=False to exclude the original output
  
  Args:
    layer_names: a list of strings, representing the names of the desired content and style layers
    
  Returns:
    A model that takes the regular vgg19 input and outputs just the content and style layers.
  
  """

  # load the the pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')

  # freeze the weights of the model's layers (make them not trainable)
  # We don't want the feature extractor to update its internal weights based on these inputs
  vgg.trainable = False
  
  # create a list of layer objects that are specified by layer_names
  outputs = [vgg.get_layer(name).output for name in layer_names]

  # create the model that outputs content and style layers only
  model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

  return model
````



**Total Loss, Content Loss, Style Loss**

$L_{total}(\mathord{ \buildrel{ \lower3pt \hbox{$ \scriptscriptstyle \rightharpoonup$}} \over p}, \mathord{ \buildrel{ \lower3pt \hbox{$ \scriptscriptstyle \rightharpoonup$}} \over a}, \mathord{ \buildrel{ \lower3pt \hbox{$ \scriptscriptstyle \rightharpoonup$}} \over x}  ) = 𝛼L_{content}(\mathord{ \buildrel{ \lower3pt \hbox{$ \scriptscriptstyle \rightharpoonup$}} \over p}, \mathord{ \buildrel{ \lower3pt \hbox{$ \scriptscriptstyle \rightharpoonup$}} \over x} ) + βL_{style}(\mathord{ \buildrel{ \lower3pt \hbox{$ \scriptscriptstyle \rightharpoonup$}} \over a}, \mathord{ \buildrel{ \lower3pt \hbox{$ \scriptscriptstyle \rightharpoonup$}} \over x} )$  

$𝛼$ is content weight and $β$ is style weight, how much we want to keep the content or style

$\mathord{ \buildrel{ \lower3pt \hbox{$ \scriptscriptstyle \rightharpoonup$}} \over p}   $ : content image (Original Photograph)

$  \mathord{ \buildrel{ \lower3pt \hbox{$ \scriptscriptstyle \rightharpoonup$}} \over x} $ : Generated image initialized to input image

$\mathord{ \buildrel{ \lower3pt \hbox{$ \scriptscriptstyle \rightharpoonup$}} \over a}$ : Style image



$E_{l} = \frac{1}{4N_{l}^2M_{l}^2}\sum_{i,j}(G_{i,j}^l-A_{i,j}^l)^2  $ 

$l$ is which layer

$G_{i,j}^l$     $A_{i,j}^l$   are Style Representation **(Gram Matrix)** of  generated image and style image

$G_{i,j}^l=\sum_{k}F_{ik}^lF_{jk}^l$ : inner product between **vectorized feature** maps $i,j$ in layer $l$



Loss: sum of element-wise square of element-wise subtraction

`tf.reduce_sum` computes the sum of elements across dimensions of a tensor.

`tf.square` computes square of x element-wise.



Use `tf.linalg.einsum` to calculate **the gram matrix** for an input tensor. In addition, calculate **the scaling factor** `num_locations` and divide the gram matrix calculation by `num_locations`.

​                                                                               $num\_locations=ℎ𝑒𝑖𝑔ℎ𝑡×𝑤𝑖𝑑𝑡ℎ$

```python
tf.linalg.einsum(
    equation, *inputs, **kwargs
)
```

This computation is defined by `equation`, a shorthand form based on Einstein summation. As an example, consider multiplying two matrices A and B to form a matrix C. The elements of C are given by: C[i,k] = sum_j A[i,j] * B[j,k]

The corresponding einsum `equation` is:  ij, jk -> ik​

In general, to convert the element-wise equation into the `equation` string, use the following procedure (intermediate strings for matrix multiplication example provided in parentheses):

1. **remove variable names, brackets, and commas**, (`ik = sum_j ij * jk`)
2. **replace "*" with ","**, (`ik = sum_j ij , jk`)
3. **drop summation signs**, and (`ik = ij, jk`)
4. **move the output to the right, while replacing "=" with "->"**. (`ij,jk->ik`)

For example:

style layer: 

H = 2 height

W = 2 width

F = 2 filters

$\begin{bmatrix}
  1&2 \\
  4&5
\end{bmatrix}\begin{bmatrix}
  5&7 \\
  2&3
\end{bmatrix}$ Two filters

Flatten into collumn vectors

$\begin{bmatrix}
 1\\
 2\\
 4\\
 5
\end{bmatrix}\begin{bmatrix}
 5\\
 7\\
 2\\
 3
\end{bmatrix}$ 

Put into a new matix A:

$A=\left [\begin{bmatrix}
 1\\
 2\\
 4\\
 5
\end{bmatrix}_{a_{1}}\begin{bmatrix}
 5\\
 7\\
 2\\
 3
\end{bmatrix}_{a_{2}}  \right ]$  

Create a new matrix G:

$G=\begin{bmatrix}
  a_1 * a_1 & a_1 * a_2   \\
  a_2 * a_1 & a_2 * a_2   
\end{bmatrix}=A^TA$  this is Gram Matrix

This feature exacted from image

```python
style_layer = tf.constant([1,2,4,5,5,7,2,3], shape = (2,2,2))
A = tf.transpose(
     tf.reshape( style_layer,
                 shape = (2,4))
)

AT = tf.transpose(A)
G = tf.matmul(AT, A)

# or
G = tf.linalg.einsum('cij,dij->cd', style_layer, style_layer)

```



````Python
def get_style_loss(features, targets):
  """Expects two images of dimension h, w, c
  
  Args:
    features: tensor with shape: (height, width, channels)
    targets: tensor with shape: (height, width, channels)

  Returns:
    style loss (scalar)
  """
  # get the average of the squared errors
  style_loss = tf.reduce_mean(tf.square(features - targets))
    
  return style_loss
  
def get_content_loss(features, targets):
  """Expects two images of dimension h, w, c
  
  Args:
    features: tensor with shape: (height, width, channels)
    targets: tensor with shape: (height, width, channels)
  
  Returns:
    content loss (scalar)
  """
  # get the sum of the squared error multiplied by a scaling factor
  content_loss = 0.5 * tf.reduce_sum(tf.square(features - targets))
    
  return content_loss

def gram_matrix(input_tensor):
  """ Calculates the gram matrix and divides by the number of locations
  Args:
    input_tensor: tensor of shape (batch, height, width, channels)
    
  Returns:
    scaled_gram: gram matrix divided by the number of locations
  """

  # calculate the gram matrix of the input tensor
  gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor) # Einstein summation
  
  # input = (batch, height, width, filters) four dimensions 
          #   b        i      j      c/d (first/second tensor)
  # output = (batch, filters, filters) 
  #            b       c(first)  d(second)    
  # tf.linalg.einsum performs matrix multiplication and reduce_sum
  
  # get the height and width of the input tensor
  input_shape = tf.shape(input_tensor) 
  height = input_shape[1] 
  width = input_shape[2] 

  # get the number of locations (height times width), and cast it as a tf.float32
  num_locations = tf.cast(height * width, tf.float32)

  # scale the gram matrix by dividing by the number of locations
  scaled_gram = gram / num_locations
    
  return scaled_gram
````



**Get the style image features**

Given the style image as input, you'll get the style features of the custom VGG model that you just created using `vgg_model()`.

- You will <u>first preprocess the image</u> using the given `preprocess_image()` function.
- You will then <u>get the outputs of the vgg model</u>.
- From the outputs, just get the style feature layers and not the content feature layer.

```python
def preprocess_image(image):
  '''centers the pixel values of a given image to use with VGG-19'''
  image = tf.cast(image, dtype=tf.float32)
  image = tf.keras.applications.vgg19.preprocess_input(image)

  return image


def get_style_image_features(image):  
  """ Get the style image features
  
  Args:
    image: an input image
    
  Returns:
    gram_style_features: the style features as gram matrices
  """
  # preprocess the image using the given preprocessing function
  preprocessed_style_image = preprocess_image(image) 

  # get the outputs from the custom vgg model that you created using vgg_model()
  outputs = vgg(preprocessed_style_image) # list style outputs

  # Get just the style feature layers (exclude the content layer)
  style_outputs = outputs[:NUM_STYLE_LAYERS] 

  # for each style layer, calculate the gram matrix for that layer and store these results in a list
  gram_style_features = [gram_matrix(style_layer) for style_layer in style_outputs] 

  return gram_style_features
```



The total loss is given by $𝐿_{𝑡𝑜𝑡𝑎𝑙}=𝛽𝐿_{𝑠𝑡𝑦𝑙𝑒}+𝛼𝐿_{𝑐𝑜𝑛𝑡𝑒𝑛𝑡}$, where $𝛽$ and $𝛼$ are weights we will give to the content and style features to generate the new image. See how it is implemented in the function below.

`tf.math.add_n` performs the same operation as `tf.math.accumulate_n`, but it waits for all of its inputs to be ready before beginning to sum. This buffering can result in higher memory consumption when inputs are ready at different times, since the minimum temporary storage required is proportional to the input size rather than the output size.

```python
tf.math.add_n(
    inputs, name=None
)
```



**Total variation loss**

One downside to the implementation above is that it produces a lot of **high frequency artifacts** (原来图片中的边缘线，勾勒图片的主要信息). You can see this when you plot the frequency variations of the image. We've defined a few helper functions below to do that.

* Decrease high frequency artifacts
* apply explicit regularization on the components that have high frequency values

Extract High Frequency Components:

```python
# high pass filter smooth the image, only allow the image value above certain threshold through
#  this algorithm determines the differences between adjacent pixels on both the x and y axes to help us regularize them.
def high_pass_x_y(image):
  x_var = image[:,:,1:,:] - image[:,:,:-1,:]
  y_var = image[:,1:,:,:] - image[:,:-1,:,:]

  return x_var, y_var
```



```python
def get_style_content_loss(style_targets, style_outputs, content_targets, 
                           content_outputs, style_weight, content_weight):
  """ Combine the style and content loss
  
  Args:
    style_targets: style features of the style image
    style_outputs: style features of the generated image
    content_targets: content features of the content image
    content_outputs: content features of the generated image
    style_weight: weight given to the style loss
    content_weight: weight given to the content loss

  Returns:
    total_loss: the combined style and content loss

  """
    
  # sum of the style losses
  style_loss = tf.add_n([ get_style_loss(style_output, style_target)
                           for style_output, style_target in zip(style_outputs, style_targets)])
  # we sum all style_layers to get style_loss
  # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。 style_output = zip(style_outputs), style_target = zip(style_targets)
  
  # Sum up the content losses
  content_loss = tf.add_n([get_content_loss(content_output, content_target)
                           for content_output, content_target in zip(content_outputs, content_targets)])

  # scale the style loss by multiplying by the style weight and dividing by the number of style layers
  style_loss = style_loss * style_weight / NUM_STYLE_LAYERS 

  # scale the content loss by multiplying by the content weight and dividing by the number of content layers
  content_loss = content_loss * content_weight / NUM_CONTENT_LAYERS 
    
  # sum up the style and content losses
  total_loss = style_loss + content_loss 

  return total_loss



def calculate_gradients(image, style_targets, content_targets, 
                        style_weight, content_weight, var_weight):
  """ Calculate the gradients of the loss with respect to the generated image
  Args:
    image: generated image
    style_targets: style features of the style image
    content_targets: content features of the content image
    style_weight: weight given to the style loss
    content_weight: weight given to the content loss
    var_weight: weight given to the total variation loss
  
  Returns:
    gradients: gradients of the loss with respect to the input image
  """
  with tf.GradientTape() as tape:
      
    # get the style image features
    style_features = get_style_image_features(image) 
      
    # get the content image features
    content_features = get_content_image_features(image) 
      
    # get the style and content loss
    loss = get_style_content_loss(style_targets, style_features, content_targets, 
                                  content_features, style_weight, content_weight) 

    # add the total variation loss
    loss += var_weight*tf.image.total_variation(image)

  # calculate gradients of loss with respect to the image
  gradients = tape.gradient(loss, image) 

  return gradients

```

Similar to model training, you will use an optimizer to update the original image from the computed gradients. Since we're dealing with images, we want to clip the values to the range we expect. That would be `[0, 255]` in this case.

```python
def update_image_with_style(image, style_targets, content_targets, style_weight, 
                            var_weight, content_weight, optimizer):
  """
  Args:
    image: generated image
    style_targets: style features of the style image
    content_targets: content features of the content image
    style_weight: weight given to the style loss
    content_weight: weight given to the content loss
    var_weight: weight given to the total variation loss
    optimizer: optimizer for updating the input image
  """

  # calculate gradients using the function that you just defined.
  gradients = calculate_gradients(image, style_targets, content_targets, 
                                  style_weight, content_weight, var_weight) 

  # apply the gradients to the given image
  optimizer.apply_gradients([(gradients, image)])  # Apply gradients to variables.

  # clip the image using the utility clip_image_values() function
  image.assign(clip_image_values(image, min_value=0.0, max_value=255.0))
```

You can now define the main loop. This will use the previous functions you just defined to **generate the stylized content image**. It does so incrementally based on the computed gradients and the number of epochs. Visualizing the output at each epoch is also useful so you can quickly see if the style transfer is working.

```python
def fit_style_transfer(style_image, content_image, style_weight=1e-2, content_weight=1e-4, 
                       var_weight=0, optimizer='adam', epochs=1, steps_per_epoch=1):
  """ Performs neural style transfer. looking like a training loop
  Args:
    style_image: image to get style features from
    content_image: image to stylize 
    style_targets: style features of the style image
    content_targets: content features of the content image
    style_weight: weight given to the style loss
    content_weight: weight given to the content loss
    var_weight: weight given to the total variation loss
    optimizer: optimizer for updating the input image
    epochs: number of epochs
    steps_per_epoch = steps per epoch
  
  Returns:
    generated_image: generated image at final epoch
    images: collection of generated images per epoch  
  """

  images = []
  step = 0

  # get the style image features 
  style_targets = get_style_image_features(style_image)
    
  # get the content image features
  content_targets = get_content_image_features(content_image)

  # initialize the generated image for updates
  generated_image = tf.cast(content_image, dtype=tf.float32)
  generated_image = tf.Variable(generated_image) 
  
  # collect the image updates starting from the content image
  images.append(content_image)
  
  # incrementally update the content image with the style features
  for n in range(epochs):
    for m in range(steps_per_epoch):
      step += 1
    
      # Update the image with the style using the function that you defined
      update_image_with_style(generated_image, style_targets, content_targets, 
                              style_weight, content_weight, var_weight, optimizer) 
    
      print(".", end='')

      if (m + 1) % 10 == 0:
        images.append(generated_image)
    
    # display the current stylized image
    clear_output(wait=True)
    display_image = tensor_to_image(generated_image)
    display_fn(display_image)

    # append to the image collection for visualization later
    images.append(generated_image)
    print("Train step: {}".format(step))
  
  # convert to uint8 (expected dtype for images with pixels in the range [0,255])
  generated_image = tf.cast(generated_image, dtype=tf.uint8)

  return generated_image, images
```



#### Fast Neural Style Transfer

```python
import tensorflow as tf
import tensorflow_hub as hub
# this will take a few minutes to load
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

content_image, style_image = load_images(content_path, style_path)

# stylize the image using the model you just downloaded 
stylized_image = hub_module(tf.image.convert_image_dtype(content_image, tf.float32), 
                            tf.image.convert_image_dtype(style_image, tf.float32))[0] # get first output
# convert the tensor to image
tensor_to_image(stylized_image)


```



### Auto-encoders and variational Auto-encoders

