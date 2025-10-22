import tensorflow.compat.v2 as tf


from tensorflow.keras import backend
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from tensorflow.keras import utils as keras_utils  # ✅ add this line
data_utils = keras_utils                           # ✅ alias for compatibility
layer_utils = keras_utils  


# isort: off
from tensorflow.python.util.tf_export import keras_export



WEIGHTS_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5"
)
WEIGHTS_PATH_NO_TOP = (
    "https://storage.googleapis.com/tensorflow/"
    "keras-applications/vgg19/"
    "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
)


def VGG19(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
):
    """Instantiates the VGG19 architecture.
    Reference:
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition](
        https://arxiv.org/abs/1409.1556) (ICLR 2015)
    For image classification use cases, see
    [this page for detailed examples](
      https://keras.io/api/applications/#usage-examples-for-image-classification-models).
    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](
      https://keras.io/guides/transfer_learning/).
    The default input size for this model is 224x224.
    Note: each Keras Application expects a specific kind of input preprocessing.
    For VGG19, call `tf.keras.applications.vgg19.preprocess_input` on your
    inputs before passing them to the model.
    `vgg19.preprocess_input` will convert the input images from RGB to BGR,
    then will zero-center each color channel with respect to the ImageNet
    dataset, without scaling.
    Args:
      include_top: whether to include the 3 fully-connected
        layers at the top of the network.
      weights: one of `None` (random initialization),
          'imagenet' (pre-training on ImageNet),
          or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
        (i.e. output of `Input()`)
        to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape
        has to be `(224, 224, 3)`
        (with `channels_last` data format)
        or `(3, 224, 224)` (with `channels_first` data format).
        It should have exactly 3 inputs channels,
        and width and height should be no smaller than 32.
        E.g. `(200, 200, 3)` would be one valid value.
      pooling: Optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional block.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional block, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
      classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
    Returns:
      A `keras.Model` instance.
    """
    if not (weights in {"imagenet", None} or tf.io.gfile.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded.  "
            f"Received: `weights={weights}.`"
        )


    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            'If using `weights` as `"imagenet"` with `include_top` '
            "as true, `classes` should be 1000.  "
            f"Received: `classes={classes}.`"
        )
    # Determine proper input shape
# ✅ Keras 3 compatible: manually handle input shape
    if input_shape is None:
      input_shape = (224, 224, 3)
    else:
      if len(input_shape) != 3 or input_shape[-1] != 3:
        raise ValueError(
            f"Expected input_shape with 3 channels, got {input_shape}"
        )



    if input_tensor is None:
        img_input =  Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input =  Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x =  Conv2D(
        64, (3, 3), padding="same", name="block1_conv1_before_activation"
    )(img_input)
    x=  Activation("relu",name="block1_conv1")(x)


    x =  Conv2D(
        64, (3, 3), padding="same", name="block1_conv2_before_activation"
    )(x)


    x=  Activation("relu",name="block1_conv2")(x)


    x =  MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)


    # Block 2
    x =  Conv2D(
        128, (3, 3), padding="same", name="block2_conv1_before_activation"
    )(x)


    x=  Activation("relu",name="block2_conv1")(x)


    x =  Conv2D(
        128, (3, 3), padding="same", name="block2_conv2_before_activation"
    )(x)


    x=  Activation("relu",name="block2_conv2")(x)


    x =  MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)


    # Block 3
    x =  Conv2D(
        256, (3, 3), padding="same", name="block3_conv1_before_activation"
    )(x)


    x=  Activation("relu",name="block3_conv1")(x)


    x =  Conv2D(
        256, (3, 3),  padding="same", name="block3_conv2_before_activation"
    )(x)


    x=  Activation("relu",name="block3_conv2")(x)


    x =  Conv2D(
        256, (3, 3), padding="same", name="block3_conv3_before_activation"
    )(x)


    x=  Activation("relu",name="block3_conv3")(x)


    x =  Conv2D(
        256, (3, 3), padding="same", name="block3_conv4_before_activation"
    )(x)


    x=  Activation("relu",name="block3_conv4")(x)


    x =  MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)


    # Block 4
    x =  Conv2D(
        512, (3, 3), padding="same", name="block4_conv1_before_activation"
    )(x)


    x=  Activation("relu",name="block4_conv1")(x)


    x =  Conv2D(
        512, (3, 3), padding="same", name="block4_conv2_before_activation"
    )(x)


    x=  Activation("relu",name="block4_conv2")(x)


    x =  Conv2D(
        512, (3, 3), padding="same", name="block4_conv3_before_activation"
    )(x)


    x=  Activation("relu",name="block4_conv3")(x)


    x =  Conv2D(
        512, (3, 3), padding="same", name="block4_conv4_before_activation"
    )(x)


    x=  Activation("relu",name="block4_conv4")(x)


    x =  MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)


    # Block 5
    x =  Conv2D(
        512, (3, 3), padding="same", name="block5_conv1_before_activation"
    )(x)


    x=  Activation("relu",name="block5_conv1")(x)


    x =  Conv2D(
        512, (3, 3), padding="same", name="block5_conv2_before_activation"
    )(x)


    x=  Activation("relu",name="block5_conv2")(x)


    x =  Conv2D(
        512, (3, 3), padding="same", name="block5_conv3_before_activation"
    )(x)


    x=  Activation("relu",name="block5_conv3")(x)


    x =  Conv2D(
        512, (3, 3), padding="same", name="block5_conv4_before_activation"
    )(x)


    x=  Activation("relu",name="block5_conv4")(x)


    x =  MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)


    if include_top:
        # Classification block
        x =  Flatten(name="flatten")(x)
        x =  Dense(4096, activation="relu", name="fc1")(x)
        x =  Dense(4096, activation="relu", name="fc2")(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x =  Dense(
            classes, activation=classifier_activation, name="predictions"
        )(x)
    else:
        if pooling == "avg":
            x =  GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x =  GlobalMaxPooling2D()(x)


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name="vgg19")


    # Load weights.
    if weights == "imagenet":
        if include_top:
            weights_path = data_utils.get_file(
                "vgg19_weights_tf_dim_ordering_tf_kernels.h5",
                WEIGHTS_PATH,
                cache_subdir="models",
                file_hash="cbe5617147190e668d6c5d5026f83318",
            )
        else:
            weights_path = data_utils.get_file(
                "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",
                WEIGHTS_PATH_NO_TOP,
                cache_subdir="models",
                file_hash="253f8cb515780f3b799900260a226db6",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)


    return model


def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="caffe"
    )
