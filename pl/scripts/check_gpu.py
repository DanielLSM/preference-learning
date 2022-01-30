# Sample python script:

# Get the Tensorflow version:
import tensorflow as tf

print("\nTensorflow version: ", tf.__version__, "\nTensorflow file: ",
      tf.__file__)

# Check that you can see the GPUs:
print('Num GPUs Available: ',
      len(tf.config.experimental.list_physical_devices('GPU')))
