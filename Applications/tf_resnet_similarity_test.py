import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from scipy.spatial.distance import cosine

def load_and_preprocess_data(img_path,target_size = (224,224)):
    img = image.load_img(img_path,target_size = target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array,axis = 0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(model,img_array):
    features = model.predict(img_array)
    return features

def compute_cosine_similarity(features_1,features_2):
    return 1-cosine(features_1,features_2)

def plot_images(img_paths,titles):
    plt.figure(figsize=(10,5))
    for i,(img_path,title) in enumerate(zip(img_paths,titles)):
        img = image.load_img(img_path,target_size = (224,224))
        plt.subplot(1,len(img_paths),i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.show()

## Example usage

base_model = tf.keras.applications.ResNet50(weights = 'imagenet',include_top = False,pooling = 'avg')
img_path_1 = 'path_to_image_1.jpg'
img_path_2 = 'path_to_image_2.jpg'

img_1 = load_and_preprocess_data(img_path_1)
img_2 = load_and_preprocess_data(img_path_2)

features_1 = extract_features(base_model,img_1)
features_2 = extract_features(base_model,img_2)

similarty = compute_cosine_similarity(features_1,features_2)