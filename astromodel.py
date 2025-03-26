from roboflow import Roboflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np

# Constelations entire dataset cause it's easy to run on collab
rf = Roboflow(api_key="lm6TWAMGtgSvLc2loZBp")
project = rf.workspace("yuri-lima-lcztr").project("constalations-classification")
dataset = project.version(1).download("tensorflow")


img_size = 224  # ResNet wants 224x224 image size
batch_size = 32 # 32 images will be used in one "iteration" or "step" of training. Changed to start from 64 when using collab

#Splits 80% for training and 20% for validation (testing).
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    dataset.location,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    dataset.location,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

#Loads ResNet50, which already knows how to recognize things.
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))


'''
ResNet50 is a pretrained model, meaning it has already been trained on millions of images from a very large dataset (ImageNet). 
These images contain a wide variety of objects, like animals, plants, and other everyday objects. 
During that training, the model learned to recognize patterns in images, such as edges, textures, shapes, and colors. 
These basic features are useful for recognizing constellations
'''

# Freeze the pre-trained layers (we only train our custom head first). We freeze the layers to keep the model's 'abilty' to detect edges, textures, shapes, and colors
for layer in base_model.layers:
    layer.trainable = False

# Add our own custom layers
x = base_model.output # We are taking whatever the ResNet model "thinks" is important from the image and passing it on to the next layers. The base_model.output is the feature map (processed representation) of the input image that the model learnt
x = GlobalAveragePooling2D()(x) # This layer takes the feature map from the model (which is a 3D array of numbers) and reduces it into a 1D array. GAP works by averaging all the values across the width and height of the feature map
x = Dense(128, activation='relu')(x) # After GAP, we add a fully connected (dense) layer with 128 neurons. meaning model now has 128 "neurons" that will help it further process and make decisions about the image. With collab (higher gpu) we can increase the neurons. 
# Although incresing neurons increases the risk of overfiiting using techinques we can curb this i.e Dropout(0.5)(x)(50% of the neurons in that layer will be randomly ignored), kernel_regularizer=l2(0.01) The l2(0.01) means that the regularization strength is 0.01. This number controls how much the model is penalized for large weights
predictions = Dense(train_data.num_classes, activation='softmax')(x) # This is the final output layer of the model. It maps the result of the previous dense layer to the number of constellations that the model is supposed to predict.
# The softmax activation function is used because it turns the raw output values into probabilities that sum to 1. Telling us which one of the classes it thinks is most likely
model = Model(inputs=base_model.input, outputs=predictions)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=val_data, epochs=5)

# Save the trained model
model.save("constellation_resnet_model.h5")

# Predict a new image
img_path = "C:/Users/DBABA/OneDrive/Documents/Data Science Projects/AstroIdentify/denoise_testing/Aquarius_input.png"

img = image.load_img(img_path, target_size=(img_size, img_size))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
class_labels = list(train_data.class_indices.keys())
print("Predicted constellation:", class_labels[np.argmax(prediction)])
