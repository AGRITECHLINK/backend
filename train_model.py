import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
import os
import json
import shutil
from sklearn.model_selection import train_test_split

# Set up paths
base_dir = 'C:/Users/JACOB MWALE/Documents/New Plant Diseases Dataset(Augmented)/train'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# Create train and val directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get all subdirectories (classes) in the new dataset
classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d not in ['train', 'val']]
print("Classes found:", classes)

# Function to check if a file is an image based on its magic number
def is_image(file_path):
    try:
        with open(file_path, 'rb') as f:
            header = f.read(10)
            return header.startswith(b'\xff\xd8') or header.startswith(b'\x89PNG')
    except:
        return False

# Split data for each class
for cls in classes:
    cls_dir = os.path.join(base_dir, cls)
    images = [f for f in os.listdir(cls_dir) if is_image(os.path.join(cls_dir, f))]
    print(f"Found {len(images)} images in class {cls}")

    if len(images) == 0:
        print(f"No images found in class {cls}")
        continue

    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    # Create class directories in train and val
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    # Move images to train and val directories
    for img in train_images:
        src = os.path.join(cls_dir, img)
        dst = os.path.join(train_dir, cls, img)
        shutil.copy(src, dst)

    for img in val_images:
        src = os.path.join(cls_dir, img)
        dst = os.path.join(val_dir, cls, img)
        shutil.copy(src, dst)

print("Data split complete.")

# Set up parameters
img_size = 224
batch_size = 32
epochs = 10

# Prepare data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical')

model = load_model('plant_disease_model.h5')

# Freeze all layers except the last few
for layer in model.layers[:-4]:
    layer.trainable = False
for layer in model.layers[-4:]:
    layer.trainable = True

# Get the number of classes in the new dataset
num_classes_new = len(train_generator.class_indices)

# Function to get the number of classes from the model
def get_num_classes(model):
    # If the model has multiple outputs, we assume the last one is for classification
    if isinstance(model.output, list):
        output = model.output[-1]
    else:
        output = model.output
    
    # Get the shape of the output
    output_shape = output.shape
    
    # If the shape is not fully defined (contains None), we need to look at the last layer
    if output_shape[-1] is None:
        last_layer = model.layers[-1]
        if isinstance(last_layer, Dense):
            return last_layer.units
        else:
            raise ValueError("Unable to determine the number of classes from the model")
    else:
        return output_shape[-1]

# Get the number of classes in the old model
try:
    num_classes_old = get_num_classes(model)
except ValueError as e:
    print(f"Error: {str(e)}")
    print("Please check your model structure and modify the script accordingly.")
    exit(1)

print(f"Number of classes in the old model: {num_classes_old}")
print(f"Number of classes in the new dataset: {num_classes_new}")

# If the new dataset has a different number of classes, replace the last layer
if num_classes_new != num_classes_old:
    print("Adjusting the model for the new number of classes...")
    # Remove the last layer
    x = model.layers[-2].output
    # Add a new classification layer with a unique name
    predictions = Dense(num_classes_new, activation='softmax', name='new_dense_layer')(x)
    model = Model(inputs=model.inputs, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size)

# Save the fine-tuned model
model.save('plant_disease_model_finetuned.keras')

# Save the new class indices
with open('class_indices_finetuned.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
