import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# Training constants
batch_size = 32
img_height = 20
img_width = 20
num_classes = 9

# Load previously trained model
model = load_model('ocr_model.h5')

# Prepare images for data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2 
)

# Prepare train and validation generators
train_generator = train_datagen.flow_from_directory(
    'data2/training_data',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    subset='training',
    class_mode='categorical'
)

validation_generator = train_datagen.flow_from_directory(
    'data/testing_data',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    subset='validation',
    class_mode='categorical'
)

with tf.device('/device:GPU:0'): # Using GPU for training calculations 


    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Continue training model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=12 # Iterations
    )

# Model evaluation
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation accuracy: {accuracy*100:.2f}%')

model.save('ocr_model_retrained.h5')
