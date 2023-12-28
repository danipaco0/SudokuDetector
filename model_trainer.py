import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# Define constants
batch_size = 32
img_height = 28
img_width = 14
num_classes = 9

# Prepare ImageDataGenerator for data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2  # Splitting 20% for validation
)

# Prepare train and validation generators
train_generator = train_datagen.flow_from_directory(
    'data/training_data',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    subset='training',
    class_mode='categorical'  # Use 'categorical' for multi-class classification
)

validation_generator = train_datagen.flow_from_directory(
    'data/testing_data',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    subset='validation',
    class_mode='categorical'
)

# Define the CNN model architecture
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=(img_height, img_width, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_uniform'),
    MaxPooling2D((2, 2)),
    Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_uniform'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu', kernel_initializer='he_uniform'),
    Dropout(0.2),
    Dense(9, activation='softmax')
])


# Compile the model
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=20
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation accuracy: {accuracy*100:.2f}%')

# Save the model
model.save('ocr_model.h5')
