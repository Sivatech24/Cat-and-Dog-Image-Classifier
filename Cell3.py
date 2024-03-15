from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set variables
train_dir = 'cats_and_dogs/train'
validation_dir = 'cats_and_dogs/validation'
test_dir = 'cats_and_dogs/test'
IMG_HEIGHT = 150
IMG_WIDTH = 150
batch_size = 32

# Create image generators
train_data_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='binary'
)

validation_data_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='binary'
)

test_data_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=50,  # There are 50 test images
    class_mode=None,
    shuffle=False
)

# Output
print("Found", train_data_gen.n, "images belonging to", train_data_gen.num_classes, "classes.")
print("Found", validation_data_gen.n, "images belonging to", validation_data_gen.num_classes, "classes.")
print("Found", test_data_gen.n, "images belonging to", len(test_data_gen.classes), "class.")
