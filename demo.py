from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "./Training"  # Change this path if needed

idg = ImageDataGenerator()

train = idg.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

print("Class indices:")
print(train.class_indices)