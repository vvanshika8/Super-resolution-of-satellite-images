import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
# Data path
data_dir = os.path.expanduser(r"C:\Users\Vanshika\Downloads\UCMerced_LandUse\Images")

# Function to load all images from the dataset
def load_ucmerced_dataset(data_dir):
    data = []
    labels = []
    
    # Iterate over the subdirectories (classes)
    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        
        # Check if it's a directory (subfolder)
        if os.path.isdir(class_path):
            # Load all images from this class folder
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    # Open and append the image
                    img = Image.open(img_path)
                    data.append(img)
                    labels.append(class_folder)  # Use the folder name as the label
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    
    return data, labels

#dataset
images, labels = load_ucmerced_dataset(data_dir)

print(f"Loaded {len(images)} images from the dataset.")

def downscale_image(image, scale=0.75, method=Image.BICUBIC, iterations=3):
    '''Perform bicubic interpolation for the specified number of iterations.'''
    for _ in range(iterations):
        width, height = image.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), method)
    return image
    
    '''Methods for Resizing:
Image.NEAREST: Nearest-neighbor interpolation (blocky).
Image.BILINEAR: Bilinear interpolation (smooth, some blur).
Image.BICUBIC: Bicubic interpolation (smooth, higher quality).
Image.LANCZOS: Lanczos interpolation (best for downscaling, retains details).'''

def load_and_downscale_dataset(data_dir, scale=0.5):
    data = []
    labels = []
    
    # Iterate over the subdirectories (classes)
    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    # Load and downscale image
                    img = Image.open(img_path)
                    downscaled_img = downscale_image(img, scale=scale)
                    data.append(downscaled_img)
                    labels.append(class_folder)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
    
    return data, labels

# Load and downscale the dataset
downscaled_images, labels = load_and_downscale_dataset(data_dir)

print(f"Loaded and downscaled {len(downscaled_images)} images from the dataset.")

'''
# Function to visualize original and downscaled images side by side
def visualize_original_and_downscaled(original_images, downscaled_images, num_images=5):
    # Limit the number of images to display
    num_images = min(num_images, len(original_images), len(downscaled_images))
    
    # Set the plot size (height is doubled for side-by-side images)
    plt.figure(figsize=(15, num_images * 2))

    for i in range(num_images):
        # Show original image
        plt.subplot(num_images, 2, 2 * i + 1)  # Two columns, so 2 * i + 1
        plt.imshow(original_images[i])
        plt.title(f"Original Image {i + 1}")
        plt.axis('off')  # Hide the axes
        
        # Show downscaled image
        plt.subplot(num_images, 2, 2 * i + 2)  # Two columns, so 2 * i + 2
        plt.imshow(downscaled_images[i])
        plt.title(f"Downscaled Image {i + 1}")
        plt.axis('off')  # Hide the axes
    plt.tight_layout()
    plt.show()

# Visualize the first 5 original and downscaled images
visualize_original_and_downscaled(images, downscaled_images, num_images=15) '''

def prepare_and_save_paired_dataset(original_images, downscaled_images, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    low_res, high_res = [], []
    
    for orig, down in tqdm(zip(original_images, downscaled_images), total=len(original_images)):
        # Convert PIL images to NumPy arrays
        orig_np = np.array(orig)
        down_np = np.array(down)

        # Check for grayscale images and convert to RGB
        if orig_np.ndim == 2:  # Grayscale original
            orig_np = np.stack([orig_np] * 3, axis=-1)
        if down_np.ndim == 2:  # Grayscale downscaled
            down_np = np.stack([down_np] * 3, axis=-1)
        
        '''# Ensure all images have consistent dimensions
        if orig_np.shape != (256, 256, 3):  # Resize original
            orig_np = np.array(orig.resize((256, 256)))
        if down_np.shape != (128, 128, 3):  # Resize downscaled
            down_np = np.array(down.resize((128, 128)))'''
        # Resize original and downscaled images before storing
        orig_resized = orig.resize((256, 256))  # Ensure original is 256x256
        down_resized = down.resize((128, 128))  # Ensure downscaled is 128x128

# Convert images to NumPy arrays
        orig_np = np.array(orig_resized)
        down_np = np.array(down_resized)

        # Debug shapes
        '''print(f"Original shape: {orig_np.shape}, Downscaled shape: {down_np.shape}")'''
        
        # Store pairs
        low_res.append(down_np)
        high_res.append(orig_np)
    
    # Save as .npy files
    np.save(os.path.join(save_dir, "low_res.npy"), np.array(low_res))
    np.save(os.path.join(save_dir, "high_res.npy"), np.array(high_res))
    print(f"Saved paired dataset to {save_dir}")
# Prepare and save the paired dataset
save_dir = os.path.expanduser(r"C:\Users\Vanshika\Downloads\UCMerced_LandUse\Paired")
prepare_and_save_paired_dataset(images, downscaled_images, save_dir)


# Load paired dataset
low_res = np.load(os.path.join(save_dir, "low_res.npy"))
high_res = np.load(os.path.join(save_dir, "high_res.npy"))

# Normalize the images to the range [0, 1]
low_res = low_res.astype('float32') / 255.0
high_res = high_res.astype('float32') / 255.0

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(low_res, high_res, test_size=0.2, random_state=42)

from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Activation, Add
from tensorflow.keras.models import Model

def build_vgg19_super_res_model(input_shape=(128, 128, 3)):
    """
    Build a super-resolution model outputting (256x256x3).
    
    Args:
        input_shape: Shape of the low-resolution input images.
        
    Returns:
        Compiled Keras model.
    """
    # Input for low-resolution images
    low_res_input = Input(shape=input_shape)

    # Upsampling layers
    x = UpSampling2D(size=(2, 2))(low_res_input)  # 128x128 -> 256x256
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    # Final high-resolution prediction
    high_res_output = Conv2D(3, (3, 3), padding='same', activation='relu')(x)  # 256x256x3

    # Model definition
    model = Model(low_res_input, high_res_output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])
    
    return model


#model
input_shape = (128, 128, 3)
super_res_model = build_vgg19_super_res_model(input_shape)
super_res_model.summary()

# Train
history = super_res_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
)
# Save the trained model
model_path = os.path.join(save_dir, "super_res_model_vgg19.h5")
super_res_model.save(model_path)
print(f"Model saved at {model_path}")