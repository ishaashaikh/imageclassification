import os
from PIL import Image
from sklearn.model_selection import train_test_split

current_directory = os.path.dirname(os.path.abspath(__file__))
IMAGE_SIZE = (224, 224)  # As required for vgg16 model
Dataset_Dir = os.path.join(current_directory, 'dataset')
Train_Dir = os.path.join(current_directory, 'train_data')
Test_Dir = os.path.join(current_directory, 'test_data')

def preprocess_and_copy_images(filenames, source_dir, dest_dir):
    for filename in filenames:
        class_name = filename.split('.')[0]
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, class_name, filename)

        img = Image.open(source_path).resize(IMAGE_SIZE)
        img.save(dest_path)

def preprocess_and_split_data(dataset_dir, train_dir, test_dir):
    # Check if train and test directories already exist
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        # Create train and test directories
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Get the list of filenames in the dataset directory
        filenames = os.listdir(dataset_dir)

        # Split the dataset into train and test sets
        train_filenames, test_filenames = train_test_split(filenames, test_size=0.3, random_state=42)

        # Create subfolders for dog and cat in train and test directories
        for subfolder in ['dog', 'cat']:
            os.makedirs(os.path.join(train_dir, subfolder), exist_ok=True)
            os.makedirs(os.path.join(test_dir, subfolder), exist_ok=True)

        # Preprocess and copy images to train directory with subfolders
        preprocess_and_copy_images(train_filenames, dataset_dir, train_dir)

        # Preprocess and copy images to test directory with subfolders
        preprocess_and_copy_images(test_filenames, dataset_dir, test_dir)
    else:
        print(f"Train and test directories already exist in {train_dir} and {test_dir}. Skipping creation.")

preprocess_and_split_data(Dataset_Dir, Train_Dir, Test_Dir)
