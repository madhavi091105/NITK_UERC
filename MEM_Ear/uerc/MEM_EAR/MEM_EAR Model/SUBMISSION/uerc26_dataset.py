import os
import csv
import pathlib
import random
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


# Base dataset class for UERC, providing common functionality for data handling
class UERCBaseDataset(Dataset):
    def __init__(self, split, root_dir, full_image_list_csv, data_split_csv, val_ratio=0.2, test_ratio=0.1):
        self.split = split
        self.root_dir = root_dir
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        if full_image_list_csv is None:
            raise ValueError("full_image_list_csv must be provided.")
        self.full_image_list_csv = os.path.join(self.root_dir, full_image_list_csv)

        if data_split_csv is None:
            raise ValueError("data_split_csv must be provided.")
        self.data_split_csv = os.path.join(self.root_dir, data_split_csv)

        # Check if the full image list CSV file exists, if not create it
        if not os.path.exists(self.full_image_list_csv):
            print(f"Full image list CSV file '{self.full_image_list_csv}' not found. Creating it...")
            self.__create_image_list()

        # Check if the data splits CSV file exists, if not create it
        if not os.path.exists(self.data_split_csv):
            print(f"Data splits CSV file '{self.data_split_csv}' not found. It will be created after generating data splits.")
            self.__create_data_split()

        self.images = self.__get_split_images()

        subjects = set(os.path.dirname(img) for img in self.images)
        classes = sorted(subjects)

        self.classes_mapping = {c: i for i, c in enumerate(classes)}
        self.num_of_classes = len(self.classes_mapping)
        self.labels = {img: self.classes_mapping[os.path.dirname(img)] for img in self.images}

        self.transforms = {}
        self.transforms['train'] = transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomRotation(30),
                                transforms.CenterCrop(224),
                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transforms['test'] = transforms.Compose([
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transforms['val'] = self.transforms['test']

    def __create_image_list(self):
        # Creates a CSV file containing the list of all images in the dataset
        img_list = sorted(pathlib.Path(self.root_dir).glob("*/*.*"))

        with open(self.full_image_list_csv, "w", newline="") as csv_file:
            cw = csv.writer(csv_file, delimiter="\t")
            for image_path in img_list:
                if image_path.suffix.lower() in ['.png', '.jpg']:
                    img_pth = str(image_path)
                    subject = os.path.basename(os.path.dirname(img_pth))
                    filename = image_path.name
                    full = os.path.join(subject, filename)

                    cw.writerow([full])

    def __get_full_image_list(self):
        # Parses the CSV file to create a dictionary mapping subjects to their images
        full_image_list = {}
        with open(self.full_image_list_csv, "r") as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                img_pth = row[0]
                subject = os.path.dirname(img_pth)
                filename = os.path.basename(img_pth)

                if subject not in full_image_list:
                    full_image_list[subject] = []
                full_image_list[subject].append(filename)

        return full_image_list

    def __create_data_split(self):
        # Randomly splits the dataset into train, validation, and test sets based on the specified ratios
        full_image_list = self.__get_full_image_list()
        subjects = list(full_image_list.keys())

        # Calculate number of subjects to select for test
        num_items_to_select = max(1, round(len(subjects) * self.test_ratio))

        # Randomly select subjects for test set
        random.shuffle(subjects)
        subjects_test = subjects[:num_items_to_select]
        subjects_train_val = subjects[num_items_to_select:]

        data = []
        # Randomly split between train and validation set
        for subject in subjects_train_val:
            images = full_image_list[subject]
            random.shuffle(images)

            num_items_to_select = max(1, round(len(images) * self.val_ratio))

            for img in images[:num_items_to_select]:
                data.append([os.path.join(subject, img), 'val'])

            for img in images[num_items_to_select:]:
                data.append([os.path.join(subject, img), 'train'])

        # Add test set
        for subject in subjects_test:
            for img in full_image_list[subject]:
                data.append([os.path.join(subject, img), 'test'])

        data = sorted(data)

        with open(self.data_split_csv, "w", newline="") as csv_file:
            cw = csv.writer(csv_file, delimiter="\t")
            for d in data:
                cw.writerow(d)

    def __get_split_images(self):
        # Reads the data splits CSV file and filters images based on the current split
        images = []
        with open(self.data_split_csv, "r") as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                if row[1] == self.split:
                    images.append(row[0])

        return images

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


# Dataset class for the identification task
class UERCDataset(UERCBaseDataset):
    def __init__(self, split, root_dir, full_image_list_csv, data_split_csv, preload_images=False):
        super(UERCDataset, self).__init__(split,root_dir, full_image_list_csv, data_split_csv)

        self.preload_images = preload_images

        if self.preload_images:
            self.preloaded_images = {}
            for img_name in self.images:
                img_path = os.path.join(self.root_dir, img_name)
                image = Image.open(img_path).convert('RGB')

                self.preloaded_images[img_name] = image

    def __len__(self):
        # Returns the total number of samples in the dataset
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieves a single image and its label by index

        # Loads the image and applies the appropriate transformations
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)

        if self.preload_images:
            image = self.preloaded_images[img_name]
        else:
            image = Image.open(img_path).convert('RGB')

        if self.transforms[self.split]:
            image = self.transforms[self.split](image)

        label = self.labels[img_name]

        return image, label


# Dataset class for verification task (pairwise image comparison)
class UERCPairwiseDataset(UERCBaseDataset):
    def __init__(self, split, root_dir, full_image_list_csv, data_split_csv, pairs_csv, num_pairs_per_subject=0, preload_images=False):
        super(UERCPairwiseDataset, self).__init__(split, root_dir, full_image_list_csv, data_split_csv)

        if pairs_csv is None:
            raise ValueError("pairs_csv must be provided.")
        self.pairs_csv = os.path.join(self.root_dir, pairs_csv)

        if not os.path.exists(self.pairs_csv):
            self.__create_pairwise_pairs(num_pairs_per_subject)

        self.preload_images = preload_images
        if self.preload_images:
            self.preloaded_images = {}
            for img_name in self.images:
                img_path = os.path.join(self.root_dir, img_name)
                image = Image.open(img_path).convert('RGB')

                self.preloaded_images[img_name] = image

        self.image_pairs, self.labels = self.__load_pairwise_pairs()

    def __create_pairwise_pairs(self, num_pairs_per_subject):
        # Creates positive and negative image pairs for pairwise comparison tasks.
        pairs = []
        labels = []

        for subject in self.classes_mapping.keys():
            images = [img for img in self.images if self.labels[img] == self.classes_mapping[subject]]

            if len(images) <= 0:
                continue

            # Get a random subset of positive pairs for the subject if num_pairs_per_subject is set
            all_positive_pairs = [(images[i], images[j]) for i in range(len(images)) for j in range(i + 1, len(images))]

            if num_pairs_per_subject > 0 and len(images) > 1:
                selected_positive_pairs = random.sample(all_positive_pairs, min(num_pairs_per_subject, len(all_positive_pairs)))
            else:
                selected_positive_pairs = all_positive_pairs

            # Get a random subset of negative pairs for the subject equivalent to num_pairs_per_subject if set, otherwise same as positive pairs
            if num_pairs_per_subject > 0:
                num_negative_pairs = num_pairs_per_subject
            else:
                num_negative_pairs = len(selected_positive_pairs)

            if num_negative_pairs <= 0:
                continue

            # Generate negative pairs
            selected_negative_pairs = []
            while len(selected_negative_pairs) < num_negative_pairs:
                other_subject = random.choice([s for s in self.classes_mapping.keys() if s != subject])
                other_images = [img for img in self.images if self.labels[img] == self.classes_mapping[other_subject]]

                if len(other_images) <= 0:
                    continue

                img1 = random.choice(images)
                img2 = random.choice(other_images)
                pair = (img1, img2)
                if pair not in selected_negative_pairs:
                    selected_negative_pairs.append(pair)

            pairs.extend(selected_positive_pairs)
            labels.extend([1] * len(selected_positive_pairs))

            pairs.extend(selected_negative_pairs)
            labels.extend([0] * len(selected_negative_pairs))

        # Sort pairs and labels together
        pairs_labels = sorted(zip(pairs, labels), key=lambda x: (x[0][0], x[0][1]))

        # Store the selected positive and negative pairs along to csv
        with open(self.pairs_csv, 'w') as f:
            for pair, label in pairs_labels:
                img1, img2 = pair
                f.write(f"{img1}\t{img2}\t{label}\n")

    def __load_pairwise_pairs(self):
        # Reads the pairs CSV file and loads the image pairs and their corresponding labels
        pairs = []
        labels = []
        with open(self.pairs_csv, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                img1, img2, label = row
                pairs.append((img1, img2))
                labels.append(int(label))

        return pairs, labels

    def __len__(self):
        # Returns the total number of image pairs in the dataset
        return len(self.image_pairs)

    def __getitem__(self, idx):
        # Retrieves a pair of images and their label by index

        # Loads and transforms a pair of images, returning them along with their label
        (img1_name, img2_name), label = self.image_pairs[idx], self.labels[idx]
        img1_path = os.path.join(self.root_dir, img1_name)
        img2_path = os.path.join(self.root_dir, img2_name)

        if self.preload_images:
            img1 = self.preloaded_images[img1_name]
            img2 = self.preloaded_images[img2_name]
        else:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')

        transform = self.transforms[self.split]
        if transform:
            img1 = transform(img1)
            img2 = transform(img2)

        return (img1, img2), label


if __name__ == "__main__":
    # Example usage of the dataset classes for training, validation, and testing
    uerc26_identification_train = UERCDataset('train', root_dir='data/public', full_image_list_csv='image_list.csv', data_split_csv='dataset_split.csv')
    print(f'Train samples: {len(uerc26_identification_train)}')

    uerc26_identification_val = UERCDataset('val', root_dir='data/public', full_image_list_csv='image_list.csv', data_split_csv='dataset_split.csv')
    print(f'Validation samples: {len(uerc26_identification_val)}')

    uerc26_identification_test = UERCDataset('test', root_dir='data/public', full_image_list_csv='image_list.csv', data_split_csv='dataset_split.csv')
    print(f'Test samples: {len(uerc26_identification_test)}')


    uerc26_pairwise_train = UERCPairwiseDataset('train', root_dir='data/public', full_image_list_csv='image_list.csv', data_split_csv='dataset_split.csv', pairs_csv='pairs_train.csv', num_pairs_per_subject=100)
    print(f'Pairwise train samples: {len(uerc26_pairwise_train)}')

    uerc26_pairwise_val = UERCPairwiseDataset('val', root_dir='data/public', full_image_list_csv='image_list.csv', data_split_csv='dataset_split.csv', pairs_csv='pairs_val.csv', num_pairs_per_subject=100)
    print(f'Pairwise validation samples: {len(uerc26_pairwise_val)}')

    uerc26_pairwise_test = UERCPairwiseDataset('test', root_dir='data/public', full_image_list_csv='image_list.csv', data_split_csv='dataset_split.csv', pairs_csv='pairs_test.csv', num_pairs_per_subject=100)
    print(f'Pairwise test samples: {len(uerc26_pairwise_test)}')
