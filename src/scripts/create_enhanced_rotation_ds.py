import os
import shutil
import random
import cv2
import albumentations as A


def read_label_file(label_file_path):
    with open(label_file_path, 'r') as f:
        lines = f.readlines()

    bboxes = []
    class_labels = []
    for line in lines:
        class_id, center_x, center_y, width, height = map(float, line.strip().split())
        bboxes.append([center_x, center_y, width, height])
        class_labels.append(int(class_id))

    return bboxes, class_labels

def write_label_file(label_file_path, bboxes, class_labels):
    with open(label_file_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            f.write(f"{class_id} {' '.join(map(str, bbox))}\n")

def process_images(images_folder, labels_folder, rotate, percent_to_rotate):
    images_files = os.listdir(images_folder)
    for img_file in images_files:
        if random.random() < percent_to_rotate:
            # Read image
            img = cv2.imread(os.path.join(images_folder, img_file))
            # Read corresponding label file
            label_file_path = os.path.join(labels_folder, img_file.replace('.JPG', '.txt'))
            bboxes, class_labels = read_label_file(label_file_path)

            # Apply augmentation
            augmented = rotate(image=img, bboxes=bboxes, class_labels=class_labels)

            # Save the image
            augmented_img = augmented['image']
            cv2.imwrite(os.path.join(images_folder, 'rotated_' + img_file), augmented_img)

            # Save the bboxes
            augmented_bboxes = augmented['bboxes']
            write_label_file(os.path.join(labels_folder, 'rotated_' + img_file.replace('.JPG', '.txt')), augmented_bboxes, class_labels)

def main(root_folder, percent_to_rotate):
    rotate = A.Compose([
        A.Rotate(limit=45, p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    for split in ['train', 'val']:
        src_images_folder = os.path.join(root_folder, 'images', split)
        src_labels_folder = os.path.join(root_folder, 'labels', split)
        
        dest_images_folder = os.path.join(root_folder, 'images', f'{split}_enhanced')
        dest_labels_folder = os.path.join(root_folder, 'labels', f'{split}_enhanced')

        # Create directories if they do not exist
        os.makedirs(dest_images_folder, exist_ok=True)
        os.makedirs(dest_labels_folder, exist_ok=True)

        # Copy images and labels from source to destination
        for filename in os.listdir(src_images_folder):
            shutil.copy2(os.path.join(src_images_folder, filename), dest_images_folder)
            label_filename = filename.rsplit('.', 1)[0] + '.txt'
            shutil.copy2(os.path.join(src_labels_folder, label_filename), dest_labels_folder)

        process_images(dest_images_folder, dest_labels_folder, rotate, percent_to_rotate)


if __name__ == "__main__":
    main('./datasets/ruler_digit_YOLOv5_PyTorch_TXT', 0.2)
    print('End')
