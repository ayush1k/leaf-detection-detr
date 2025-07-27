Leaf Detection using Detection Transformers (DETR)
This project demonstrates how to fine-tune a pre-trained Detection Transformer (DETR) model for leaf detection using a custom dataset. The code is optimized for training in a Google Colab environment, leveraging its GPU resources (specifically T4 GPUs with Automatic Mixed Precision and DataParallel).

1. Project Overview
The goal of this project is to build an object detection model capable of identifying and localizing leaves within images. It utilizes the DETR architecture, which simplifies the object detection pipeline by directly predicting bounding boxes and class labels using a Transformer encoder-decoder.

2. Features
End-to-End Object Detection: Fine-tunes a DETR model for leaf detection.

Custom Dataset Integration: Includes a RealLeafDataset class to load annotations from a CSV file and images from a custom folder structure.

Google Colab / Google Drive Compatibility: Designed to run seamlessly in Google Colab, with dataset loading from Google Drive.

GPU Optimization: Leverages torch.nn.DataParallel for multi-GPU training (e.g., T4x2 in Colab) and Automatic Mixed Precision (AMP) for faster and more memory-efficient training.

Model Saving & Loading: Demonstrates how to save the fine-tuned model and image processor for later use, and how to load them for inference.

Visualization: Provides a separate script to visualize detected bounding boxes on test images.

3. Dataset Information
The code is configured to work with a dataset that has the following structure and annotation format:

Dataset Structure
Your dataset folder (e.g., leaf-detection) should be placed directly in your Google Drive (e.g., /content/drive/MyDrive/leaf-detection/). Inside this folder, the expected structure is:

/content/drive/MyDrive/leaf-detection/


├── train/
│   ├── LEAF_0001.jpg
│   ├── LEAF_0002.jpg
│   └── ... (All your training images are directly in the 'train' folder)
├── test/
│   └── leaf/
│       ├── TEST_001.jpg
│       ├── TEST_002.jpg
│       └── ... (All your test images are in the 'test/leaf' folder)
└── train.csv (Your annotation file for training images)



train.csv Format
The train.csv file should contain the bounding box annotations for your training images. The code specifically expects the following columns:

image_id: The filename of the image (e.g., LEAF_0009.jpg). This must exactly match the filenames in the train/ directory.

bbox: A string representation of a Python list containing the bounding box coordinates in [x_min, y_min, width, height] format (e.g., "[473, 273, 289, 335]").

Example train.csv snippet:

image_id,width,height,bbox
LEAF_0009.jpg,1024,1024,"[473, 273, 289, 335]"
LEAF_0009.jpg,1024,1024,"[588, 516, 272, 318]"
LEAF_0010.jpg,1024,1024,"[1, 59, 170, 366]"
...

Important Note: The code assumes all detected objects belong to a single class: 'leaf'. If your dataset has multiple classes, you'll need to adjust the dataset_categories variable and the class_name = 'leaf' assignment in RealLeafDataset.__getitem__.

4. Setup Instructions (Google Colab)
Open a new Google Colab Notebook.

Change Runtime Type: Go to Runtime -> Change runtime type -> Select GPU as the hardware accelerator.

Mount Google Drive: Run the first cell in the provided code (from google.colab import drive; drive.mount('/content/drive')) and follow the prompts to authorize Google Drive access.

Upload Dataset to Google Drive: Upload your leaf-detection dataset (with the structure described above) to your Google Drive. For example, place the leaf-detection folder directly inside MyDrive.

If you uploaded a .rar file: You must extract it in Colab first. You can use a command like !unrar x /content/drive/MyDrive/archive.rar /content/drive/MyDrive/ (you might need to install unrar: !apt-get install unrar). Ensure the extracted folder matches the expected leaf-detection name.

Install Libraries: Run the !pip install ... command at the beginning of the script.

5. How to Run the Training Code (detr-leaf-detection-code)
Copy the entire code from the detr-leaf-detection-code immersive into a Colab cell.

Adjust get_dataset_paths:

Locate the get_dataset_paths function.

Crucially, update the base_dataset_root variable to the exact path of your leaf-detection folder within your Google Drive. For example, if it's directly in MyDrive, keep it as "/content/drive/MyDrive/leaf-detection". If it's in a subfolder like MyProjects/leaf-data, change it to "/content/drive/MyDrive/MyProjects/leaf-data/leaf-detection".

Run the Cell: Execute the Colab cell containing the code.

The script will:

Print the configured dataset paths.

Load annotations from train.csv and images from train/.

Split the training data into training and validation sets.

Initialize and adapt the DETR model for leaf detection.

Start the training loop for 50 epochs, showing progress bars and loss values.

Save the trained model and image processor to /content/drive/MyDrive/fine_tuned_detr_leaf_detector.

6. How to Run Inference and Visualization (detr-leaf-detection-visualization)
This separate code block allows you to test your trained model on new images and visualize the bounding box detections.

Copy the entire code from the detr-leaf-detection-visualization immersive into a new Colab cell (or a new notebook).

Ensure Drive is Mounted: If running in a new session, re-mount Google Drive.

Adjust get_dataset_paths: Similar to the training script, ensure the base_dataset_root in the get_dataset_paths function within this visualization script points correctly to your leaf-detection dataset.

Adjust model_load_path:

Locate the model_load_path variable (around line 100 in the visualization script).

Set this to the exact path where your trained model was saved. This defaults to "/content/drive/MyDrive/fine_tuned_detr_leaf_detector".

Control Visualization: Modify num_test_images_to_visualize to control how many test images from your test/ folder will be processed and displayed.

Run the Cell: Execute the Colab cell.

The script will:

Load the saved model and image processor.

Iterate through the specified number of test images.

Perform inference on each image.

Display the image with detected bounding boxes (if any) and their confidence scores.

7. Important Notes & Troubleshooting
CSV Column Names: The RealLeafDataset expects image_id and bbox columns. If your CSV uses different names, you must update the ann['image_id'] and ann['bbox'] accessors in RealLeafDataset.__getitem__.

Bounding Box Format: The bbox column must contain a string that ast.literal_eval can parse into a list of four numbers: [x_min, y_min, width, height].

Invalid Bounding Boxes: The code includes a check (if width <= 0 or height <= 0: continue) to skip invalid bounding boxes (zero or negative dimensions) from your CSV. If you see warnings about "Invalid bounding box dimensions," it means your dataset contains such annotations.

Inference Threshold: The visualization script uses a default threshold=0.3. If you want to see only very confident detections, increase this value (e.g., to 0.7). If you're seeing no detections, ensure your model is sufficiently trained and try lowering this threshold further (e.g., 0.1) for debugging.

Epochs: 50 epochs is a good start, but for optimal performance on your specific dataset, you might need to train for significantly more epochs (e.g., 100, 200, or more). Monitor the training and validation loss to determine when the model converges.

GPU Memory: If you encounter "CUDA out of memory" errors, try reducing the batch_size_per_gpu (e.g., from 4 to 2 or 1).

FutureWarning for autocast: This is a minor warning from PyTorch about a deprecated syntax. The code still works, but you can update torch.cuda.amp.autocast() to torch.amp.autocast('cuda') if you wish.

Feel free to experiment with the parameters and monitor your model's performance!