import cv2
import os
import random
import gdown
import config
from data.SKUs import productSku,productSkuList,competitorsSkuList
import logging
import zipfile
logger = logging.getLogger(__name__)

class SaveOutputImage:
    """
    A utility class to draw bounding boxes, labels, and confidence scores on images 
    for visualization of object detection results.
    """

    def __init__(self):
        """
        Initializes the SaveOutputImage instance and assigns unique colors to PepsiCo SKU labels.
        """
        self.colors = self.GenerateColors(productSkuList)

    def GenerateColors(self, labels):
        """
        Generates a unique random color for each label.

        Parameters:
        - labels (list[str]): A list of label names.

        Returns:
        - dict: A dictionary mapping each label to a unique color (RGB tuple).
        """
        colors = {}
        for label in labels:
            color = tuple(random.choices(range(256), k=3))  # Generate random RGB color
            colors[label] = color
        return colors

    def DrawRectanglesWithLabels(self, ImagePath, boxes, confidences, labels, OutputPath):
        """
        Draws bounding boxes, labels, and confidence scores on the image and saves it.

        Parameters:
        - ImagePath (str): The path to the input image.
        - boxes (list[list[int]]): List of bounding boxes, each as [x1, y1, x2, y2].
        - confidences (list[float]): Confidence scores for each detection.
        - labels (list[str]): Labels for each detection.
        - OutputPath (str): The path to save the annotated image.
        """
        image = cv2.imread(ImagePath)

        for box, confidence, label in zip(boxes, confidences, labels):
            x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
            
            # Determine color based on label category
            if label in productSkuList:
                color = (0, 255, 0)  # GREEN
            elif label in competitorsSkuList:
                color = (0, 0, 255)  # RED
            else:
                color = (255, 0, 0)  # BLUE
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Create label text with confidence score
            LabelText = f'{label} {confidence:.2f}'

            # Calculate text size and position
            (w, h), _ = cv2.getTextSize(LabelText, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(image, LabelText, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imwrite(OutputPath, image)

    def DrawRectangles(self, ImagePath, boxes, confidences, labels, OutputPath):
        """
        Draws bounding boxes (without labels) on the image and saves it.

        Parameters:
        - ImagePath (str): The path to the input image.
        - boxes (list[list[int]]): List of bounding boxes, each as [x1, y1, x2, y2].
        - confidences (list[float]): Confidence scores for each detection.
        - labels (list[str]): Labels for each detection.
        - OutputPath (str): The path to save the annotated image.
        """
        image = cv2.imread(ImagePath)

        for box, confidence, label in zip(boxes, confidences, labels):
            x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
            
            # Determine color based on label category
            if label in productSkuList:
                color = (0, 255, 0)  # GREEN
            elif label in competitorsSkuList:
                color = (0, 0, 255)  # RED
            else:
                color = (255, 0, 0)  # BLUE
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Modify output path to prevent overwriting
        OutputPathImage = OutputPath.split(".")[0]
        ImageFormat = OutputPath.split(".")[-1]
        OutputPath = f"{OutputPathImage}_1.{ImageFormat}"
        cv2.imwrite(OutputPath, image)

    def SaveOutput(self, BoundingBoxes, confidences, SkuIndexes, SkuIndexesDict, SkuNames, InputImagePath, SaveOutputImagePath):
        """
        Saves the output image with bounding boxes and labels drawn on it.

        Parameters:
        - BoundingBoxes (torch.Tensor): Tensor of bounding boxes in [x1, y1, x2, y2] format.
        - confidences (torch.Tensor): Tensor of confidence scores for each detection.
        - SkuIndexes (list[int]): List of SKU indexes corresponding to detections.
        - SkuIndexesDict (dict): Dictionary mapping SKU indexes to instance counts.
        - SkuNames (dict): Dictionary mapping SKU indexes to label names.
        - InputImagePath (str): Path to the input image file.
        - SaveOutputImagePath (str): Path to save the annotated image.
        """
        boxes = BoundingBoxes.tolist()
        confidences = confidences.tolist()
        labels = [SkuNames[int(sku)] for sku in SkuIndexes]

        # Draw and save annotated images
        self.DrawRectangles(InputImagePath, boxes, confidences, labels, SaveOutputImagePath)
        self.DrawRectanglesWithLabels(InputImagePath, boxes, confidences, labels, SaveOutputImagePath)


def download_and_unzip(model_url, output_path, extract_to):
    logger.info(f"Downloading Model weights in zip format")
    gdown.download(model_url, output_path, quiet=False, fuzzy=True)
    
    if output_path.endswith('.zip'):
        logger.info(f"Unzipping Model weights .....")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Unzipped model file successfully! .....")
        os.remove(output_path)
    else:
        print('Downloaded file is not a zip file.')
        logger.error(f"Downloaded model file is not a zip file.")


modelsConfig = [
    {
        "url": config.ROW_MODEL_URL,
        "download_path": config.DOWNLOAD_ROW_MODEL_PATH,
        "extract_path": config.EXTRACT_ROW_MODEL_DIR,
    },
    {
        "url": config.SECTION_MODEL_URL,
        "download_path": config.DOWNLOAD_SECTION_MODEL_PATH,
        "extract_path": config.EXTRACT_SECTION_MODEL_DIR,
    },
    {
        "url": config.PRODUCT_MODEL_URL,
        "download_path": config.DOWNLOAD_PRODUCT_MODEL_PATH,
        "extract_path": config.EXTRACT_PRODUCT_MODEL_DIR,
    },
]
