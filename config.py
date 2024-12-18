import os
import glob

######################## PATHS ###############################
# data dir
DATA_DIR = os.path.join(os.getcwd(), 'data')
# input image dir
INPUT_IMAGES_DIR = os.path.join(DATA_DIR, 'input_images')
# save row-wise images
ROW_WISE_IMAGES_DIR = os.path.join(DATA_DIR, 'row_wise_images')
# save section-wise images
SECTION_WISE_IMAGES_DIR = os.path.join(DATA_DIR, 'section_wise_images')
# save product-wise images
PRODUCT_WISE_IMAGES_DIR = os.path.join(DATA_DIR, 'product_wise_images')
# final output images
FINAL_OUTPUT_DIR = os.path.join(DATA_DIR, 'final_output_images')

# Model weights dir
MODEL_WEIGHTS_DIR = os.path.join(DATA_DIR, 'checkpoints')
# product_detection_weight model configuration
ROW_WISE_MODEL_WEIGHTS = os.path.join(MODEL_WEIGHTS_DIR, 'row_wise_weight/weights/last.pt')
# row_wise_weight model configuration
SECTION_WISE_MODEL_WEIGHTS = os.path.join(MODEL_WEIGHTS_DIR, 'section_wise_weight/weights/last.pt')
# product_detection_weight model configuration
PRODUCT_SEGMENTATION_MODEL_WEIGHTS = os.path.join(MODEL_WEIGHTS_DIR, 'product_wise_weight/weights/last.pt')




# Model parameters
INTERSECTION_OVER_UNION = 0.65
CONFIDENCE_THRESHOLD_FOR_DETECTION = 0.3
CONFIDENCE_THRESHOLD = 0.5
OUTPUT_IMAGE_SIZE = 720

# Download Model weights
# Example usage
ROW_MODEL_URL = 'https://drive.google.com/file/d/18v-cfpOfN2MBPqeJSUUW1OKCCCh2XoxN/view?usp=sharing'
SECTION_MODEL_URL = 'https://drive.google.com/uc?id=16BnnDI-wrfRZQwHK-1kOUtZ7GSo3PGCz'
PRODUCT_SEG_MODEL_URL = 'https://drive.google.COM/uc?id=1-1Q6J9Z'
PRODUCT_MODEL_URL = 'https://drive.google.com/uc?id=16BnnDI-wrfRZQwHK-1kOUtZ7GSo3PGCz' 
DOWNLOAD_ROW_MODEL_PATH = 'data/row.zip'  # Local path for the downloaded zip file
DOWNLOAD_SECTION_MODEL_PATH = 'data/section.zip'  # Local path for the downloaded zip file
DOWNLOAD_PRODUCT_SEG_MODEL_PATH = 'data/product_seg.zip'  # Local path for the downloaded zip file
DOWNLOAD_PRODUCT_MODEL_PATH = 'data/product.zip'  # Local path for the downloaded zip file
EXTRACT_ROW_MODEL_DIR = 'data/checkpoints/row_wise_weight/'  # Directory where the zip file should be extracted 
EXTRACT_SECTION_MODEL_DIR = 'data/checkpoints/section_wise_weight/'  # Directory where the zip file should be extracted
EXTRACT_PRODUCT_SEG_MODEL_DIR = 'data/checkpoints/product_wise_weight/'  # Directory where the zip file should be extracted
EXTRACT_PRODUCT_MODEL_DIR = 'data/checkpoints/product_detection_weight/'  # Directory where the zip file should be extracted
