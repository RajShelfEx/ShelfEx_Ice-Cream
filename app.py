import os
import cv2
import time
import base64
import config
import logging
import requests
import platform
import subprocess
from flask import Flask, request, jsonify
from utils import download_and_unzip,modelsConfig
from mainPipeline import Result
from config import FINAL_OUTPUT_DIR

app = Flask(__name__)

# ******************************** CHECK MODEL DIRECTORY EXIST OR NOT *****************************
# modelWeightsDir = config.MODEL_WEIGHTS_DIR
# if not os.path.exists(modelWeightsDir):
#     # Loop through each model and download/unzip if necessary
#     for model in modelsConfig:
#         download_and_unzip(
#             model_url=model["url"],
#             output_path=model["download_path"],
#             extract_to=model["extract_path"],
#         )
# **************************************************************************************************

productRecognition = Result()

# Initialize logger
logger = logging.getLogger(__name__)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s')

def clearLoggerFile():
    # Define log file path
    logFilePath = 'app.log'
    # Check log file size
    if os.path.exists(logFilePath):
        fileSizeMb = os.path.getsize(logFilePath) / 1024  #  (1024 * 1024) --> this is for 1 MB
        if fileSizeMb > 500:  # If log file is greater than 500 KB
            # Empty the log file using redirection operator
            with open(logFilePath, 'w') as file:
                subprocess.run(['echo', '-n'], stdout=file)

            # Optional: Log that the log file was cleared
            logger.info("Log file exceeded 500 KB and was cleared.")

def checkDir(dirPath):
    ####################### IMAGE DIRECTORY ###########################

    # Check if InputImage directory exists, if not create it
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
        logging.info(f'Created directory: {dirPath}')
    else:
        logging.info(f'Directory already exists: {dirPath}')
        try:
            # Determine the OS and run the appropriate command
            if platform.system() == 'Windows':
                command = f'rmdir /S /Q "{dirPath}" && mkdir "{dirPath}"'
            else:
                command = f'rm -rf {dirPath}/*'
            
            subprocess.run(command, shell=True, check=True)
            
        except subprocess.CalledProcessError as e:
            logging.info(f"Failed to delete contents of {dirPath}. Reason: {e}")
        
def imageToBase64(outputImagePath):
    """
    Convert image to Base64 link

    Args:
        outputImagePath (str): Detected image path

    Returns:
        str : Image Base64 link
    """
    with open(outputImagePath, 'rb') as imageFile:
        encodedImage = base64.b64encode(imageFile.read()).decode('utf-8')
    return encodedImage

@app.route('/ShelfEx', methods=['POST'])
def shelfEx():
    if request.method == 'POST':
        # image url list from the JSON request
        imageUrlList = request.json.get('url')
        try:
            # Clear app.log file if greater than 500 KB
            clearLoggerFile()
            start = time.time()
            # check input image dir
            inputImageDir = config.INPUT_IMAGES_DIR
            checkDir(inputImageDir)
            
            # # check save image dir
            # saveImageDir = config.DETECTED_IMAGE_DIR
            # checkDir(saveImageDir)
            
            finalResult = []
            detection = {}
            # iterate each image url
            for imageUrl in imageUrlList:
                # Download image
                response = requests.get(imageUrl)
                if response.status_code == 200:
                    # extract image name
                    imageName = imageUrl.split('/')[-1]
                    imagePath = os.path.join(inputImageDir, f'{imageName}')
                    with open(imagePath, 'wb') as f:
                        f.write(response.content)
                    logging.info(f'Image downloaded and saved to {imagePath}')

                ############################ DETECTION PROCESS #################################
                detectionResult = productRecognition.main()
            
                ####################### SINGLE BOTTLE DETECTION PROCESS ###########################
                # finalResult = obj_singleBottle_detection_process.PredictAllBottlesBoundingBoxes(detectionResult)
                logging.info(f'Detection Results: {detectionResult}')
                ############################### FINAL RESULTS ################################
            
                # # convert output image-with-label into base64 
                # outputImagePathWithLabel = f"{finalOutputDir}/{imageName}"
                # encodedImage = imageToBase64(outputImagePathWithLabel)
                # detectionResult["imageWithLabel"] = encodedImage

                # # convert output image-without-label into base64
                imgName = imageName.split(".")[0]
                imgFormat = imageName.split(".")[-1]
                outputImagePath = f"{FINAL_OUTPUT_DIR}/outputImage.{imgFormat}"
                encodedImage = imageToBase64(outputImagePath)
                #detection["image"] = encodedImage
                detection["result"] = detectionResult
                # Add Result in finalResult
                finalResult.append(detection)
            
            totalTime = time.time() - start
            logging.info(f'Detection Time : {totalTime}')
            return jsonify({"detectionResult": finalResult}), 200
            
        except Exception as e:
            logging.error(f'[EXCEPTION]: {e}')
            return jsonify(f"Image could not be processed. Please try again.")
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
