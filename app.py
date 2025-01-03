import os
import time
import base64
import config
import logging
import requests
import subprocess
from flask import Flask, request, jsonify
from utils import download_and_unzip,modelsConfig, checkDir
from mainPipeline import Result
from config import FINAL_OUTPUT_DIR
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ******************************** CHECK MODEL DIRECTORY EXIST OR NOT *****************************
modelWeightsDir = config.MODEL_WEIGHTS_DIR
if not os.path.exists(modelWeightsDir):
    checkDir("data")
    # Loop through each model and download/unzip if necessary
    for model in modelsConfig:
        download_and_unzip(
            model_url=model["url"],
            output_path=model["download_path"],
            extract_to=model["extract_path"],
        )
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
            # check row-wise image dir
            checkDir(config.ROW_WISE_IMAGES_DIR)
            # check section-wise image dir
            checkDir(config.SECTION_WISE_IMAGES_DIR)
            # check product-wise image dir
            checkDir(config.PRODUCT_WISE_IMAGES_DIR)
            # check finaloutput image dir
            checkDir(config.FINAL_OUTPUT_DIR)

            finalResult = []
            detection = {}
            inputImage = ''
            # iterate each image url
            for imageUrl in imageUrlList:
                # Download image
                response = requests.get(imageUrl)
                if response.status_code == 200:
                    # extract image name
                    imageName = imageUrl.split('/')[-1]
                    imagePath = os.path.join(inputImageDir, f'{imageName}')
                    inputImage = imagePath
                    with open(imagePath, 'wb') as f:
                        f.write(response.content)
                    logging.info(f'Image downloaded and saved to {imagePath}')

                ############################ DETECTION PROCESS #################################
                detectionResult = productRecognition.main()
                print(detectionResult)
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
                outputImagePath = f"{FINAL_OUTPUT_DIR}/{imgName}.{imgFormat}"
                encodedImage = imageToBase64(outputImagePath)

                detection["Input-Image"] = encodedImage
                detection["result"] = detectionResult
                # Add Result in finalResult
                encodedInputImage = imageToBase64(inputImage)
                detectionResult['image'] = encodedInputImage
                detectionResult['image_with_label'] = encodedImage
                finalResult.append(detectionResult)

            totalTime = time.time() - start
            logging.info(f'Detection Time : {totalTime}')
            return jsonify({"Detection Result": finalResult}), 200

        except Exception as e:
            logging.error(f'[EXCEPTION]: {e}')
            return jsonify(f"Image could not be processed. Please try again.{e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
