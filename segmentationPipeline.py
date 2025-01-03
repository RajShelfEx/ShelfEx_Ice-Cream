import os
import logging
import numpy as np
import cv2
from ultralytics import YOLO
import config
from data.SKUs import rowSku, sectionSku, productSku

logger = logging.getLogger(__name__)
class segmentation:
    def __init__(self):
        self.rowModel = YOLO(config.ROW_WISE_MODEL_WEIGHTS)
        self.sectionModel = YOLO(config.SECTION_WISE_MODEL_WEIGHTS)
        self.productModel = YOLO(config.PRODUCT_SEGMENTATION_MODEL_WEIGHTS)
        self.inputImageDir = config.INPUT_IMAGES_DIR
        self.rowImageDir = config.ROW_WISE_IMAGES_DIR
        self.sectionImageDir = config.SECTION_WISE_IMAGES_DIR
        self.productImageDir = config.PRODUCT_WISE_IMAGES_DIR
        self.finalOutputDir = config.FINAL_OUTPUT_DIR
        self.IOU = config.INTERSECTION_OVER_UNION
        self.CONF = config.CONFIDENCE_THRESHOLD
        self.ImageSize = config.OUTPUT_IMAGE_SIZE
        self.rowSku = rowSku
        self.sectionSku = sectionSku
        self.productSKU = productSku

    def process_masks(self, model, imagePath, skuMapping):
        """
        Common method to process masks from the model's prediction.
        This method uses the specified SKU mapping (row or section).
        """
        results = model.predict(imagePath, save=False)[0]
        orig_img = results.orig_img
        try:
            masks = results.masks.data
            confidences = results.boxes.conf
            SkuIndexes = results.boxes.cls.int().tolist()
        except AttributeError as e:
            logging.error(f"Error processing masks: {e}")
            return orig_img, [], [],[]

        class_names = results.names
        processed_masks = []
        for i, mask in enumerate(masks):
            if hasattr(results.boxes, 'cls'):
                class_id = results.boxes.cls[i]
                if int(class_id) in skuMapping:
                    class_name = skuMapping[int(class_id)]
                else:
                    raise KeyError(f"Class ID {class_id} not found in the specified SKU mapping.")
            else:
                raise ValueError("Class IDs are not available in the results.")

            mask_np = mask.cpu().numpy()
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            mask_resized = cv2.resize(mask_binary, (orig_img.shape[1], orig_img.shape[0]))
            processed_masks.append((mask_resized, class_name))

        return orig_img, processed_masks,confidences, SkuIndexes
    
    def rowWiseSegmentation(self):
        os.makedirs(self.rowImageDir, exist_ok=True)

        for imageName in os.listdir(self.inputImageDir):
            imagePath = os.path.join(self.inputImageDir, imageName)
            if imageName.lower().endswith(('.jpg', '.jpeg', '.png')):
                logging.info(f"Processing image: {imagePath}")
                orig_img, processed_masks, _ , _= self.process_masks(self.rowModel, imagePath, self.rowSku)
                for mask, class_name in processed_masks:
                    white_background = np.ones_like(orig_img) * 255
                    for c in range(3):
                        white_background[:, :, c] = np.where(mask == 1, orig_img[:, :, c], 255)
                    output_path = os.path.join(self.rowImageDir, f"{class_name}.jpg")
                    cv2.imwrite(output_path, white_background, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    logging.info(f"Saved segmented image for SKU '{class_name}' at: {output_path}")

    def sectionWiseSegmentation(self):
            """
            Perform section-wise segmentation on all images in the row-wise directory
            and save only unique masks based on IoU and confidence score.
            Update and return the filtered `processed_masks` and mask data as a list of lists.
            """
            os.makedirs(self.sectionImageDir, exist_ok=True)

            # Dictionary to store updated processed masks for each image
            filtered_processed_masks = {}
            mask_data_list = []  # List to store mask data for all images

            for imageName in os.listdir(self.rowImageDir):
                imagePath = os.path.join(self.rowImageDir, imageName)

                if imageName.lower().endswith(('.jpg', '.jpeg', '.png')):
                    logging.info(f"Processing image: {imagePath}")
                    orig_img, processed_masks, confidences, _ = self.process_masks(
                        self.sectionModel, imagePath, self.sectionSku
                    )

                    # Store unique masks
                    unique_masks = []

                    for i, (mask, class_name) in enumerate(processed_masks):
                        is_duplicate = False

                        for j, (u_mask, _, u_conf) in enumerate(unique_masks):
                            # Calculate IoU between the current mask and unique masks
                            intersection = np.logical_and(mask, u_mask).sum()
                            union = np.logical_or(mask, u_mask).sum()
                            iou = intersection / union if union > 0 else 0

                            # If IoU exceeds threshold, treat it as duplicate
                            if iou > self.IOU:
                                is_duplicate = True
                                # Keep the mask with the higher confidence
                                if confidences[i] > u_conf:
                                    unique_masks[j] = (mask, class_name, confidences[i])
                                break

                        if not is_duplicate:
                            unique_masks.append((mask, class_name, confidences[i]))

                    # Save the unique masks and update processed_masks
                    updated_processed_masks = []
                    for mask, class_name, confidence in unique_masks:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            cropped_img = orig_img[y:y + h, x:x + w]
                            black_background = np.zeros((h, w, 3), dtype=np.uint8)
                            black_background[:h, :w] = cropped_img
                            output_path = os.path.join(self.sectionImageDir, f"{class_name}%.jpg")
                            cv2.imwrite(output_path, black_background, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            logging.info(f"Saved unique cropped image for SKU '{class_name}' at: {output_path}")

                        # Update processed_masks with the filtered masks
                        updated_processed_masks.append((mask, class_name))

                    # Save the updated processed masks for the current image
                    filtered_processed_masks[imageName] = updated_processed_masks
            return filtered_processed_masks
    
    def productSegmentation(self, section_dict):
        """
        Perform section-wise segmentation on all images in the sdection-wise directory
        and save only unique masks based on IoU and confidence score.
        Update and return the filtered `processed_masks`.
        """
        os.makedirs(self.productImageDir, exist_ok=True)

        # Dictionary to store updated processed masks for each image
        filtered_processed_masks = {}
        all_detection_results = {}
        # Check if input directory is empty
        if not os.listdir(self.sectionImageDir):
            logging.warning(f"No images found in directory: {self.sectionImageDir}")
            return filtered_processed_masks

        for imageName in os.listdir(self.sectionImageDir):
            imagePath = os.path.join(self.sectionImageDir, imageName)

            if imageName.lower().endswith(('.jpg', '.jpeg', '.png')):
                logging.info(f"Processing image: {imagePath}")
                (orig_img, processed_masks, confidences,skuIndex) = self.process_masks(
                    self.productModel, imagePath, self.productSKU
                )

                try:
                    SkuNames = productSku[skuIndex[0]]
                except Exception as e:
                    logger.error(f"Error in getting SKU Name: {e}")
                    SkuNames = "Product is blured or not in SKU List!"

                all_detection_results[imageName.split('.')[0]] = SkuNames
                # Store unique masks
                unique_masks = []

                for i, (mask, class_name) in enumerate(processed_masks):
                    is_duplicate = False
                    for j, (u_mask, _, u_conf) in enumerate(unique_masks):
                        # Validate mask shapes before IoU calculation
                        if mask.shape != u_mask.shape:
                            logging.warning(f"Mask shapes do not match: {mask.shape} vs {u_mask.shape}")
                            continue

                        # Calculate IoU
                        intersection = np.logical_and(mask > 0, u_mask > 0).sum()
                        union = np.logical_or(mask > 0, u_mask > 0).sum()
                        iou = intersection / union if union > 0 else 0

                        # If IoU exceeds threshold, treat it as duplicate
                        if iou > self.IOU:
                            is_duplicate = True
                            # Keep the mask with the higher confidence
                            if confidences[i] > u_conf:
                                unique_masks[j] = (mask, class_name, confidences[i])
                            break

                    if not is_duplicate:
                        unique_masks.append((mask, class_name, confidences[i]))

                # Save the unique masks as images
                for mask, class_name, confidence in unique_masks:
                    white_background = np.ones_like(orig_img) * 255  # Create a white background
                    for c in range(3):  # Apply mask to each color channel
                        white_background[:, :, c] = np.where(mask > 0, orig_img[:, :, c], white_background[:, :, c])

                    # Save the image
                    base_name = os.path.splitext(imageName)[0]
                    output_path = os.path.join(self.productImageDir, f"{base_name}_{class_name}.jpg")
                    cv2.imwrite(output_path, white_background, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    logging.info(f"Saved segmented image for SKU '{class_name}' at: {output_path}")

        return  all_detection_results
                                      
