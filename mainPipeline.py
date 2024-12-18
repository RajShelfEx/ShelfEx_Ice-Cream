import os
import cv2
from PIL import Image
from ultralytics import YOLO
from collections import Counter
import numpy as np
from segmentationPipeline import segmentation
from temp.detectionPipeline import detection
import config
import shutil


class Result:
    def __init__(self):
        # Instantiate the ObjectDetection class
        self.segment = segmentation()
        self.input_dir = config.INPUT_IMAGES_DIR
        self.row_dir = config.ROW_WISE_IMAGES_DIR
        self.section_dir = config.SECTION_WISE_IMAGES_DIR
        self.product_dir = config.PRODUCT_WISE_IMAGES_DIR 
        self.final_output_dir = config.FINAL_OUTPUT_DIR

    def clear_directories(self):
        directories = [self.row_dir, self.section_dir, self.product_dir, self.final_output_dir]
        for directory in directories:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    def Save_output_image(self, mask_data_dict):
        # create dictionary of section name and mask
        section_dict = {}
        for key, value in mask_data_dict.items():
            for ele in value:
                section_name = ele[1]
                section_mask = ele[0]
                section_dict[section_name] = section_mask

        # Create list of masks
        mask_data_list = []
        for key, value in section_dict.items():
            mask_data_list.append(value)
        
        # Ensure the output directory exists
        if not os.path.exists(self.final_output_dir):
            os.makedirs(self.final_output_dir)

        # Read to draw section-mask on input image
        files = os.listdir(self.input_dir)

        # Filter out non-image files (you can add more formats if needed)
        image_file = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Get the image path
        image_path = os.path.join(self.input_dir, image_file[0])

        # Read the input image to draw section
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image {image_path}")
            return


        colored_canvas = np.zeros_like(image)

        # Generate distinct colors for each mask
        num_colors = 256  # Max colors
        colors = [
            tuple(np.random.choice(range(256), size=3)) for _ in range(num_colors)
        ]
        # Process the predicted masks
        color_index = 0
        for mask in mask_data_list:  # Process each mask for the current image
            # Create a color mask for the current mask
            mask_color = np.zeros_like(image)
            color = colors[color_index % len(colors)]  # Cycle through colors
            color_index += 1
            mask_color[mask > 0] = color

            # Add the colored mask to the canvas
            colored_canvas = cv2.addWeighted(colored_canvas, 1.0, mask_color, 1.0, 0)

        # Overlay the colored canvas on the original image
        overlayed_image = cv2.addWeighted(image, 0.7, colored_canvas, 0.3, 0)


        # Save the result
        output_path = os.path.join(self.final_output_dir, "outputImage.jpg")
        cv2.imwrite(output_path, overlayed_image)
        print(f"Saved: {output_path}")
        return section_dict


    def main(self):
        # Perform row-wise segmentation on all images in the input directory
        self.clear_directories()
        self.segment.rowWiseSegmentation()
        sec_masks =self.segment.sectionWiseSegmentation()
        section_dict = self.Save_output_image(sec_masks)
        output=self.segment.productSegmentation(section_dict)
        return output

    
    


if __name__ == "__main__":
    # Initialize the Result class
     
    res = Result()
    res.clear_directories()
    # Run the main function
    print(res.main())
    
