import cv2
import os
import numpy as np
from ultralytics import YOLO
from segmentationPipeline import segmentation

obj_segmentation = segmentation()
sec_masks = obj_segmentation.sectionWiseSegmentation()
final_output_dir = "data/final_output_images"

def Save_output_image(mask_data_dict):
    mask_data_list = []
    for key, value in mask_data_dict.items():
        mask_data_list.append(value)

    input_image_path = "data/input_images/22.jpg"
    # Read the input image to draw section
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Unable to load image {input_image_path}")
        return

    # Create a blank mask to combine all masks and a blank canvas for colored masks
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    colored_canvas = np.zeros_like(image)

    # Generate distinct colors for each mask
    num_colors = 256  # Max colors
    colors = [
        tuple(np.random.choice(range(256), size=3)) for _ in range(num_colors)
    ]
    # Process the predicted masks
    color_index = 0
    
    for mask in mask_data_list:  # Process each mask for the current image
        # mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)  # Convert mask to uint8
        # resized_mask = cv2.resize(mask_np, (image.shape[1], image.shape[0]))  # Resize mask to match image size

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
    output_path = os.path.join(final_output_dir, "outputimage.jpg")
    cv2.imwrite(output_path, overlayed_image)
    print(f"Saved: {output_path}")


section_dict = {}
for key, value in sec_masks.items():
    for ele in value:
        section_name = ele[1]
        section_mask = ele[0]
        section_dict[section_name] = section_mask

Save_output_image(section_dict)
    
   

# Load a pretrained YOLO11n model
# model = YOLO("data/checkpoints/product_detection_weight/weights/last.pt")

# results = model("data/section_wise_images", save = True)  # list of 2 Results objects
