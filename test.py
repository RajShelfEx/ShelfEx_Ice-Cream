import csv
import pandas as pd
from utils import save_to_csv , csv_to_dict  # save_to_csv is a function
from SKUs import resuit_format # resuit_format is a dictionary
from mainPipeline import Result




# productRecognition = Result()
# detectionResult = productRecognition.main()

detectionResult = {
            "A1-75-%": "Filur-54-ml",
            "A2-25-%": "Solero-Exotic-90-ml",
            "A3-50-%": "Solero-Exotic-90-ml",
            "A4-25-%": "Product is blured or not in SKU List!",
            "A5-25-%": "Product is blured or not in SKU List!",
            "B1-50-%": "HB-Pineapple-Split-75-ml",
            "B2-75-%": "HB-Pineapple-Split-75-ml",
            "B3-75-%": "Zapp-58-ml",
            "B5-75-%": "Product is blured or not in SKU List!",
            "C1-75-%": "Green-Fresh",
            "C2-75-%": "Green-Fresh",
            "C3-75-%": "Nogger-Classic-88",
            "C4-50-%": "Cornetto-Soft-Stracciatella"
}

# take result format dict
final_image_result = resuit_format.copy()
# update result format dict according to detection result
for key, value in detectionResult.items():
    if key != "image":
        bin_name = key.split("-")[0]
        bin_occ = key.split("-")[1]
        sku_name = value
        index = resuit_format["Bin"].index(bin_name)
        # update result
        final_image_result["Bin_Occupancy"][index] = f"{bin_occ} %"
        final_image_result["SKU_Detection"][index] = "Yes"
        # if sku_name == "Product is blured or not in SKU List!":
        #     final_image_result["SKU_Name"][index] = "Not Identified"
        final_image_result["SKU_Name"][index] = sku_name
        

# Call the function
save_to_csv(final_image_result, "data/result_format.csv")
# Example usage
csv_file = "data/result_format.csv"  # Replace with the actual file path
final_result_dict = csv_to_dict(csv_file)
for key, value in final_result_dict.items():
    print(f"{key}: {value}")
    print("------------------------------------------")
    