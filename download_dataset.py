# !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key=<API-KEY>)
project = rf.workspace("sample-aoipo").project("masterdataset")
version = project.version(3)
dataset = version.download("yolov11")
                
