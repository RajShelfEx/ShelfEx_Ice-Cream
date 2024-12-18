# !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="4BMHBXI4YPtMpL0lEfm3")
project = rf.workspace("sample-aoipo").project("masterdataset")
version = project.version(3)
dataset = version.download("yolov11")
                
