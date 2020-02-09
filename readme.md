## CPD net for salient object detection

Pytorch implementation of CPD(CVPR 2019), some codes(part of
network structure definition) are from repository
 https://github.com/wuzhe71/CPD ,this project is for practice.
 
 ### Usage
* Train
    1. Put corresponding dataset in ./input/
        * training images(RGB, jpg format): ./input/train/raw/
        * training masks(gray, png format): ./input/train/mask/
        * validation images(RGB, jpg format): ./input/test/raw/
        * validation masks(gray, png format): ./input/test/mask/
    2. Run train.py, if you want to change some parameters,
    see train.py for detail.
* Inference
    1. Put inference data in ./inference/
        * inference images(RGB, jpg format): ./inference
    2. Run inference.py, output saliency maps will be in
    ./output directory.

### Requirements
Original running environment:
* Python 3.7.5
* Pytorch 1.3.1
* TorchVision 0.2.1
* pillow 7.0.0

Detailed requirements are listed in requirements.txt.
