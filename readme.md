# EAST : An Efficient and Accurate Scene Text Detector

### Introduction
This is a tensorflow implemention of EAST. I only reimplement the RBOX part of the paper, which achieves an F1 score
of 80.8 on the ICDAR 2015 dataset (which is about two points better than the result of pvanet in the paper, see http://rrc.cvc.uab.es/?ch=4&com=evaluation&task=1). The running speed is about 150ms (network) + 300ms (NMS) per image on a K40 card. The nms part is too slow because of the use of shapely in python, and can be further improved.

Thanks for the author's ([@zxytim](https://github.com/zxytim)) help!
Please site his [paper](https://arxiv.org/abs/1704.03155v2) if you find this useful.

### Contents
1. [Installation](#installation)
2. [Download](#download)
3. [Test](#train)
4. [Train](#test)
5. [Examples](#examples)

### Installation
1. I think any version of tensorflow version > 1.0 should be ok.

### Download
1. Models trained on ICDAR 2013 (training set) + ICDAR 2015 (training set): [BaiduYun link](http://pan.baidu.com/s/1jHWDrYQ)
2. Resnet V1 50 provided by tensorflow slim: [slim resnet v1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)

### Train
If you want to train the model, you should provide the dataset path, in the dataset path, a separate gt text file should be provided for each image
and run

```
python multigpu_train.py --gpu_list=0 --input_size=512 --batch_size=14 --checkpoint_path=/tmp/east_icdar2015_resnet_v1_50_rbox/ \
--text_scale=512 --training_data_path=/data/ocr/icdar2015/ --geometry=RBOX --learning_rate=0.0001 --num_readers=24 \
--pretrained_model_path=/tmp/resnet_v1_50.ckpt
```

If you have more than one gpu, you can pass gpu ids to gpu_list

**Note: you should change the gt text file of icdar2015's filename to img_\*.txt instead of gt_img_\*.txt(or you can change the code in icdar.py), and some extra characters should be removed from the file.
See the examples in training_samples/**

### Test
run
```
python eval.py --test_data_path=/tmp/images/ --gpu_list=0 --checkpoint_path=/tmp/east_icdar2015_resnet_v1_50_rbox/ \
--output_path=/tmp/
```

a text file will be then written to the output path.


### Examples
Here is some test examples on icdar2015, enjoy the beautiful text boxes!
![image_1](demo_images/img_2.jpg)
![image_2](demo_images/img_10.jpg)
![image_3](demo_images/img_14.jpg)
![image_4](demo_images/img_26.jpg)
![image_5](demo_images/img_75.jpg)

Please let me know if you encounter any issues(my email boostczc@gmail dot com).
