## Introduction：

This repository holds the codebase, dataset and models for the paper:

**Egocentric Human Pose Estimation using Head-mounted mmWave Radar**.

The structure of the repository is listed as following:
```
|-- Config
|   |-- config.py
|   |-- config_demo.py
|-- Net
|   |-- GCN.py
|   |-- IMU_Net.py
|   |-- Lower_Net.py
|   |-- Upper_Net.py
|-- Processor
|   |-- Test
|   |   |-- Demo_test.py
|   |-- Train
|       |-- Train_IMU.py
|       |-- Train_Lower.py
|       |-- Train_Upper.py
|-- Resource
|   |-- Img
|   |-- Pretrained_model
|   |-- Sample_data
|-- Util
|   |-- Universal_Util
|   |   |-- Dataset.py
|   |   |-- Dataset_sample.py
|   |   |-- Utils.py
|   |   |-- Utils_demo.py
|   |-- Visual_Util
|       |-- draw3Dpose.py
|-- main.py
|-- README.md
|-- requirement.txt
```

## Visualization：

Todo

## Requirements:

- Python3 (>3.5)
- PyTorch
- Other Python libraries can be installed by `pip install -r requirements.txt`

****Tips:** If you don't want to set up the environment locally, you can use Google's Colab service to run
this [notebook](https://colab.research.google.com/drive/1Y8gPFRGWQudmVw7DBKW0JkgehBoV7BoH?usp=sharing).**

## Installation:

``` shell
git clone https://github.com/yenanjing/mmEgo_Rev.git
```

## Data:

The pre-processed sample data is stored in the `Resource/Sample_data` directory, which includes thirteen actions
collected in the paper. Additional data will be placed in Dropbox in the future.

## Test:

### Quantitative results

You can use the following command to test with pose estimation quantization results.

```shell
python main.py --infer
```

Expected Terminal Output as follows:

```shell
data load end
835it [00:28, 29.50it/s]
Average Joint Localization Error(cm): 3.893234612654426
Average UpperBody Joint Localization Error(cm): 3.507117400849294
Average LowerBody Joint Localization Error(cm): 4.487715154930861
Average Joint Rotation Error(°): 5.3738645146242865
Per Joint Localization Error(cm): [3.35941255 2.87198341 2.56157758 2.30499098 2.34241374 2.80576303
 4.48244299 6.71278707 2.31875466 2.69022188 4.41651893 6.65339315
 3.10652336 4.25521534 5.23230264 5.73526735 3.0540133  4.12044566
 5.01231217 5.38564139 2.33594403]
```

And the plot window displaying the error bar chart for each joint, is as follows:
<p align="center">
    <img src="Resource/Img/infer_result.png", width="1200">
</p>

### Qualitative results

You can use the following command to test with pose estimation visualization results

```shell
python main.py --infer --vis
```

The expected output should be a comparison between the predicted skeleton and the ground truth skeleton for each frame,
as shown below:
<p align="center">
    <img src="Resource/Img/visualization.png", width="1200">
</p>

## Train:

You can use the following commands to train the IMU_Net, Upper_Net, Lower_Net, respectively.

```shell
# train IMU_Net
python main.py --train --network IMU_Net

# train Upper_Net
python main.py --train --network Upper_Net

# train Lower_Net
python main.py --train --network Lower_Net
```

For any question, feel free to contact.