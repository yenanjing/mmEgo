## Introduction：

This repository holds the codebase, dataset and models for the paper:
**Egocentric Human Pose Estimation using Head-mounted mmWave Radar**

## Visualization：

Todo

## Requirements:

- Python3 (>3.5)
- PyTorch
- Other Python libraries can be installed by `pip install -r requirements.txt`

## Installation:

``` shell
git clone https://github.com/yenanjing/mmEgo_Rev.git
```

## Data:

The pre-processed sample data is stored in the `Resource/Sample_data` directory, which includes thirteen actions collected in the
paper. Additional data will be placed in Dropbox in the future.

## Test:

You can use the following commands to test.

```shell
# with pose estimation quantization results
python main.py --infer

# with pose estimation visualization results
python main.py --infer --vis
```

## Train:

You can use the following commands to train the IMU_Net, Upper_Net, Lower_Net, respectively.

```shell
# train IMU_Net
python main.py --train --network=IMU_Net

# train Upper_Net
python main.py --train --network=Upper_Net

# train Lower_Net
python main.py --train --network=Lower_Net
