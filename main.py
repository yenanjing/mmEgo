import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')
    parser.add_argument('--network', type=str, choices=['IMU_Net', 'Upper_Net', 'Lower_Net'],
                        help='Choose a network: IMU_Net, Upper_Net, Lower_Net')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--infer', action='store_true', help='Perform inference')
    parser.add_argument('--vis', action='store_true', help='Visualization')

    args = parser.parse_args()
    if args.train:
        if args.network == 'IMU_Net':
            from Processor.Train.Train_IMU import MMEgo

            processor = MMEgo()
            processor.train_imu()
        if args.network == 'Upper_Net':
            from Processor.Train.Train_Upper import MMEgo

            processor = MMEgo()
            processor.train_upper()
        if args.network == 'Lower_Net':
            from Processor.Train.Train_Lower import MMEgo

            processor = MMEgo()
            processor.train_lower()

    elif args.infer:
        # 调用推理类或模块的代码
        from Processor.Test.Demo_test import MMEgo

        processor = MMEgo()
        if args.vis:
            processor.eval_all_skeleton()
        else:
            processor.eval_model()
