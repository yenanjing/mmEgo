import argparse
from Config.config import Config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')
    parser.add_argument('--network', type=str, choices=['IMU_Net', 'Upper_Net', 'Lower_Net'],
                        help='Choose a network: IMU_Net, Upper_Net, Lower_Net')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--infer', action='store_true', help='Perform inference')
    parser.add_argument('--vis', action='store_true', help='Visualization')
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, help="device: [cuda:%d, cpu]")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument('--log_dir', type=int, help='Path to save the model and report')
    parser.add_argument('--load_IMU_path', type=str, help='Path to load IMU_Net')
    parser.add_argument('--load_Upper_path', type=str, help='Path to load Upper_Net')
    parser.add_argument('--load_Lower_path', type=str, help='Path to load Lower_Net')



    args = parser.parse_args()
    if args.train:
        if args.epochs is not None:
            Config.epochs = args.epochs
        if args.log_dir is not None:
            Config.Idx = args.log_dir
        if args.device is not None:
            Config.device = args.device
        if args.load_IMU_path is not None:
            Config.model_IMU_path = args.load_IMU_path
        if args.load_Upper_path is not None:
            Config.model_upper_path = args.load_Upper_path
        if args.load_Lower_path is not None:
            Config.model_lower_path = args.load_Lower_path

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
