import argparse
from time import time
import torch
import utility
import model_helper
import os


def get_input_args():
    parser = argparse.ArgumentParser()
    valid_archs = {'densenet121', 'densenet161', 'vgg13_bn', 'resnet18'}
    parser.add_argument('data_dir', type=str, help='Set directory to training images')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Set directory to save checkpoints')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Set learning rate hyperparameter')
    parser.add_argument('--hidden_units', type=int, default=500, help='Set number of hidden units hyperparameter')
    parser.add_argument('--epochs', type=int, default=4, help='Set number of epochs')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Set use GPU for training')
    parser.add_argument('--num_threads', type=int, default=16,
                        help='Set number of threads used to train model when using CPU')
    parser.add_argument('--arch', dest='arch', default='densenet121', action='store', choices=valid_archs,
                        help='Set architecture to use for training')

    parser.set_defaults(gpu=False)

    return parser.parse_args()


def main():
    start_time = time()
    in_args = get_input_args()
    use_gpu = torch.cuda.is_available() and in_args.gpu

    print("Training on {} using {}".format(
        "GPU" if use_gpu else "CPU", in_args.arch))

    print("Architecture:{}, Learning rate:{}, Hidden Units:{}, Epochs:{}".format(
        in_args.arch, in_args.learning_rate, in_args.hidden_units, in_args.epochs))

    dataloaders, class_to_idx = model_helper.get_dataloders(in_args.data_dir)

    model, optimizer, criterion = model_helper.create_model(in_args.arch,
                                                            in_args.learning_rate,
                                                            in_args.hidden_units,
                                                            class_to_idx)

    if use_gpu:
        model.cuda()
        criterion.cuda()
    else:
        torch.set_num_threads(in_args.num_threads)

    model_helper.train(model, criterion, optimizer,
                       in_args.epochs,
                       dataloaders['training'],
                       dataloaders['validation'],
                       use_gpu)

    if in_args.save_dir:
        if not os.path.exists(in_args.save_dir):
            os.makedirs(in_args.save_dir)

        file_path = in_args.save_dir + '/' + in_args.arch + '_checkpoint.pth'
    else:
        file_path = in_args.arch + '_checkpoint.pth'

    model_helper.save_checkpoint(file_path,
                                 model, optimizer,
                                 in_args.arch,
                                 in_args.learning_rate,
                                 in_args.hidden_units,
                                 in_args.epochs)

    test_loss, accuracy = model_helper.validate(
        model, criterion, dataloaders['testing'], use_gpu)
    print("Test Accuracy: {:.3f}".format(accuracy))

    end_time = time()
    utility.print_elapsed_time(end_time - start_time)


if __name__ == "__main__":
    main()