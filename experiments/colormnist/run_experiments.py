from experiments.colormnist.local_experiments import ColorMNISTTest
import argparse


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('number', type=int, help='')
    args = parser.parse_args()

    if args.number == 1:
        ColorMNISTTest().colormnist_cmr()
    elif args.number == 2:
        ColorMNISTTest().colormnist_competitors()


if __name__ == '__main__':
    main()


