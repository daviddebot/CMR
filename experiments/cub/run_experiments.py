from experiments.cub.local_experiments import CUBTest
import argparse


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('number', type=int, help='')
    args = parser.parse_args()

    if args.number == 1:
        CUBTest().cub_cmr()
    elif args.number == 2:
        CUBTest().cub_competitors()

if __name__ == '__main__':
    main()


