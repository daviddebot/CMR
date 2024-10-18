from experiments.celeba.local_experiments import CelebATest
import argparse


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('number', type=int, help='')
    args = parser.parse_args()

    if args.number == 1:
        CelebATest().celeba_cmr()
    elif args.number == 2:
        CelebATest().celeba_competitors()


if __name__ == '__main__':
    main()


