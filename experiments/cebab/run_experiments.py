from experiments.cebab.local_experiments import CebabTest
import argparse


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('number', type=int, help='')
    args = parser.parse_args()

    if args.number == 1:
        CebabTest().cebab_cmr()
    elif args.number == 2:
        CebabTest().cebab_competitors()
    elif args.number == 3:
        CebabTest().cebab_cmr_ablation2()
    elif args.number == 4:
        CebabTest().cebab_cmr_ablation1()


if __name__ == '__main__':
    main()


