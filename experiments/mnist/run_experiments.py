from experiments.mnist.local_experiments import MNISTTest
import argparse


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('number', type=int, help='')
    args = parser.parse_args()

    if args.number == 1:
        MNISTTest().mnist_cmr()
    elif args.number == 2:
        MNISTTest().mnist_competitors()
    elif args.number == 3:
        MNISTTest().mnist_rule_interventions()
    elif args.number == 4:
        MNISTTest().mnist_cmr_incomplete_concept_set()
    elif args.number == 5:
        MNISTTest().mnist_comps_incomplete_concept_set()
    elif args.number == 6:
        MNISTTest().mnist_interventions()


if __name__ == '__main__':
    main()


