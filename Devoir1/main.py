import argparse
import solver_naive
import solver_advanced
import time
from utils import Instance


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--agent', type=str, default='naive')
    parser.add_argument('--group', type=str, default='easy')
    parser.add_argument('--number', type=int, default=1)
    parser.add_argument('--outdir', type=str, default='solutions')
    parser.add_argument('--visualisation_file', type=str, default='visualization')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    instance = Instance(args.group,args.number)

    print("***********************************************************")
    print("[INFO] Start the solving: artwall design")
    print("[INFO] input file: %s" % instance.filepath)
    print("[INFO] output directory: %s" % args.outdir)
    print("[INFO] visualization file: %s" % args.visualisation_file)
    print("[INFO] number of artpieces: %s" % (instance.n))
    print(f"[INFO] wall size (WxH): {instance.wall.width()}x{instance.wall.height()}")
    print("***********************************************************")

    start_time = time.time()

    # Méthode à implémenter
    if args.agent == "naive":
        # assign a different time slot for each course
        solution = solver_naive.solve(instance)
    elif args.agent == "advanced":
        # Your nice agent
        solution = solver_advanced.solve(instance)
    else:
        raise Exception("This agent does not exist")


    solving_time = round((time.time() - start_time) / 60,2)

    # You can disable the display if you do not want to generate the visualization
    instance.visualize_solution(solution, visualisation_file=args.visualisation_file)
    #
    instance.save_solution(solution, args.outdir)
    print("***********************************************************")
    print("[INFO] Solution obtained")
    print("[INFO] Execution time : %s minutes" % solving_time)
    print("[INFO] Number of walls : %s" % solution.nwalls)
    print("[INFO] Sanity check passed : %s" % instance.is_valid_solution(solution))
    print("***********************************************************")
