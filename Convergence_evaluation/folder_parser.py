import argparse
import os
from analyze import PMFAnalyzer


def parse(path):
    # parser = argparse.ArgumentParser(
    #     description="path for folder containing abf1.hist.czar.pmf and abf1.hist.count files"
    # )
    # args = parser.parse_args()
    # path = args.path
    print(path)
    with open(path + 'results.dat', 'w') as f:
        for folder in os.listdir(path):
            if folder.startswith('reference'):
                continue
            elif folder.startswith('common'):
                continue
            if not os.path.isdir(path + folder):
                continue

            try:
                if not os.path.isfile(path + folder + '/output/abf_00.abf1.hist.czar.pmf'):
                    filename = path + folder + '/output/window1.abf1.hist.'
                else:
                    filename = path + folder + '/output/abf_00.abf1.hist.'
                analyzer = PMFAnalyzer(
                    filename + 'czar.pmf',
                    filename + 'count',
                    slope_thresh=0.01,
                    n_recent=5,
                    use_final_rmsd=False,
                    count_std_thresh=None

                )

                print('{} {} {}'.format(*folder.split('_'), analyzer.convergence_idx))
                f.write('{} {} {}\n'.format(*folder.split('_'), analyzer.convergence_idx))

            except:
                print('{} {} {}'.format(*folder.split('_'), 'err'))
                f.write('{} {} {}\n'.format(*folder.split('_'), 'err'))

            # analyzer.plot(
            #     font_size=20,
            #     annotation_fs=20,
            #     show_annotations=False,
            #     save_path=path + folder + 'convergence.png',
            #     dpi=300,
            #
            # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="path for folder containing abf1.hist.czar.pmf and abf1.hist.count files"
    )
    parser.add_argument('path',
                        help='Path to folders')
    args = parser.parse_args()
    path = args.path
    parse(path)
