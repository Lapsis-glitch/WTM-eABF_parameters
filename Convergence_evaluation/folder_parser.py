import argparse
import os
from analyze_ND import PMFAnalyzer


def parse(path, reference_pmf):
    print(path)

    with open(path + 'results.dat', 'w') as f:
        for folder in os.listdir(path):

            # Skip special folders
            if folder.startswith('reference'):
                continue
            if folder.startswith('common'):
                continue
            if not os.path.isdir(path + folder):
                continue

            try:
                # Detect which filename prefix is present
                if not os.path.isfile(path + folder + '/output/abf_00.abf1.hist.czar.pmf'):
                    filename = path + folder + '/output/window1.abf1.hist.'
                else:
                    filename = path + folder + '/output/abf_00.abf1.hist.'

                analyzer = PMFAnalyzer(
                    filename + 'czar.pmf',
                    filename + 'count',
                    slope_thresh=0.01,
                    n_recent=5,
                    use_sliding_window=False,      # NEW API
                    count_std_thresh=None,
                    reference_pmf_file=reference_pmf,
                    rmsd_thresh=0.25               # optional: fixed RMSD threshold
                )

                print('{} {} {}'.format(*folder.split('_'), analyzer.convergence_idx))
                f.write('{} {} {}\n'.format(*folder.split('_'), analyzer.convergence_idx))

            except Exception as e:
                print('{} {} {}'.format(*folder.split('_'), 'err'))
                f.write('{} {} {}\n'.format(*folder.split('_'), 'err'))

            # Optional plotting:
            # analyzer.plot(
            #     font_size=20,
            #     annotation_fs=20,
            #     show_annotations=False,
            #     save_path=path + folder + 'convergence.png',
            #     dpi=300,
            # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Path to folder containing ABF PMF/count histories"
    )
    parser.add_argument('path', help='Path to folders')
    parser.add_argument('--reference-pmf',
                        type=str, default=None,
                        help='Optional external PMF file to use as RMSD reference')

    args = parser.parse_args()
    parse(args.path, args.reference_pmf)