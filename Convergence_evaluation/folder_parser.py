import argparse
import os
from analyze_ND import PMFAnalyzer


def parse(path, reference_pmf):
    """Scan immediate subdirectories and evaluate convergence using PMFAnalyzer."""
    print(path)

    results_file = os.path.join(path, 'results.dat')
    with open(results_file, 'w') as f:
        for folder in os.listdir(path):
            # Skip special folders and non-directories
            if folder.startswith(('reference', 'common')):
                continue
            folder_path = os.path.join(path, folder)
            if not os.path.isdir(folder_path):
                continue

            try:
                # Detect which filename prefix is present using os.path.join
                candidate1 = os.path.join(folder_path, 'output', 'abf_00.abf1.hist.czar.pmf')
                if os.path.isfile(candidate1):
                    filename = os.path.join(folder_path, 'output', 'abf_00.abf1.hist.')
                else:
                    filename = os.path.join(folder_path, 'output', 'window1.abf1.hist.')

                analyzer = PMFAnalyzer(
                    filename + 'czar.pmf',
                    filename + 'count',
                    slope_thresh=0.01,
                    n_recent=5,
                    use_sliding_window=False,
                    count_std_thresh=None,
                    reference_pmf_file=reference_pmf,
                    rmsd_thresh=0.25
                )

                print('{} {} {}'.format(*folder.split('_'), analyzer.convergence_idx))
                f.write('{} {} {}\n'.format(*folder.split('_'), analyzer.convergence_idx))

            except Exception as e:
                print('{} {} {}'.format(*folder.split('_'), 'err'))
                f.write('{} {} {}\n'.format(*folder.split('_'), 'err'))



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