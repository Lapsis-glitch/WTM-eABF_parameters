import argparse
import os
import numpy as np
from collections import defaultdict
from analyze_ND import PMFAnalyzer


def extract_base_name(folder):
    """Remove _seed_X suffix."""
    if "_seed_" in folder:
        return folder.rsplit("_seed_", 1)[0]
    return folder


def parse(path, reference_pmf):
    print(path)

    grouped_results = defaultdict(list)

    # First pass: compute convergence per folder
    for folder in os.listdir(path):
        if folder.startswith(('reference', 'common')):
            continue

        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue

        base_name = extract_base_name(folder)

        try:
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
                rmsd_thresh=0.25,
                use_ref_and_slope=True
            )

            grouped_results[base_name].append(analyzer.convergence_idx)

        except Exception:
            # skip errors for aggregation (or you could log them)
            continue

    # Second pass: compute stats and write output
    results_file = os.path.join(path, 'results.dat')
    with open(results_file, 'w') as f:
        for base_name, values in grouped_results.items():
            # Remove None or invalid entries
            clean_values = [v for v in values if v is not None]

            if len(clean_values) == 0:
                print(f"{base_name} skipped (no valid values)")
                continue

            mean_val = np.mean(clean_values)
            std_val = np.std(clean_values)



            print(f"{base_name} {mean_val:.3f} {std_val:.3f} (n={len(clean_values)})")
            f.write(f"{base_name} {mean_val:.3f} {std_val:.3f} {len(clean_values)}\n")


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