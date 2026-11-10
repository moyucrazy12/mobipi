"""Compare recorded numerical behavior; source datasets are always read-only."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def compare(before, after):
    names = []
    exact, maximum = True, 0.
    with h5py.File(before, "r") as reference, h5py.File(after, "r") as candidate:
        reference.visititems(lambda name, item: names.append(name) if isinstance(item, h5py.Dataset) else None)
        other = []
        candidate.visititems(lambda name, item: other.append(name) if isinstance(item, h5py.Dataset) else None)
        if names != other:
            raise AssertionError("Recording dataset keys differ")
        for name in names:
            a, b = reference[name][:], candidate[name][:]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise AssertionError(f"{name}: shape or dtype differs")
            exact = exact and np.array_equal(a, b)
            if np.issubdtype(a.dtype, np.floating):
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=0, err_msg=name)
                if a.size:
                    maximum = max(maximum, float(np.max(np.abs(a - b))))
            else:
                np.testing.assert_array_equal(a, b, err_msg=name)
    return dict(before=str(before), after=str(after), dataset_arrays=len(names),
                all_arrays_bitwise_equal=bool(exact), maximum_float_difference=maximum,
                scope="All dataset arrays, including images; file-path metadata attributes excluded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = compare(args.before, args.after)
    text = json.dumps(result, indent=2) + "\n"
    if args.report:
        with args.report.open("x") as stream:
            stream.write(text)
    print(text)


if __name__ == "__main__":
    main()
