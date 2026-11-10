"""Check historical CLI entry points, optionally comparing pre-move help text."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    commands = json.loads(Path(__file__).with_name("fixtures").joinpath("cli_commands.json").read_text())
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(root / path) for path in
                                      (".", "external/robomimic", "external/mimicgen", "external/robocasa"))

    def check(path):
        result = subprocess.run([sys.executable, path, "--help"], env=env,
                                capture_output=True, text=True, timeout=60)
        return dict(script=path, exit=result.returncode, stdout=result.stdout, stderr=result.stderr)

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(check, commands))
    failed = any(result["exit"] for result in results)
    if args.reference:
        reference = {item["script"]: item for item in json.loads(args.reference.read_text())}
        mapping = json.loads((root / "docs/codebase_cleanup/migration.json").read_text())
        aliases = {old[:-3].replace("/", "."): new[:-3].replace("/", ".")
                   for old, new in mapping.items() if old.endswith(".py")}

        def normalize(text):
            for old, new in sorted(aliases.items(), key=lambda pair: -len(pair[0])):
                text = text.replace(old, new)
            return text

        for result in results:
            result["help_matches_reference_after_module_rename"] = (
                normalize(result["stdout"]) == normalize(reference[result["script"]]["stdout"]))
            failed = failed or not result["help_matches_reference_after_module_rename"]
    with args.report.open("x") as stream:
        json.dump(results, stream, indent=2)
        stream.write("\n")
    print(f"Checked {len(results)} historical CLI entry points; failures={sum(bool(r['exit']) for r in results)}")
    for result in results:
        if result["exit"]:
            print(result["script"], result["stderr"][-1200:])
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
