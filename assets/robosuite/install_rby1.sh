#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: install_rby1.sh /path/to/conda/environment

Example:
  ./assets/robosuite/install_rby1.sh ~/anaconda3/envs/mobipi
EOF
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="$1"

if [[ ! -d "$CONDA_ENV" ]]; then
    echo "Conda environment does not exist: $CONDA_ENV" >&2
    exit 1
fi

CONDA_ENV="$(cd -- "$CONDA_ENV" && pwd)"
PYTHON_BIN="$CONDA_ENV/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found in Conda environment: $PYTHON_BIN" >&2
    exit 1
fi

ROBOSUITE_VERSION="$("$PYTHON_BIN" - <<'PY'
from importlib.metadata import version
print(version("robosuite"))
PY
)"

if [[ "$ROBOSUITE_VERSION" != "1.5.0" ]]; then
    echo "Unsupported robosuite version: $ROBOSUITE_VERSION" >&2
    echo "This installer is tested with robosuite 1.5.0." >&2
    exit 1
fi

ROBOSUITE_DIR="$("$PYTHON_BIN" - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("robosuite")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("robosuite is not installed")

print(Path(next(iter(spec.submodule_search_locations))).resolve())
PY
)"

REQUIRED_SOURCES=(
    "$SOURCE_DIR/models/assets/robots/rby1a/rby1a_1.2.xml"
    "$SOURCE_DIR/models/assets/bases/rby1_mount.xml"
    "$SOURCE_DIR/models/robots/manipulators/rby1_robot.py"
    "$SOURCE_DIR/models/grippers/rby1_gripper.py"
    "$SOURCE_DIR/controllers/config/robots/default_rby1.json"
)

for source_file in "${REQUIRED_SOURCES[@]}"; do
    if [[ ! -f "$source_file" ]]; then
        echo "Required RB-Y1 source file is missing: $source_file" >&2
        exit 1
    fi
done

PATCH_TARGETS=(
    "$ROBOSUITE_DIR/models/robots/manipulators/__init__.py"
    "$ROBOSUITE_DIR/models/grippers/__init__.py"
    "$ROBOSUITE_DIR/robots/__init__.py"
    "$ROBOSUITE_DIR/controllers/parts/mobile_base/joint_vel.py"
)

for target_file in "${PATCH_TARGETS[@]}"; do
    if [[ ! -f "$target_file" ]]; then
        echo "Required robosuite file is missing: $target_file" >&2
        exit 1
    fi
done

"$PYTHON_BIN" - "$ROBOSUITE_DIR" <<'PY'
from pathlib import Path
import shutil
import sys


robosuite_dir = Path(sys.argv[1])
manipulators_init = robosuite_dir / "models/robots/manipulators/__init__.py"
grippers_init = robosuite_dir / "models/grippers/__init__.py"
robots_init = robosuite_dir / "robots/__init__.py"
joint_vel = robosuite_dir / "controllers/parts/mobile_base/joint_vel.py"


def insert_before(text, marker, addition, path):
    if addition.strip() in text:
        return text
    position = text.find(marker)
    if position < 0:
        raise RuntimeError(f"Could not find {marker!r} in {path}")
    return text[:position] + addition + text[position:]


updated = {}

text = manipulators_init.read_text()
import_line = "from .rby1_robot import RBY1\n"
if import_line.strip() not in text:
    if text and not text.endswith("\n"):
        text += "\n"
    text += import_line
updated[manipulators_init] = text

text = grippers_init.read_text()
text = insert_before(
    text,
    "GRIPPER_MAPPING",
    "from .rby1_gripper import RBY1Gripper\n\n",
    grippers_init,
)
if '"RBY1Gripper"' not in text and "'RBY1Gripper'" not in text:
    text = insert_before(
        text,
        "GRIPPER_MAPPING = {",
        "",
        grippers_init,
    )
    opening = text.find("{", text.find("GRIPPER_MAPPING"))
    text = text[: opening + 1] + '\n    "RBY1Gripper": RBY1Gripper,' + text[opening + 1 :]
updated[grippers_init] = text

text = robots_init.read_text()
if "WheeledRobot" not in text:
    raise RuntimeError(f"WheeledRobot is not available in {robots_init}")
if '"RBY1"' not in text and "'RBY1'" not in text:
    marker = "ROBOT_CLASS_MAPPING"
    mapping = text.find(marker)
    if mapping < 0:
        raise RuntimeError(f"Could not find {marker!r} in {robots_init}")
    opening = text.find("{", mapping)
    if opening < 0:
        raise RuntimeError(f"Could not find mapping opening brace in {robots_init}")
    text = text[: opening + 1] + '\n    "RBY1": WheeledRobot,' + text[opening + 1 :]
updated[robots_init] = text

text = joint_vel.read_text()
replacement_marker = "base_action = np.asarray(action).copy()"
if replacement_marker not in text:
    lines = text.splitlines(keepends=True)
    start_marker = "base_action = np.copy([action[i] for i in [1, 0, 2]])"
    end_marker = "base_action[1] = -x * np.sin(theta) + y * np.cos(theta)"
    starts = [index for index, line in enumerate(lines) if line.strip() == start_marker]
    if len(starts) != 1:
        raise RuntimeError(
            f"Expected one legacy mobile-base block in {joint_vel}, found {len(starts)}"
        )
    start = starts[0]
    ends = [
        index
        for index in range(start, len(lines))
        if lines[index].strip() == end_marker
    ]
    if not ends:
        raise RuntimeError(f"Could not find the end of the mobile-base block in {joint_vel}")
    end = ends[0]
    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    replacement = [
        f"{indent}if jnt_dim == 3:\n",
        f"{indent}    base_action = np.copy([action[i] for i in [1, 0, 2]])\n",
        f"{indent}    # input raw base action is delta relative to current pose of base\n",
        f"{indent}    # controller expects deltas relative to initial pose of base at start of episode\n",
        f"{indent}    # transform deltas from current base pose coordinates to initial base pose coordinates\n",
        f"{indent}    x, y = base_action[0:2]\n",
        "\n",
        f"{indent}    # do the reverse of theta rotation\n",
        f"{indent}    base_action[0] = x * np.cos(theta) + y * np.sin(theta)\n",
        f"{indent}    base_action[1] = -x * np.sin(theta) + y * np.cos(theta)\n",
        f"{indent}else:\n",
        f"{indent}    base_action = np.asarray(action).copy()\n",
    ]
    lines[start : end + 1] = replacement
    text = "".join(lines)
updated[joint_vel] = text

# Validate every edit before changing any installed file.
assert "from .rby1_robot import RBY1" in updated[manipulators_init]
assert "from .rby1_gripper import RBY1Gripper" in updated[grippers_init]
assert '"RBY1Gripper": RBY1Gripper' in updated[grippers_init]
assert '"RBY1": WheeledRobot' in updated[robots_init]
assert replacement_marker in updated[joint_vel]

for path, contents in updated.items():
    backup = path.with_name(path.name + ".pre-rby1")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(contents)
PY

mkdir -p \
    "$ROBOSUITE_DIR/models/assets/robots/rby1a" \
    "$ROBOSUITE_DIR/models/assets/bases" \
    "$ROBOSUITE_DIR/models/robots/manipulators" \
    "$ROBOSUITE_DIR/models/grippers" \
    "$ROBOSUITE_DIR/controllers/config/robots"

cp -a "$SOURCE_DIR/models/assets/robots/rby1a/." \
    "$ROBOSUITE_DIR/models/assets/robots/rby1a/"
install -m 0644 "$SOURCE_DIR/models/assets/bases/rby1_mount.xml" \
    "$ROBOSUITE_DIR/models/assets/bases/rby1_mount.xml"
install -m 0644 "$SOURCE_DIR/models/robots/manipulators/rby1_robot.py" \
    "$ROBOSUITE_DIR/models/robots/manipulators/rby1_robot.py"
install -m 0644 "$SOURCE_DIR/models/grippers/rby1_gripper.py" \
    "$ROBOSUITE_DIR/models/grippers/rby1_gripper.py"
install -m 0644 "$SOURCE_DIR/controllers/config/robots/default_rby1.json" \
    "$ROBOSUITE_DIR/controllers/config/robots/default_rby1.json"

echo "RB-Y1 files installed into: $ROBOSUITE_DIR"
echo "Original patched files have a one-time .pre-rby1 backup beside them."
echo "Run the import test with:"
echo "  $PYTHON_BIN $SOURCE_DIR/rby1_import_test.py"
