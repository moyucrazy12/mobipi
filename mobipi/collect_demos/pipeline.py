"""Resumable task-independent orchestration of existing demonstration tools."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

from mobipi.collect_demos.experiment_paths import artifact_path
import tempfile

import h5py
from mobipi.collect_demos.pipeline_tasks import TASKS


def save_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2)
        temporary = Path(stream.name)
    temporary.replace(path)


@contextmanager
def run_lock(root):
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another pipeline process owns this output directory") from None
        yield


def merge(datasets, output, validator, prepared=False, metadata=None):
    """Copy accepted episodes unchanged, remapping local source indices by name."""
    if output.exists():
        return
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        # Preserve an interrupted merge; rebuild from immutable attempt files.
        index = 0
        while temporary.with_suffix(f".interrupted_{index}").exists():
            index += 1
        temporary.rename(temporary.with_suffix(f".interrupted_{index}"))
    count, total, canonical = 0, 0, []
    with h5py.File(temporary, "x") as dest:
        data = dest.create_group("data")
        for path in datasets:
            with h5py.File(path, "r") as src:
                local = json.loads(src["data"].attrs.get("source_demo_keys", "[]"))
                for name in local:
                    if name not in canonical:
                        canonical.append(name)
                if count == 0:
                    for key, value in src["data"].attrs.items():
                        data.attrs[key] = value
                for name in sorted(src["data"], key=lambda key: int(key.split("_")[-1])):
                    src.copy(src["data"][name], data, name=f"demo_{count}")
                    episode = data[f"demo_{count}"]
                    episode.attrs["pipeline_source_file"] = str(path)
                    episode.attrs["pipeline_source_episode"] = name
                    if metadata and str(path) in metadata:
                        episode.attrs["pipeline_attempt_metadata"] = json.dumps(metadata[str(path)])
                    if "src_demo_inds" in episode:
                        if not local:
                            raise ValueError("Missing source_demo_keys for source index remapping")
                        episode["src_demo_inds"][...] = [canonical.index(local[int(i)]) for i in episode["src_demo_inds"][:]]
                    total += int(episode.attrs["num_samples"])
                    count += 1
        data.attrs["total"] = total
        if canonical:
            data.attrs["source_demo_keys"] = json.dumps(canonical)
    validator(temporary, prepared=prepared)
    temporary.rename(output)


class Pipeline:
    def __init__(self, root, task, settings, resume=False):
        self.root, self.task = root, task
        manifest = root / "manifest.json"
        if manifest.exists():
            if not resume:
                raise FileExistsError("Run exists; use --resume")
            self.state = json.loads(manifest.read_text())
            previous = dict(self.state["settings"])
            for key in ("seed_attempts", "augment_attempts"):
                if settings.get(key, 0) >= previous.get(key, 0):
                    previous[key] = settings[key]
            if settings != previous:
                raise ValueError("Resume settings differ from saved run")
            self.state["settings"] = settings
            self.save()
        else:
            if any(p.name != ".lock" for p in root.iterdir()):
                raise FileExistsError("New run directory must be empty")
            self.state = {"settings": settings, "active_arm": task.active_arm,
                          "attempts": {"collect": [], "augment": []}, "completed": []}
            self.save()
        self.settings = settings

    def save(self):
        save_json(self.root / "manifest.json", self.state)

    def path(self, stage):
        imported = self.settings.get({"collect": "seed_dataset", "prepare": "prepared_dataset",
                                     "merge": "augmented_dataset"}.get(stage, ""))
        return Path(imported) if imported else self.root / {
            "collect": "seeds.hdf5", "prepare": "source.hdf5", "merge": "generated.hdf5",
            "convert": "policy.hdf5"}[stage]

    def check(self, path, prepared=False):
        return self.task.call("validate", path, prepared=prepared)

    def candidate(self, stage):
        """Reserve a new artifact directory; interrupted files remain available."""
        parent = self.root / "artifacts" / stage
        parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="build_", dir=parent)) / "dataset.hdf5"

    def attempts(self, stage):
        records = self.state["attempts"][stage]
        target = self.settings["seeds" if stage == "collect" else "augment"]
        budget = self.settings["seed_attempts" if stage == "collect" else "augment_attempts"]
        config = self.settings["seed_config" if stage == "collect" else "augmentation_config"]
        source = self.path("prepare") if stage == "augment" else None
        for record in records:
            if record["status"] == "running":
                candidate = Path(record["dataset"])
                try:
                    self.check(candidate, prepared=stage == "augment")
                    record["status"] = "accepted"
                except (OSError, ValueError, KeyError, AssertionError) as error:
                    record.update(status="interrupted", reason=str(error))
                self.save()
        while sum(r["status"] == "accepted" for r in records) < target and len(records) < budget:
            index = len(records)
            directory = self.root / "attempts" / stage / f"attempt_{index:06d}"
            directory.parent.mkdir(parents=True, exist_ok=True)
            effective, metadata = self.task.call("attempt_config", config, stage, index, source,
                                                self.settings["translation"], self.settings["rotation_deg"])
            candidate = directory / ("seeds.hdf5" if stage == "collect" else "generated.hdf5")
            record = dict(index=index, status="running", directory=str(directory), dataset=str(candidate),
                          configuration=effective, **metadata)
            records.append(record)
            self.save()
            try:
                if stage == "collect":
                    self.task.call("collect", effective, directory, 1, 1)
                else:
                    self.task.call("augment", source, directory, effective, 1, 1)
                self.check(candidate, prepared=stage == "augment")
                record["status"] = "accepted"
            except Exception as error:
                # Some existing collectors raise after flushing a complete episode.
                try:
                    self.check(candidate, prepared=stage == "augment")
                    record.update(status="accepted", warning=str(error))
                except (OSError, ValueError, KeyError, AssertionError):
                    record.update(status="rejected", reason=str(error))
            finally:
                self.save()
        accepted = [Path(r["dataset"]) for r in records if r["status"] == "accepted"]
        if len(accepted) < target:
            raise RuntimeError(f"{stage}: {len(accepted)}/{target} successes; finite attempt budget {budget} reached")
        return accepted

    def run(self, stage, convert=False):
        stages = ["collect", "prepare", "augment", "merge", "validate"] if stage == "all" else [stage]
        if stage == "all" and self.settings.get("augmented_dataset"):
            stages = ["merge", "validate"]
        elif stage == "all" and self.settings.get("prepared_dataset"):
            stages.remove("collect")
        if stage == "all" and convert:
            stages.append("convert")
        for current in stages:
            if current in self.state["completed"]:
                continue
            if current == "collect":
                if not self.settings.get("seed_dataset"):
                    merge(self.attempts("collect"), self.path("collect"), self.check)
                else:
                    self.check(self.path("collect"))
            elif current == "prepare":
                if self.settings.get("prepared_dataset") and not self.path("prepare").is_file():
                    raise FileNotFoundError("Imported prepared dataset must already exist")
                if not self.path("prepare").exists():
                    candidate = self.candidate("prepare")
                    self.task.call("prepare", self.path("collect"), candidate)
                    self.check(candidate, prepared=True)
                    candidate.rename(self.path("prepare"))
                self.check(self.path("prepare"), prepared=True)
            elif current == "augment":
                if self.settings.get("augmented_dataset"):
                    self.check(self.path("merge"), prepared=True)
                else:
                    self.attempts("augment")
            elif current == "merge":
                if not self.settings.get("augmented_dataset"):
                    accepted = [Path(r["dataset"]) for r in self.state["attempts"]["augment"] if r["status"] == "accepted"]
                    if len(accepted) != self.settings["augment"]:
                        raise ValueError("Run augment stage to obtain the requested success count before merging")
                    provenance = {r["dataset"]: {k: v for k, v in r.items() if k != "configuration"}
                                  for r in self.state["attempts"]["augment"] if r["status"] == "accepted"}
                    merge(accepted, self.path("merge"), self.check, prepared=True, metadata=provenance)
            elif current == "validate":
                rows = self.check(self.path("merge"), prepared=True)
                if len(rows) != self.settings["augment"]:
                    raise ValueError("Merged episode count does not match target")
                save_json(self.root / "validation.json", rows)
                with h5py.File(self.path("merge"), "r") as f:
                    episodes = list(f["data"].values())
                    lengths = [int(ep.attrs["num_samples"]) for ep in episodes]
                    source_keys = json.loads(f["data"].attrs.get("source_demo_keys", "[]"))
                    sources = {source_keys[int(i)] for ep in episodes if "src_demo_inds" in ep for i in ep["src_demo_inds"][:]}
                    self.state["dataset_statistics"] = dict(episodes=len(episodes), total_steps=sum(lengths),
                        length_min=min(lengths), length_max=max(lengths), length_mean=sum(lengths)/len(lengths),
                        unique_source_ids=sorted(sources), path=str(self.path("merge")))
            elif current == "convert":
                if not self.task.convert:
                    raise ValueError("Task has no policy converter")
                if not self.path("convert").exists():
                    candidate = self.candidate("convert")
                    report = self.task.call("convert", self.path("merge"), candidate)
                    report.update(dataset=str(self.path("convert")), conversion_artifact=str(candidate))
                    save_json(self.root / "conversion.json", report)
                    # Keep the converter's recorded artifact path valid too.
                    os.link(candidate, self.path("convert"))
                elif not (self.root / "conversion.json").exists():
                    raise ValueError("Existing policy output has no conversion report")
            self.state["completed"].append(current)
            self.save()
        return self.summary()

    def summary(self):
        summary = {"task": self.settings["task"], "active_arm": self.task.active_arm,
                   "completed": self.state["completed"], "stages": {},
                   "dataset": self.state.get("dataset_statistics")}
        for name, records in self.state["attempts"].items():
            accepted = sum(r["status"] == "accepted" for r in records)
            summary["stages"][name] = dict(attempts=len(records), successful=accepted,
                                          success_rate=accepted / len(records) if records else None)
        save_json(self.root / "summary.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--stage", choices=["all", "collect", "prepare", "augment", "merge", "validate", "convert"], default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--convert", action="store_true")
    for name in ("seeds", "augment", "seed-attempts", "augment-attempts"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--translation", type=float, help="Task translation limit in metres (faucet: XY)")
    parser.add_argument("--rotation-deg", type=float, help="Task rotation limit in degrees (faucet: yaw)")
    for name in ("seed-dataset", "prepared-dataset", "augmented-dataset", "seed-config", "augmentation-config"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    root = (args.output or artifact_path("demos", f"demos_{args.task}")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    task = TASKS[args.task]
    manifest = root / "manifest.json"
    saved = json.loads(manifest.read_text())["settings"] if manifest.exists() else {}
    if args.resume and not saved:
        parser.error("No saved run exists at --output")
    settings = dict(saved) if args.resume else {"task": args.task, "seeds": 10, "augment": 100,
        "translation": .02, "rotation_deg": 2., "seed_dataset": None, "prepared_dataset": None, "augmented_dataset": None,
        "seed_config": json.loads(Path(task.seed_config).read_text()),
        "augmentation_config": json.loads(Path(task.augmentation_config).read_text())}
    settings["task"] = args.task
    for key in ("seeds", "augment", "seed_attempts", "augment_attempts", "translation", "rotation_deg"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    for key in ("seed_dataset", "prepared_dataset", "augmented_dataset", "seed_config", "augmentation_config"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = json.loads(value.read_text()) if key.endswith("config") else str(value.resolve())
    settings.setdefault("seed_attempts", 3 * settings["seeds"])
    settings.setdefault("augment_attempts", 3 * settings["augment"])
    for key in ("seed_dataset", "prepared_dataset", "augmented_dataset"):
        if settings.get(key) and not Path(settings[key]).is_file():
            parser.error(f"{key} must reference an existing read-only input file")
    if not 1 <= settings["seeds"] <= settings["seed_attempts"] or not 1 <= settings["augment"] <= settings["augment_attempts"]:
        parser.error("Success targets must be positive and no larger than attempt budgets")
    with run_lock(root):
        pipeline = Pipeline(root, task, settings, resume=args.resume)
        try:
            print(json.dumps(pipeline.run(args.stage, convert=args.convert), indent=2))
        finally:
            pipeline.summary()


if __name__ == "__main__":
    main()
