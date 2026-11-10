"""Offline presentation report from faucet result.json and control/tracking logs."""
import argparse
import json
from pathlib import Path

from mobipi.collect_demos.experiment_paths import new_output

import numpy as np
from scipy.spatial.transform import Rotation

from robomimic.scripts.report_rby1_cartesian_evaluation import markdown_table
from robomimic.utils.rby1_evaluation import atomic_json, write_csv


def read_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stats(values):
    return dict(samples=len(values), mean=float(np.mean(values)) if len(values) else None,
                maximum=float(np.max(values)) if len(values) else None,
                p95=float(np.percentile(values, 95)) if len(values) else None)


def completion(result, steps, controls, actions):
    """Reconstruct full criteria from aligned saved states; never use rollout length as completion time."""
    if not result.get("full_success"):
        return None, None, "Not a successful episode"
    release = result.get("release_step")
    if release not in steps or not actions:
        return None, None, "Missing release pose or gripper commands"
    origin = np.asarray(steps[release]["actual_tcp"])[:3, 3]
    queries = np.array([row["control_step"] for row in actions])
    horizon = result["rollout_length"]
    satisfied = []
    for index in range(release, horizon + 1):
        row = steps.get(index)
        if row is None:
            return None, None, f"Missing post-control state at step {index}"
        # control.jsonl[s] is the PRE-state for command s+1, hence POST-state s.
        contact = controls[index]["contact"] if index < len(controls) else result.get("final_contact")
        query = int(np.searchsorted(queries, index - 1, side="right") - 1)
        if contact is None or query < 0:
            return None, None, f"Missing contact or gripper state at step {index}"
        grip = actions[query]["action"][-1]
        distance = np.linalg.norm(np.asarray(row["actual_tcp"])[:3, 3] - origin)
        satisfied.append(bool(row["handle_state"]["water_on"] and not contact and grip < 0 and distance >= .08))
    yes = np.flatnonzero(satisfied)
    if not len(yes) or not satisfied[-1]:
        return None, None, "Saved full_success disagrees with reconstructed final criteria"
    no = np.flatnonzero(np.logical_not(satisfied))
    first = release + int(yes[0])
    sustained = release + int(no[-1] + 1) if len(no) else release
    return first, sustained, "Reconstructed, 1-based post-control steps; sustained means criteria hold through final horizon"


def report(source, output):
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Report output must be outside the original evaluation directory")
    folders = sorted(source.glob("seed_*"), key=lambda path: int(path.name.split("_")[-1]))
    if not folders:
        raise ValueError("No seed directories found")
    output.mkdir(parents=True, exist_ok=False)
    rows, inventory, missing_results, all_position, all_orientation = [], {}, [], [], []
    for folder in folders:
        inventory[folder.name] = {p.name: dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                                  for p in sorted(folder.iterdir()) if p.is_file()}
        if not (folder / "result.json").exists():
            missing_results.append(folder.name)
            continue
        result = json.loads((folder / "result.json").read_text())
        track = read_lines(folder / "tracking.jsonl")
        steps = {int(row["control_step"]): row for row in track if row.get("event") == "step"}
        assert len(steps) == sum(row.get("event") == "step" for row in track), "Duplicate control measurements"
        assert all(1 <= step <= result["rollout_length"] for step in steps)
        position, orientation = [], []
        for step in steps.values():
            if "actual_tcp" not in step or "desired_tcp" not in step:
                continue
            actual, desired = np.asarray(step["actual_tcp"]), np.asarray(step["desired_tcp"])
            if not (np.isfinite(actual).all() and np.isfinite(desired).all()):
                continue
            p = float(np.linalg.norm(actual[:3, 3] - desired[:3, 3]))
            angle = float(Rotation.from_matrix(desired[:3, :3] @ actual[:3, :3].T).magnitude())
            # Verify the saved post-action measurements; ignore bounded_ik and
            # ik_predicted_* solver residuals completely.
            if step.get("position_error") is not None:
                assert abs(step["position_error"] - p) < 1e-9
            if step.get("orientation_error") is not None:
                assert abs(step["orientation_error"] - angle) < 1e-9
            position.append(1000 * p)
            orientation.append(float(np.rad2deg(angle)))
        first, sustained, note = completion(result, steps, read_lines(folder / "control.jsonl"),
                                             read_lines(folder / "policy_actions.jsonl"))
        error = (result.get("error") or "").lower()
        collision = any(term in error for term in ("collision", "disallowed arm contact", "overlaps sink basin"))
        # Counts are terminated episodes, not solver calls with converged=false.
        ik_failure = not collision and any(term in error for term in ("ik", "joint limit", "joint trajectory", "discontinuous"))
        timeout = bool(result.get("timeout", result.get("termination") == "horizon"))
        row = {key: result.get(key) for key in ("diffusion_seed", "environment_seed", "checkpoint", "checkpoint_path",
            "full_success", "activation", "physical_contact", "release", "retreat", "final_water_on", "rollout_length",
            "activation_step", "release_step", "final_faucet_angle", "maximum_faucet_angle", "final_retreat_m", "termination", "error")}
        row.update(collision=collision, ik_failure=ik_failure, timeout=timeout,
                   unsuccessful_timeout=timeout and not bool(result.get("full_success")),
                   fixture_offset=json.dumps(result.get("fixture_offset_xyz_yaw")),
                   first_full_criteria_step=first, sustained_full_criteria_step=sustained, completion_note=note,
                   position_samples=len(position), orientation_samples=len(orientation),
                   missing_tracking_steps=result["rollout_length"] - len(position),
                   position_mean_mm=stats(position)["mean"], position_max_mm=stats(position)["maximum"],
                   position_p95_mm=stats(position)["p95"], orientation_mean_deg=stats(orientation)["mean"],
                   orientation_max_deg=stats(orientation)["maximum"], video=str(folder / "rollout.mp4"))
        rows.append(row)
        all_position.extend(position)
        all_orientation.extend(orientation)
    if not rows:
        raise ValueError("No completed result files")
    assert len({r["diffusion_seed"] for r in rows}) == len(rows)
    n = len(rows)
    rate_keys = ["full_success", "activation", "physical_contact", "release", "retreat", "final_water_on"]
    rates = {key: dict(count=sum(bool(r[key]) for r in rows), total=n,
                      rate=sum(bool(r[key]) for r in rows)/n, missing=sum(r[key] is None for r in rows)) for key in rate_keys}
    counts = {key: sum(r[key] for r in rows) for key in ("collision", "ik_failure", "timeout", "unsuccessful_timeout")}
    counts["other_runtime_failure"] = sum(bool(r["error"]) and not r["collision"] and not r["ik_failure"] for r in rows)
    completed = [r["sustained_full_criteria_step"] for r in rows if r["full_success"] and r["sustained_full_criteria_step"] is not None]
    summary = dict(source=str(source), seed_directories=len(folders), completed_results=n, missing_results=missing_results,
                   rates=rates, failure_and_horizon_counts=counts, position_tracking_mm=stats(all_position),
                   orientation_tracking_deg=stats(all_orientation), executed_control_steps=sum(r["rollout_length"] for r in rows),
                   missing_tracking_steps=sum(r["missing_tracking_steps"] for r in rows),
                   completion_steps=stats(completed), rows=rows,
                   inventory=inventory, environment_seeds=sorted({r["environment_seed"] for r in rows}),
                   fixture_offsets=sorted({r["fixture_offset"] for r in rows}),
                   checkpoint_paths=sorted({r["checkpoint_path"] for r in rows}),
                   methods=dict(tracking="Pooled post-control actual TCP vs executed desired TCP, event=step only; solver residuals excluded",
                                timeout="Recorded fixed-horizon termination, includes successful runs; unsuccessful_timeout is reported separately",
                                completion="Reconstructed first and sustained full-criteria steps on final-success episodes from aligned saved poses/contact/water/gripper; no trajectory resimulation"))
    atomic_json(output / "summary.json", summary)
    write_csv(output / "per_seed.csv", rows, list(rows[0]))
    write_csv(output / "metrics.csv", [dict(metric=k, **v) for k, v in rates.items()], ["metric", "count", "total", "rate", "missing"])
    labels = dict(full_success="Full task success", activation="Faucet activation", physical_contact="Physical contact",
                  release="Release after activation", retreat="Safe final retreat", final_water_on="Water on at end")
    sections = ["# RB-Y1 TurnOnSinkFaucet — half-time evaluation", "",
        f"Epoch 14; {n} diffusion seeds; 100 training demonstrations (77 train / 23 validation, supplied experiment description).",
        f"Source: `{source}`. Environment seeds: {summary['environment_seeds']}; fixture offsets: {summary['fixture_offsets']}.",
        "Fixed base, nominal fixture, 2,500-control-step horizon (125 s at 20 Hz). Diffusion-seed variation at one scene does not measure unseen-layout or base-pose generalization.", "",
        "## Task outcomes", "", markdown_table(["Metric", "Count", "Rate"],
          [[labels[k], f"{v['count']}/{n}", f"{v['rate']:.1%}"] for k, v in rates.items()]), "",
        "Full success requires activation through physical contact, release, ≥8 cm final retreat, no final contact, an open-gripper command, final water on and no runtime failure. Rates above are across all result files, not conditional stage rates.", "",
        "## Failures and fixed-horizon termination", "", markdown_table(["Classification", "Episodes"],
          [["Collision / disallowed contact", counts['collision']], ["IK/controller failure (excluding collision)", counts['ik_failure']],
           ["Other runtime failure", counts['other_runtime_failure']], ["Recorded timeout / horizon reached", counts['timeout']],
           ["Unsuccessful timeout / horizon reached", counts['unsuccessful_timeout']]]), "",
        "Successful runs also use timeout=true because evaluation never stops at task completion. Horizon counts overlap success counts. Collision and IK classifications use the terminal error message; pre-action collision=false fields do not override a later collision exception. Nonconverged optimizer calls are not counted as IK failures.", "",
        "## TCP tracking", "", markdown_table(["Measurement", "Mean", "Maximum", "95th percentile", "Samples"],
          [["Position (mm)", *[summary['position_tracking_mm'][k] for k in ('mean','maximum','p95','samples')]],
           ["Orientation (degrees)", *[summary['orientation_tracking_deg'][k] for k in ('mean','maximum','p95','samples')]]]), "",
        f"Recomputed from actual and desired poses in post-control tracking event=step records. Pooled sample weighting; longer episodes contribute more samples. Coverage: {len(all_position):,}/{summary['executed_control_steps']:,} executed control steps; missing: {summary['missing_tracking_steps']:,}. Missing values are null/—, never zero. No IK solver residuals or pre-action control errors are substituted. Tracking is against the retimed executed TCP target, not the original policy waypoint.", "",
        "## Steps to completion (successful episodes)", "", markdown_table(
          ["Seed", "Activation", "Release", "First full criteria", "Sustained full criteria", "Recorded episode length"],
          [[r[k] for k in ('diffusion_seed','activation_step','release_step','first_full_criteria_step','sustained_full_criteria_step','rollout_length')]
           for r in rows if r['full_success']]), "",
        f"Mean sustained-completion step: {summary['completion_steps']['mean']}; maximum: {summary['completion_steps']['maximum']} (successful episodes only).", "",
        "Steps are 1-based post-control measurements. Full-completion time was not directly recorded; the two full-criteria columns are reconstructed. Sustained means all criteria remain true through the saved final horizon. Post-step contact is taken from the next command's pre-state (final contact from result.json); gripper commands are aligned by policy control_step. Failed runs have no successful completion time; their episode length is not used as one.", "",
        "## Per-seed outcomes", "", markdown_table(["Seed", "Full", "Activated", "Contact", "Released", "Retreated", "Steps", "Termination"],
          [[r['diffusion_seed'], *['Yes' if r[k] else 'No' for k in ('full_success','activation','physical_contact','release','retreat')],
            r['rollout_length'], 'Collision' if r['collision'] else 'IK failure' if r['ik_failure'] else r['termination']] for r in rows]), "",
        "## Availability and provenance", "",
        "Every seed includes result.json, config.json, rollout.mp4, policy_actions.jsonl, control.jsonl and tracking.jsonl. The original evaluator writes per-rollout results and a basic aggregate but no presentation report. The existing door reporter requires rollouts.csv plus microwave-specific fields and success semantics; it cannot consume this faucet format. This reporter reuses its Markdown tables and the common atomic JSON/CSV exporters.",
        f"Missing result files: {missing_results or 'none'}. Missing post-control tracking samples: {summary['missing_tracking_steps']}. Completion reconstruction notes: " + "; ".join(f"seed {r['diffusion_seed']}: {r['completion_note']}" for r in rows if r['full_success']),
        "Checkpoint provenance: " + "; ".join(f"`{p}`" for p in summary['checkpoint_paths']), "",
        "Files: summary.json (full metrics/provenance), per_seed.csv, metrics.csv, outcome_rates.png/.svg, tcp_tracking.png/.svg. Original inputs were read only; no simulations, evaluations or checkpoint loads were performed."]
    (output / "report.md").write_text("\n".join(sections) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    keys = rate_keys[:-1]
    bars = ax.bar([labels[k].replace(' after activation','') for k in keys], [rates[k]['rate']*100 for k in keys], color="#3278a6")
    ax.bar_label(bars, labels=[f"{rates[k]['count']}/{n} ({rates[k]['rate']:.1%})" for k in keys], padding=4)
    ax.set(ylabel="Rollouts (%)", ylim=(0, 108), title=f"RB-Y1 faucet policy, epoch 14 — {n} diffusion seeds")
    ax.tick_params(axis='x', labelrotation=12)
    for suffix in ('png','svg'):
        fig.savefig(output / f"outcome_rates.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, values, unit in zip(axes, (all_position, all_orientation), ("Position error (mm)", "Orientation error (degrees)")):
        if values:
            quantiles = np.linspace(0, 1, 1001)
            ax.plot(np.quantile(values, quantiles), quantiles*100, color="#3278a6")
            ax.axvline(np.percentile(values, 95), color="#d07a22", linestyle="--", label="95th percentile")
            ax.set(xlabel=unit, ylabel="Measured control steps (%)", ylim=(0, 100))
            ax.legend()
        else:
            ax.text(.5, .5, "Tracking unavailable", ha="center")
    fig.suptitle("Actual TCP tracking versus executed target (pooled samples)")
    for suffix in ('png','svg'):
        fig.savefig(output / f"tcp_tracking.{suffix}", dpi=180)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ('completed_results','rates','failure_and_horizon_counts',
        'position_tracking_mm','orientation_tracking_deg','missing_tracking_steps','completion_steps')}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Epoch directory containing seed_*/result.json")
    parser.add_argument("--output", type=Path, default=new_output("reports", "faucet_evaluation"), help="New, separate report directory")
    args = parser.parse_args()
    report(args.source, args.output)
