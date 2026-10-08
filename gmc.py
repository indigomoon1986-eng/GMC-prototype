#!/usr/bin/env python3
"""
Geometric Manifold Computing
Curvature-Governed Resilience Research Prototype
Steven Leake / Monarch X / Sophia

Python 3.10+, standard library only.

Run:
    python gmc.py simulate
    python gmc.py verify
    python gmc.py test
    python gmc.py replay --file results/receipts.json
    python gmc.py write-rust

Boundary:
    Single-process distributed-system simulation.
    Graph-Laplacian curvature proxy, not intrinsic Ricci curvature.
    No physical validation, real infrastructure actuation, or universal
    anomaly-detection guarantee.
    Rust source is provided; compilation requires a Rust toolchain.
"""

import argparse
import copy
import hashlib
import json
import math
import pathlib
import random
import unittest
from dataclasses import asdict, dataclass

MODEL = "gmc.graph-resilience/1"


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digest(value):
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class Policy:
    max_transfer: float = 0.05
    energy_tolerance: float = 1e-12
    anomaly_threshold: float = 0.08
    operator: str = "steven"
    capability: str = "simulate.transfer"

    def validate(self):
        for value in (
            self.max_transfer,
            self.energy_tolerance,
            self.anomaly_threshold,
        ):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("invalid policy number")
        if self.max_transfer <= 0:
            raise ValueError("invalid transfer budget")
        if self.energy_tolerance < 0 or self.anomaly_threshold < 0:
            raise ValueError("negative policy tolerance")
        if not self.operator or self.capability != "simulate.transfer":
            raise ValueError("invalid simulation authority")


def validate(state):
    if not isinstance(state, dict):
        raise ValueError("state must be an object")
    if set(state) != {"model", "epoch", "nodes", "edges"}:
        raise ValueError("unknown state structure")
    if state["model"] != MODEL:
        raise ValueError("unknown model")
    if type(state["epoch"]) is not int or state["epoch"] < 0:
        raise ValueError("invalid epoch")
    nodes = state["nodes"]
    edges = state["edges"]
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("empty or invalid node list")
    if not isinstance(edges, list):
        raise ValueError("invalid edge list")
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError("invalid node")
        if set(node) != {"id", "load", "capacity", "x"}:
            raise ValueError("invalid node structure")
        if type(node["id"]) is not int or node["id"] != index:
            raise ValueError("ordered contiguous node identities required")
        for key in ("load", "capacity", "x"):
            value = node[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("invalid telemetry")
        if node["capacity"] <= 0:
            raise ValueError("nonpositive capacity")
        if not 0 <= node["load"] <= node["capacity"]:
            raise ValueError("capacity violation")
    seen = set()
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError("invalid edge structure")
        a, b = edge
        if (
            type(a) is not int
            or type(b) is not int
            or not 0 <= a < b < len(nodes)
            or (a, b) in seen
        ):
            raise ValueError("invalid or duplicate edge")
        seen.add((a, b))
    return state


def field(state):
    return [
        node["load"] / node["capacity"]
        for node in state["nodes"]
    ]


def mass(state):
    return math.fsum(node["load"] for node in state["nodes"])


def energy(state):
    utilization = field(state)
    return math.fsum(
        0.5 * (utilization[a] - utilization[b]) ** 2
        for a, b in state["edges"]
    )


def geometry(state):
    validate(state)
    utilization = field(state)
    count = len(utilization)
    curvature = [0.0] * count
    degree = [0] * count
    distances = [[] for _ in range(count)]
    for a, b in state["edges"]:
        delta = utilization[b] - utilization[a]
        curvature[a] += delta
        curvature[b] -= delta
        degree[a] += 1
        degree[b] += 1
        distance = abs(
            state["nodes"][a]["x"] - state["nodes"][b]["x"]
        )
        distances[a].append(distance)
        distances[b].append(distance)
    epsilon = 1e-6
    density = [
        1.0 / (math.fsum(values) / len(values) + epsilon)
        if values else 0.0
        for values in distances
    ]
    inverse = [
        1.0 / (value + epsilon) if value else 0.0
        for value in density
    ]
    total = math.fsum(inverse)
    weights = [
        value / total if total else 0.0
        for value in inverse
    ]
    contrast = [
        weights[a] * abs(utilization[a] - utilization[b])
        for a, b in state["edges"]
    ]
    return {
        "curvature": curvature,
        "degree": degree,
        "density": density,
        "weights": weights,
        "contrast": contrast,
        "isolated": [
            index for index, value in enumerate(degree)
            if value == 0
        ],
    }


def detect(state, baseline_curvature, policy):
    policy.validate()
    current = geometry(state)["curvature"]
    if len(current) != len(baseline_curvature):
        raise ValueError("baseline shape mismatch")
    if any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in baseline_curvature
    ):
        raise ValueError("invalid baseline")
    score = max(
        abs(current[index] - baseline_curvature[index])
        for index in range(len(current))
    )
    return {
        "score": score,
        "threshold": policy.anomaly_threshold,
        "detected": score > policy.anomaly_threshold,
    }


def proposals(state, policy):
    validate(state)
    policy.validate()
    metrics = geometry(state)
    curvature = metrics["curvature"]
    degree = metrics["degree"]
    gradient = [
        -curvature[index] / node["capacity"]
        for index, node in enumerate(state["nodes"])
    ]
    state_digest = digest(state)
    actions = []
    for a, b in state["edges"]:
        source, destination = (
            (a, b) if gradient[a] > gradient[b] else (b, a)
        )
        source_node = state["nodes"][source]
        destination_node = state["nodes"][destination]
        c1 = source_node["capacity"]
        c2 = destination_node["capacity"]
        hessian = (
            degree[source] / c1 ** 2
            + degree[destination] / c2 ** 2
            + 2.0 / (c1 * c2)
        )
        amount = min(
            policy.max_transfer,
            abs(gradient[source] - gradient[destination]) / hessian,
            source_node["load"],
            c2 - destination_node["load"],
        )
        if amount <= 1e-12:
            continue
        actions.append({
            "src": source,
            "dst": destination,
            "amount": amount,
            "epoch": state["epoch"],
            "state_digest": state_digest,
            "operator": policy.operator,
            "capability": policy.capability,
        })
    return actions


def admit(state, action, policy):
    unchanged = copy.deepcopy(state)
    try:
        validate(state)
        policy.validate()
        if not isinstance(action, dict) or set(action) != {
            "src",
            "dst",
            "amount",
            "epoch",
            "state_digest",
            "operator",
            "capability",
        }:
            raise ValueError("action shape")
        source = action["src"]
        destination = action["dst"]
        amount = action["amount"]
        if type(source) is not int or type(destination) is not int:
            raise ValueError("node identity")
        if sorted([source, destination]) not in state["edges"]:
            raise ValueError("edge permission")
        if (
            type(action["epoch"]) is not int
            or action["epoch"] != state["epoch"]
            or action["state_digest"] != digest(state)
        ):
            raise ValueError("stale snapshot")
        if (
            action["operator"] != policy.operator
            or action["capability"] != policy.capability
        ):
            raise ValueError("authority")
        if (
            type(amount) not in (int, float)
            or not math.isfinite(amount)
            or not 0 < amount <= policy.max_transfer
        ):
            raise ValueError("budget")
        after = copy.deepcopy(state)
        after["nodes"][source]["load"] -= amount
        after["nodes"][destination]["load"] += amount
        after["epoch"] += 1
        validate(after)
        if abs(mass(after) - mass(state)) > 1e-10:
            raise ValueError("conservation")
        if energy(after) > energy(state) + policy.energy_tolerance:
            raise ValueError("energy increases")
        return after, "admitted"
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OverflowError,
        ZeroDivisionError,
    ) as error:
        return unchanged, "rejected:" + str(error)


class Engine:
    def __init__(self, state, policy=None):
        validate(state)
        self.policy = policy or Policy()
        self.policy.validate()
        self.initial = copy.deepcopy(state)
        self.state = copy.deepcopy(state)
        self.receipts = []

    def execute(self, action):
        before = copy.deepcopy(self.state)
        try:
            canonical(action)
        except (ValueError, TypeError, OverflowError):
            action = {
                "invalid_repr": repr(action),
            }
        action = copy.deepcopy(action)
        after, status = admit(before, action, self.policy)
        receipt = {
            "schema": "monarch.gmc.receipt/2",
            "sequence": len(self.receipts),
            "prior_root": (
                self.receipts[-1]["receipt_digest"]
                if self.receipts else None
            ),
            "model": MODEL,
            "policy": asdict(self.policy),
            "before": before,
            "action": action,
            "after": after,
            "status": status,
            "environment": "python-simulation",
            "physical_validation": False,
        }
        receipt["receipt_digest"] = digest(receipt)
        self.receipts.append(receipt)
        self.state = after
        return receipt

    def step(self):
        for edge in list(self.state["edges"]):
            action = next(
                (
                    candidate
                    for candidate in proposals(self.state, self.policy)
                    if sorted([
                        candidate["src"],
                        candidate["dst"],
                    ]) == edge
                ),
                None,
            )
            if action is not None:
                self.execute(action)
        return energy(self.state)


def replay(initial, receipts, expected_policy=None, expected_root=None):
    validate(initial)
    state = copy.deepcopy(initial)
    previous = None
    for index, receipt in enumerate(receipts):
        body = {
            key: value
            for key, value in receipt.items()
            if key != "receipt_digest"
        }
        if (
            digest(body) != receipt["receipt_digest"]
            or receipt["sequence"] != index
            or receipt["prior_root"] != previous
            or receipt["before"] != state
            or receipt["schema"] != "monarch.gmc.receipt/2"
            or receipt["model"] != MODEL
            or receipt["environment"] != "python-simulation"
            or receipt["physical_validation"] is not False
        ):
            raise ValueError("receipt integrity")
        policy = Policy(**receipt["policy"])
        policy.validate()
        if expected_policy is not None and policy != expected_policy:
            raise ValueError("policy mismatch")
        after, status = admit(state, receipt["action"], policy)
        if status != receipt["status"] or after != receipt["after"]:
            raise ValueError("semantic replay")
        state = after
        previous = receipt["receipt_digest"]
    if expected_root is not None and previous != expected_root:
        raise ValueError("trusted root mismatch")
    return state


def ring(count=12):
    if type(count) is not int or count < 3:
        raise ValueError("ring requires at least three nodes")
    return {
        "model": MODEL,
        "epoch": 0,
        "nodes": [
            {
                "id": index,
                "load": 0.5,
                "capacity": 1.0,
                "x": index / count,
            }
            for index in range(count)
        ],
        "edges": sorted(
            [[index, index + 1] for index in range(count - 1)]
            + [[0, count - 1]]
        ),
    }


def research_gates(state):
    results = []
    try:
        validate(state)
    except (ValueError, TypeError) as error:
        return [{
            "gate": 1,
            "name": "manifold_admissibility",
            "status": "FAIL",
            "reason": str(error),
        }]
    results.append({
        "gate": 1,
        "name": "manifold_admissibility",
        "status": "PASS",
        "qualification": (
            "Graph validity only; smooth manifold existence unproved."
        ),
    })
    metrics = geometry(state)
    density_pass = (
        not metrics["isolated"]
        and abs(math.fsum(metrics["weights"]) - 1.0) < 1e-10
    )
    results.append({
        "gate": 2,
        "name": "density_bias_correction",
        "status": "PASS" if density_pass else "FAIL",
        "isolated": metrics["isolated"],
    })
    if not density_pass:
        return results
    utilization = field(state)
    reconstruction_error = max(
        (
            abs(
                contrast / metrics["weights"][a]
                - abs(utilization[a] - utilization[b])
            )
            for (a, b), contrast in zip(
                state["edges"],
                metrics["contrast"],
            )
        ),
        default=0.0,
    )
    results.append({
        "gate": 3,
        "name": "contrast_reconstruction",
        "status": (
            "PASS" if reconstruction_error < 1e-10 else "FAIL"
        ),
        "max_error": reconstruction_error,
        "qualification": "Numerical consistency, not semantic truth.",
    })
    if reconstruction_error >= 1e-10:
        return results
    errors = []
    for count in (32, 64, 128):
        spacing = 2.0 * math.pi / count
        values = [
            math.sin(index * spacing)
            for index in range(count)
        ]
        errors.append(max(
            abs(
                (
                    values[(index - 1) % count]
                    - 2.0 * values[index]
                    + values[(index + 1) % count]
                ) / spacing ** 2
                + values[index]
            )
            for index in range(count)
        ))
    convergence_pass = (
        errors[2] < errors[1] < errors[0]
        and errors[2] < 0.001
    )
    results.append({
        "gate": 4,
        "name": "curvature_convergence",
        "status": "PASS" if convergence_pass else "FAIL",
        "resolutions": [32, 64, 128],
        "max_errors": errors,
        "qualification": (
            "Periodic finite-difference benchmark only; "
            "not arbitrary-graph or physical validation."
        ),
    })
    return results


def simulate(seed=1599, sweeps=240):
    rng = random.Random(seed)
    report = {
        "model": MODEL,
        "seed": seed,
        "sweeps": sweeps,
        "environment": "single-process-simulation",
        "physical_validation": False,
        "research_gates": research_gates(ring()),
        "scenarios": {},
    }
    logs = {}
    for scenario in (
        "overload",
        "capacity_degradation",
        "partition",
        "adversarial",
        "unobservable_failure",
    ):
        state = ring()
        baseline = geometry(state)["curvature"]
        if scenario == "overload":
            state["nodes"][0]["load"] = 0.95
            state["nodes"][1]["load"] = 0.05
        elif scenario == "capacity_degradation":
            state["nodes"][0]["capacity"] = 0.55
        elif scenario == "partition":
            state["edges"] = [
                edge for edge in state["edges"]
                if edge not in [[0, 11], [5, 6]]
            ]
            for node in state["nodes"]:
                node["load"] = rng.uniform(0.1, 0.9)
        engine = Engine(state)
        initial_energy = energy(state)
        detection = detect(state, baseline, engine.policy)
        if scenario == "adversarial":
            action = {
                "src": 0,
                "dst": 1,
                "amount": float("nan"),
                "epoch": state["epoch"],
                "state_digest": digest(state),
                "operator": "steven",
                "capability": "simulate.transfer",
            }
            engine.execute(action)
            action["amount"] = 0.01
            action["operator"] = "sophia"
            engine.execute(action)
        trajectory = [initial_energy]
        for _ in range(sweeps):
            trajectory.append(engine.step())
        root = (
            engine.receipts[-1]["receipt_digest"]
            if engine.receipts else None
        )
        recovered = replay(
            engine.initial,
            engine.receipts,
            expected_policy=engine.policy,
            expected_root=root,
        )
        if recovered != engine.state:
            raise AssertionError("replay disagreement")
        final_energy = energy(engine.state)
        report["scenarios"][scenario] = {
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_reduction_fraction": (
                1.0 - final_energy / initial_energy
                if initial_energy else 0.0
            ),
            "detection": detection,
            "receipts": len(engine.receipts),
            "rejected": sum(
                receipt["status"].startswith("rejected:")
                for receipt in engine.receipts
            ),
            "mass_error": abs(
                mass(engine.state) - mass(engine.initial)
            ),
            "final_utilization": field(engine.state),
            "trajectory": trajectory,
            "replay_verified": True,
        }
        logs[scenario] = {
            "initial": engine.initial,
            "policy": asdict(engine.policy),
            "root": root,
            "receipts": engine.receipts,
        }
    return report, logs


RUST_CARGO = """
[package]
name = "gmc-kernel"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["rlib", "cdylib"]
"""

RUST_SOURCE = r'''
// Numerical simulation kernel; does not authenticate or authorize effects.

#[derive(Clone, Debug)]
pub struct State {
    pub load: Vec<f64>,
    pub capacity: Vec<f64>,
    pub edges: Vec<(usize, usize)>,
}

impl State {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.load.is_empty() || self.load.len() != self.capacity.len() {
            return Err("shape");
        }
        for (&load, &capacity) in self.load.iter().zip(&self.capacity) {
            if !load.is_finite()
                || !capacity.is_finite()
                || capacity <= 0.0
                || load < 0.0
                || load > capacity
            {
                return Err("telemetry");
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for &(a, b) in &self.edges {
            if a >= b || b >= self.load.len() || !seen.insert((a, b)) {
                return Err("edge");
            }
        }
        Ok(())
    }

    pub fn curvature(&self) -> Result<Vec<f64>, &'static str> {
        self.validate()?;
        let mut result = vec![0.0; self.load.len()];
        for &(a, b) in &self.edges {
            let delta =
                self.load[b] / self.capacity[b]
                - self.load[a] / self.capacity[a];
            result[a] += delta;
            result[b] -= delta;
        }
        Ok(result)
    }

    pub fn energy(&self) -> Result<f64, &'static str> {
        self.validate()?;
        Ok(self.edges.iter().map(|&(a, b)| {
            let delta =
                self.load[a] / self.capacity[a]
                - self.load[b] / self.capacity[b];
            delta * delta / 2.0
        }).sum())
    }

    pub fn transfer(
        &self,
        source: usize,
        destination: usize,
        amount: f64,
        budget: f64,
    ) -> Result<Self, &'static str> {
        self.validate()?;
        if !budget.is_finite()
            || budget <= 0.0
            || !amount.is_finite()
            || amount <= 0.0
            || amount > budget
        {
            return Err("budget");
        }
        let edge = if source < destination {
            (source, destination)
        } else {
            (destination, source)
        };
        if !self.edges.contains(&edge) {
            return Err("edge permission");
        }
        let mut after = self.clone();
        after.load[source] -= amount;
        after.load[destination] += amount;
        after.validate()?;
        let before_mass: f64 = self.load.iter().sum();
        let after_mass: f64 = after.load.iter().sum();
        if (after_mass - before_mass).abs() > 1e-10 {
            return Err("conservation");
        }
        if after.energy()? > self.energy()? + 1e-12 {
            return Err("energy");
        }
        Ok(after)
    }
}

// Caller must supply valid, non-overlapping buffers of declared lengths.
#[no_mangle]
pub unsafe extern "C" fn gmc_curvature(
    n: usize,
    m: usize,
    load: *const f64,
    capacity: *const f64,
    edges: *const usize,
    output: *mut f64,
) -> i32 {
    if n == 0
        || n > 1_000_000
        || m > 4_000_000
        || load.is_null()
        || capacity.is_null()
        || output.is_null()
        || (m > 0 && edges.is_null())
    {
        return -1;
    }
    let edge_data = if m == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(edges, 2 * m)
    };
    let state = State {
        load: std::slice::from_raw_parts(load, n).to_vec(),
        capacity: std::slice::from_raw_parts(capacity, n).to_vec(),
        edges: edge_data
            .chunks_exact(2)
            .map(|edge| (edge[0], edge[1]))
            .collect(),
    };
    match state.curvature() {
        Ok(values) => {
            std::ptr::copy_nonoverlapping(values.as_ptr(), output, n);
            0
        }
        Err(_) => -2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> State {
        State {
            load: vec![0.9, 0.1],
            capacity: vec![1.0, 1.0],
            edges: vec![(0, 1)],
        }
    }

    #[test]
    fn curvature_matches_reference() {
        assert_eq!(state().curvature().unwrap(), vec![-0.8, 0.8]);
    }

    #[test]
    fn transfer_reduces_energy() {
        let before = state();
        let after = before.transfer(0, 1, 0.05, 0.05).unwrap();
        assert!(after.energy().unwrap() < before.energy().unwrap());
    }

    #[test]
    fn nonfinite_transfer_is_refused() {
        assert!(state().transfer(0, 1, f64::NAN, 0.05).is_err());
    }

    #[test]
    fn contradictory_transfer_is_refused() {
        assert!(state().transfer(1, 0, 0.05, 0.05).is_err());
    }
}
'''


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.logs = simulate()

    def test_determinism(self):
        report, logs = simulate()
        self.assertEqual(digest(report), digest(self.report))
        self.assertEqual(digest(logs), digest(self.logs))

    def test_research_gates(self):
        self.assertTrue(all(
            gate["status"] == "PASS"
            for gate in research_gates(ring())
        ))

    def test_fail_closed_pipeline(self):
        state = ring()
        state["nodes"][0]["capacity"] = 0
        results = research_gates(state)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "FAIL")

    def test_isolation(self):
        state = ring()
        state["edges"] = []
        self.assertEqual(research_gates(state)[-1]["status"], "FAIL")

    def test_tamper_detection(self):
        log = copy.deepcopy(self.logs["overload"])
        log["receipts"][0]["after"]["nodes"][0]["load"] += 0.01
        with self.assertRaises(ValueError):
            replay(log["initial"], log["receipts"])

    def test_stale_and_unauthorized_actions(self):
        state = ring()
        state["nodes"][0]["load"] = 0.9
        engine = Engine(state)
        action = proposals(state, engine.policy)[0]
        self.assertEqual(engine.execute(action)["status"], "admitted")
        self.assertTrue(
            engine.execute(action)["status"].startswith("rejected:")
        )
        action = proposals(engine.state, engine.policy)[0]
        action["operator"] = "sophia"
        self.assertIn("authority", engine.execute(action)["status"])

    def test_adversarial_numbers(self):
        state = ring()
        state["nodes"][0]["load"] = 0.9
        engine = Engine(state)
        for value in (
            float("nan"),
            float("inf"),
            -0.1,
            1e9,
            True,
        ):
            action = proposals(state, engine.policy)[0]
            action["amount"] = value
            self.assertTrue(
                engine.execute(action)["status"].startswith("rejected:")
            )
            self.assertEqual(engine.state, state)
        self.assertEqual(
            replay(state, engine.receipts, expected_policy=engine.policy),
            state,
        )

    def test_partition_limit(self):
        values = self.report["scenarios"]["partition"]["final_utilization"]
        self.assertLess(max(values[:6]) - min(values[:6]), 0.001)
        self.assertLess(max(values[6:]) - min(values[6:]), 0.001)
        self.assertGreater(abs(values[0] - values[6]), 0.01)

    def test_random_conservation_and_descent(self):
        rng = random.Random(42)
        for _ in range(30):
            state = ring()
            for node in state["nodes"]:
                node["load"] = rng.random()
            engine = Engine(state)
            initial_mass = mass(state)
            previous_energy = energy(state)
            for _ in range(12):
                current_energy = engine.step()
                self.assertLessEqual(
                    current_energy,
                    previous_energy + 1e-10,
                )
                self.assertAlmostEqual(
                    mass(engine.state),
                    initial_mass,
                )
                previous_energy = current_energy

    def test_observability_boundary(self):
        result = self.report["scenarios"]["unobservable_failure"]
        self.assertFalse(result["detection"]["detected"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("simulate", "verify", "replay", "test", "write-rust"),
    )
    parser.add_argument("--output", default="results")
    parser.add_argument("--file")
    parser.add_argument("--seed", type=int, default=1599)
    parser.add_argument("--sweeps", type=int, default=240)
    args = parser.parse_args()
    if args.command == "test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.command == "verify":
        print(json.dumps(research_gates(ring()), indent=2))
        return
    if args.command == "write-rust":
        root = pathlib.Path("rust")
        (root / "src").mkdir(parents=True, exist_ok=True)
        (root / "Cargo.toml").write_text(RUST_CARGO, encoding="utf-8")
        (root / "src" / "lib.rs").write_text(
            RUST_SOURCE,
            encoding="utf-8",
        )
        print("cargo test --manifest-path rust/Cargo.toml")
        print("cargo build --release --manifest-path rust/Cargo.toml")
        return
    if args.command == "replay":
        if not args.file:
            parser.error("replay requires --file")
        logs = json.loads(pathlib.Path(args.file).read_text())
        for name, log in logs.items():
            replay(
                log["initial"],
                log["receipts"],
                expected_policy=Policy(**log["policy"]),
                expected_root=log["root"],
            )
            print(name + ": verified")
        return
    if args.sweeps < 0:
        parser.error("--sweeps must be nonnegative")
    report, logs = simulate(args.seed, args.sweeps)
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "simulation.json").write_bytes(canonical(report))
    (output / "receipts.json").write_bytes(canonical(logs))
    summary = {
        name: {
            key: value
            for key, value in scenario.items()
            if key not in ("trajectory", "final_utilization")
        }
        for name, scenario in report["scenarios"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
