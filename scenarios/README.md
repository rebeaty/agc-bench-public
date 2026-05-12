# `scenarios/` - HELM dataset loaders

Each file defines one HELM `Scenario` for an AGC-Bench benchmark. A scenario is
responsible for loading or constructing benchmark instances, formatting inputs,
and attaching references or metadata needed by metrics.

The companion run definition for each scenario is in `run_specs/`, and the
paper-vs-implementation audit for each benchmark is in `audit/fidelity/`.
