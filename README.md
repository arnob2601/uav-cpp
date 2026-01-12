# Underwater Autonomous Vehicle Coverage Path Planning
A project exploring path planning for underwater autonomous vehicles facing drift and localization errors.

## Features
- Lawnmower path planning
- Drift simulation
- Localization error simulation
- Coverage visualization
- TODOs:
    - [ ] Add localization action to correct error
    - [ ] Test on Linux

## Installation
The project requires `Python 3` and uses `uv` for dependency management.

Steps:
1. Install `uv` if not already installed: guide [here](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the repository
`git clone git@github.com:arnob2601/uav-cpp.git`

### Windows
`Chocolatey` first and then using that `GNU Make` need to be installed for windows powershell for running the makefile.

## Usage
Run the simulation
`uv run scripts/simulation.py`

### Sample Outputs

| Considering perfect localization and no drift | Considering drift and localization error |
|--------------|---------------|
| ![](assets/coverage_no_drift.png) | ![](assets/coverage_with_drift.png) |


Run tests:
```bash
make test
```

By default, tests are run in parallel using all available cores. To specify the number of workers, use the `N` argument:
```bash
make test N=4
```