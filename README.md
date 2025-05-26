# Dynamic Language Model

A spiking neural network model for encoding letter sequences using NEST 3.8, trained on the TinyShakespeare dataset. The model uses STDP to learn letter transitions, with 100% synaptic connectivity and optimized memory usage.

## Features
- Processes up to 900,000 letter tokens.
- 5 ms spike spacing, 1–2 spikes per letter.
- Low RAM usage (~20–30% of 7.7 GiB).
- Generates raster plots for spike visualization.

## Requirements
- Python 3.13
- NEST 3.8
- Conda environment
- Libraries: `numpy`, `matplotlib`, `datasets`

## Installation
```bash
conda create -n nest python=3.13
conda activate nest
conda install nest-simulator=3.8 numpy matplotlib datasets
Usage

bash
python letterSTDPNetwork.py > output.log

Output
Logs: logs/letter_detections_*.txt (spikes, weights).
Plots: letter_raster_*.png (spike raster and timeline).

License
MIT License

Author
Rob Freeman
