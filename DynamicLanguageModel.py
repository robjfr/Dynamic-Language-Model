import nest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from datasets import load_dataset
import logging
import os
import time as tm
import string
import pickle

# Setup logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(
    filename=os.path.join(log_dir, f"letter_detections_{int(tm.time())}.txt"),
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    filemode="w"
)
logger = logging.getLogger()

# Letter-to-neuron mapping
letter_to_neuron = {letter: i for i, letter in enumerate(string.ascii_lowercase)}
neuron_to_letter = {i: letter for letter, i in letter_to_neuron.items()}
num_neurons = len(letter_to_neuron)
print(f"Number of neurons: {num_neurons}")

# Load TinyShakespeare dataset
dataset = load_dataset("tiny_shakespeare", split="train", trust_remote_code=True)
text_data = "".join([sample["text"] for sample in dataset]).lower()
letters = [ch for ch in text_data if ch in letter_to_neuron][:900000]
print(f"Total letters processed: {len(letters)}")
print(f"Unique letters: {len(set(letters))}")
logger.info(f"First 20 letters: {letters[:20]}")

# NEST setup
nest.ResetKernel()
nest.SetKernelStatus({
    "resolution": 0.1,
    "min_delay": 0.1,
    "max_delay": 10.0,
    "overwrite_files": True,
    "print_time": False,
    "rng_seed": int(tm.time() * 1000) % (2**32)
})

# Create neurons
neurons = nest.Create("iaf_psc_alpha", num_neurons, params={
    "V_th": -60.0,
    "V_reset": -70.0,
    "t_ref": 3.0,
    "V_m": -70.0,
    "E_L": -70.0,
    "C_m": 250.0,
    "tau_m": 10.0
})
print(f"Neurons type: {type(neurons)}, Content: {neurons}")
logger.info(f"Neurons: {neurons}")

# Create spike generators
generators = nest.Create("spike_generator", num_neurons)

# Connect generators to neurons
for i in range(num_neurons):
    nest.Connect(
        generators[i:i+1],
        neurons[i:i+1],
        "one_to_one",
        syn_spec={"weight": 1000.0, "delay": 0.1}
    )
print("Created generator connections")

# Setup STDP synapses (100% connectivity)
nest.SetDefaults("stdp_synapse", {
    "Wmax": 100.0,
    "lambda": 0.002,
    "tau_plus": 20.0
})
for i in range(num_neurons):
    for j in range(num_neurons):
        if i != j:
            nest.Connect(
                neurons[i:i+1],
                neurons[j:j+1],
                "one_to_one",
                syn_spec={
                    "synapse_model": "stdp_synapse",
                    "weight": 1.0 if np.random.random() < 0.5 else -1.0,
                    "delay": 1.0
                }
            )
print("Created STDP connections")
logger.info(f"STDP connections: {len(nest.GetConnections(neurons, neurons))}")

# Assign spike times (5 ms spacing, 1.5 ms offset)
spike_times = defaultdict(list)
resolution = nest.resolution
for i, letter in enumerate(letters):
    neuron_id = letter_to_neuron[letter]
    t_rounded = round((i * 5.0 + 1.5) / resolution) * resolution
    spike_times[neuron_id].append(t_rounded)
    if i < 10:
        print(f"Letter {i}: {letter}, Time {t_rounded:.1f} ms, Neuron {neuron_id}")

# Set spike times and clear memory
expected_spikes = 0
for j in range(num_neurons):
    times = spike_times[j]
    valid_times = sorted(set([t for t in times if np.isfinite(t) and t >= resolution]))
    if not valid_times:
        valid_times = [2 * resolution]
    nest.SetStatus(generators[j:j+1], {"spike_times": valid_times})
    expected_spikes += len(valid_times)
    print(f"Neuron {j} ({neuron_to_letter[j]}): {len(valid_times)} spikes")
print(f"Expected spikes: {expected_spikes}")
spike_times.clear()

# Create spike recorder (0–1000 ms)
spike_recorder = nest.Create("spike_recorder", params={
    "record_to": "memory",
    "start": 0.0,
    "stop": 1000.0
})
nest.Connect(neurons, spike_recorder, "all_to_all")
print(f"Neuron-to-recorder connections: {nest.GetConnections(neurons, spike_recorder)}")

# Simulate
sim_time = (len(letters) * 5.0) + 505.0
sim_time = round(sim_time / 1.0) * 1.0
logger.info(f"Simulation time: {sim_time} ms")
nest.Simulate(sim_time)

# Process spikes
status = nest.GetStatus(spike_recorder)[0]
n_events = status.get("n_events", 0)
if n_events == 0:
    print("Warning: No spikes recorded in spike_recorder")
events = status.get("events", {"times": [], "senders": []})
total_spikes = len(events["times"])
print(f"Total spikes recorded (0-1000 ms): {total_spikes}")
print(f"Sample events: {events['times'][:10]}, {events['senders'][:10]}")
logger.info(f"Events data: times={events['times'][:10]}, senders={events['senders'][:10]}")

# Plot raster and timeline
all_spikes = [(time, sender - 1) for time, sender in zip(events["times"], events["senders"]) if 1 <= sender <= num_neurons]
all_spikes.sort(key=lambda x: x[0])
letter_time_labels = [(neuron_to_letter[n], t) for t, n in all_spikes]
logger.info(f"First 20 spikes: {all_spikes[:20]}")
logger.info(f"First 20 letter labels: {letter_time_labels[:20]}")

if len(events["times"]) > 0:
    fig, (ax_raster, ax_timeline) = plt.subplots(2, 1, figsize=(15, 7),
                                                gridspec_kw={'height_ratios': [8, 1]}, sharex=True)
    spike_times_by_neuron = [[] for _ in range(num_neurons)]
    for t, n in all_spikes:
        spike_times_by_neuron[n].append(t)
    ax_raster.eventplot(spike_times_by_neuron, colors='black', linelengths=0.8)
    ax_raster.set_ylabel("Neuron (letter)")
    ax_raster.set_yticks(list(neuron_to_letter.keys()))
    ax_raster.set_yticklabels(list(neuron_to_letter.values()))
    ax_raster.set_title("Spike Raster Encoding of TinyShakespeare Letters (5 ms spacing)")
    ax_raster.grid(True, linestyle='--', alpha=0.3)
    for letter, time in letter_time_labels:
        ax_timeline.text(time, 0.5, letter, ha='center', va='center', fontsize=8)
    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_yticks([])
    ax_timeline.set_xlabel("Time (ms)")
    ax_timeline.set_xlim(0, 800)
    ax_timeline.set_title("Letter Timeline")
    plot_filename = f"letter_raster_{int(tm.time())}.png"
    try:
        plt.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        logger.info(f"Plot error: {e}")
    plt.close()
else:
    print("No spikes to plot")
all_spikes.clear()
letter_time_labels.clear()

# Save weights
connections = nest.GetConnections(neurons, neurons, synapse_model="stdp_synapse")
weights = nest.GetStatus(connections, "weight")
source_target_pairs = [(c.source, c.target) for c in connections]
weight_dict = {(src, tgt): w for (src, tgt), w in zip(source_target_pairs, weights)}
with open("weights_final.pkl", "wb") as f:
    pickle.dump(weight_dict, f)
print(f"Saved weights to weights_final.pkl")
logger.info(f"Saved weights: {list(weight_dict.items())[:50]}")

# Process STDP weights
logger.info(f"Number of STDP connections: {len(connections)}")
weight_changes = []
for (source, target), weight in zip(source_target_pairs, weights):
    if abs(weight) > 1.1:
        weight_changes.append((neuron_to_letter[source-1], neuron_to_letter[target-1], weight))
print(f"Number of significant weight changes: {len(weight_changes)}")
print(f"Sample weight changes: {weight_changes[:10]}")
logger.info(f"Weight changes: {weight_changes[:50]}")
