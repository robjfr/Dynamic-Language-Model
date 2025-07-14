import nest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, CheckButtons, TextBox, Slider
from matplotlib.patches import Patch
from collections import defaultdict

class InteractiveNeuralSimulation:
    def __init__(self):
        self.show_synapses = False
        self.show_active_synapses_only = True
        self.min_two_incoming = False
        self.synapse_threshold = 0.1
        self.current_weight_scale = 1.0
        self.e_to_i_weight = 200.0
        self.i_to_e_weight = -400.0
        self.e_to_i_connections = None
        self.i_to_e_connections = None
        
        # Poisson input parameters - lower values since it's just supplemental
        self.base_poisson_rate = 50.0   # Lower starting rate
        self.poisson_increment = 25.0   # Smaller increment
        self.max_poisson_rate = 300.0  # Maximum rate in Hz
        self.min_spikes_criterion = 1   # Minimum spikes required from phoneme group
        
        self.setup_network()
        self.setup_visualization()
        self.current_trial = 0
        self.is_paused = True
        self.trial_data = []
        self.weight_history = []
        self.current_weights = None
        self.phoneme_indices = [0]
        self.last_valid_phoneme_indices = [0]
        self.trials_executed = False

    def setup_network(self):
        """Initialize the neural network"""
        nest.ResetKernel()
        
        self.phoneme_dict = {
            "hello": ["HH", "AH", "L", "OW"],
            "world": ["W", "ER", "L", "D"],
            "test": ["T", "EH", "S", "T"],
            "spike": ["S", "P", "AY", "K"],
            "neuron": ["N", "UH", "R", "AA", "N"]
        }
        
        word = input("Enter a word: ").lower()
        self.word = word
        self.phonemes = [p for p in self.phoneme_dict.get(word, list(word.upper())) if p != " "]
        self.n_phonemes = len(self.phonemes)
        
        self.n_hidden = 600
        #self.sim_time = 500.0
        self.sim_time = 300.0 # sim doesn't continue after stimulus at this point
        self.n_trials = 50
        self.phoneme_interval = 10.0
        self.fanout = 40
        
        # Create Poisson generators for each hidden neuron
        self.poisson_generators = nest.Create('poisson_generator', self.n_hidden)
        for pg in self.poisson_generators:
            pg.rate = 0.0  # Start with no input
        
        self.phoneme_spikegens = nest.Create('spike_generator', self.n_phonemes)
        self.phoneme_parrots = nest.Create('parrot_neuron', self.n_phonemes)
        nest.Connect(self.phoneme_spikegens, self.phoneme_parrots, 'one_to_one')
        
        self.hidden = nest.Create('iaf_psc_alpha', self.n_hidden)
        self.hidden.tau_m = 20.0
        self.hidden.t_ref = 2.0
        
        # Connect Poisson generators to hidden neurons
        nest.Connect(self.poisson_generators, self.hidden, 'one_to_one', 
                    syn_spec={'weight': 100.0, 'delay': 1.0})
        
        # Add inhibitory neurons
        self.n_inhibitory = int(self.n_hidden * 0.2)
        self.inhibitory = nest.Create('iaf_psc_alpha', self.n_inhibitory)
        self.inhibitory.tau_m = 10.0
        self.inhibitory.t_ref = 1.0
        self.inhibitory.V_th = -50.0
        
        self.setup_connectivity()
        self.setup_inhibitory_connections()
        
        self.rec_phoneme = nest.Create('spike_recorder')
        self.rec_hidden = nest.Create('spike_recorder')
        self.rec_inhibitory = nest.Create('spike_recorder')
        nest.Connect(self.phoneme_parrots, self.rec_phoneme)
        nest.Connect(self.hidden, self.rec_hidden)
        nest.Connect(self.inhibitory, self.rec_inhibitory)
        
    def setup_connectivity(self):
        """Setup network connectivity"""
        self.unique_phonemes = sorted(set(self.phonemes))
        n_unique_phonemes = len(self.unique_phonemes)
        
        self.phoneme_to_hidden_group = {}
        target_ids = self.hidden.get('global_id')
        
        for i, phoneme in enumerate(self.unique_phonemes):
            start = i * self.fanout
            end = start + self.fanout
            gids = target_ids[start:end]
            self.phoneme_to_hidden_group[phoneme] = nest.NodeCollection(gids)
        
        self.group_map = {}
        for phoneme, gids in self.phoneme_to_hidden_group.items():
            for gid in gids.get('global_id'):
                self.group_map[gid] = phoneme
        
        self.setup_feedforward_connections()
        self.setup_recurrent_connections()
        
    def setup_feedforward_connections(self):
        """Setup feedforward connections from phonemes to hidden neurons"""
        source_ids = self.phoneme_parrots.get('global_id')
        
        sources = []
        targets = []
        delays = []
        self.feedforward_phoneme_map = []
        
        for i, phoneme in enumerate(self.phonemes):
            src_gid = source_ids[i]
            tgt_gids = self.phoneme_to_hidden_group[phoneme].get('global_id')
            
            for tgt_gid in tgt_gids:
                delay = np.random.uniform(1.0, 10.0)
                sources.append(src_gid)
                targets.append(tgt_gid)
                delays.append(delay)
                self.feedforward_phoneme_map.append(i)
        
        # Start with strong feedforward weights - they're the primary drivers
        weights = np.clip(np.random.normal(loc=800.0, scale=100.0, size=len(sources)), 0.0, 1000.0)
        
        nest.Connect(
            sources, targets,
            conn_spec={"rule": "one_to_one"},
            syn_spec={"weight": weights, "delay": delays}
        )
        
        self.feedforward_connections = nest.GetConnections(
            source=self.phoneme_parrots,
            target=self.hidden
        )
        
    def setup_recurrent_connections(self):
        """Setup recurrent connections between hidden neurons"""
        nest.CopyModel("stdp_synapse", "recurrent_stdp", {
            "weight": 50.0,
            "Wmax": 100.0,
            "delay": 1.0,
            "tau_plus": 20.0,
            "mu_plus": 0.0,
            "mu_minus": 0.0,
            "lambda": 0.05
        })
        
        group_size = self.fanout
        groups = []
        for phoneme in self.unique_phonemes:
            gids = self.phoneme_to_hidden_group[phoneme]
            groups.append(gids[:group_size])
        
        subset_size = 15
        sources_ex = []
        targets_ex = []
        
        for i in range(len(groups)):
            for j in range(len(groups)):
                tgt_group = groups[j]
                src_group = groups[i]
                
                tgt_subset = np.random.choice(tgt_group.get('global_id'), size=subset_size, replace=False)
                src_subset = np.random.choice(src_group.get('global_id'), size=group_size // 2, replace=False)
                
                for src in src_subset:
                    for tgt in tgt_subset:
                        sources_ex.append(src)
                        targets_ex.append(tgt)
        
        sources_ex = np.array(sources_ex)
        targets_ex = np.array(targets_ex)
        autapse_mask = sources_ex != targets_ex
        self.sources_ex = sources_ex[autapse_mask]
        self.targets_ex = targets_ex[autapse_mask]
        
        initial_weights = np.clip(np.random.uniform(50.0, 100.0, size=len(self.sources_ex)), 0.0, 200.0)
        initial_delays = np.random.uniform(1.0, 2.0, size=len(self.sources_ex))
        
        nest.Connect(
            self.sources_ex, self.targets_ex,
            conn_spec={"rule": "one_to_one"},
            syn_spec={
                "synapse_model": "recurrent_stdp",
                "weight": initial_weights,
                "delay": initial_delays
            }
        )
        
        self.recurrent_connections = nest.GetConnections(
            source=nest.NodeCollection(sorted(set(self.sources_ex))),
            target=nest.NodeCollection(sorted(set(self.targets_ex))),
            synapse_model="recurrent_stdp"
        )

    def setup_inhibitory_connections(self):
        """Setup E→I and I→E connections"""
        nest.Connect(
            self.hidden, self.inhibitory,
            conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.2},
            syn_spec={'weight': self.e_to_i_weight, 'delay': 1.0}
        )
        
        nest.Connect(
            self.inhibitory, self.hidden,
            conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.4},
            syn_spec={'weight': self.i_to_e_weight, 'delay': 1.0}
        )
        
        self.e_to_i_connections = nest.GetConnections(self.hidden, self.inhibitory)
        self.i_to_e_connections = nest.GetConnections(self.inhibitory, self.hidden)

    def get_active_synapses(self, spike_times, spike_senders, time_window=20.0):
        """Identify synapses that are active based on spike timing and weight threshold"""
        active_synapses = []
        
        conn_data = nest.GetStatus(self.recurrent_connections, 
                                   keys=['source', 'target', 'weight', 'delay'])
        
        spike_lookup = defaultdict(list)
        for time, sender in zip(spike_times, spike_senders):
            spike_lookup[sender].append(time)
        
        for conn_tuple in conn_data:
            source_gid, target_gid, weight, delay = conn_tuple
            
            if weight < self.synapse_threshold:
                continue
                
            source_spikes = spike_lookup.get(source_gid, [])
            target_spikes = spike_lookup.get(target_gid, [])
            
            for src_time in source_spikes:
                expected_arrival = src_time + delay
                for tgt_time in target_spikes:
                    if expected_arrival <= tgt_time <= expected_arrival + time_window:
                        active_synapses.append({
                            'source': source_gid,
                            'target': target_gid,
                            'weight': weight,
                            'delay': delay,
                            'pre_time': src_time,
                            'post_time': tgt_time,
                            'transmission_time': expected_arrival
                        })
                        break
        
        return active_synapses

    def setup_visualization(self):
        """Setup the interactive visualization"""
        self.fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 4, height_ratios=[2, 1], width_ratios=[3, 1, 1, 1], 
                               hspace=0.3, wspace=0.3)
        
        self.ax_raster = self.fig.add_subplot(gs[0, 0:2])
        self.ax_phoneme = self.fig.add_subplot(gs[0, 2])
        self.ax_weights = self.fig.add_subplot(gs[1, 0])
        self.ax_feedforward_weights = self.fig.add_subplot(gs[1, 1])
        self.ax_poisson_rates = self.fig.add_subplot(gs[1, 2])
        
        self.ax_controls = self.fig.add_subplot(gs[:, 3])
        self.ax_controls.set_xlim(0, 1)
        self.ax_controls.set_ylim(0, 1)
        self.ax_controls.axis('off')
        
        self.setup_controls()
        
        self.phoneme_color_map = {
            label: cm.get_cmap('tab10')(i) 
            for i, label in enumerate(self.unique_phonemes)
        }
        self.phoneme_index_colors = {
            i: cm.get_cmap('tab10')(self.unique_phonemes.index(self.phonemes[i]) % 10)
            for i in range(self.n_phonemes)
        }
        self.hidden_gid_to_phoneme = {
            gid: phoneme for phoneme, group in self.phoneme_to_hidden_group.items()
            for gid in group.get('global_id')
        }





    def setup_controls(self):
        """Setup control widgets"""
        ax_play = plt.axes([0.82, 0.85, 0.15, 0.04])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.toggle_play)
        
        ax_step_back = plt.axes([0.82, 0.80, 0.07, 0.04])
        self.btn_step_back = Button(ax_step_back, '◀')
        self.btn_step_back.on_clicked(self.step_back)
        
        ax_step_forward = plt.axes([0.90, 0.80, 0.07, 0.04])
        self.btn_step_forward = Button(ax_step_forward, '▶')
        self.btn_step_forward.on_clicked(self.step_forward)
        
        ax_run_trials = plt.axes([0.82, 0.75, 0.15, 0.04])
        self.btn_run_trials = Button(ax_run_trials, 'Run Trial 1')
        self.btn_run_trials.on_clicked(self.run_trials_button_clicked)
        
        # Add trial info text
        ax_trial_info = plt.axes([0.82, 0.70, 0.15, 0.04])
        ax_trial_info.axis('off')
        self.trial_info_text = ax_trial_info.text(0.5, 0.5, 'Trials: 0/10', 
                                                  ha='center', va='center',
                                                  transform=ax_trial_info.transAxes,
                                                  fontsize=10, fontweight='bold')
        
        # Feedforward weight control
        ax_feedforward_weight = plt.axes([0.82, 0.65, 0.15, 0.04])
        self.feedforward_weight_slider = Slider(ax_feedforward_weight, 'FF Weight', 0.0, 1000.0, 
                                                 valinit=800.0)
        self.feedforward_weight_slider.on_changed(self.on_feedforward_weight_changed)
        
        ax_phoneme_index = plt.axes([0.82, 0.60, 0.15, 0.04])
        self.phoneme_index_textbox = TextBox(ax_phoneme_index, 'Phoneme Idx', initial='0')
        self.phoneme_index_textbox.on_submit(self.update_phoneme_index)
        
        # Poisson rate controls
        ax_base_rate = plt.axes([0.82, 0.55, 0.15, 0.04])
        self.base_rate_slider = Slider(ax_base_rate, 'Poisson Base', 0.0, 200.0, 
                                       valinit=self.base_poisson_rate)
        self.base_rate_slider.on_changed(self.update_base_rate)
        
        ax_rate_increment = plt.axes([0.82, 0.50, 0.15, 0.04])
        self.rate_increment_slider = Slider(ax_rate_increment, 'Poisson Inc', 0.0, 100.0, 
                                           valinit=self.poisson_increment)
        self.rate_increment_slider.on_changed(self.update_rate_increment)
        
        # Add minimum spikes criterion slider
        ax_min_spikes = plt.axes([0.82, 0.45, 0.15, 0.04])
        self.min_spikes_slider = Slider(ax_min_spikes, 'Min Spikes', 1, 10, 
                                       valinit=self.min_spikes_criterion, valfmt='%d')
        self.min_spikes_slider.on_changed(self.update_min_spikes_criterion)
        
        ax_weight_plus = plt.axes([0.82, 0.35, 0.07, 0.04])
        self.btn_weight_plus = Button(ax_weight_plus, 'Weight +')
        self.btn_weight_plus.on_clicked(lambda event: self.update_weights(1.2))
        
        ax_weight_minus = plt.axes([0.90, 0.35, 0.07, 0.04])
        self.btn_weight_minus = Button(ax_weight_minus, 'Weight -')
        self.btn_weight_minus.on_clicked(lambda event: self.update_weights(0.8))
        
        ax_weight_scale_text = plt.axes([0.82, 0.30, 0.15, 0.04])
        self.weight_scale_text = TextBox(ax_weight_scale_text, 'Scale', initial='1.00', 
                                       color='0.95', hovercolor='0.95')
        self.weight_scale_text.set_val(f"{self.current_weight_scale:.2f}")
        self.weight_scale_text.text_disp.set_color('black')
        self.weight_scale_text.ax.set_facecolor('white')
        
        ax_syn_threshold = plt.axes([0.82, 0.25, 0.15, 0.04])
        self.synapse_threshold_slider = Slider(ax_syn_threshold, 'Syn Threshold', 0.0, 100.0, 
                                             valinit=self.synapse_threshold)
        self.synapse_threshold_slider.on_changed(self.update_synapse_threshold)
        
        ax_e_to_i = plt.axes([0.82, 0.20, 0.15, 0.04])
        self.e_to_i_slider = Slider(ax_e_to_i, 'E→I Weight', 0.0, 500.0, 
                                   valinit=self.e_to_i_weight)
        self.e_to_i_slider.on_changed(self.update_e_to_i_weight)
        
        ax_i_to_e = plt.axes([0.82, 0.15, 0.15, 0.04])
        self.i_to_e_slider = Slider(ax_i_to_e, 'I→E Weight', -800.0, 0.0, 
                                   valinit=self.i_to_e_weight)
        self.i_to_e_slider.on_changed(self.update_i_to_e_weight)
        
        self.setup_synapse_options()
















    def setup_synapse_options(self):
        """Setup synapse visualization options"""
        labels = ['Show Synapses', 'Active Only', 'Weight Scaling', 'Min 2 Incoming']
        actives = [False, True, False, False]
        
        rax = plt.axes([0.82, 0.05, 0.15, 0.15])
        self.synapse_check = CheckButtons(rax, labels, actives)
        self.synapse_check.on_clicked(self.update_synapse_options)
        
        self.ax_controls.text(0.1, 0.95, 'Controls', fontsize=12, fontweight='bold')
        self.ax_controls.text(0.1, 0.15, 'Synapse Options', fontsize=12, fontweight='bold')

#    def update_base_rate(self, val):
#        """Update base Poisson rate"""
#        self.base_poisson_rate = val
#        print(f"Base Poisson rate updated to: {val} Hz")
#
#    def update_rate_increment(self, val):
#        """Update Poisson rate increment"""
#        self.poisson_increment = val
#        print(f"Poisson rate increment updated to: {val} Hz/ms")

    # Grok 3 suggested unpdate functions
    def update_base_rate(self, val):
        """Update the base Poisson rate for all Poisson generators"""
        self.base_poisson_rate = val
        print(f"Updated base Poisson rate to: {val} Hz")
        # Update Poisson generators if they exist
        if hasattr(self, 'poisson_generators'):
            for pg in self.poisson_generators:
                pg.rate = val
        # Update display if trials have been executed
        if self.trial_data:
            self.update_display()

    def update_rate_increment(self, val):
        """Update the Poisson rate increment"""
        self.poisson_increment = val
        print(f"Updated Poisson rate increment to: {val} Hz")
        # Update display if trials have been executed
        if self.trial_data:
            self.update_display()

    def update_feedforward_weights(self, phoneme_indices, new_weight):
        """Update feedforward weights for a list of phoneme indices"""
        print(f"Updating feedforward weights for phoneme indices {phoneme_indices} to {new_weight}")
        updated = False

        # Always get current weights from NEST to ensure we have the latest state
        current_ff_weights = np.array(nest.GetStatus(self.feedforward_connections, keys="weight"))

        for phoneme_index in phoneme_indices:
            if phoneme_index < 0 or phoneme_index >= self.n_phonemes:
                print(f"Invalid phoneme index: {phoneme_index}. Must be between 0 and {self.n_phonemes-1}.")
                continue

            phoneme = self.phonemes[phoneme_index]
            parrot_gid = self.phoneme_parrots.get('global_id')[phoneme_index]
            parrot = nest.NodeCollection([parrot_gid])

            # Get ALL connections from this phoneme parrot to ANY hidden neuron
            connections = nest.GetConnections(source=parrot, target=self.hidden)

            if connections:
                print(f"Found {len(connections)} connections for phoneme {phoneme} (index {phoneme_index})")
                # Update NEST weights
                nest.SetStatus(connections, {"weight": float(new_weight)})

                # Update the current weights array
                conn_indices = [i for i, idx in enumerate(self.feedforward_phoneme_map) if idx == phoneme_index]
                print(f"Updating {len(conn_indices)} weight entries for phoneme {phoneme}")

                for idx in conn_indices:
                    current_ff_weights[idx] = new_weight

                updated = True
            else:
                print(f"No connections found for phoneme {phoneme} (index {phoneme_index})")

        # Update all trial data if trials have been executed
        if updated and self.trial_data:
            for trial_idx in range(len(self.trial_data)):
                self.trial_data[trial_idx]['feedforward_weights'] = current_ff_weights.copy()
            print(f"Updated feedforward weights in all {len(self.trial_data)} trials")

        # Update the feedforward weights display
        if updated:
            self.update_feedforward_histogram(current_ff_weights)

        # Update full display if trials have been executed
        if self.trials_executed:
            self.update_display()

    def update_phoneme_index(self, text):
        """Update the phoneme indices from the textbox, supporting range input"""
        try:
            if '-' in text:
                start, end = map(int, text.split('-'))
                if start < 0 or end >= self.n_phonemes or start > end:
                    print(f"Invalid range: {text}. Must be between 0 and {self.n_phonemes-1}.")
                    self.phoneme_index_textbox.set_val(f"{self.last_valid_phoneme_indices[0]}" if len(self.last_valid_phoneme_indices) == 1 else f"{self.last_valid_phoneme_indices[0]}-{self.last_valid_phoneme_indices[-1]}")
                    return
                self.phoneme_indices = list(range(start, end + 1))
            else:
                index = int(text)
                if 0 <= index < self.n_phonemes:
                    self.phoneme_indices = [index]
                else:
                    print(f"Invalid phoneme index: {index}. Must be between 0 and {self.n_phonemes-1}.")
                    self.phoneme_index_textbox.set_val(f"{self.last_valid_phoneme_indices[0]}")
                    return
            self.last_valid_phoneme_indices = self.phoneme_indices.copy()
            print(f"Updated phoneme indices: {self.phoneme_indices}")
            # Apply current feedforward weight to new indices
            current_weight = self.feedforward_weight_slider.val
            self.update_feedforward_weights(self.phoneme_indices, current_weight)
        except ValueError:
            print("Invalid input: Phoneme index must be an integer or range (e.g., '2' or '2-5').")
            self.phoneme_index_textbox.set_val(f"{self.last_valid_phoneme_indices[0]}")




    def on_feedforward_weight_changed(self, val):
        """Handle changes to the feedforward weight slider"""
        self.update_feedforward_weights(self.phoneme_indices, val)
        

    def update_feedforward_histogram(self, weights):
        """Update the feedforward weights display as a histogram with position-based stats"""
        self.ax_feedforward_weights.clear()

        # Create histogram
        self.ax_feedforward_weights.hist(weights, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)

        # Add statistics
        mean_weight = np.mean(weights)
        std_weight = np.std(weights)

        # Group weights by position index
        position_weight_data = {}
        for i in range(self.n_phonemes):
            position_weight_data[i] = []

        # Group weights by position using the feedforward_phoneme_map
        for i, weight in enumerate(weights):
            position_index = self.feedforward_phoneme_map[i]
            position_weight_data[position_index].append(weight)

        # Add text overlay with position-specific stats
        text_y = 0.95
        line_height = 0.06
        for pos_idx in range(self.n_phonemes):
            if position_weight_data[pos_idx]:
                phoneme = self.phonemes[pos_idx]
                # Use color based on the phoneme value for visual consistency
                color = self.phoneme_color_map[phoneme]
                pos_mean = np.mean(position_weight_data[pos_idx])
                # Show position index, phoneme, and mean weight
                self.ax_feedforward_weights.text(0.02, text_y, f"Pos {pos_idx} ({phoneme}): μ={pos_mean:.1f}",
                                               transform=self.ax_feedforward_weights.transAxes,
                                               fontsize=9, color=color, fontweight='bold')
                text_y -= line_height

        # Highlight currently selected positions
        if hasattr(self, 'phoneme_indices'):
            selected_info = [f"{i}({self.phonemes[i]})" for i in self.phoneme_indices]
            highlight_text = f"Selected: {', '.join(selected_info)}"
            self.ax_feedforward_weights.text(0.98, 0.02, highlight_text,
                                           ha='right', va='bottom',
                                           fontweight='bold', fontsize=9,
                                           bbox=dict(boxstyle="round,pad=0.3",
                                                   facecolor="yellow", alpha=0.7),
                                           transform=self.ax_feedforward_weights.transAxes)

        self.ax_feedforward_weights.set_title(f"Feedforward Weights by Position (μ={mean_weight:.1f}, σ={std_weight:.1f})")
        self.ax_feedforward_weights.set_xlabel("Weight")
        self.ax_feedforward_weights.set_ylabel("Frequency")

        self.fig.canvas.draw_idle()



    def update_min_spikes_criterion(self, val):
        """Update minimum spikes criterion"""
        self.min_spikes_criterion = int(val)
        print(f"Minimum spikes criterion updated to: {self.min_spikes_criterion}")

    def run_single_trial_with_adaptive_poisson(self, trial):
        """Run a single trial with adaptive Poisson input"""
        print(f"Running trial {trial} with adaptive Poisson input (min spikes criterion: {self.min_spikes_criterion})")
        nest.SetStatus(self.rec_phoneme, {'n_events': 0})
        nest.SetStatus(self.rec_hidden, {'n_events': 0})
        nest.SetStatus(self.rec_inhibitory, {'n_events': 0})
        
        offset = trial * self.sim_time
        
        # Set ALL phoneme spike times BEFORE simulation starts
        for i in range(self.n_phonemes):
            spike_time = offset + i * self.phoneme_interval + 5.0
            self.phoneme_spikegens[i].spike_times = [spike_time]
            print(f"Set phoneme {i} ({self.phonemes[i]}) to spike at time {spike_time}")
        
        # Track which phoneme POSITIONS have spiked (by index, not by phoneme string)
        phoneme_position_spike_times = {}
        # Track spike counts for each position
        phoneme_position_spike_counts = defaultdict(int)
        poisson_rate_history = []
        
        # Set initial Poisson rates
        current_rate = self.base_poisson_rate
        for pg in self.poisson_generators:
            pg.rate = current_rate
        
        # Track current phoneme being presented
        current_phoneme_idx = -1
        current_phoneme_window_start = 0
        
        # Run simulation in 1ms steps
        for ms in range(int(self.sim_time)):
            current_time = offset + ms
            
            # Check if we're entering a new phoneme window
            for i in range(self.n_phonemes):
                phoneme_time = i * self.phoneme_interval + 5.0
                if ms == int(phoneme_time):
                    current_phoneme_idx = i
                    current_phoneme_window_start = ms
                    # Reset Poisson rate to base when new phoneme starts
                    current_rate = self.base_poisson_rate
                    for pg in self.poisson_generators:
                        pg.rate = current_rate
                    print(f"Starting window for phoneme position {i} ({self.phonemes[i]}) at ms {ms}")
            
            # Simulate 1ms
            nest.Simulate(1.0)
            
            # Only process if we're in a phoneme window
            if current_phoneme_idx >= 0:
                current_phoneme = self.phonemes[current_phoneme_idx]
                
                # Check if the current phoneme position has enough spikes
                hidden_events = self.rec_hidden.events
                if len(hidden_events['times']) > 0:
                    # Get spikes from this millisecond
                    recent_spikes_mask = hidden_events['times'] > current_time
                    recent_senders = hidden_events['senders'][recent_spikes_mask]
                    
                    # Count spikes from current phoneme group
                    for sender in recent_senders:
                        phoneme_group = self.hidden_gid_to_phoneme.get(sender)
                        if phoneme_group == current_phoneme:
                            phoneme_position_spike_counts[current_phoneme_idx] += 1
                            
                            # Check if we've reached the criterion
                            if (phoneme_position_spike_counts[current_phoneme_idx] >= self.min_spikes_criterion and 
                                current_phoneme_idx not in phoneme_position_spike_times):
                                phoneme_position_spike_times[current_phoneme_idx] = current_time
                                print(f"Phoneme position {current_phoneme_idx} ({current_phoneme}) reached {self.min_spikes_criterion} spikes at time {current_time}")
                                # Reset to base rate after successful spike criterion met
                                current_rate = self.base_poisson_rate
                                for pg in self.poisson_generators:
                                    pg.rate = current_rate
                                break
                
                # Only increase Poisson if:
                # 1. Current phoneme position hasn't reached spike criterion yet
                # 2. We're within the time window for this phoneme (before next phoneme)
                next_phoneme_time = ((current_phoneme_idx + 1) * self.phoneme_interval + 5.0 
                                    if current_phoneme_idx < self.n_phonemes - 1 
                                    else self.sim_time)
                
                if (current_phoneme_idx not in phoneme_position_spike_times and 
                    ms < int(next_phoneme_time) - 1):  # Leave 1ms buffer before next phoneme
                    # Increase rate only if current phoneme position hasn't fired enough
                    current_rate = min(self.max_poisson_rate, current_rate + self.poisson_increment)
                    for pg in self.poisson_generators:
                        pg.rate = current_rate
            
            poisson_rate_history.append((current_time, current_rate))
        
        # Get final results
        recurrent_weights = nest.GetStatus(self.recurrent_connections, keys="weight")
        feedforward_weights = nest.GetStatus(self.feedforward_connections, keys="weight")
        
        trial_data = {
            'trial': trial,
            'hidden_times': self.rec_hidden.events['times'].copy(),
            'hidden_senders': self.rec_hidden.events['senders'].copy(),
            'phoneme_times': self.rec_phoneme.events['times'].copy(),
            'phoneme_senders': self.rec_phoneme.events['senders'].copy(),
            'inhibitory_times': self.rec_inhibitory.events['times'].copy(),
            'inhibitory_senders': self.rec_inhibitory.events['senders'].copy(),
            'recurrent_weights': np.array(recurrent_weights),
            'feedforward_weights': np.array(feedforward_weights),
            'poisson_rate_history': poisson_rate_history,
            'simulated': True
        }
        
        print(f"Trial {trial}: {len(trial_data['hidden_times'])} hidden spikes")
        print(f"Phoneme positions that reached criterion: {sorted(phoneme_position_spike_times.keys())}")
        print(f"Final spike counts by position: {dict(phoneme_position_spike_counts)}")
        return trial_data







    def run_trials_button_clicked(self, event):
        """Handle the Run Trials button click - runs ONE trial"""
        if len(self.trial_data) < self.n_trials:
            trial_num = len(self.trial_data)
            print(f"Running trial {trial_num + 1}/{self.n_trials} with adaptive Poisson input...")
            
            # Run a single trial
            trial_data = self.run_single_trial_with_adaptive_poisson(trial_num)
            self.trial_data.append(trial_data)
            
            # Store weight history
            self.weight_history.append({
                'trial': trial_num,
                'recurrent_weights': trial_data['recurrent_weights'].copy(),
                'feedforward_weights': trial_data['feedforward_weights'].copy()
            })
            
            # Update current trial to show the new one
            self.current_trial = trial_num
            self.trials_executed = True
            
            # Update button text
            if len(self.trial_data) >= self.n_trials:
                self.btn_run_trials.label.set_text('Reset All')
            else:
                self.btn_run_trials.label.set_text(f'Run Trial {len(self.trial_data) + 1}')
            
            self.update_display()
        else:
            # Reset everything
            print("Resetting all trials...")
            self.trial_data = []
            self.weight_history = []
            self.current_trial = 0
            self.trials_executed = False
            self.reset_simulation()
            self.btn_run_trials.label.set_text('Run Trial 1')
            self.show_no_trials_message()

    def reset_simulation(self):
        """Reset the NEST simulation to initial state"""
        print("Resetting simulation...")
        nest.SetStatus(self.rec_phoneme, {'n_events': 0})
        nest.SetStatus(self.rec_hidden, {'n_events': 0})
        nest.SetStatus(self.rec_inhibitory, {'n_events': 0})
        
        nest.ResetKernel()
        self.setup_network()
        
        # Reapply current settings
        current_e_to_i = self.e_to_i_slider.val
        current_i_to_e = self.i_to_e_slider.val
        
        if self.e_to_i_connections:
            nest.SetStatus(self.e_to_i_connections, {'weight': float(current_e_to_i)})
        if self.i_to_e_connections:
            nest.SetStatus(self.i_to_e_connections, {'weight': float(current_i_to_e)})

    def run_all_trials(self):
        """This method is no longer used - kept for compatibility"""
        print("Note: Trials now run one at a time. Use the 'Run Trial' button.")

    def update_poisson_rate_plot(self, trial_data):
        """Update the Poisson rate history plot"""
        self.ax_poisson_rates.clear()
        
        if 'poisson_rate_history' in trial_data:
            times, rates = zip(*trial_data['poisson_rate_history'])
            self.ax_poisson_rates.plot(times, rates, 'b-', linewidth=1)
            self.ax_poisson_rates.set_title(f"Poisson Rate History (Trial {self.current_trial + 1})")
            self.ax_poisson_rates.set_xlabel("Time (ms)")
            self.ax_poisson_rates.set_ylabel("Rate (Hz)")
            self.ax_poisson_rates.grid(True, alpha=0.3)
            
            # Add phoneme markers
            offset = self.current_trial * self.sim_time
            for i in range(self.n_phonemes):
                phoneme_time = offset + i * self.phoneme_interval + 5.0
                self.ax_poisson_rates.axvline(x=phoneme_time, color='red', 
                                             linestyle='--', alpha=0.5)
                self.ax_poisson_rates.text(phoneme_time, max(rates)*0.9, 
                                         self.phonemes[i], ha='center', 
                                         fontsize=8, color='red')
            
            # Add statistics
            max_rate = max(rates)
            final_rate = rates[-1]
            self.ax_poisson_rates.text(0.98, 0.95, f"Max: {max_rate:.0f} Hz\nFinal: {final_rate:.0f} Hz", 
                                     ha='right', va='top', transform=self.ax_poisson_rates.transAxes,
                                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                                     fontsize=9)

    def update_display(self):
        """Update all visualization components"""
        if not self.trial_data or self.current_trial >= len(self.trial_data):
            self.show_no_trials_message()
            return
            
        current_data = self.trial_data[self.current_trial]
        self.update_raster_plot(current_data)
        self.update_phoneme_plot(current_data)
        self.update_weight_plots(current_data)
        self.update_poisson_rate_plot(current_data)
        
        # Update feedforward weights histogram
        if 'feedforward_weights' in current_data:
            self.update_feedforward_histogram(current_data['feedforward_weights'])
        
        # Update trial info text
        if hasattr(self, 'trial_info_text'):
            self.trial_info_text.set_text(f'Trials: {len(self.trial_data)}/{self.n_trials}')
        
        self.fig.canvas.draw_idle()







    def show_no_trials_message(self):
        """Show a message when no trials have been executed yet"""
        self.ax_raster.clear()
        self.ax_phoneme.clear()
        self.ax_weights.clear()
        self.ax_feedforward_weights.clear()
        self.ax_poisson_rates.clear()
        
        self.ax_raster.text(0.5, 0.5, 
                           f"Neural Simulation with Feedforward + Poisson Input\n" +
                           f"Word: '{self.word}'\n" +
                           f"Phonemes: {self.phonemes}\n\n" +
                           "1. Set feedforward weights for specific phonemes\n" +
                           "2. Adjust recurrent weights with Weight +/- buttons\n" +
                           "3. Adjust Poisson background input if needed\n" +
                           "4. Click 'Run Trial 1' to start\n\n" +
                           "Feedforward connections drive the sequence\n" +
                           "Poisson input provides supplemental activation", 
                           ha='center', va='center', fontsize=12, 
                           transform=self.ax_raster.transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor="lightblue", alpha=0.7))
        
        self.ax_raster.set_title(f"Neural Simulation - Word: '{self.word}'")
        self.ax_raster.set_xlim(0, 1)
        self.ax_raster.set_ylim(0, 1)
        self.ax_raster.axis('off')
        
        self.ax_phoneme.text(0.5, 0.5, "Phoneme Activity\n(Run trial to see data)", 
                            ha='center', va='center', transform=self.ax_phoneme.transAxes)
        self.ax_phoneme.set_title("Phoneme Activity")
        
        # Show current recurrent weights even before trials
        current_recurrent_weights = np.array(nest.GetStatus(self.recurrent_connections, keys="weight"))
        self.update_recurrent_weights_display(current_recurrent_weights)
        
        # Show current feedforward weights even before trials
        current_ff_weights = np.array(nest.GetStatus(self.feedforward_connections, keys="weight"))
        self.update_feedforward_histogram(current_ff_weights)
        
        self.ax_poisson_rates.text(0.5, 0.5, "Poisson Rate History\n(Run trial to see data)", 
                                  ha='center', va='center', transform=self.ax_poisson_rates.transAxes)
        self.ax_poisson_rates.set_title("Poisson Rate History")
        
        # Update trial info text
        if hasattr(self, 'trial_info_text'):
            self.trial_info_text.set_text(f'Trials: {len(self.trial_data)}/{self.n_trials}')
        
        self.fig.canvas.draw_idle()



    def update_recurrent_weights_display(self, weights):
        """Update the recurrent weights histogram display"""
        self.ax_weights.clear()
        self.ax_weights.hist(weights, bins=50, color='skyblue', edgecolor='black')
        self.ax_weights.set_title(f"Recurrent Weights (Scale: {self.current_weight_scale:.2f})")
        self.ax_weights.set_xlabel("Weight")
        self.ax_weights.set_ylabel("Frequency")
        
        # Add statistics
        mean_weight = np.mean(weights)
        std_weight = np.std(weights)
        self.ax_weights.text(0.98, 0.95, f"μ={mean_weight:.1f}\nσ={std_weight:.1f}", 
                           ha='right', va='top', transform=self.ax_weights.transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    def update_e_to_i_weight(self, val):
        """Update excitatory to inhibitory connection weights"""
        self.e_to_i_weight = val
        if self.e_to_i_connections:
            nest.SetStatus(self.e_to_i_connections, {'weight': float(val)})
            print(f"Updated E→I weights to: {val}")
            if self.trial_data:
                self.update_display()

    def update_i_to_e_weight(self, val):
        """Update inhibitory to excitatory connection weights"""
        self.i_to_e_weight = val
        if self.i_to_e_connections:
            nest.SetStatus(self.i_to_e_connections, {'weight': float(val)})
            print(f"Updated I→E weights to: {val}")
            if self.trial_data:
                self.update_display()

    def update_synapse_options(self, label):
        """Update synapse visualization options"""
        if label == 'Show Synapses':
            self.show_synapses = not self.show_synapses
            print(f"Show Synapses toggled: {self.show_synapses}")
        elif label == 'Active Only':
            self.show_active_synapses_only = not self.show_active_synapses_only
            print(f"Active Only toggled: {self.show_active_synapses_only}")
        elif label == 'Weight Scaling':
            self.weight_scaled_lines = not getattr(self, 'weight_scaled_lines', False)
            print(f"Weight Scaling toggled: {self.weight_scaled_lines}")
        elif label == 'Min 2 Incoming':
            self.min_two_incoming = not self.min_two_incoming
            print(f"Min 2 Incoming toggled: {self.min_two_incoming}")
        
        self.update_display()
        
    def update_synapse_threshold(self, val):
        """Update synapse threshold"""
        self.synapse_threshold = val
        print(f"Synapse threshold updated to: {self.synapse_threshold}")
        self.update_display()








    def update_weights(self, scale):
        """Update recurrent connection weights by scaling factor"""
        print(f"Scaling recurrent weights by {scale}")
        self.current_weight_scale *= scale
        
        # Get current weights from NEST
        current_recurrent_weights = np.array(nest.GetStatus(self.recurrent_connections, keys="weight"))
        
        # Scale the weights
        new_recurrent_weights = current_recurrent_weights * scale
        new_recurrent_weights = np.clip(new_recurrent_weights, 0.0, 200.0)
        
        # Update NEST
        weight_dicts = [{"weight": float(w)} for w in new_recurrent_weights]
        nest.SetStatus(self.recurrent_connections, weight_dicts)
        
        # Update any existing trial data
        if self.trial_data and self.current_trial < len(self.trial_data):
            for trial_idx in range(len(self.trial_data)):
                self.trial_data[trial_idx]['recurrent_weights'] = new_recurrent_weights.copy()
        
        self.update_weight_scale_text()
        print(f"Updated recurrent weights. New mean: {np.mean(new_recurrent_weights):.1f}")
        
        # Update display
        if self.trial_data:
            self.update_display()
        else:
            # Update just the recurrent weights display
            self.update_recurrent_weights_display(new_recurrent_weights)
            self.fig.canvas.draw_idle()















    def update_weight_scale_text(self):
        """Update the displayed weight scale value"""
        self.weight_scale_text.set_val(f"{self.current_weight_scale:.2f}")

    def draw_synapses_on_raster(self, spike_times, spike_senders, hidden_id_to_y):
        """Draw synaptic connections on the raster plot"""
        if not self.show_synapses:
            return
            
        print(f"Drawing synapses, show_active_only={self.show_active_synapses_only}")
        
        if self.show_active_synapses_only:
            active_synapses = self.get_active_synapses(spike_times, spike_senders)
            print(f"Found {len(active_synapses)} active synapses")
            
            incoming_count = defaultdict(int)
            for synapse in active_synapses:
                incoming_count[synapse['target']] += 1
            
            filtered_synapses = active_synapses
            if self.min_two_incoming:
                filtered_synapses = [
                    synapse for synapse in active_synapses
                    if incoming_count[synapse['target']] >= 2
                ]
                print(f"Filtered to {len(filtered_synapses)} synapses with min 2 incoming")
            
            plotted_synapses = 0
            for synapse in filtered_synapses:
                src_gid = synapse['source']
                tgt_gid = synapse['target']
                weight = synapse['weight']
                pre_time = synapse['pre_time']
                post_time = synapse['post_time']
                
                if src_gid in hidden_id_to_y and tgt_gid in hidden_id_to_y:
                    src_y = hidden_id_to_y[src_gid]
                    tgt_y = hidden_id_to_y[tgt_gid]
                    
                    alpha = min(weight / 100.0, 1.0)
                    linewidth = 0.5 + (weight / 100.0)
                    
                    self.ax_raster.plot([pre_time, post_time], [src_y, tgt_y], 
                                        'r-', alpha=alpha, linewidth=linewidth)
                    
                    mid_time = (pre_time + post_time) / 2
                    mid_y = (src_y + tgt_y) / 2
                    dx = post_time - pre_time
                    dy = tgt_y - src_y
                    
                    if dx != 0:
                        self.ax_raster.annotate('', xy=(mid_time + dx*0.1, mid_y + dy*0.1), 
                                                xytext=(mid_time, mid_y),
                                                arrowprops=dict(arrowstyle='->', 
                                                                color='red', alpha=alpha,
                                                                lw=0.5))
                    plotted_synapses += 1
            
            print(f"Plotted {plotted_synapses} active synapses")
        else:
            try:
                conn_data = nest.GetStatus(self.recurrent_connections, 
                                           keys=['source', 'target', 'weight'])
                
                trial_start_time = self.current_trial * self.sim_time if hasattr(self, 'current_trial') else 0
                trial_end_time = trial_start_time + self.sim_time
                
                plotted_synapses = 0
                for conn_tuple in conn_data:
                    src_gid, tgt_gid, weight = conn_tuple
                    
                    if weight < self.synapse_threshold:
                        continue
                        
                    if src_gid in hidden_id_to_y and tgt_gid in hidden_id_to_y:
                        src_y = hidden_id_to_y[src_gid]
                        tgt_y = hidden_id_to_y[tgt_gid]
                        
                        alpha = min(weight / 100.0, 0.3)
                        linewidth = 0.3 + (weight / 200.0)
                        color = 'blue' if weight > 75 else 'gray'
                        self.ax_raster.plot([trial_start_time, trial_end_time], [src_y, tgt_y], 
                                            color=color, alpha=alpha, linewidth=linewidth)
                        plotted_synapses += 1
                
                print(f"Plotted {plotted_synapses} static synapses")
            except Exception as e:
                print(f"Error drawing static synapses: {e}")

    def toggle_play(self, event):
        """Toggle play/pause for reviewing existing trials"""
        if not self.trial_data:
            print("No trials have been run yet. Click 'Run Trial 1' first.")
            return
            
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_play.label.set_text('Play')
        else:
            self.btn_play.label.set_text('Pause')
            self.animate_trials()

    def animate_trials(self):
        """Animate through existing trials"""
        if not self.is_paused and self.current_trial < len(self.trial_data) - 1:
            self.step_forward(None)
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(0.5)
            if not self.is_paused:
                self.animate_trials()

    def step_forward(self, event):
        """Step forward to next trial (if it exists)"""
        if not self.trial_data:
            print("No trials have been run yet. Click 'Run Trial 1' first.")
            return
            
        if self.current_trial < len(self.trial_data) - 1:
            self.current_trial += 1
            print(f"Viewing trial {self.current_trial + 1}/{len(self.trial_data)}")
            self.update_display()
        else:
            print(f"Already at last trial ({len(self.trial_data)}). Run more trials to continue.")

    def step_back(self, event):
        """Step back to previous trial"""
        if not self.trial_data:
            print("No trials have been run yet. Click 'Run Trial 1' first.")
            return
            
        if self.current_trial > 0:
            self.current_trial -= 1
            print(f"Viewing trial {self.current_trial + 1}/{len(self.trial_data)}")
            self.update_display()
        else:
            print("Already at first trial.")










    def update_raster_plot(self, trial_data):
        """Update the main raster plot - showing only spiking neurons"""
        self.ax_raster.clear()
        
        print(f"Trial {self.current_trial}: Hidden spikes: {len(trial_data['hidden_times'])}, "
              f"Phoneme spikes: {len(trial_data['phoneme_times'])}, "
              f"Inhibitory spikes: {len(trial_data['inhibitory_times'])}")
        
        phoneme_times = trial_data['phoneme_times']
        phoneme_senders = trial_data['phoneme_senders']
        hidden_times = trial_data['hidden_times']
        hidden_senders = trial_data['hidden_senders']
        inhibitory_times = trial_data['inhibitory_times']
        inhibitory_senders = trial_data['inhibitory_senders']
        
        unique_phoneme_senders = sorted(set(phoneme_senders)) if len(phoneme_senders) > 0 else []
        unique_hidden_senders = sorted(set(hidden_senders)) if len(hidden_senders) > 0 else []
        unique_inhibitory_senders = sorted(set(inhibitory_senders)) if len(inhibitory_senders) > 0 else []
        
        # Sort hidden neurons by their phoneme group order
        # Create ordering based on first appearance of each phoneme in the sequence
        phoneme_order = []
        for phoneme in self.phonemes:
            if phoneme not in phoneme_order:
                phoneme_order.append(phoneme)
        
        # Sort hidden senders by their phoneme group
        def hidden_sort_key(gid):
            phoneme = self.hidden_gid_to_phoneme.get(gid, '')
            if phoneme in phoneme_order:
                return (phoneme_order.index(phoneme), gid)  # Sort by phoneme order, then by GID
            else:
                return (len(phoneme_order), gid)  # Put unknown phonemes at the end
        
        unique_hidden_senders = sorted(unique_hidden_senders, key=hidden_sort_key)
        
        print(f"Unique spiking neurons - Phoneme: {len(unique_phoneme_senders)}, "
              f"Hidden: {len(unique_hidden_senders)}, Inhibitory: {len(unique_inhibitory_senders)}")
        
        current_y = 0
        phoneme_gid_to_y = {}
        hidden_gid_to_y = {}
        inhibitory_gid_to_y = {}
        
        for gid in unique_phoneme_senders:
            phoneme_gid_to_y[gid] = current_y
            current_y += 1
        
        if unique_phoneme_senders and (unique_hidden_senders or unique_inhibitory_senders):
            current_y += 1
        
        # Track where each phoneme group starts for visual separation
        last_phoneme = None
        phoneme_group_separators = []
        
        for gid in unique_hidden_senders:
            hidden_gid_to_y[gid] = current_y
            current_phoneme = self.hidden_gid_to_phoneme.get(gid)
            
            # Add separator line when phoneme group changes
            if last_phoneme is not None and current_phoneme != last_phoneme:
                phoneme_group_separators.append(current_y - 0.5)
            
            last_phoneme = current_phoneme
            current_y += 1
        
        if unique_hidden_senders and unique_inhibitory_senders:
            current_y += 1
        
        for gid in unique_inhibitory_senders:
            inhibitory_gid_to_y[gid] = current_y
            current_y += 1
        
        phoneme_plot_count = 0
        if len(phoneme_times) > 0:
            for time, sender in zip(phoneme_times, phoneme_senders):
                if sender in phoneme_gid_to_y:
                    phoneme_idx = sender - self.phoneme_parrots.get('global_id')[0]
                    if 0 <= phoneme_idx < len(self.phonemes):
                        phoneme_text = self.phonemes[phoneme_idx]
                        color = self.phoneme_index_colors.get(phoneme_idx, 'black')
                        y_pos = phoneme_gid_to_y[sender]
                        
                        self.ax_raster.text(time, y_pos, phoneme_text, 
                                             ha='center', va='center', 
                                             fontsize=10, fontweight='bold',
                                             color=color)
                        phoneme_plot_count += 1

        hidden_plot_count = 0
        if len(hidden_times) > 0:
            for time, sender in zip(hidden_times, hidden_senders):
                if sender in hidden_gid_to_y:
                    y_pos = hidden_gid_to_y[sender]
                    phoneme_label = self.hidden_gid_to_phoneme.get(sender, 'H')
                    color = self.phoneme_color_map.get(phoneme_label, 'black')
                    
                    self.ax_raster.text(time, y_pos, phoneme_label, 
                                         ha='center', va='center', 
                                         fontsize=8, color=color)
                    hidden_plot_count += 1

        inhibitory_plot_count = 0
        if len(inhibitory_times) > 0:
            for time, sender in zip(inhibitory_times, inhibitory_senders):
                if sender in inhibitory_gid_to_y:
                    y_pos = inhibitory_gid_to_y[sender]
                    self.ax_raster.plot(time, y_pos, 'gx', markersize=4)
                    inhibitory_plot_count += 1

        print(f"Plotted - Phoneme: {phoneme_plot_count}, Hidden: {hidden_plot_count}, Inhibitory: {inhibitory_plot_count}")

        if current_y == 0:
            trial_start_time = self.current_trial * self.sim_time
            trial_end_time = trial_start_time + self.sim_time
            self.ax_raster.text((trial_start_time + trial_end_time)/2, 0, "No spikes in this trial", 
                               ha='center', va='center', fontsize=12, color='red')
            self.ax_raster.set_xlim(trial_start_time, trial_end_time)
            self.ax_raster.set_ylim(-0.5, 0.5)
            self.ax_raster.set_xlabel("Time (ms)")
            self.ax_raster.set_ylabel("Spiking Neurons")
            self.ax_raster.set_title(f"Raster Plot - Trial {self.current_trial + 1}/{self.n_trials} (No Activity)")
            return

        trial_start_time = self.current_trial * self.sim_time
        trial_end_time = trial_start_time + self.sim_time
        
        self.ax_raster.set_xlim(trial_start_time, trial_end_time)
        self.ax_raster.set_ylim(-0.5, current_y - 0.5)
        self.ax_raster.set_xlabel("Time (ms)")
        self.ax_raster.set_ylabel("Spiking Neurons")
        self.ax_raster.set_title(f"Raster Plot - Trial {self.current_trial + 1}/{self.n_trials} (Spiking Neurons Only)")

        y_ticks = []
        y_labels = []
        
        for gid in unique_phoneme_senders:
            phoneme_idx = gid - self.phoneme_parrots.get('global_id')[0]
            if 0 <= phoneme_idx < len(self.phonemes):
                y_ticks.append(phoneme_gid_to_y[gid])
                y_labels.append(f"P{phoneme_idx} ({self.phonemes[phoneme_idx]})")
        
        for gid in unique_hidden_senders:
            phoneme_label = self.hidden_gid_to_phoneme.get(gid, 'H')
            y_ticks.append(hidden_gid_to_y[gid])
            y_labels.append(f"H{gid} ({phoneme_label})")
        
        for gid in unique_inhibitory_senders:
            y_ticks.append(inhibitory_gid_to_y[gid])
            y_labels.append(f"I{gid}")
        
        if y_ticks:
            self.ax_raster.set_yticks(y_ticks)
            self.ax_raster.set_yticklabels(y_labels, fontsize=8)
            
            # Add horizontal lines to separate neuron types
            if unique_phoneme_senders and (unique_hidden_senders or unique_inhibitory_senders):
                separator_y = max(phoneme_gid_to_y.values()) + 0.5
                self.ax_raster.axhline(y=separator_y, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            
            # Add separators between phoneme groups
            for sep_y in phoneme_group_separators:
                self.ax_raster.axhline(y=sep_y, color='lightgray', linestyle=':', linewidth=0.5, alpha=0.5)
            
            if unique_hidden_senders and unique_inhibitory_senders:
                separator_y = max(hidden_gid_to_y.values()) + 0.5
                self.ax_raster.axhline(y=separator_y, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

        if len(hidden_times) > 0 and hasattr(self, 'show_synapses') and self.show_synapses:
            print(f"Drawing synapses for {len(hidden_gid_to_y)} spiking hidden neurons")
            self.draw_synapses_on_raster(hidden_times, hidden_senders, hidden_gid_to_y)

        self.fig.canvas.draw_idle()






    def update_phoneme_plot(self, trial_data):
        """Update the phoneme activity plot"""
        self.ax_phoneme.clear()
        
        phoneme_spikes = trial_data['phoneme_senders']
        hidden_spikes = trial_data['hidden_senders']
        
        hidden_spike_phonemes = [self.hidden_gid_to_phoneme.get(gid, None) for gid in hidden_spikes]
        
        phoneme_spike_counts = defaultdict(int)
        for phoneme_label in hidden_spike_phonemes:
            if phoneme_label:
                phoneme_spike_counts[phoneme_label] += 1
        
        labels = self.unique_phonemes
        counts = [phoneme_spike_counts[label] for label in labels]
        colors = [self.phoneme_color_map[label] for label in labels]

        if labels:
            self.ax_phoneme.bar(labels, counts, color=colors)
            self.ax_phoneme.set_ylabel("Spike Count (Hidden Neurons)")
            self.ax_phoneme.set_title(f"Hidden Neuron Activity by Phoneme Group (Trial {self.current_trial + 1})")
            self.ax_phoneme.tick_params(axis='x', rotation=45)
        else:
            self.ax_phoneme.text(0.5, 0.5, "No Hidden Spikes", ha='center', va='center', transform=self.ax_phoneme.transAxes)

    def update_weight_plots(self, trial_data):
        """Update the weight distribution plot"""
        self.ax_weights.clear()
        
        # Recurrent Weights
        recurrent_weights = trial_data['recurrent_weights']
        self.ax_weights.hist(recurrent_weights, bins=50, color='skyblue', edgecolor='black')
        self.ax_weights.set_title(f"Recurrent Weights (Trial {self.current_trial + 1})")
        self.ax_weights.set_xlabel("Weight")
        self.ax_weights.set_ylabel("Frequency")
        
        # Add statistics
        mean_weight = np.mean(recurrent_weights)
        std_weight = np.std(recurrent_weights)
        self.ax_weights.text(0.98, 0.95, f"μ={mean_weight:.1f}\nσ={std_weight:.1f}", 
                           ha='right', va='top', transform=self.ax_weights.transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        self.fig.tight_layout(rect=[0, 0, 0.8, 1])

# Main execution
if __name__ == "__main__":
    sim = InteractiveNeuralSimulation()
    sim.show_no_trials_message()
    plt.show()
