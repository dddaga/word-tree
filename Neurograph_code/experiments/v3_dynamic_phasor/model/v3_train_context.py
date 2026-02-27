"""
V3 Training Context
Complete training orchestrator for Wine Quality with all V3 components.
Forward + backward propagation, auxiliary losses, evaluation metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from core.high_res_tables import HighResolutionLookupTables
from core.node_store import NodeStore
from core.graph import build_static_graph
from core.modular_forward_engine import VectorizedForwardEngine
from core.activation_table import VectorizedActivationTable
from modules.class_encoding import generate_fixed_class_encodings
from modules.classification_loss import ClassificationLoss

from experiments.v3_dynamic_phasor.components.complex_signal import ComplexSignalComputer
from experiments.v3_dynamic_phasor.components.signal_normalization import PhasorNormalization
from experiments.v3_dynamic_phasor.components.radiation_diversity import RadiationDiversityTracker
from experiments.v3_dynamic_phasor.components.load_balancer import RadiationLoadBalancer
from experiments.v3_dynamic_phasor.components.uncertainty import MCDropoutUncertainty
from experiments.v3_dynamic_phasor.model.wine_input_adapter import WineInputAdapter
from experiments.v3_dynamic_phasor.model.v3_phase_cell import V3PhaseCell
from experiments.v3_dynamic_phasor.model.v3_forward_engine import V3ForwardEngine


class V3TrainContext:
    """Complete training context for V3 Dynamic Phasor experiment on Wine Quality."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get("device", "cpu")

        # Feature flags (for ablation)
        self.enable_complex = config.get("complex_activation", {}).get("enabled", True)
        self.enable_normalization = config.get("normalization", {}).get("enabled", True)
        self.enable_diversity = config.get("radiation_diversity", {}).get("enabled", True)
        self.enable_balance = config.get("load_balance", {}).get("enabled", True)
        self.enable_uncertainty = config.get("uncertainty", {}).get("enabled", True)

        self._setup_core()
        self._setup_graph()
        self._setup_input()
        self._setup_output()
        self._setup_v3_components()
        self._setup_forward_engine()

        # Training state
        self.current_epoch = 0
        self.training_losses: List[float] = []
        self.aux_diversity_losses: List[float] = []
        self.aux_balance_losses: List[float] = []
        self.total_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []

        print(f"V3 Training Context initialized")
        print(f"  Complex: {self.enable_complex}, Norm: {self.enable_normalization}")
        print(f"  Diversity: {self.enable_diversity}, Balance: {self.enable_balance}")
        print(f"  Uncertainty: {self.enable_uncertainty}")
        print(f"  Parameters: {self.count_parameters():,}")

    # ------------------------------------------------------------------ setup
    def _setup_core(self) -> None:
        arch = self.config.get("architecture", {})
        res = self.config.get("resolution", {})

        self.phase_bins = res.get("phase_bins", 512)
        self.mag_bins = res.get("mag_bins", 1024)
        self.vector_dim = arch.get("vector_dim", 5)
        self.total_nodes = arch.get("total_nodes", 200)
        self.num_input = arch.get("input_nodes", 22)
        self.num_output = arch.get("output_nodes", 6)

        self.lookup_tables = HighResolutionLookupTables(
            phase_bins=self.phase_bins,
            mag_bins=self.mag_bins,
            device=self.device,
        )

        gamma = self.config.get("complex_activation", {}).get("gamma", 1.0)
        self.phase_cell = V3PhaseCell(
            vector_dim=self.vector_dim,
            lookup_tables=self.lookup_tables,
            gamma=gamma,
            enable_complex=self.enable_complex,
            enable_normalization=self.enable_normalization,
            mag_bins=self.mag_bins,
        )

    def _setup_graph(self) -> None:
        arch = self.config.get("architecture", {})
        gs = self.config.get("graph_structure", {})
        self.graph_df = build_static_graph(
            total_nodes=self.total_nodes,
            num_input_nodes=self.num_input,
            num_output_nodes=self.num_output,
            vector_dim=self.vector_dim,
            phase_bins=self.phase_bins,
            mag_bins=self.mag_bins,
            cardinality=gs.get("cardinality", 4),
            seed=self.config.get("dataset", {}).get("seed", 42),
        )
        self.node_store = NodeStore(
            self.graph_df, self.vector_dim, self.phase_bins, self.mag_bins,
        )
        self.input_nodes = [f"n{i}" for i in range(self.num_input)]
        self.output_nodes = [f"n{i}" for i in range(self.total_nodes - self.num_output, self.total_nodes)]

    def _setup_input(self) -> None:
        ds = self.config.get("dataset", {})
        self.input_adapter = WineInputAdapter(
            input_dim=11,
            num_input_nodes=self.num_input,
            vector_dim=self.vector_dim,
            phase_bins=self.phase_bins,
            mag_bins=self.mag_bins,
            device=self.device,
            test_size=ds.get("test_size", 0.30),
            val_fraction=ds.get("val_fraction", 0.50),
            seed=ds.get("seed", 42),
        )
        self.num_classes = self.input_adapter.num_classes

    def _setup_output(self) -> None:
        self.class_encodings = generate_fixed_class_encodings(
            phase_bins=self.phase_bins,
            mag_bins=self.mag_bins,
            vector_dim=self.vector_dim,
            seed=42,
        )
        self.loss_fn = ClassificationLoss(
            num_classes=self.num_classes,
            temperature=1.0,
            label_smoothing=0.0,
        )

        # Learning rates
        tr = self.config.get("training", {})
        self.phase_lr = tr.get("phase_learning_rate", 0.015)
        self.mag_lr = tr.get("magnitude_learning_rate", 0.012)

    def _setup_v3_components(self) -> None:
        norm_cfg = self.config.get("normalization", {})
        clamp_cfg = norm_cfg.get("magnitude_clamping", {})
        clamp_range = clamp_cfg.get("range", [0.1, 0.9])

        self.normalizer = PhasorNormalization(
            mag_bins=self.mag_bins,
            clamp_low_frac=clamp_range[0],
            clamp_high_frac=clamp_range[1],
        )

        div_cfg = self.config.get("radiation_diversity", {})
        self.diversity_tracker = RadiationDiversityTracker(
            total_nodes=self.total_nodes,
            alpha=div_cfg.get("alpha", 0.01),
        )

        lb_cfg = self.config.get("load_balance", {})
        self.load_balancer = RadiationLoadBalancer(
            total_nodes=self.total_nodes,
            beta=lb_cfg.get("beta", 0.001),
            capacity_limit=lb_cfg.get("capacity_limit", 5),
            temperature_init=lb_cfg.get("temperature_init", 1.0),
            temperature_min=lb_cfg.get("temperature_min", 0.1),
            temperature_anneal_rate=lb_cfg.get("temperature_anneal_rate", 0.995),
        )

        unc_cfg = self.config.get("uncertainty", {})
        self.uncertainty = MCDropoutUncertainty(
            vector_dim=self.vector_dim,
            phase_bins=self.phase_bins,
            mag_bins=self.mag_bins,
            dropout_rate=unc_cfg.get("dropout_rate", 0.1),
            num_mc_samples=unc_cfg.get("num_mc_samples", 10),
        )

    def _setup_forward_engine(self) -> None:
        fp = self.config.get("forward_pass", {})
        base_engine = VectorizedForwardEngine(
            graph_df=self.graph_df,
            node_store=self.node_store,
            phase_cell=self.phase_cell,
            lookup_table=self.lookup_tables,
            max_nodes=self.total_nodes,
            vector_dim=self.vector_dim,
            phase_bins=self.phase_bins,
            mag_bins=self.mag_bins,
            max_timesteps=fp.get("max_timesteps", 30),
            decay_factor=fp.get("decay_factor", 0.6),
            min_strength=fp.get("min_activation_strength", 1.0),
            top_k_neighbors=fp.get("top_k_neighbors", 4),
            radiation_batch_size=fp.get("radiation_batch_size", 128),
            min_output_activation_timesteps=fp.get("min_output_activation_timesteps", 2),
            use_radiation=fp.get("use_radiation", True),
            device=self.device,
            verbose=False,
        )

        self.forward_engine = V3ForwardEngine(
            base_engine=base_engine,
            node_store=self.node_store,
            normalizer=self.normalizer,
            diversity_tracker=self.diversity_tracker,
            load_balancer=self.load_balancer,
            enable_diversity=self.enable_diversity,
            enable_balance=self.enable_balance,
            enable_normalization=self.enable_normalization,
        )

    # ------------------------------------------------------------------ forward
    def forward_pass(self, input_context: Dict) -> Dict[str, torch.Tensor]:
        """Run forward pass and extract output signals."""
        activation_table = self.forward_engine.forward_pass(input_context)

        output_signals = {}
        for node_id in self.output_nodes:
            try:
                phase = self.node_store.get_phase(node_id)
                mag = self.node_store.get_mag(node_id)

                if self.node_store.is_node_active(node_id):
                    signal = self.lookup_tables.get_signal_vector(phase, mag)
                else:
                    # Use null activation if node didn't activate
                    if self.enable_uncertainty:
                        null_p, null_m = self.uncertainty.get_null_activation()
                        signal = self.lookup_tables.get_signal_vector(null_p, null_m)
                    else:
                        signal = torch.zeros(self.vector_dim, device=phase.device)

                output_signals[node_id] = signal
            except (KeyError, AttributeError):
                output_signals[node_id] = torch.zeros(self.vector_dim)

        return output_signals

    def compute_logits(self, output_signals: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute class logits from output signals via cosine similarity."""
        return self.loss_fn.compute_logits_from_signals(
            output_signals, self.class_encodings, self.lookup_tables,
        )

    # ------------------------------------------------------------------ backward
    def backward_pass(
        self, logits: torch.Tensor, target_label: int, output_signals: Dict[str, torch.Tensor],
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute loss and backpropagate gradients to discrete parameters.

        Returns:
            node_gradients: {node_id: (phase_grad, mag_grad)}
        """
        target = torch.tensor([target_label], device=logits.device)
        primary_loss = self.loss_fn(logits, target)

        # Compute upstream gradient from logits
        probs = F.softmax(logits, dim=-1).squeeze(0)
        target_one_hot = F.one_hot(target, self.num_classes).float().squeeze(0)
        logit_grad = probs - target_one_hot  # [num_classes]

        node_gradients = {}
        for i, node_id in enumerate(self.output_nodes):
            if node_id not in output_signals:
                continue
            signal = output_signals[node_id]
            if signal.abs().sum() < 1e-12:
                continue

            # Gradient of logit wrt signal (via cosine similarity)
            # Approximate: use the class encoding that is closest
            class_id = i if i < self.num_classes else 0
            if class_id in self.class_encodings:
                class_phase, class_mag = self.class_encodings[class_id]
                class_signal = self.lookup_tables.get_signal_vector(class_phase, class_mag)

                # d(cosine_sim)/d(signal) approx
                norm_s = signal.norm() + 1e-8
                norm_c = class_signal.norm() + 1e-8
                upstream = logit_grad[class_id] * (class_signal / (norm_s * norm_c))

                phase = self.node_store.get_phase(node_id)
                mag = self.node_store.get_mag(node_id)
                pg, mg = self.lookup_tables.compute_signal_gradients(phase, mag, upstream)
                node_gradients[node_id] = (pg, mg)

        return node_gradients, primary_loss

    def apply_updates(self, node_gradients: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Apply discrete parameter updates from gradients."""
        for node_id, (pg, mg) in node_gradients.items():
            phase_updates, mag_updates = self.lookup_tables.quantize_gradients_to_discrete_updates(
                pg, mg,
                phase_learning_rate=self.phase_lr,
                magnitude_learning_rate=self.mag_lr,
                node_id=node_id,
            )
            current_phase = self.node_store.get_phase(node_id)
            current_mag = self.node_store.get_mag(node_id)
            new_phase, new_mag = self.lookup_tables.apply_discrete_updates(
                current_phase, current_mag, phase_updates, mag_updates,
            )
            self.node_store.phase_table[node_id].data.copy_(new_phase)
            self.node_store.mag_table[node_id].data.copy_(new_mag)

    # ------------------------------------------------------------------ train
    def train_single_sample(self, sample_idx: int, dataset: str = "train") -> Tuple[float, float]:
        """Train on a single sample. Returns (total_loss, accuracy)."""
        # Get input context
        input_context, target_label = self.input_adapter.get_input_context(
            sample_idx, self.input_nodes, dataset=dataset,
        )

        # Apply dropout if enabled
        if self.enable_uncertainty:
            for node_id in input_context:
                p, m = input_context[node_id]
                p, m = self.uncertainty.apply_phase_dropout(p, m, training=True)
                input_context[node_id] = (p, m)

        # Forward
        output_signals = self.forward_pass(input_context)
        logits = self.compute_logits(output_signals)

        # Backward
        node_gradients, primary_loss = self.backward_pass(logits, target_label, output_signals)

        # Apply updates
        self.apply_updates(node_gradients)

        # Compute auxiliary losses
        div_loss = self.diversity_tracker.compute_entropy_loss() if self.enable_diversity else torch.tensor(0.0)
        bal_loss = self.load_balancer.compute_balance_loss() if self.enable_balance else torch.tensor(0.0)

        total_loss = primary_loss.item() + div_loss.item() + bal_loss.item()

        # Accuracy
        pred = torch.argmax(logits.squeeze(0) if logits.dim() > 1 else logits).item()
        accuracy = 1.0 if pred == target_label else 0.0

        return total_loss, accuracy

    def train_epoch(self) -> Tuple[float, float, float, float]:
        """Train for one epoch. Returns (avg_total_loss, avg_primary_loss, avg_div_loss, avg_bal_loss)."""
        num_train = len(self.input_adapter.X_train)
        indices = np.random.permutation(num_train)

        batch_size = self.config.get("training", {}).get("batch_size", 64)
        total_loss_sum = 0.0
        correct = 0
        count = 0

        for i in range(min(batch_size, num_train)):
            idx = int(indices[i])
            loss, acc = self.train_single_sample(idx)
            total_loss_sum += loss
            correct += acc
            count += 1

        avg_loss = total_loss_sum / max(count, 1)
        avg_acc = correct / max(count, 1)

        # Auxiliary losses for logging
        div_loss = self.diversity_tracker.compute_entropy_loss().item() if self.enable_diversity else 0.0
        bal_loss = self.load_balancer.compute_balance_loss().item() if self.enable_balance else 0.0

        return avg_loss, div_loss, bal_loss, avg_acc

    def train(self, num_epochs: Optional[int] = None) -> Dict:
        """Full training loop."""
        if num_epochs is None:
            num_epochs = self.config.get("training", {}).get("epochs", 100)

        print(f"\nStarting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.perf_counter()

            avg_loss, div_loss, bal_loss, train_acc = self.train_epoch()

            epoch_time = time.perf_counter() - epoch_start

            self.training_losses.append(avg_loss - div_loss - bal_loss)
            self.aux_diversity_losses.append(div_loss)
            self.aux_balance_losses.append(bal_loss)
            self.total_losses.append(avg_loss)
            self.train_accuracies.append(train_acc)

            # Validation
            val_acc = 0.0
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                val_metrics = self.evaluate("val")
                val_acc = val_metrics["accuracy"]
                self.val_accuracies.append(val_acc)
            else:
                self.val_accuracies.append(self.val_accuracies[-1] if self.val_accuracies else 0.0)

            # Anneal load balancer temperature
            if self.enable_balance:
                self.load_balancer.anneal_temperature()

            # Reset diversity tracker per epoch
            if self.enable_diversity:
                self.diversity_tracker.reset()

            # Reset load balancer epoch counts
            if self.enable_balance:
                self.load_balancer.reset_epoch()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch+1:3d}/{num_epochs} | "
                    f"Loss: {avg_loss:.4f} (div={div_loss:.4f} bal={bal_loss:.4f}) | "
                    f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
                    f"Time: {epoch_time:.2f}s"
                )

        return {
            "training_losses": self.training_losses,
            "aux_diversity_losses": self.aux_diversity_losses,
            "aux_balance_losses": self.aux_balance_losses,
            "total_losses": self.total_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
        }

    # ------------------------------------------------------------------ eval
    def evaluate(self, dataset: str = "test", max_samples: Optional[int] = None) -> Dict:
        """Evaluate on a dataset split. Returns accuracy, F1, precision, recall, confusion."""
        if dataset == "val":
            X, y = self.input_adapter.X_val, self.input_adapter.y_val
        elif dataset == "test":
            X, y = self.input_adapter.X_test, self.input_adapter.y_test
        else:
            X, y = self.input_adapter.X_train, self.input_adapter.y_train

        n = len(X)
        if max_samples is not None:
            n = min(n, max_samples)

        all_preds = []
        all_labels = []

        for i in range(n):
            input_context, label = self.input_adapter.get_input_context(
                i, self.input_nodes, dataset=dataset,
            )
            output_signals = self.forward_pass(input_context)
            logits = self.compute_logits(output_signals)
            pred = torch.argmax(logits.squeeze(0) if logits.dim() > 1 else logits).item()
            all_preds.append(pred)
            all_labels.append(label)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = (all_preds == all_labels).mean()
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        return {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "confusion_matrix": cm.tolist(),
            "num_samples": n,
        }

    def get_logits_for_calibration(self, dataset: str = "val") -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Collect logits and labels for temperature calibration."""
        if dataset == "val":
            X, y = self.input_adapter.X_val, self.input_adapter.y_val
        else:
            X, y = self.input_adapter.X_test, self.input_adapter.y_test

        logits_list = []
        labels_list = []
        for i in range(len(X)):
            input_context, label = self.input_adapter.get_input_context(
                i, self.input_nodes, dataset=dataset,
            )
            output_signals = self.forward_pass(input_context)
            logits = self.compute_logits(output_signals)
            if logits.dim() > 1:
                logits = logits.squeeze(0)
            logits_list.append(logits.detach())
            labels_list.append(torch.tensor(label))

        return logits_list, labels_list

    # ------------------------------------------------------------------ util
    def count_parameters(self) -> int:
        """Total parameter count across all components."""
        total = 0
        # Node store
        for p in self.node_store.parameters():
            total += p.numel()
        # Input adapter
        for p in self.input_adapter.parameters():
            total += p.numel()
        # Uncertainty (null activation + temperature)
        if self.enable_uncertainty:
            for p in self.uncertainty.parameters():
                total += p.numel()
        return total

    def get_complexity_report(self) -> Dict:
        """Space and time complexity summary."""
        # Space
        param_count = self.count_parameters()
        memory_bytes = sum(
            p.numel() * p.element_size()
            for p in list(self.node_store.parameters()) + list(self.input_adapter.parameters())
        )
        if self.enable_uncertainty:
            memory_bytes += sum(p.numel() * p.element_size() for p in self.uncertainty.parameters())

        lt_memory = self.lookup_tables.estimate_memory_usage() * 1024 * 1024  # MB -> bytes

        # Time
        timing = self.forward_engine.get_timing_stats()

        return {
            "space_complexity": {
                "total_parameters": param_count,
                "memory_bytes": int(memory_bytes),
                "memory_mb": memory_bytes / (1024 * 1024),
                "lookup_table_memory_mb": self.lookup_tables.estimate_memory_usage(),
                "breakdown": {
                    "node_store": sum(p.numel() for p in self.node_store.parameters()),
                    "input_adapter": sum(p.numel() for p in self.input_adapter.parameters()),
                    "lookup_tables_bytes": int(lt_memory),
                },
            },
            "time_complexity": timing,
        }
