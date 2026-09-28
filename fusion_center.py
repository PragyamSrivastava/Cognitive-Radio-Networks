"""
fusion_center.py — Cooperative Spectrum Sensing Simulation.
Simulates multiple sensing nodes with independent noise realizations,
and implements OR-rule and Majority Voting fusion at a central decision point.
"""

import numpy as np
import config as cfg
from signal_generator import _add_awgn, GENERATORS
from spectrogram_processor import iq_to_spectrogram
from cnn_model import predict_with_occupancy


class SensingNode:
    """
    Simulates a single cognitive radio sensing node.
    
    Each node receives the same base signal but with an independent
    AWGN noise realization, simulating spatially distributed sensors.
    """
    
    def __init__(self, node_id, model):
        """
        Parameters:
            node_id: Unique integer identifier for this node.
            model: Trained Keras CNN model for classification.
        """
        self.node_id = node_id
        self.model = model
    
    def sense(self, clean_signal, snr_db, threshold=None):
        """
        Perform local spectrum sensing on a signal.
        
        The node adds its own independent noise realization, converts to
        spectrogram, and runs CNN inference.
        
        Parameters:
            clean_signal: Clean IQ signal (complex array).
            snr_db: Target SNR in dB for this node's observation.
            threshold: Confidence threshold for occupied decision.
        
        Returns:
            Dict with keys:
                - 'node_id': This node's ID
                - 'predicted_class': Predicted signal type index
                - 'predicted_name': Predicted signal type name
                - 'confidence': Max softmax probability
                - 'occupied': Boolean occupied flag
                - 'probabilities': Full softmax vector
        """
        threshold = threshold or cfg.OCCUPIED_THRESHOLD
        
        # Independent noise realization for this node
        noisy_signal = _add_awgn(clean_signal, snr_db)
        
        # Convert to spectrogram
        spec = iq_to_spectrogram(noisy_signal)
        spec_input = spec.reshape(1, *cfg.SPEC_SIZE, 1).astype(np.float32)
        
        # CNN inference
        result = predict_with_occupancy(self.model, spec_input, threshold)
        
        return {
            'node_id': self.node_id,
            'predicted_class': int(result['class_indices'][0]),
            'predicted_name': result['class_names'][0],
            'confidence': float(result['confidences'][0]),
            'occupied': bool(result['occupied'][0]),
            'probabilities': result['probabilities'][0],
        }


class FusionCenter:
    """
    Fusion Center for Cooperative Spectrum Sensing.
    
    Aggregates binary decisions from multiple sensing nodes and applies
    fusion rules (OR, Majority Voting) to make a global decision.
    """
    
    def __init__(self, model, num_nodes=None):
        """
        Parameters:
            model: Trained Keras CNN model (shared across nodes).
            num_nodes: Number of simulated sensing nodes.
        """
        num_nodes = num_nodes or cfg.NUM_NODES
        self.nodes = [SensingNode(i, model) for i in range(num_nodes)]
        self.num_nodes = num_nodes
    
    def collect_decisions(self, clean_signal, snr_db, threshold=None):
        """
        Have all nodes independently sense the same signal.
        
        Parameters:
            clean_signal: Clean IQ signal.
            snr_db: Target SNR in dB.
            threshold: Confidence threshold.
        
        Returns:
            List of node decision dicts.
        """
        decisions = []
        for node in self.nodes:
            decision = node.sense(clean_signal, snr_db, threshold)
            decisions.append(decision)
        return decisions
    
    @staticmethod
    def fuse_or(decisions):
        """
        OR Rule: Channel is occupied if ANY node detects a signal.
        
        Maximizes probability of detection at the cost of higher false alarm rate.
        
        Parameters:
            decisions: List of node decision dicts.
        
        Returns:
            Dict with global decision and metadata.
        """
        occupied_flags = [d['occupied'] for d in decisions]
        global_occupied = any(occupied_flags)
        
        # Average confidence across detecting nodes (or all if none detected)
        detecting = [d['confidence'] for d in decisions if d['occupied']]
        avg_confidence = np.mean(detecting) if detecting else np.mean(
            [d['confidence'] for d in decisions]
        )
        
        return {
            'rule': 'OR',
            'global_occupied': global_occupied,
            'avg_confidence': float(avg_confidence),
            'node_decisions': occupied_flags,
            'num_detecting': sum(occupied_flags),
        }
    
    @staticmethod
    def fuse_majority(decisions):
        """
        Majority Voting Rule: Channel is occupied if MORE THAN HALF of nodes detect.
        
        Balanced approach between OR and AND rules.
        
        Parameters:
            decisions: List of node decision dicts.
        
        Returns:
            Dict with global decision and metadata.
        """
        occupied_flags = [d['occupied'] for d in decisions]
        num_detecting = sum(occupied_flags)
        majority_threshold = len(decisions) / 2
        global_occupied = num_detecting > majority_threshold
        
        avg_confidence = np.mean([d['confidence'] for d in decisions])
        
        return {
            'rule': 'MAJORITY',
            'global_occupied': global_occupied,
            'avg_confidence': float(avg_confidence),
            'node_decisions': occupied_flags,
            'num_detecting': num_detecting,
        }
    
    def cooperative_sense(self, clean_signal, snr_db, threshold=None):
        """
        Full cooperative sensing pipeline.
        
        Collects individual node decisions, then applies both fusion rules.
        
        Parameters:
            clean_signal: Clean IQ signal.
            snr_db: Target SNR in dB.
            threshold: Confidence threshold.
        
        Returns:
            Dict with individual decisions and both fusion results.
        """
        decisions = self.collect_decisions(clean_signal, snr_db, threshold)
        
        return {
            'individual_decisions': decisions,
            'or_fusion': self.fuse_or(decisions),
            'majority_fusion': self.fuse_majority(decisions),
        }


def simulate_cooperative_sensing(model, snr_range=None, num_trials=100,
                                 threshold_range=None):
    """
    Run a full cooperative sensing simulation across SNR and threshold values.
    
    Used for generating ROC curves comparing individual vs. cooperative performance.
    
    Parameters:
        model: Trained Keras CNN model.
        snr_range: Array of SNR values in dB.
        num_trials: Number of Monte Carlo trials per (SNR, threshold) pair.
        threshold_range: Array of confidence thresholds to sweep.
    
    Returns:
        results: Dict with Pd and Pf values for individual, OR, and majority rules,
                 indexed by SNR.
    """
    snr_range = snr_range if snr_range is not None else cfg.SNR_RANGE_DB
    if threshold_range is None:
        threshold_range = np.linspace(0.1, 0.99, 30)
    
    fusion_center = FusionCenter(model)
    results = {}
    
    for snr_db in snr_range:
        snr_key = float(snr_db)
        results[snr_key] = {
            'thresholds': threshold_range.tolist(),
            'individual': {'pd': [], 'pf': []},
            'or_rule': {'pd': [], 'pf': []},
            'majority_rule': {'pd': [], 'pf': []},
        }
        
        for thresh in threshold_range:
            # Counters for detection and false alarm
            ind_detect = 0
            ind_false_alarm = 0
            or_detect = 0
            or_false_alarm = 0
            maj_detect = 0
            maj_false_alarm = 0
            
            n_signal = 0  # Trials where signal is present (H1)
            n_noise = 0   # Trials where only noise is present (H0)
            
            for trial in range(num_trials):
                # H1: Signal present — randomly pick a modulation type
                class_name = cfg.CLASSES[np.random.randint(len(cfg.CLASSES))]
                clean_signal = GENERATORS[class_name]()
                
                result = fusion_center.cooperative_sense(clean_signal, snr_db, thresh)
                
                # Individual node detection (average across nodes)
                for d in result['individual_decisions']:
                    if d['occupied']:
                        ind_detect += 1
                n_signal += len(result['individual_decisions'])
                
                if result['or_fusion']['global_occupied']:
                    or_detect += 1
                if result['majority_fusion']['global_occupied']:
                    maj_detect += 1
                
                # H0: Noise only
                noise_only = _add_awgn(np.zeros(cfg.NUM_SAMPLES, dtype=complex), snr_db=100)
                noise_spec = iq_to_spectrogram(noise_only)
                noise_input = noise_spec.reshape(1, *cfg.SPEC_SIZE, 1).astype(np.float32)
                noise_pred = predict_with_occupancy(model, noise_input, thresh)
                
                if noise_pred['occupied'][0]:
                    ind_false_alarm += 1
                n_noise += 1
                
                # Cooperative false alarm for noise-only
                noise_decisions = []
                for node in fusion_center.nodes:
                    noise_realization = _add_awgn(
                        np.zeros(cfg.NUM_SAMPLES, dtype=complex), snr_db=100
                    )
                    ns = iq_to_spectrogram(noise_realization)
                    ni = ns.reshape(1, *cfg.SPEC_SIZE, 1).astype(np.float32)
                    nr = predict_with_occupancy(model, ni, thresh)
                    noise_decisions.append({
                        'occupied': bool(nr['occupied'][0]),
                        'confidence': float(nr['confidences'][0]),
                    })
                
                or_fa = FusionCenter.fuse_or(noise_decisions)
                maj_fa = FusionCenter.fuse_majority(noise_decisions)
                
                if or_fa['global_occupied']:
                    or_false_alarm += 1
                if maj_fa['global_occupied']:
                    maj_false_alarm += 1
            
            # Compute probabilities
            pd_ind = ind_detect / max(n_signal, 1)
            pf_ind = ind_false_alarm / max(n_noise, 1)
            pd_or = or_detect / max(num_trials, 1)
            pf_or = or_false_alarm / max(num_trials, 1)
            pd_maj = maj_detect / max(num_trials, 1)
            pf_maj = maj_false_alarm / max(num_trials, 1)
            
            results[snr_key]['individual']['pd'].append(pd_ind)
            results[snr_key]['individual']['pf'].append(pf_ind)
            results[snr_key]['or_rule']['pd'].append(pd_or)
            results[snr_key]['or_rule']['pf'].append(pf_or)
            results[snr_key]['majority_rule']['pd'].append(pd_maj)
            results[snr_key]['majority_rule']['pf'].append(pf_maj)
        
        print(f"  SNR = {snr_db:+.0f} dB — simulation complete")
    
    return results


if __name__ == "__main__":
    print("Fusion center module loaded.")
    print(f"Configured for {cfg.NUM_NODES} cooperative sensing nodes.")
    print("Run main.py for full simulation.")
