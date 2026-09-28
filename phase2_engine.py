"""
phase2_engine.py — Advanced Phase 2 Engine for Cognitive Radio Networks:
1. Dynamic Spectrum Access & Autonomous Frequency Hopping Policy
2. Soft Decision Fusion (Weighted Likelihood / Maximal Ratio Combining)
3. Defense against Malicious Nodes (Spectrum Sensing Data Falsification - SSDF)
"""

import numpy as np
import config as cfg
from signal_generator import generate_fm, generate_am, generate_bpsk, _add_awgn


# ─────────────────────────────────────────────────────────────────────────────
# 1. Autonomous Dynamic Frequency Hopping Engine
# ─────────────────────────────────────────────────────────────────────────────
class AutonomousHoppingEngine:
    """
    Simulates a Secondary User (SU) dynamically accessing the spectrum.
    If a Primary User (PU) appears on the active channel, the SU autonomously
    hops to the highest-quality vacant channel.
    """
    def __init__(self, channels=None):
        self.channels = channels or [88.5, 91.5, 93.5, 95.0, 98.3, 100.1, 104.0]
        self.current_channel = 98.3  # Initial vacant channel
        self.hop_history = []

    def evaluate_and_hop(self, channel_states):
        """
        channel_states: dict {freq: {'occupied': bool, 'snr': float, 'confidence': float}}
        Returns: decision dict with hop action
        """
        curr_state = channel_states.get(self.current_channel, {'occupied': False, 'snr': 20.0})
        
        if curr_state['occupied']:
            # Channel is now occupied by Primary User -> Must EVACUATE!
            vacant_candidates = [
                f for f, s in channel_states.items() 
                if not s['occupied'] and f != self.current_channel
            ]
            if vacant_candidates:
                # Pick the cleanest vacant channel (lowest noise)
                next_channel = vacant_candidates[0]
                action = f"⚠️ PU DETECTED on {self.current_channel} MHz! Autonomously hopped to VACANT {next_channel} MHz."
                old_channel = self.current_channel
                self.current_channel = next_channel
                self.hop_history.append((old_channel, next_channel, "PU Collision Avoidance"))
                return {
                    "action": "HOP",
                    "old_channel": old_channel,
                    "new_channel": next_channel,
                    "reason": "Primary User Collision Avoidance",
                    "status": action
                }
            else:
                return {
                    "action": "BACKOFF",
                    "old_channel": self.current_channel,
                    "new_channel": None,
                    "reason": "Spectrum Full (No Vacant Channels)",
                    "status": "🛑 All channels occupied! Secondary User backing off."
                }
        else:
            return {
                "action": "MAINTAIN",
                "old_channel": self.current_channel,
                "new_channel": self.current_channel,
                "reason": "Channel Clear",
                "status": f"🟢 Channel {self.current_channel} MHz is VACANT. Maintaining transmission."
            }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Soft Decision Fusion Center
# ─────────────────────────────────────────────────────────────────────────────
class SoftFusionCenter:
    """
    Advanced Soft Decision Fusion Center.
    Instead of 1-bit binary decisions, nodes transmit their continuous softmax
    probability vectors or energy statistics to the Fusion Center.
    """
    @staticmethod
    def soft_combining(node_confidences, weights=None):
        """
        Calculates Weighted Average Confidence (Soft Combining):
        Decision statistic Lambda = Sum(w_i * P_i)
        """
        num_nodes = len(node_confidences)
        if weights is None:
            weights = np.ones(num_nodes) / num_nodes
        else:
            weights = np.array(weights) / np.sum(weights)

        soft_statistic = float(np.sum(weights * np.array(node_confidences)))
        global_decision = soft_statistic >= cfg.OCCUPIED_THRESHOLD

        return {
            "fusion_type": "Soft Decision (Weighted Confidence)",
            "soft_statistic": soft_statistic,
            "global_occupied": global_decision,
            "weights_used": weights.tolist()
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Malicious Node & Security Defense (SSDF Mitigation)
# ─────────────────────────────────────────────────────────────────────────────
class SecureCooperativeFusion:
    """
    Defends against Spectrum Sensing Data Falsification (SSDF) attacks
    where a rogue/malicious sensing node intentionally sends false reports.
    Uses dynamic reputation/trust scoring to isolate rogue nodes.
    """
    def __init__(self, num_nodes=3):
        self.num_nodes = num_nodes
        self.trust_scores = np.ones(num_nodes) * 1.0  # Trust score 0.0 to 1.0

    def robust_fuse(self, node_reports, malicious_node_idx=None):
        """
        node_reports: list of dicts [{'node_id': i, 'occupied': bool, 'confidence': float}]
        malicious_node_idx: optional node to simulate as an attacker (always lies)
        """
        # Inject attack if specified
        reports = []
        for r in node_reports:
            r_copy = dict(r)
            if malicious_node_idx is not None and r['node_id'] == malicious_node_idx:
                # Invert decision (Attacker lies to fool the network)
                r_copy['occupied'] = not r['occupied']
                r_copy['confidence'] = 0.95 if r_copy['occupied'] else 0.10
                r_copy['is_attacker'] = True
            else:
                r_copy['is_attacker'] = False
            reports.append(r_copy)

        # Preliminary consensus using trust-weighted voting
        weights = self.trust_scores / np.sum(self.trust_scores)
        weighted_votes = np.sum([
            weights[i] * (1.0 if reports[i]['occupied'] else 0.0) 
            for i in range(self.num_nodes)
        ])
        consensus = weighted_votes >= 0.5

        # Update trust scores based on deviation from consensus
        for i in range(self.num_nodes):
            if reports[i]['occupied'] == consensus:
                self.trust_scores[i] = min(1.0, self.trust_scores[i] + 0.05)
            else:
                self.trust_scores[i] = max(0.05, self.trust_scores[i] - 0.25)

        return {
            "consensus_occupied": bool(consensus),
            "reports": reports,
            "trust_scores": self.trust_scores.tolist(),
            "isolated_nodes": [i for i, t in enumerate(self.trust_scores) if t < 0.3]
        }
