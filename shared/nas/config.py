# config.py
# Function: Define the configuration of the GA

import sys
import numpy as np
import json
import hashlib
sys.path.append('../utils/')
sys.path.append('../cifar10data/')
sys.path.append('../cifar100data/')
sys.path.append('../stl10/')

import math
import numpy as np
import torch.nn.parallel
import torch.optim

from utils.pytorch_utils import *

# ======================================
# ===== Genetic Architecture Config =====
# ======================================

class GeneticArchSearchConfig():
    """
    LST config generator with:
      1) per-layer reduction factor (RF) for both connection & side,
      2) connection vector includes input-ladder connection layer(s),
      3) hard rule: if a side layer is empty, its paired connection must be empty,
      4) rf_level: 0=independent, 1=per-layer conn RF mirrors side RF, 2=single global RF,
      5) rf_anchor for input-ladder & upsample: 'random'|'follow'|'fixed'.
    """

    def __init__(self, search_space=None, target_hardware=None, ref_value=None,
                 rl_batch_size=10, reward_beta=800, rf_level=2,
                 rf_anchor='follow', rf_fixed=None, **kwargs):

        self.search_space = search_space
        self.target_hardware = target_hardware
        self.ref_value = ref_value
        self.batch_size = rl_batch_size

        self.rf_level = rf_level                  # 0 (various) / 1 (same con and side) / 2 (global)
        self.rf_anchor = rf_anchor                # 'random' | 'follow' | 'fixed'
        self.rf_fixed = rf_fixed                  # if anchor=fixed, can be rf value or rf index

        self._baseline = None
        self.history = {}

        # NOTE: input_ladder indicates how many extra connection slots beyond backbone layers
        # self.mapping = {
        #     'vgg':      {'archi': 13,                      'input_ladder': 1},
        #     'resnet':   {'archi': {'side': 20, 'connection': 17}, 'input_ladder': 1},
        #     'alexnet':  {'archi': 5,                       'input_ladder': 1},
        #     'vit-base': {'archi': 12,                      'input_ladder': 1},
        #     't5-base': {'archi': {'side': 24, 'connection': 24}, 'input_ladder': 2},
        # }
        self.mapping = {'vgg': {'archi': 13, 'droppable_layers': [0, 2, 4, 5, 7, 8, 9, 10, 11, 12], 'input_ladder': 1},
                        # layer index start from 0.  [1, 3, 5, 6, 8, 9, 10, 11, 12]
                        'resnet': {'archi': {'side': 20, 'connection': 17},
                                   'droppable_layers': [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15], 'input_ladder': 1},
                        # [1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16] must -1
                        'alexnet': {'archi': 5, 'droppable_layers': [4, 6], 'input_ladder': 1},
                        'vit-base': {'archi': 12, 'droppable_layers': [i for i in range(0, 13)], 'input_ladder': 1},
                        't5-base': {'archi': {'side': 24, 'connection': 24}, 'droppable_layers': [i for i in range(1, 25)], 'input_ladder': 2},
                        'llama': {'archi': 32, 'droppable_layers': list(range(0, 33)), 'input_ladder': 1},
                        }

        self.nas_score_mapping = {
            'resnet': 10.6,
            'vgg': 12.0,
            'alexnet': 9000.0 / 1024,
            'vit-base': 12.0,
            't5-base': 12.0,
            'llama': 12.0,  # placeholder; tune after NAS experiments
        }
        self.latency_mapping = {
            'resnet': 1.9,
            'vgg': 12.0,
            'alexnet': 0.7,
            'vit-base': 240.0,
            't5-base': 190.3,
            'llama': 1000.0,  # placeholder; measure on mammoth
        }
        self.reward_beta = reward_beta

        self.model_name = None
        print(f'reward beta: {self.reward_beta}, rf_level: {self.rf_level}, rf_anchor: {self.rf_anchor}')

    @property
    def config(self):
        config = {'type': type(self)}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    # ===== reward functions (unchanged) =====

    def calculate_reward(self, net_info, verbose=True):
        if 'score' in net_info and 'latency' in net_info:
            return self.calculate_reward_score_latency(net_info, verbose)
        elif 'score' in net_info and 'flops' in net_info:
            return self.calculate_reward_score_flops(net_info, verbose)
        elif 'acc' in net_info and 'latency' in net_info:
            return self.calculate_reward_score_acc_latency(net_info, verbose)
        elif 'acc' in net_info and 'flops' in net_info:
            return self.calculate_reward_score_acc_flops(net_info, verbose)
        else:
            raise NotImplementedError

    def calculate_reward_score_latency(self, net_info, verbose=True):
        alpha = 0.7
        base_score = self.nas_score_mapping[self.model_name]
        score = abs(net_info['score'])
        delta_score = (score - base_score) / base_score

        base_latency = self.latency_mapping[self.model_name]
        latency = net_info['latency']
        delta_latency = (latency - base_latency) / base_latency

        reward = alpha * delta_score * self.reward_beta - (1 - alpha) * delta_latency
        if verbose:
            print(f'score: {score: .2f}, latency: {latency: .2f}, delta score: {delta_score*100: .1f}%, delta latency: {delta_latency*100: .1f}%, reward: {reward:.2f}')
        return reward, delta_score, delta_latency

    def calculate_reward_score_flops(self, net_info, verbose=True):
        alpha = 0.7
        base_score = self.nas_score_mapping[self.model_name]
        score = abs(net_info['score'])
        delta_score = (score - base_score) / base_score

        base_latency = self.latency_mapping[self.model_name]
        flops = net_info['flops']
        predicted_latency = flops / 1e9 * 200 + base_latency

        delta_latency = (predicted_latency - base_latency) / base_latency
        reward = alpha * delta_score * self.reward_beta - (1 - alpha) * delta_latency
        if verbose:
            print(f'predicted latency: {predicted_latency:.2f} ms, delta score: {delta_score*100: .1f}%, delta latency: {delta_latency*100: .1f}%, reward: {reward:.2f}')
        return reward, delta_score, delta_latency

    def calculate_reward_score_acc_flops(self, net_info, verbose=True):
        alpha = 0.7
        accuracy = abs(net_info['accuracy'])
        delta_score = accuracy / 10000.0

        base_latency = self.latency_mapping[self.model_name]
        flops = net_info['flops']
        predicted_latency = flops / 1e9 * 200 + base_latency

        delta_latency = (predicted_latency - base_latency) / base_latency
        reward = alpha * delta_score * self.reward_beta - (1 - alpha) * delta_latency
        if verbose:
            print(f'predicted latency: {predicted_latency:.2f} ms, accuracy: {accuracy: .1f}%, delta latency: {delta_latency*100: .1f}%, reward: {reward:.2f}')
        return reward, accuracy, delta_latency

    def calculate_reward_score_acc_latency(self, net_info, verbose=True):
        alpha = 0.7
        accuracy = abs(net_info['accuracy'])
        delta_score = accuracy / 10000.0

        base_latency = self.latency_mapping[self.model_name]
        latency = net_info['latency']
        delta_latency = (latency - base_latency) / base_latency

        reward = alpha * delta_score * self.reward_beta - (1 - alpha) * delta_latency
        if verbose:
            print(f'latency: {latency:.2f} ms, accuracy: {accuracy: .1f}%, delta latency: {delta_latency*100: .1f}%, reward: {reward:.2f}')
        return reward, accuracy, delta_latency

    @property
    def baseline(self):
        return self._baseline

    @baseline.setter
    def baseline(self, value):
        self._baseline = value

    # ==============================
    # ===== LST Config (Core) ======
    # ==============================

    @staticmethod
    def _weighted_choice(length):
        """Bias towards index 0 (empty) while keeping others possible."""
        if length == 1:
            return [0]
        choices = list(range(length))
        w0 = 0.5
        w_other = (1.0 - w0) / (length - 1)
        weights = [w0] + [w_other] * (length - 1)
        return random.choices(choices, weights=weights, k=1)

    # ---- helpers for constraints / RF coupling ----

    def _counts(self):
        archi = self.mapping[self.model_name]['archi']
        num_side = archi['side'] if isinstance(archi, dict) else archi
        num_conn = archi['connection'] if isinstance(archi, dict) else archi
        num_input = self.mapping[self.model_name].get('input_ladder', 1)
        return num_side, num_conn, num_input

    def _enforce_side_conn_empty(self, side_init, connection_init):
        """If side[i] is EMPTY (0), connection[i] must be EMPTY (0). Only for backbone-aligned pairs."""
        _, num_conn, _ = self._counts()
        for i in range(num_conn):
            if side_init[i] == 0 and connection_init[i] != 0:
                connection_init[i] = 0

    def _first_last_non_empty_side_rf(self, rf_side_idx, side_init):
        """Return (rf_first_non_empty, rf_last_non_empty) or (None, None)."""
        rf1 = rfL = None
        for i, op in enumerate(side_init):
            if op != 0:
                rf1 = rf_side_idx[i]
                break
        for i in range(len(side_init) - 1, -1, -1):
            if side_init[i] != 0:
                rfL = rf_side_idx[i]
                break
        return rf1, rfL

    def _rf_index_of(self, value_or_index, rf_pool):
        """Accept either raw RF value or RF index; return valid index into rf_pool (clipped)."""
        if value_or_index is None:
            return None
        # exact index
        if isinstance(value_or_index, int) and 0 <= value_or_index < len(rf_pool):
            return value_or_index
        # try to map a value
        try:
            return rf_pool.index(value_or_index)
        except Exception:
            # clip to nearest by value distance
            arr = np.array(rf_pool, dtype=float)
            val = float(value_or_index)
            return int(np.abs(arr - val).argmin())

    def _enforce_rf_level_and_anchor(self, rf_conn_idx, rf_side_idx, side_init, upsample, rf_pool):
        """
        Apply rf_level coupling and rf_anchor policy (input-ladder & upsample).
        Returns possibly updated (rf_conn_idx, rf_side_idx, upsample).
        """
        num_side, num_conn, num_input = self._counts()

        # rf_level
        if self.rf_level == 2:
            # single global RF: choose existing upsample as source if present, else first side rf
            if upsample is None or not (0 <= upsample < len(rf_pool)):
                g = rf_side_idx[0] if len(rf_side_idx) else 0
            else:
                g = upsample
            # If fixed provided, override
            g_fixed = self._rf_index_of(self.rf_fixed, rf_pool) if self.rf_anchor == 'fixed' else None
            if g_fixed is not None:
                g = g_fixed
            rf_conn_idx[:] = [g] * len(rf_conn_idx)
            rf_side_idx[:] = [g] * len(rf_side_idx)
            upsample = g

        elif self.rf_level == 1:
            # per-layer: conn RFs mirror side RFs for backbone-aligned pairs
            for i in range(num_conn):
                rf_conn_idx[i] = rf_side_idx[i]

        # rf_anchor for input-ladder & upsample
        if self.rf_anchor == 'fixed':
            idx = self._rf_index_of(self.rf_fixed, rf_pool)
            if idx is None:
                idx = 0
            # input-ladder
            for j in range(num_input):
                rf_conn_idx[num_conn + j] = idx
            # upsample
            upsample = idx

        elif self.rf_anchor == 'follow':
            if self.model_name == 'resnet': # the last 3 layer is for downsample in the list
                rf_first, rf_last = self._first_last_non_empty_side_rf(rf_side_idx[:-3], side_init[:-3])
            else:
                rf_first, rf_last = self._first_last_non_empty_side_rf(rf_side_idx, side_init)

            # input-ladder follows first non-empty (if exists)
            if rf_first is not None:
                for j in range(num_input):
                    rf_conn_idx[num_conn + j] = rf_first
            # upsample follows last non-empty (if exists)
            if rf_last is not None:
                upsample = rf_last

        # else: 'random' → nothing special (already randomized)

        # If rf_level==1 and side has no non-empty layer yet, leave anchors as-is.
        # Always return in-bounds ints
        rf_conn_idx[:] = [int(np.clip(r, 0, len(rf_pool) - 1)) for r in rf_conn_idx]
        rf_side_idx[:] = [int(np.clip(r, 0, len(rf_pool) - 1)) for r in rf_side_idx]
        upsample = int(np.clip(upsample, 0, len(rf_pool) - 1))
        return rf_conn_idx, rf_side_idx, upsample

    # ---- generation steps ----

    def ladder_config_gen_step_1(self, search_space, rf=None):
        """
        Build a raw (unencoded) config with per-layer RF indices.
        Includes input-ladder connection(s).
        """
        assert self.model_name in self.mapping, f"Unknown model: {self.model_name}"
        num_side, num_conn, num_input = self._counts()

        # pools
        rf_pool = search_space['reduction_factor']
        assert len(rf_pool) > 0
        conn_ops = search_space['connection']
        side_ops = search_space['side']

        # decide base RF index to use if rf_level == 2 or if rf is specified
        rf_idx_choice = None
        if rf is not None and rf in rf_pool:
            rf_idx_choice = rf_pool.index(rf)

        # per-layer RF indices
        if self.rf_level == 2:
            # single global RF
            g = rf_idx_choice if rf_idx_choice is not None else random.randrange(len(rf_pool))
            rf_conn_idx = [g] * (num_conn + num_input)
            rf_side_idx = [g] * num_side
            upsample = g
        else:
            # independent or coupled later
            if rf_idx_choice is None:
                rf_conn_idx = [random.randrange(len(rf_pool)) for _ in range(num_conn + num_input)]
                rf_side_idx = [random.randrange(len(rf_pool)) for _ in range(num_side)]
                upsample = random.randrange(len(rf_pool))
            else:
                rf_conn_idx = [rf_idx_choice for _ in range(num_conn + num_input)]
                rf_side_idx = [rf_idx_choice for _ in range(num_side)]
                upsample = rf_idx_choice

        # per-layer op indices
        # backbone-aligned connections (bias toward empty if >1 op)
        # connection_init = [self._weighted_choice(len(conn_ops))[0] for _ in range(num_conn)]

        empty_rate = 0.2
        # connection ops
        conn_candidates = [i for i, _ in enumerate(conn_ops)]
        conn_weights = [empty_rate] + [(1 - empty_rate) / (len(conn_candidates) - 1)] * (len(conn_candidates) - 1)
        connection_init = [random.choices(conn_candidates, weights=conn_weights, k=1)[0] for _ in range(num_conn)]

        # side ops
        side_candidates = [i for i, _ in enumerate(side_ops)]
        side_weights = [empty_rate] + [(1 - empty_rate) / (len(side_candidates) - 1)] * (len(side_candidates) - 1)
        side_init = [random.choices(side_candidates, weights=side_weights, k=1)[0] for _ in range(num_side)]

        # connection_init = [random.choices([i for i, _ in enumerate(conn_ops)])[0] for _ in range(num_conn)]
        # side_init = [random.choices([i for i, _ in enumerate(side_ops)])[0] for _ in range(num_side)]

        # model-specific seeds
        if self.model_name == 'resnet':
            if num_side >= 3:
                side_init[-3:] = [max(1, s) for s in side_init[-3:]]
            connection_input = [1 for _ in range(num_input)]  # cannot be 0
        else:
            connection_input = [random.randint(1, len(conn_ops) - 1) for _ in range(num_input)]  # cannot be 0

        # append input-ladder connection(s)
        connection_init = connection_init + connection_input

        # --- enforce core coupling + rf semantics at init ---
        self._enforce_side_conn_empty(side_init, connection_init)
        rf_conn_idx, rf_side_idx, upsample = self._enforce_rf_level_and_anchor(
            rf_conn_idx, rf_side_idx, side_init, upsample, rf_pool
        )

        return {
            'rf_conn_idx': rf_conn_idx,
            'rf_side_idx': rf_side_idx,
            'connection_init': connection_init,
            'side_init': side_init,
            'upsample': upsample,
        }

    def ladder_config_gen_step_2(self, config_dict):
        """
        Apply architectural limitations (no droppable_layers):
          - side empty -> paired connection empty
          - model-specific invariants, but NEVER override the empty-coupling
          - re-apply rf coupling/anchor
        """
        rf_conn_idx = list(config_dict['rf_conn_idx'])
        rf_side_idx = list(config_dict['rf_side_idx'])
        connection_init = list(config_dict['connection_init'])
        side_init = list(config_dict['side_init'])
        upsample = config_dict['upsample']

        num_side, num_conn, num_input = self._counts()
        rf_pool = self.search_space['reduction_factor']

        droppable = self.mapping[self.model_name].get('droppable_layers', [])
        # print('before:', side_init)
        side_init = [side_type if i in droppable else 1 for i, side_type in enumerate(side_init)]
        # print('after :', side_init)


        # model-specific invariants (conditional on side non-empty)
        if self.model_name == 'resnet':
            # side downsample must stay non-zero
            if num_side >= 3:side_init[-3:] = [max(1, s) for s in side_init[-3:]]
            # input ladder must stay non-zero
            connection_init[-1] = max(1, connection_init[-1])
            # last connection non-zero
            connection_init[-2] = max(1, connection_init[-2])
            side_init[-4] = max(1, side_init[-4])
        elif self.model_name in ['vit-base', 'vgg', 'alexnet', 'llama']:
            # input ladder non-zero
            connection_init[-1] = max(1, connection_init[-1])
            # last connection non-zero
            connection_init[-2] = max(1, connection_init[-2])
            side_init[-1] = max(1, side_init[-1])
            # last backbone connection non-zero **only if** paired side non-empty
            last_backbone = num_conn - 1
            if last_backbone >= 0 and side_init[last_backbone] != 0:
                connection_init[last_backbone] = max(1, connection_init[last_backbone])
        elif self.model_name == 't5-base':
            # TWO input ladders: [..., enc_input, dec_input] at the tail → both must be non-zero
            if self.mapping['t5-base']['input_ladder'] != 2:
                raise ValueError("t5-base expects 2 input-ladder connectors.")
            connection_init[-2] = max(1, connection_init[-2])  # enc input ladder
            connection_init[-1] = max(1, connection_init[-1])  # dec input ladder
            connection_init[11] = max(1, connection_init[11])  # enc last layer
            side_init[11] = max(1, side_init[11])  # last enc side layer
            connection_init[-3] = max(1, connection_init[-3])  # dec last layer
            side_init[-1] = max(1, side_init[-1])  # last dec side layer
            last_backbone = num_conn - 1
            if last_backbone >= 0 and side_init[last_backbone] != 0:
                connection_init[last_backbone] = max(1, connection_init[last_backbone])
        else:
            raise NotImplementedError

        # core coupling
        self._enforce_side_conn_empty(side_init, connection_init)



        # rf-level + anchor
        rf_conn_idx, rf_side_idx, upsample = self._enforce_rf_level_and_anchor(
            rf_conn_idx, rf_side_idx, side_init, upsample, rf_pool
        )

        return {
            'rf_conn_idx': rf_conn_idx,
            'rf_side_idx': rf_side_idx,
            'connection_init': connection_init,
            'side_init': side_init,
            'upsample': upsample,
        }

    def ladder_config_gen_step_3(self, init_config, search_space):
        """Encode per-layer (rf_idx, op_idx) -> single scalar per layer."""
        conn_ops = search_space['connection']
        side_ops = search_space['side']

        rf_conn_idx = init_config['rf_conn_idx']
        rf_side_idx = init_config['rf_side_idx']
        connection_cfg = init_config['connection_init']
        side_cfg = init_config['side_init']

        encoded_conn = [rf_conn_idx[i] * len(conn_ops) + connection_cfg[i] for i in range(len(connection_cfg))]
        encoded_side = [rf_side_idx[i] * len(side_ops) + side_cfg[i] for i in range(len(side_cfg))]

        return {
            'connection': encoded_conn,
            'side': encoded_side,
            'upsample': init_config['upsample'],
            'all': encoded_conn + encoded_side + [init_config['upsample']],
            # keep raw for debug/printing
            'rf_conn_idx': rf_conn_idx,
            'rf_side_idx': rf_side_idx,
            'connection_init': connection_cfg,
            'side_init': side_cfg,
        }

    # ===== GA ops =====

    def crossover(self, parents, offspring_size):
        """Two-point crossover on paired (connection, side, rf) genes; re-apply constraints & rf policies.

        connection[i] and side[i] are tightly coupled by the cascade rule, so both lists are
        cut at the same two points. Cuts are drawn fresh per offspring pair for diversity.
        Two-point preserves contiguous active-layer blocks better than uniform crossover,
        which matters because the cascade structure makes consecutive genes semantically cohesive.
        """
        offsprings = []
        n = len(parents[0]['connection_init'])  # Lc == Ls by construction

        for k in range(offspring_size):
            p1 = parents[k % len(parents)]
            p2 = parents[(k + 1) % len(parents)]
            child = {}

            # Draw two distinct cut points per pair; sort so i <= j
            if n > 1:
                i, j = sorted(np.random.choice(n, size=2, replace=False))
            else:
                i = j = 0

            # Swap the [i:j] segment from p2 into p1; same cut for all coupled fields
            def splice(a, b):
                return a[:i] + b[i:j] + a[j:]

            child['connection_init'] = splice(p1['connection_init'], p2['connection_init'])
            child['side_init']       = splice(p1['side_init'],       p2['side_init'])
            child['rf_conn_idx']     = splice(p1['rf_conn_idx'],     p2['rf_conn_idx'])
            child['rf_side_idx']     = splice(p1['rf_side_idx'],     p2['rf_side_idx'])

            # upsample inherit random
            child['upsample'] = random.choice([p1['upsample'], p2['upsample']])

            # re-apply limitations + rf policies
            fixed = self.ladder_config_gen_step_2(child)
            offsprings.append(fixed)
        return offsprings

    def mutation(self, offspring_list, sigma, forbid_List, search_space):
        """
        Domain-aware, unbiased discrete mutation:
          - scale step std by domain size,
          - rint() (unbiased) instead of astype(int) (truncation),
          - wrap-around modulo domain to avoid edge bias,
          - re-apply architectural invariants + rf policies (unchanged).
        """
        Rf = len(search_space['reduction_factor'])
        Oc = len(search_space['connection'])
        Os = len(search_space['side'])

        num_side, num_conn, num_input = self._counts()
        rf_pool = self.search_space['reduction_factor']

        # Interpret 'sigma' as a coarse knob; normalize by domain sizes.
        # Typical initial sigma≈8 → strong moves across most of the domain; then decays.
        sigma_frac = float(min(1.0, max(0.0, sigma / 8.0)))  # map to [0,1]

        # per-domain step std (in index units)
        std_conn = max(1.0, sigma_frac * max(1, Oc - 1))
        std_side = max(1.0, sigma_frac * max(1, Os - 1))
        std_rf = max(1.0, 0.25 * sigma_frac * max(1, Rf - 1))  # RF moves are gentler (¼)

        for ch in offspring_list:
            # ---- mutate connection ops (incl. input-ladder tail) ----
            if 'connection_init' not in forbid_List:
                conn = np.asarray(ch['connection_init'], dtype=int)
                # unbiased integer step with wrap-around
                noise = np.rint(np.random.normal(0.0, std_conn, size=conn.size)).astype(int)
                conn = (conn + noise) % Oc
                ch['connection_init'] = conn.tolist()

            # ---- mutate side ops ----
            if 'side_init' not in forbid_List:
                side = np.asarray(ch['side_init'], dtype=int)
                noise = np.rint(np.random.normal(0.0, std_side, size=side.size)).astype(int)
                side = (side + noise) % Os
                ch['side_init'] = side.tolist()

            # ---- mutate RF indices ----
            if self.rf_level == 2:
                # mutate a single global RF and propagate
                g = int(ch['upsample'])
                g = int((g + int(np.rint(np.random.normal(0.0, std_rf)))) % Rf)
                ch['upsample'] = g
                ch['rf_conn_idx'] = [g] * (num_conn + num_input)
                ch['rf_side_idx'] = [g] * num_side
            else:
                if 'rf_conn_idx' not in forbid_List:
                    rfc = np.asarray(ch['rf_conn_idx'], dtype=int)
                    noise = np.rint(np.random.normal(0.0, std_rf, size=rfc.size)).astype(int)
                    rfc = (rfc + noise) % Rf
                    ch['rf_conn_idx'] = rfc.tolist()

                if 'rf_side_idx' not in forbid_List:
                    rfs = np.asarray(ch['rf_side_idx'], dtype=int)
                    noise = np.rint(np.random.normal(0.0, std_rf, size=rfs.size)).astype(int)
                    rfs = (rfs + noise) % Rf
                    ch['rf_side_idx'] = rfs.tolist()

                # occasional upsample flip
                if random.random() < 0.3:
                    ch['upsample'] = random.randrange(Rf)

            # ===== re-apply architectural limitations & rf policies (UNCHANGED) =====
            # side empty -> connection empty
            self._enforce_side_conn_empty(ch['side_init'], ch['connection_init'])

            # model-specific invariants (conditional)
            if self.model_name == 'resnet':
                if num_side >= 3:
                    ch['side_init'][-3:] = [max(1, s) for s in ch['side_init'][-3:]]
                ch['connection_init'][-1] = max(1, ch['connection_init'][-1])  # input ladder
                ch['connection_init'][-2] = max(1, ch['connection_init'][-2])
                ch['side_init'][-4] = max(1, ch['side_init'][-4])
            elif self.model_name in ['vit-base', 'vgg', 'alexnet', 'llama']:
                ch['connection_init'][-1] = max(1, ch['connection_init'][-1])  # input ladder
                ch['connection_init'][-2] = max(1, ch['connection_init'][-2])
                ch['side_init'][-1] = max(1, ch['side_init'][-1])
                last_backbone = num_conn - 1
                if last_backbone >= 0 and ch['side_init'][last_backbone] != 0:
                    ch['connection_init'][last_backbone] = max(1, ch['connection_init'][last_backbone])
            elif self.model_name == 't5-base':
                ch['connection_init'][-2] = max(1, ch['connection_init'][-2])
                ch['connection_init'][-1] = max(1, ch['connection_init'][-1])
                ch['connection_init'][-3] = max(1, ch['connection_init'][-3])
                ch['connection_init'][11] = max(1, ch['connection_init'][11])
                ch['side_init'][11] = max(1, ch['side_init'][11])
                ch['side_init'][-1] = max(1, ch['side_init'][-1])
                last_backbone = num_conn - 1
                if last_backbone >= 0 and ch['side_init'][last_backbone] != 0:
                    ch['connection_init'][last_backbone] = max(1, ch['connection_init'][last_backbone])
            else:
                raise NotImplementedError

            # rf-level coupling & rf_anchor
            ch['rf_conn_idx'], ch['rf_side_idx'], ch['upsample'] = self._enforce_rf_level_and_anchor(
                ch['rf_conn_idx'], ch['rf_side_idx'], ch['side_init'], ch['upsample'], rf_pool
            )

        return offspring_list

    def mutation2(self, offspring_list, sigma, forbid_List, search_space):
        """
        Mutate ops and RF indices, then re-apply:
          - side-empty ⇒ connection-empty
          - model-specific invariants (conditional)
          - rf_level coupling & rf_anchor for input-ladder/upsample
        """
        Rf = len(search_space['reduction_factor'])
        Oc = len(search_space['connection'])
        Os = len(search_space['side'])

        num_side, num_conn, num_input = self._counts()
        rf_pool = self.search_space['reduction_factor']

        for ch in offspring_list:
            # mutate connection ops (including input ladder entries at the end)
            if 'connection_init' not in forbid_List:
                conn = np.array(ch['connection_init'])
                noise = np.random.normal(0, sigma, num_conn + num_input).astype(int)
                conn = np.clip(conn + noise, 0, Oc - 1)
                ch['connection_init'] = conn.tolist()

            # mutate side ops
            if 'side_init' not in forbid_List:
                side = np.array(ch['side_init'])
                noise = np.random.normal(0, sigma, num_side).astype(int)
                side = np.clip(side + noise, 0, Os - 1)
                ch['side_init'] = side.tolist()

            # mutate RF indices
            if self.rf_level == 2:
                # mutate a single global RF and propagate
                g = ch['upsample']  # treat upsample as the global seed
                g = int(np.clip(g + int(np.round(np.random.normal(0, sigma/4))), 0, Rf - 1))
                ch['upsample'] = g
                ch['rf_conn_idx'] = [g] * (num_conn + num_input)
                ch['rf_side_idx'] = [g] * num_side
            else:
                if 'rf_conn_idx' not in forbid_List:
                    rfc = np.array(ch['rf_conn_idx'])
                    noise = np.random.normal(0, sigma/4, num_conn + num_input).astype(int)
                    rfc = np.clip(rfc + noise, 0, Rf - 1)
                    ch['rf_conn_idx'] = rfc.tolist()

                if 'rf_side_idx' not in forbid_List:
                    rfs = np.array(ch['rf_side_idx'])
                    noise = np.random.normal(0, sigma/4, num_side).astype(int)
                    rfs = np.clip(rfs + noise, 0, Rf - 1)
                    ch['rf_side_idx'] = rfs.tolist()

                # occasional upsample flip
                if random.random() < 0.3:
                    ch['upsample'] = random.randrange(Rf)

            # ===== re-apply architectural limitations & rf policies =====
            # side empty -> connection empty
            self._enforce_side_conn_empty(ch['side_init'], ch['connection_init'])

            # model-specific invariants (conditional)
            if self.model_name == 'resnet':
                if num_side >= 3:
                    ch['side_init'][-3:] = [max(1, s) for s in ch['side_init'][-3:]]
                ch['connection_init'][-1] = max(1, ch['connection_init'][-1])  # input ladder
                ch['connection_init'][-2] = max(1, ch['connection_init'][-2])
                ch['side_init'][-4] = max(1, ch['side_init'][-4])
            elif self.model_name in ['vit-base', 'vgg', 'alexnet', 'llama']:
                ch['connection_init'][-1] = max(1, ch['connection_init'][-1])  # input ladder
                ch['connection_init'][-2] = max(1, ch['connection_init'][-2])
                ch['side_init'][-1] = max(1, ch['side_init'][-1])
                last_backbone = num_conn - 1
                if last_backbone >= 0 and ch['side_init'][last_backbone] != 0:
                    ch['connection_init'][last_backbone] = max(1, ch['connection_init'][last_backbone])
            elif self.model_name == 't5-base':
                # two input ladder + two last connection layer
                ch['connection_init'][-2] = max(1, ch['connection_init'][-2])
                ch['connection_init'][-1] = max(1, ch['connection_init'][-1])
                ch['connection_init'][-3] = max(1, ch['connection_init'][-3])
                ch['connection_init'][11] = max(1, ch['connection_init'][11])
                ch['side_init'][11] = max(1, ch['side_init'][11])
                ch['side_init'][-1] = max(1, ch['side_init'][-1])
                last_backbone = num_conn - 1
                if last_backbone >= 0 and ch['side_init'][last_backbone] != 0:
                    ch['connection_init'][last_backbone] = max(1, ch['connection_init'][last_backbone])
            else:
                raise NotImplementedError

            # rf-level coupling & rf_anchor
            ch['rf_conn_idx'], ch['rf_side_idx'], ch['upsample'] = self._enforce_rf_level_and_anchor(
                ch['rf_conn_idx'], ch['rf_side_idx'], ch['side_init'], ch['upsample'], rf_pool
            )

        return offspring_list

    def ladder_config_once_for_all(self, search_space, rf=None):
        if rf is not None:
            print(f'generate random LST config with fixed reduction factor: {rf}')
        init_cfg = self.ladder_config_gen_step_1(search_space, rf=rf)
        fixed_cfg = self.ladder_config_gen_step_2(init_cfg)
        final_cfg = self.ladder_config_gen_step_3(fixed_cfg, search_space)
        return final_cfg

    def get_good_ladder_config(self, search_space, rf=4, con_type='regular'):
        """
        Quick sanity config: non-empty everywhere (respecting morphisms),
        consistent with rf_level & rf_anchor.
        """
        rf_pool = search_space['reduction_factor']
        rf_idx = rf_pool.index(rf) if rf in rf_pool else 0
        con_idx = search_space['connection'].index(con_type)

        num_side, num_conn, num_input = self._counts()

        # start with all-ones (non-empty), then enforce policies
        connection_init = [con_idx for _ in range(num_conn)] + [con_idx for _ in range(num_input)]
        side_init = [1 for _ in range(num_side)]

        # base RFs
        if self.rf_level == 2:
            rf_conn_idx = [rf_idx] * (num_conn + num_input)
            rf_side_idx = [rf_idx] * num_side
            upsample = rf_idx
        else:
            rf_conn_idx = [rf_idx] * (num_conn + num_input)
            rf_side_idx = [rf_idx] * num_side
            upsample = rf_idx

        # enforce everything once
        cfg = {
            'rf_conn_idx': rf_conn_idx,
            'rf_side_idx': rf_side_idx,
            'connection_init': connection_init,
            'side_init': side_init,
            'upsample': upsample,
        }
        fixed_cfg = self.ladder_config_gen_step_2(cfg)
        final_cfg = self.ladder_config_gen_step_3(fixed_cfg, search_space)
        return final_cfg

    #  select mating pool algorithm
    def _topk_indices(self, fitness, n_sel):
        f = np.asarray(fitness, dtype=float).copy()
        idxs = []
        for _ in range(min(n_sel, len(f))):
            i = int(np.argmax(f))
            idxs.append(i)
            f[i] = -np.inf
        return idxs

    # ---- NSGA-II helpers ----
    def _nsga2_fast_non_dominated_sort(self, scores, lats):
        N = len(scores)
        S = [set() for _ in range(N)]
        n = np.zeros(N, dtype=int)
        fronts = [[]]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dom_i = (scores[i] >= scores[j] and lats[i] <= lats[j]) and \
                        ((scores[i] > scores[j]) or (lats[i] < lats[j]))
                dom_j = (scores[j] >= scores[i] and lats[j] <= lats[i]) and \
                        ((scores[j] > scores[i]) or (lats[j] < lats[i]))
                if dom_i:
                    S[i].add(j)
                elif dom_j:
                    n[i] += 1
            if n[i] == 0:
                fronts[0].append(i)
        f = 0
        while fronts[f]:
            Q = []
            for i in fronts[f]:
                for j in S[i]:
                    n[j] -= 1
                    if n[j] == 0:
                        Q.append(j)
            f += 1
            fronts.append(Q)
        fronts.pop()  # last empty
        return fronts

    def _nsga2_crowding_distance(self, idxs, scores, lats):
        if not idxs:
            return {}
        dist = {i: 0.0 for i in idxs}

        # score (maximize)
        s_idx = sorted(idxs, key=lambda i: scores[i])
        smin, smax = scores[s_idx[0]], scores[s_idx[-1]]
        dist[s_idx[0]] = dist[s_idx[-1]] = float('inf')
        if smax > smin:
            rng = (smax - smin)
            for k in range(1, len(s_idx) - 1):
                dist[s_idx[k]] += (scores[s_idx[k + 1]] - scores[s_idx[k - 1]]) / rng

        # latency (minimize)
        l_idx = sorted(idxs, key=lambda i: lats[i])
        lmin, lmax = lats[l_idx[0]], lats[l_idx[-1]]
        dist[l_idx[0]] = dist[l_idx[-1]] = float('inf')
        if lmax > lmin:
            rng = (lmax - lmin)
            for k in range(1, len(l_idx) - 1):
                # inverted because smaller latency is better
                dist[l_idx[k]] += (lats[l_idx[k - 1]] - lats[l_idx[k + 1]]) / rng
        return dist

    def _nsga2_indices(self, net_info_list, n_sel, elite_frac=0.2, fallback_fitness=None):
        scores = np.array([d['score'] for d in net_info_list], dtype=float)
        lats = np.array([d['latency'] for d in net_info_list], dtype=float)
        fronts = self._nsga2_fast_non_dominated_sort(scores, lats)

        n_elite = int(max(1, round(np.clip(elite_frac, 0.10, 0.20) * n_sel)))
        selected = []

        # 1) take elites from first front by crowding distance
        if fronts:
            cd = self._nsga2_crowding_distance(fronts[0], scores, lats)
            elites = sorted(fronts[0], key=lambda i: cd[i], reverse=True)[:n_elite]
            selected.extend(elites)

        # 2) fill remaining by walking fronts (high crowding first)
        remaining = n_sel - len(selected)
        for f in fronts:
            pool = [i for i in f if i not in selected]
            if not pool:
                continue
            cd = self._nsga2_crowding_distance(pool, scores, lats)
            order = sorted(pool, key=lambda i: cd[i], reverse=True)
            take = min(len(order), remaining)
            selected.extend(order[:take])
            remaining -= take
            if remaining == 0:
                break

        # 3) emergency pad with scalar fitness if still short
        if remaining > 0 and fallback_fitness is not None:
            rest = np.argsort(-np.asarray(fallback_fitness))  # descending
            for i in rest:
                if int(i) not in selected:
                    selected.append(int(i))
                    remaining -= 1
                    if remaining == 0:
                        break
        return selected[:n_sel]

    # ---- Tournament / SUS (single-objective) ----
    def _tournament_indices(self, fitness, n_sel, k=3, unique=False):
        N = len(fitness)
        rng = np.random.default_rng()
        chosen, available = [], set(range(N))
        f = np.asarray(fitness, dtype=float)
        for _ in range(n_sel):
            if unique and len(available) < max(1, k):
                unique = False
            pool_from = list(available if unique else range(N))
            pool = rng.choice(pool_from, size=min(k, N), replace=False)
            winner = int(pool[np.argmax(f[pool])])
            chosen.append(winner)
            if unique:
                available.discard(winner)
        return chosen

    def _sus_indices(self, fitness, n_sel):
        f = np.asarray(fitness, dtype=float)
        f = f - np.min(f) + 1e-12  # shift to non-negative
        total = float(np.sum(f))
        if total <= 0:
            return list(np.random.default_rng().choice(len(f), size=n_sel, replace=True))
        step = total / n_sel
        start = np.random.default_rng().uniform(0, step)
        pointers = start + step * np.arange(n_sel)

        idxs, cum, i = [], 0.0, 0
        for p in pointers:
            while cum + f[i] < p:
                cum += f[i]
                i = (i + 1) % len(f)
            idxs.append(i)
        return idxs

    # ---- Random immigrant (fresh sample using your init pipeline) ----
    def _random_sample_parent(self):
        """Create a fresh random parent using your existing init logic."""
        init_cfg = self.ladder_config_gen_step_1(self.search_space)
        fixed = self.ladder_config_gen_step_2(init_cfg)
        return fixed

    # ---- Public API ----
    def select_mating_pool(
            self,
            parent_list,
            fitness_list,
            n_mating,
            net_info_list=None,
            method="topk",  # "topk" | "nsga2" | "tournament" | "sus"
            elite_frac=0.2,  # NSGA-II only
            immigrants_frac=0.10,  # % of random immigrants included
            tournament_k=3,  # tournament size
            unique_tournament=False
    ):
        """
        Versatile selection with optional random immigrants.

        - "topk":           classic elitist top-k on fitness_list.
        - "nsga2":          multi-objective (score ↑, latency ↓) + crowding.
        - "tournament":     k-way tournament on fitness_list.
        - "sus":            stochastic universal sampling on fitness_list.

        Returns: list of selected parents (length = n_mating).
        """
        N = len(parent_list)
        assert N > 0 and n_mating > 0

        # how many slots to fill by selection vs immigrants
        n_imm = int(round(max(0.0, min(0.9, immigrants_frac)) * n_mating))
        n_sel = max(1, n_mating - n_imm)

        # choose indices by method
        method = (method or "topk").lower()
        if method == "topk":
            idxs = self._topk_indices(fitness_list, n_sel)

        elif method == "nsga2":
            if net_info_list is None:
                # fall back to scalar top-k if multi-obj metrics missing
                idxs = self._topk_indices(fitness_list, n_sel)
            else:
                idxs = self._nsga2_indices(
                    net_info_list=net_info_list,
                    n_sel=n_sel,
                    elite_frac=elite_frac,
                    fallback_fitness=fitness_list
                )

        elif method == "tournament":
            idxs = self._tournament_indices(fitness_list, n_sel, k=tournament_k, unique=unique_tournament)

        elif method == "sus":
            idxs = self._sus_indices(fitness_list, n_sel)

        else:
            # safe fallback
            idxs = self._topk_indices(fitness_list, n_sel)

        selected = [parent_list[i] for i in idxs]

        # add random immigrants
        if n_imm > 0:
            immigrants = [self._random_sample_parent() for _ in range(n_imm)]
            selected = selected + immigrants

        return selected[:n_mating]

    def config_to_readable_text(self, ladder_config, search_space, verbose=True):
        """
        Pretty print, showing per-layer RF next to each op.
        Marks input-ladder connection entries at the tail.
        """
        conn_ops = search_space['connection']
        side_ops = search_space['side']
        rf_pool = search_space['reduction_factor']

        connection = ladder_config['connection']
        side = ladder_config['side']
        upsample_idx = ladder_config['upsample']

        num_side, num_conn, num_input = self._counts()

        def decode(encoded, ops):
            op_idx = encoded % len(ops)
            rf_idx = encoded // len(ops)
            return ops[op_idx], rf_pool[rf_idx]

        conn_lines = []
        for i, e in enumerate(connection):
            op, rf = decode(e, conn_ops)
            tag = f"[Conn {i}]" if i < num_conn else f"[InputConn {i - num_conn}]"
            conn_lines.append(f"{tag} {op} (rf={rf})")

        side_lines = []
        for i, e in enumerate(side):
            op, rf = decode(e, side_ops)
            side_lines.append(f"[Side {i}] {op} (rf={rf})")

        if verbose:
            print('-' * 12 + ' ladder (per-layer rf) ' + '-' * 12)
            print(f"head rf index: {upsample_idx} -> rf={rf_pool[upsample_idx]} | "
                  f"rf_level={self.rf_level}, rf_anchor={self.rf_anchor}")
            width = 56
            print(f'{"connection":<{width}} | {"side":<{width}}')
            M = max(len(conn_lines), len(side_lines))
            for i in range(M):
                t1 = conn_lines[i] if i < len(conn_lines) else ""
                t2 = side_lines[i] if i < len(side_lines) else ""
                print(f'{t1:<{width}} | {t2:<{width}}')
            print('-' * 80)
        return {'upsample': upsample_idx, 'connection_layers': conn_lines, 'side_layers': side_lines}


if __name__ == '__main__':
    # cifar10 = Cifar10RunConfig()
    # print(cifar10.data_config)

    train = cifar10.train_loader
    print("Length of Train Loader: ", len(train))
    test = cifar10.test_loader
    print("Length of Test Loader: ", len(test))

    ga = GeneticArchSearchConfig()
    print(ga.config)
