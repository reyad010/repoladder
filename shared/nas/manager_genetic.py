# Description: Manager for running the NAS algorithm

import sys
sys.path.append('../utils/')

import os, time, json, copy
from datetime import timedelta

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from nas.config import GeneticArchSearchConfig
from utils.pytorch_utils import *
from utils.model_deploy import *
from nas.nas_score import _ZC_METHODS
import socket

# =========================
# ===== Run  Manager  =====
# =========================
class RunManager:
    def __init__(self, path, net, run_config, out_log=True, model_training=False):
        self.path = path
        self.net = net
        self.run_config = run_config
        self.out_log = out_log

        self._logs_path, self._save_path = None, None
        self.best_acc = 0
        self.start_epoch = 0
        self.model_training = model_training

        # Initialize model
        # load_path = f"./state_dict/{self.net.model_name}_{self.run_config.dataset}.pth"
        # if os.path.exists(load_path) and not self.model_training:
        #     state_dict = torch.load(load_path, map_location=torch.device("cuda:0"))
        #     self.net.init_model(state_dict)
        #     print("Model loaded successfully")
        # else:
        #     print("Model loaded with ranodm weights")

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.net = torch.nn.DataParallel(self.net)
            self.net.to(self.device)
            cudnn.benchmark = True
        else:
            raise ValueError

        # net info
        self.print_net_info()

        self.criterion = nn.CrossEntropyLoss()
        if self.run_config is not None:
            if self.run_config.no_decay_keys:
                keys = self.run_config.no_decay_keys.split('#')
                self.optimizer = self.run_config.build_optimizer([
                    self.net.module.get_parameters(keys, mode='exclude'),
                    self.net.module.get_parameters(keys, mode='include'),
                ])
            else:
                self.optimizer = self.run_config.build_optimizer(self.net.module.weight_parameters())

        self.unique_number = None

    # -------- save path and log path --------
    @property
    def save_path(self):
        if self._save_path is None:
            save_path = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return self._logs_path

    # -------- net info --------
    def net_inference_latency(self, net=None, device='gpu-cpu-nonblocking', channels_last=False, gpu='0', **kwargs):
        dup_net = net
        dup_net = dup_net.module if hasattr(dup_net, 'module') else dup_net
        inference_latency = measure_inference_latency(dup_net, device, channels_last=channels_last, gpu=gpu, **kwargs)
        return inference_latency

    def net_model_size(self):
        model_size = self.net.module.get_model_size()
        return model_size

    def print_net_info(self):
        if self.out_log:
            print(self.net)

        model_size = 0
        if self.out_log:
            model_size = self.net_model_size()
            print('Model size: {}'.format(model_size))

        net_info = {'archi': str(self.net), 'param': '%.2fM' % (model_size / 1e6)}
        with open('%s/net_info.txt' % self.logs_path, 'w') as fout:
            fout.write(json.dumps(net_info, indent=4) + '\n')

    # -------- save and load models --------
    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.net.module.state_dict()}
        if model_name is None:
            model_name = 'checkpoint.pth.tar'
        checkpoint['dataset'] = self.run_config.dataset
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)
        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline().rstrip('\n')
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = '%s/checkpoint.pth.tar' % self.save_path
                with open(latest_fname, 'w') as fout:
                    fout.write(model_fname + '\n')
            if self.out_log:
                print('=> loading checkpoint from %s' % model_fname)
            checkpoint = torch.load(model_fname) if torch.cuda.is_available() else torch.load(model_fname, map_location='cpu')
            self.net.module.load_state_dict(checkpoint['state_dict'])
            new_manual_seed = int(time.time())
            torch.manual_seed(new_manual_seed)
            torch.cuda.manual_seed_all(new_manual_seed)
            np.random.seed(new_manual_seed)
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                self.best_acc = checkpoint['best_acc']
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.out_log:
                print("=> loaded checkpoint '{}'".format(model_fname))
        except Exception:
            if self.out_log:
                print('fail to load checkpoint from %s' % self.save_path)

    def save_config(self, print_info=True):
        os.makedirs(self.path, exist_ok=True)
        net_save_path = os.path.join(self.path, 'net.config')
        json.dump(self.net.module.config, open(net_save_path, 'w'), indent=4)
        if print_info:
            print('Network configs dump to %s' % net_save_path)
        run_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4)
        if print_info:
            print('Run configs dump to %s' % run_save_path)

    # -------- train & validate --------
    def write_log(self, log_str, prefix, should_print=True):
        if prefix in ['valid', 'latency_test']:
            with open(os.path.join(self.logs_path, 'valid_console.txt'), 'a') as fout:
                fout.write(log_str + '\n'); fout.flush()
        if prefix in ['valid', 'latency_test', 'train']:
            with open(os.path.join(self.logs_path, 'train_console.txt'), 'a') as fout:
                if prefix in ['valid', 'latency_test']:
                    fout.write('=' * 10)
                fout.write(log_str + '\n'); fout.flush()
        if should_print:
            print(log_str)

    # -------- phantom Jacobian score (kept for reference) --------
    def score(self, net=None, x=None, target=None):
        grads = {}
        for name, param in net.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone()
        net.zero_grad()
        x.requires_grad_(True)
        y = net(x); temp_y = y.clone()
        y.backward(torch.ones_like(y))
        jacobs = x.grad.detach().reshape(x.size(0), -1).cpu().numpy()
        corrs = np.corrcoef(jacobs)
        corrs_ori = copy.deepcopy(corrs)
        if np.isnan(corrs).any():
            if np.isfinite(corrs).any():
                print("------Warning------: NaN values found. Replacing NaNs with mean of finite values.")
                finite_mean = np.nanmean(corrs[np.isfinite(corrs)])
                corrs = np.where(np.isnan(corrs), finite_mean, corrs)
                np.save("jacobs.npy", jacobs); np.save("corrs.npy", corrs); np.save("corrs_ori.npy", corrs_ori)
                np.save("temp_y.npy", temp_y.detach().cpu().numpy())
            else:
                print("------Warning------: No finite values, using default mean 0.")
                corrs = np.where(np.isnan(corrs), 0, corrs)
                np.save("jacobs.npy", jacobs); np.save("corrs.npy", corrs)
                np.save("temp_y.npy", temp_y.detach().cpu().numpy())
        elif np.isinf(corrs).any():
            print("------Warning------: Inf values found. Clamping to finite range.")
            max_val = np.max(corrs[np.isfinite(corrs)]); min_val = np.min(corrs[np.isfinite(corrs)])
            corrs = np.where(corrs == np.inf, max_val, corrs)
            corrs = np.where(corrs == -np.inf, min_val, corrs)
        v, _ = np.linalg.eig(corrs)
        v = np.where(v <= 0, 1e-5, v)
        k = 1e-5
        score = -np.sum(np.log(v + k) + 1. / (v + k))
        for name, param in net.named_parameters():
            if name in grads: param.grad = grads[name]
        return score

    def nas_score(self, net, x, target, score_type='naswot', **kwargs):
        return _ZC_METHODS[score_type](net, x, target, **kwargs)

    def acc_score(self):
        nBatch = len(self.run_config.train_loader); epoch = 0
        previous_state_dict = self.net.state_dict()
        self.summarize_trainable_ratio(self.net)
        def train_log_func(epoch_, i, batch_time, data_time, losses, top1, top5, lr):
            batch_log = ('Train [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\tLoss {losses.val:.4f} ({losses.avg:.4f})\t'
                         'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'
                         ).format(epoch_ + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time, losses=losses, top1=top1)
            batch_log += '\tlr {lr:.5f}'.format(lr=lr); return batch_log
        print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
        end = time.time()
        train_top1, train_top5 = self.train_one_epoch(
            lambda i: self.run_config.adjust_learning_rate(self.optimizer, epoch, i, nBatch),
            lambda i, batch_time, data_time, losses, top1, top5, new_lr:
                train_log_func(1, i, batch_time, data_time, losses, top1, top5, new_lr),
        )
        time_per_epoch = time.time() - end
        seconds_left = int((self.run_config.n_epochs - epoch - 1) * time_per_epoch)
        print('Time per epoch: %s, Est. complete in: %s' % (str(timedelta(seconds=time_per_epoch)),
                                                            str(timedelta(seconds=seconds_left))))
        val_loss, val_acc, val_acc5 = self.validate(is_test=False, return_top5=True)
        is_best = val_acc > self.best_acc
        self.best_acc = max(self.best_acc, val_acc)
        val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})'.format(
            epoch + 1, self.run_config.n_epochs, val_loss, val_acc, self.best_acc)
        val_log += '\tTrain top-1 {top1.avg:.3f}'.format(top1=train_top1)
        self.net.load_state_dict(previous_state_dict)
        return val_acc if isinstance(val_acc, float) else val_acc.item()

    def summarize_trainable_ratio(self, model: nn.Module):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = 100.0 * trainable_params / total_params if total_params > 0 else float('nan')
        for name, params in model.named_parameters():
            if params.requires_grad:
                print(f'trainable layer: {name}, params: {params.numel()}')
        print(f"{'Total params:':<20} {total_params / 1e6:.2f} M")
        print(f"{'Trainable params:':<20} {trainable_params / 1e6:.2f} M")
        print(f"{'Trainable %:':<20} {ratio:.2f}%")
        return total_params, trainable_params, ratio

# ===============================
# ===== Arch Search Manager =====
# ===============================
class ArchSearchRunManager:
    def __init__(self, args, super_net, run_config, arch_search_config: GeneticArchSearchConfig, loader, model_training=False):
        self.loader = loader
        self.run_manager = RunManager(args.path, super_net, run_config, False, model_training)
        self.arch_search_config = arch_search_config
        self.arch_search_config.model_name = super_net.model_name

        self.reward_metric = ['score', 'latency']  # accuracy/score, latency/flops
        self.channels_last = True if self.arch_search_config.model_name in ['vgg', 'resnet'] else False
        print(f'channels_last: {self.channels_last}')
        self.nas_attn_mask = None  # set for NLP models; None for vision

        self.gpu = args.gpu
        self.score_type = 'naswot'
        self.n_generation = args.n_generation
        self.mutation_sigma = float(args.mutation_sigma)
        self.npop = args.npop  # population size

        # Optional knobs (if not present in config, fall back)
        self.selection_method = getattr(self.arch_search_config, 'selection_method', 'topk')
        self.elite_frac = getattr(self.arch_search_config, 'elite_frac', 0.2)
        self.immigrants_frac = getattr(self.arch_search_config, 'immigrants_frac', 0.25)
        self.tournament_k = getattr(self.arch_search_config, 'tournament_k', 3)

        # bookkeeping
        self.host_name = socket.gethostname()
        self.test_device = getattr(args, 'test_device', 'single')
        print(f'host: {self.host_name}, device: {self.test_device}')
        self.init_latency_test = True
        self.nas_data = None
        self.n_generation_list = []  # for plotting

        # global best + Pareto hall-of-fame
        self.best_reward_so_far = -float('inf')
        self.best_config_so_far = None
        self.best_info_so_far = None
        self.hall_of_fame = []  # list of {'key','config','info','reward'}
        self.count = 0
        print(f'immigrants_frac: {self.immigrants_frac}')

    # ---- helpers for HoF ----
    def _config_key(self, cfg: dict) -> str:
        return json.dumps(cfg['all'], sort_keys=True)

    def _dominates(self, a: dict, b: dict) -> bool:
        s, l = self.reward_metric[0], self.reward_metric[1]
        better_or_equal = (a[s] >= b[s]) and (a[l] <= b[l])
        strictly_better = (a[s] > b[s]) or (a[l] < b[l])
        return better_or_equal and strictly_better

    def _update_hof(self, cfg: dict, info: dict, reward: float):
        key = self._config_key(cfg)
        for e in self.hall_of_fame:
            if e['key'] == key:
                if reward > e['reward']:
                    e['config'] = copy.deepcopy(cfg); e['info'] = info; e['reward'] = float(reward)
                return
        dominated_idxs = []
        for i, e in enumerate(self.hall_of_fame):
            if self._dominates(info, e['info']):
                dominated_idxs.append(i)
            elif self._dominates(e['info'], info):
                return
        for i in reversed(dominated_idxs):
            self.hall_of_fame.pop(i)
        self.hall_of_fame.append({'key': key, 'config': copy.deepcopy(cfg), 'info': info, 'reward': float(reward)})

    @property
    def net(self):
        return self.run_manager.net.module

    def write_log(self, log_str, prefix, should_print=True, end='\n'):
        with open(os.path.join(self.run_manager.logs_path, '%s.log' % prefix), 'a') as fout:
            fout.write(log_str + end); fout.flush()
        if should_print: print(log_str)

    def validate(self):
        self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.run_manager.run_config.test_batch_size
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = False
        if hasattr(self.net, 'set_chosen_op_active'):
            self.net.set_chosen_op_active(); self.net.unused_modules_off()
        valid_res = self.run_manager.validate(is_test=False, use_train_mode=False, return_top5=True)
        model_size = self.run_manager.net_model_size()
        if hasattr(self.net, 'set_chosen_op_active'):
            self.net.unused_modules_back()
        return valid_res, model_size

    # ---- main GA loop (simplified) ----
    def train(self, fix_net_weights=False):
        print("Dataset: ", self.loader.task)

        # population & GA params
        npop = self.npop
        n_generation = self.n_generation
        n_mating = int(npop / 2)

        # simple mutation decay
        sigma = float(self.mutation_sigma)
        sigma_decay = 0.85
        sigma_floor = 0.20

        # init population
        parents_init = [self.arch_search_config.ladder_config_gen_step_1(self.net.search_space) for _ in range(npop)]
        parents_fixed = [self.arch_search_config.ladder_config_gen_step_2(p) for p in parents_init]
        parents = [self.arch_search_config.ladder_config_gen_step_3(p, self.net.search_space) for p in parents_fixed]
        final_config = None
        final_config_init = None

        # data_loader = self.run_manager.run_config.train_loader
        self.net.freeze_params()  # freeze supernet weights during architecture search

        for gen_idx in range(n_generation):
            self.nth_generation_score = []
            print(f'Generation {gen_idx + 1}/{n_generation}  (sigma={sigma:.2f})')

            # evaluate current population
            t0 = time.time()
            fitness_list, net_info_list = self.genetic_update_step(parents)
            elapsed = time.time() - t0

            # update global best + hall-of-fame
            for cfg_i, info_i, rew_i in zip(parents, net_info_list, fitness_list):
                if rew_i > self.best_reward_so_far:
                    self.best_reward_so_far = float(rew_i)
                    self.best_config_so_far = copy.deepcopy(cfg_i)
                    self.best_info_so_far = info_i
                self._update_hof(cfg_i, info_i, float(rew_i))

            # record per-gen for plotting
            self.n_generation_list.append(copy.deepcopy(self.nth_generation_score))

            # log gen summary
            mean_reward = float(np.mean(fitness_list)) if len(fitness_list) else float('nan')
            print(f'  [GA] time={elapsed:.3f}s  mean_reward={mean_reward:.4f}  best={np.max(fitness_list):.4f}')

            # best of this gen (for info)
            best_idx = int(np.argmax(np.asarray(fitness_list)))
            final_config = parents[best_idx]
            final_config_init = parents_init[best_idx]
            print(f"  [Best@Gen] reward={fitness_list[best_idx]:.4f}, "
                  f"{self.reward_metric[0]}={net_info_list[best_idx][self.reward_metric[0]]:.4f}, "
                  f"{self.reward_metric[1]}={net_info_list[best_idx][self.reward_metric[1]]}")

            # selection (method from config), with immigrants
            parents_sel = self.arch_search_config.select_mating_pool(
                parent_list=parents,
                fitness_list=fitness_list,
                n_mating=n_mating,
                net_info_list=net_info_list,
                method=self.selection_method,      # "topk" | "nsga2" | "tournament" | "sus"
                elite_frac=self.elite_frac,
                immigrants_frac=self.immigrants_frac,
                tournament_k=self.tournament_k
            )

            # crossover + mutation
            offspring_cross = self.arch_search_config.crossover(parents_sel, npop - n_mating)
            offspring_mut = self.arch_search_config.mutation(
                offspring_cross, sigma, forbid_List=[], search_space=self.net.search_space
            )

            # next generation population
            parents_init = parents_sel + offspring_mut
            parents_fixed = [self.arch_search_config.ladder_config_gen_step_2(p) for p in parents_init]
            parents = [self.arch_search_config.ladder_config_gen_step_3(p, self.net.search_space) for p in parents_fixed]

            # decay sigma
            sigma = max(sigma_floor, sigma * sigma_decay)

        # end of GA
        print(f'trend of nas score and latency: {self.n_generation_list}')
        hof_net_info_list = [e["info"] for e in getattr(self, "hall_of_fame", [])]
        print(f"HOF net_info list: {hof_net_info_list}")
        print(f'final config init: {final_config_init}')
        print(f'final config: {final_config}')

        # replace with global best across all gens
        if self.best_config_so_far is not None:
            final_config = self.best_config_so_far
            print(f'[GLOBAL BEST] reward={self.best_reward_so_far:.4f}, '
                  f"{self.reward_metric[0]}={self.best_info_so_far[self.reward_metric[0]]:.4f}, "
                  f"{self.reward_metric[1]}={self.best_info_so_far[self.reward_metric[1]]}")

        # persist hall-of-fame (Pareto frontier)
        try:
            hof_serializable = [
                {
                    'reward': e['reward'],
                    self.reward_metric[0]: e['info'][self.reward_metric[0]],
                    self.reward_metric[1]: e['info'][self.reward_metric[1]],
                    'config': e['config'],
                } for e in self.hall_of_fame
            ]
            with open(os.path.join(self.run_manager.logs_path, 'hall_of_fame.json'), 'w') as f:
                json.dump(hof_serializable, f, indent=2)
            print(f'HOF size={len(self.hall_of_fame)} saved to hall_of_fame.json')
        except Exception:
            pass

        print(f'best config: {final_config}')
        # inform tee_server to close
        from utils.tensor_transfer_channel import send_value
        send_value('iter', ['__done__'], 'list')
        return final_config

    # ---- evaluate one population ----
    def genetic_update_step(self, parents_list):
        assert isinstance(self.arch_search_config, GeneticArchSearchConfig)
        self.run_manager.net.train()

        # one batch cached for NASWOT
        if self.nas_data is None:
            raw = next(iter(self.loader.test_loader))
            if isinstance(raw, dict):
                # NLP loader returns a dict: {input_ids, attention_mask, labels}
                self.nas_data = [
                    raw['input_ids'].to(self.run_manager.device),
                    raw['labels'].to(self.run_manager.device),
                ]
                self.nas_attn_mask = raw['attention_mask'].to(self.run_manager.device)
            else:
                self.nas_data = list(raw)
                self.nas_data[0] = self.nas_data[0].to(self.run_manager.device)
                self.nas_data[1] = self.nas_data[1].to(self.run_manager.device)
                self.nas_attn_mask = None

        reward_buffer, net_info_buffer = [], []
        test_device = self.test_device  # still supported by measure_inference_latency
        print('='*50 + f'Genetic Algorithm' + '='*50 + '\n')
        for cfg in parents_list:
            print('=' * 50 + f"[backbone] cfg {self.count + 1}" + '=' * 50, flush=True)
            print(f'cfg: {self.count + 1}: {cfg}')
            self.count += 1
            torch.cuda.empty_cache()
            # materialize a normal net from sampled config

            Net = self.net.convert_to_normal_net(cfg['all'])
            if hasattr(Net, 'init_weight'):
                print(f'Net init weight')
                Net.init_weight()
            if getattr(Net, '_supernet_ready', False):
                # Backbone is a shared reference already on GPU and frozen;
                # side layers are already on the correct device from the super-net.
                # Calling .to() or .freeze_backbone() would traverse 7B params for nothing.
                Net.backbone.eval()
            else:
                Net.to('cuda:0')
                Net.freeze_backbone()

            # metric-1: NASWOT score (or accuracy)
            try:
                if self.reward_metric[0] == 'score':
                    score_kwargs = {}
                    if self.nas_attn_mask is not None:
                        score_kwargs['attention_mask'] = self.nas_attn_mask
                    metric1 = self.run_manager.nas_score(Net, self.nas_data[0], self.nas_data[1], self.score_type, **score_kwargs)
                elif self.reward_metric[0] == 'accuracy':
                    metric1 = self.run_manager.acc_score()
                else:
                    metric1 = 0.0
            except Exception as e:
                print(f'nas score issue for: {e}')
                self.arch_search_config.config_to_readable_text(cfg, self.net.search_space)
                print(cfg)
                raise

            # metric-2: latency / flops
            try:
                if self.reward_metric[1] == 'flops':
                    metric2 = self.net.get_flops(self.nas_data[0], verbose=False)
                elif self.reward_metric[1] == 'latency':
                    metric2 = self.run_manager.net_inference_latency(
                        net=Net, device=test_device, channels_last=self.channels_last, gpu=self.gpu,
                        lst_config=cfg, share_root='./file_share',
                        transfer_type=self.net.transfer_type, init_latency_test=self.init_latency_test
                    )[test_device]
                    # metric2
                    self.init_latency_test = False
                else:
                    metric2 = 0.0
            except Exception as e:
                print(f'latency issue for: {e}')
                self.arch_search_config.config_to_readable_text(cfg, self.net.search_space)
                print(cfg)
                raise

            net_info = {self.reward_metric[0]: metric1, self.reward_metric[1]: metric2}
            net_info_buffer.append(net_info)

            reward, delta_a, delta_b = self.arch_search_config.calculate_reward(net_info)
            if delta_a < 0: delta_a = 1e-6
            if delta_b < 0: delta_b = 1e-6
            self.nth_generation_score.append((delta_a, delta_b, net_info))

            reward_buffer.append(reward)
            del Net
        return reward_buffer, net_info_buffer
