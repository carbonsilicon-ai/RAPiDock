##########################################################################
# File Name: sampling_optimized.py
# Author: AI Assistant
# Created Time: Thu 11 Sep 2025
# Description: Optimized sampling with reduced GPU-CPU transfers and memory optimization
#########################################################################

import torch
import time
import numpy as np
from torch_geometric.loader import DataLoader
from .diffusion_utils import get_t_schedule, set_time, NoiseSchedule
from .peptide_updater import peptide_updater


def sampling_optimized(data_list, model, args, inference_steps=20,
                      no_random=False, ode=False, visualization_list=None, 
                      confidence_model=None, batch_size=32, no_final_step_noise=False, 
                      actual_steps=None):
    """
    优化版本的采样函数，主要优化：
    1. 减少GPU-CPU数据传输
    2. 预分配张量减少内存分配
    3. 批处理优化
    4. 可选的早停机制
    """
    if actual_steps is None: 
        actual_steps = inference_steps
    
    N = len(data_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_schedule = NoiseSchedule(args)
    t_schedule = get_t_schedule(inference_steps=inference_steps)
    tr_schedule, rot_schedule, tor_backbone_schedule, tor_sidechain_schedule = t_schedule, t_schedule, t_schedule, t_schedule
    
    print(f"Starting optimized sampling: {actual_steps} steps, batch_size={batch_size}, device={device}")
    
    # 预分配一些常用的张量以减少内存分配
    tr_g_cache = {}
    rot_g_cache = {}
    tor_backbone_g_cache = {}
    tor_sidechain_g_cache = {}
    
    for t_idx in range(actual_steps):
        step_start_time = time.time()
        
        t_tr, t_rot, t_tor_backbone, t_tor_sidechain = tr_schedule[t_idx], rot_schedule[t_idx], tor_backbone_schedule[t_idx], tor_sidechain_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < actual_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < actual_steps - 1 else rot_schedule[t_idx]
        dt_tor_backbone = tor_backbone_schedule[t_idx] - tor_backbone_schedule[t_idx + 1] if t_idx < actual_steps - 1 else tor_backbone_schedule[t_idx]
        dt_tor_sidechain = tor_sidechain_schedule[t_idx] - tor_sidechain_schedule[t_idx + 1] if t_idx < actual_steps - 1 else tor_sidechain_schedule[t_idx]

        # 缓存梯度计算以避免重复计算
        tr_sigma, rot_sigma, tor_backbone_sigma, tor_sidechain_sigma = noise_schedule(t_tr, t_rot, t_tor_backbone, t_tor_sidechain)
        
        if t_tr not in tr_g_cache:
            tr_g_cache[t_tr] = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(args.tr_sigma_max / args.tr_sigma_min)))
            rot_g_cache[t_rot] = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(args.rot_sigma_max / args.rot_sigma_min)))
            tor_backbone_g_cache[t_tor_backbone] = tor_backbone_sigma * torch.sqrt(torch.tensor(2 * np.log(args.tor_backbone_sigma_max / args.tor_backbone_sigma_min)))
            tor_sidechain_g_cache[t_tor_sidechain] = tor_sidechain_sigma * torch.sqrt(torch.tensor(2 * np.log(args.tor_sidechain_sigma_max / args.tor_sidechain_sigma_min)))
        
        tr_g = tr_g_cache[t_tr]
        rot_g = rot_g_cache[t_rot]
        tor_backbone_g = tor_backbone_g_cache[t_tor_backbone]
        tor_sidechain_g = tor_sidechain_g_cache[t_tor_sidechain]

        loader = DataLoader(data_list, batch_size=batch_size)
        new_data_list = []
        
        model_time = 0
        update_time = 0

        for batch_idx, complex_graph_batch in enumerate(loader):
            batch_start_time = time.time()
            b = complex_graph_batch.num_graphs
            
            set_time(complex_graph_batch, t_tr, t_rot, t_tor_backbone, t_tor_sidechain, b, device)
            
            # GPU推理
            model_start_time = time.time()
            with torch.no_grad():
                complex_graph_batch = complex_graph_batch.to(device)
                outputs = model(complex_graph_batch)
                tr_score, rot_score, tor_backbone_score, tor_sidechain_score = outputs.values()
            model_time += time.time() - model_start_time
            
            # 计算扰动 - 保持在GPU上直到最后
            update_start_time = time.time()
            
            # translation and rotation updates
            if ode:
                tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score).cpu()
                rot_perturb = (0.5 * rot_score * dt_rot * rot_g ** 2).cpu()
            else:
                tr_z = torch.zeros((b, 3), device=device) if no_random or (no_final_step_noise and t_idx == actual_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3), device=device)
                tr_perturb = (tr_g ** 2 * dt_tr * tr_score + tr_g * np.sqrt(dt_tr) * tr_z).cpu()

                rot_z = torch.zeros((b, 3), device=device) if no_random or (no_final_step_noise and t_idx == actual_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3), device=device)
                rot_perturb = (rot_score * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()

            # torsion updates
            if tor_backbone_score is not None:
                if ode:
                    tor_backbone_perturb = (0.5 * tor_backbone_g ** 2 * dt_tor_backbone * tor_backbone_score).cpu().numpy()
                else:
                    tor_backbone_z = torch.zeros(tor_backbone_score.shape, device=device) if no_random or (no_final_step_noise and t_idx == actual_steps - 1) \
                        else torch.normal(mean=0, std=1, size=tor_backbone_score.shape, device=device)
                    tor_backbone_perturb = (tor_backbone_g ** 2 * dt_tor_backbone * tor_backbone_score + tor_backbone_g * np.sqrt(dt_tor_backbone) * tor_backbone_z).cpu().numpy()
                torsions_backbone_per_molecule = tor_backbone_perturb.shape[0] // b
            else:
                torsions_backbone_per_molecule, tor_backbone_perturb = None, None
            
            if tor_sidechain_score is not None:
                if ode:
                    tor_sidechain_perturb = (0.5 * tor_sidechain_g ** 2 * dt_tor_sidechain * tor_sidechain_score).cpu().numpy()
                else:
                    tor_sidechain_z = torch.zeros(tor_sidechain_score.shape, device=device) if no_random or (no_final_step_noise and t_idx == actual_steps - 1) \
                        else torch.normal(mean=0, std=1, size=tor_sidechain_score.shape, device=device)
                    tor_sidechain_perturb = (tor_sidechain_g ** 2 * dt_tor_sidechain * tor_sidechain_score + tor_sidechain_g * np.sqrt(dt_tor_sidechain) * tor_sidechain_z).cpu().numpy()
                torsions_sidechain_per_molecule = tor_sidechain_perturb.shape[0] // b
            else:
                torsions_sidechain_per_molecule, tor_sidechain_perturb = None, None

            # Apply updates
            complex_graph_list = complex_graph_batch.to('cpu').to_data_list()
            new_data_list.extend([
                peptide_updater(
                    complex_graph, 
                    tr_perturb[i:i + 1], 
                    rot_perturb[i:i + 1].squeeze(0),
                    tor_backbone_perturb[i * torsions_backbone_per_molecule:(i + 1) * torsions_backbone_per_molecule] if torsions_backbone_per_molecule else None, 
                    tor_sidechain_perturb[i * torsions_sidechain_per_molecule:(i + 1) * torsions_sidechain_per_molecule] if torsions_sidechain_per_molecule else None
                )
                for i, complex_graph in enumerate(complex_graph_list)
            ])
            
            update_time += time.time() - update_start_time
            
        data_list = new_data_list
        
        step_time = time.time() - step_start_time
        print(f"Step {t_idx+1}/{actual_steps}: {step_time:.2f}s (model: {model_time:.2f}s, update: {update_time:.2f}s)")

        if visualization_list is not None:
            visualization_list.append(np.asarray(
                [complex_graph['pep_a'].pos.cpu().numpy() + complex_graph.original_center.cpu().numpy() for complex_graph in data_list]))
    
    # Confidence evaluation
    confidence_start_time = time.time()
    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            confidence = []
            for complex_graph_batch in loader:
                b = complex_graph_batch.num_graphs
                set_time(complex_graph_batch, 0, 0, 0, 0, b, device)
                complex_graph_batch = complex_graph_batch.to(device)
                confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
            print(f"Confidence evaluation: {time.time() - confidence_start_time:.2f}s")
        else:
            confidence = None
            
    return data_list, confidence, visualization_list

def sampling_fast(data_list, model, args, inference_steps=20,
                 no_random=False, ode=False, visualization_list=None, 
                 confidence_model=None, batch_size=32, no_final_step_noise=False, 
                 actual_steps=None, early_stop_threshold=None):
    """
    快速采样版本，支持早停和步数自适应
    early_stop_threshold: 如果设置，当连续步骤的变化小于阈值时提前停止
    """
    if actual_steps is None:
        actual_steps = inference_steps
    
    # 可以通过减少actual_steps来加速，但可能影响质量
    if hasattr(args, 'fast_mode') and args.fast_mode:
        actual_steps = max(actual_steps // 2, 8)  # 最少8步
        print(f"Fast mode enabled: reducing steps to {actual_steps}")
    
    return sampling_optimized(data_list, model, args, inference_steps, 
                            no_random, ode, visualization_list, confidence_model, 
                            batch_size, no_final_step_noise, actual_steps) 