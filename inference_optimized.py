##########################################################################
# File Name: inference_optimized.py
# Author: AI Assistant  
# Created Time: Thu 11 Sep 2025
# Description: Optimized inference with batch GPU processing and async CPU scoring
#########################################################################

import os
import copy
import yaml
import torch
import MDAnalysis
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from io import StringIO
from argparse import Namespace
from MDAnalysis.coordinates.memory import MemoryReader
from torch_geometric.loader import DataListLoader
from utils.inference_parsing import get_parser
from utils.utils import get_model, ExponentialMovingAverage
from utils.inference_utils import InferenceDataset, set_nones
from utils.peptide_updater import randomize_position
from utils.sampling import sampling
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

warnings.filterwarnings("ignore")

def load_model(score_model_args, ckpt_path, device):
    model = get_model(score_model_args, no_parallel=True)
    state_dict = torch.load(
        ckpt_path, map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict["model"], strict=True)
    model = model.to(device)

    ema_weights = ExponentialMovingAverage(
        model.parameters(), decay=score_model_args.ema_rate
    )
    ema_weights.load_state_dict(state_dict["ema_weights"], device=device)
    ema_weights.copy_to(model.parameters())
    return model

def load_config(args):
    if args.config:
        # content in config file will cover the cmd input content
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value

def prepare_data(args, score_model_args):
    if args.protein_peptide_csv is not None:
        df = pd.read_csv(args.protein_peptide_csv)
        complex_name_list = set_nones(df["complex_name"].tolist())
        protein_description_list = set_nones(df["protein_description"].tolist())
        peptide_description_list = set_nones(df["peptide_description"].tolist())
    else:
        complex_name_list = [args.complex_name]
        protein_description_list = [args.protein_description]
        peptide_description_list = [args.peptide_description]
    
    complex_name_list = [
        name if name is not None else f"complex_{i}"
        for i, name in enumerate(complex_name_list)
    ]
    for name in complex_name_list:
        write_dir = f"{args.output_dir}/{name}"
        os.makedirs(write_dir, exist_ok=True)
    
    # preprocessing of initial proteins and peptides into geometric graphs
    return InferenceDataset(
        output_dir=args.output_dir,
        complex_name_list=complex_name_list,
        protein_description_list=protein_description_list,
        peptide_description_list=peptide_description_list,
        lm_embeddings=score_model_args.esm_embeddings_path_train is not None,
        lm_embeddings_pep=score_model_args.esm_embeddings_peptide_train is not None,
        conformation_type=args.conformation_type,
        conformation_partial=args.conformation_partial,
    )

def prepare_data_list(original_complex_graph, N):
    data_list = []
    nums = []
    if len(original_complex_graph["peptide_inits"]) == 1:
        data_list = [copy.deepcopy(original_complex_graph) for _ in range(N)]
    elif len(original_complex_graph["peptide_inits"]) > 1:
         for i, peptide_init in enumerate(
                    original_complex_graph["peptide_inits"]
                ):
            if i !=0:
                original_complex_graph["pep_a"].pos = (
                torch.from_numpy(
                    MDAnalysis.Universe(peptide_init).atoms.positions
                )
                - original_complex_graph.original_center
            )
            num = N - sum(nums) if i == len(original_complex_graph["peptide_inits"]) - 1 else round(
                original_complex_graph["partials"][i] / sum(original_complex_graph["partials"]) * N
            )
            nums.append(num)
            data_list.extend([copy.deepcopy(original_complex_graph) for _ in range(num)])
    return data_list

def save_predictions_async(write_dir, predict_pos, original_complex_graph, args, confidence):
    """异步版本的预测保存函数，在单独线程中运行CPU密集型任务"""
    raw_pdb = MDAnalysis.Universe(StringIO(original_complex_graph["pep"].noh_mda), format="pdb")
    peptide_unrelaxed_files = []
    
    re_order = None
    # reorder predictions based on confidence output
    if confidence is not None:
        confidence = confidence.cpu().numpy()
        re_order = np.argsort(confidence)[::-1]
        confidence = confidence[re_order]
        predict_pos = predict_pos[re_order]

    for rank, pos in enumerate(predict_pos):
        raw_pdb.atoms.positions = pos
        file_name = f"rank{rank+1}_{args.scoring_function}.pdb" if confidence is not None else f"rank{rank+1}.pdb"
        peptide_unrelaxed_file = os.path.join(write_dir, file_name)
        peptide_unrelaxed_files.append(peptide_unrelaxed_file)
        raw_pdb.atoms.write(peptide_unrelaxed_file)

    if args.scoring_function == "ref2015" or args.fastrelax:
        from utils.pyrosetta_utils import relax_score
        relaxed_poses = [peptide.replace(".pdb", "_relaxed.pdb") for peptide in peptide_unrelaxed_files]
        protein_raw_file = f"{write_dir}/{os.path.basename(write_dir)}_protein_raw.pdb"

        with multiprocessing.Pool(args.cpu) as pool:
            ref2015_scores = pool.map(
                relax_score,
                zip(
                    [protein_raw_file] * len(peptide_unrelaxed_files),
                    peptide_unrelaxed_files,
                    relaxed_poses,
                    [args.scoring_function == "ref2015"] * len(peptide_unrelaxed_files),
                ),
            )
        if ref2015_scores and ref2015_scores[0] is not None:
            re_order = np.argsort(ref2015_scores)
            score_results = [['file','ref2015score']]
            for rank, order in enumerate(re_order):
                os.rename(relaxed_poses[order], os.path.join(write_dir, f"rank{rank+1}_{args.scoring_function}.pdb"))
                score_results.append([f"rank{rank+1}_{args.scoring_function}", f"{ref2015_scores[order]:.2f}"])
            open(os.path.join(write_dir, "ref2015_score.csv"),'w').write('\n'.join([','.join(i) for i in score_results]))
    
    return re_order if re_order is not None else 0

def process_complex_gpu_only(model, confidence_model, score_model_args, args, original_complex_graph):
    """只进行GPU采样，返回结果供后续CPU处理"""
    # data_list_prepare
    N = args.N
    data_list = prepare_data_list(original_complex_graph, N)
    randomize_position(data_list, False, score_model_args.tr_sigma_max)

    print('len data_list', len(data_list), args.batch_size)

    start_time = time.time()
    data_list, confidence, visualization_list = sampling(
        data_list=data_list,
        model=model,
        args=score_model_args,
        batch_size=args.batch_size,
        no_final_step_noise=args.no_final_step_noise,
        inference_steps=args.inference_steps,
        actual_steps=(
            args.actual_steps
            if args.actual_steps is not None
            else args.inference_steps
        ),
        visualization_list=None,
        confidence_model=confidence_model,
    )
    gpu_time = time.time() - start_time
    print(f"GPU sampling time: {gpu_time:.2f} seconds")

    predict_pos = np.asarray(
        [
            complex_graph["pep_a"].pos.cpu().numpy()
            + original_complex_graph.original_center.cpu().numpy()
            for complex_graph in data_list
        ]
    )
    
    return predict_pos, confidence, visualization_list

def process_batch_gpu(model, confidence_model, score_model_args, args, batch_data):
    """批量处理多个复合物的GPU采样"""
    all_data_list = []
    complex_info = []
    
    # 准备所有数据
    for original_complex_graph, write_dir, complex_name in batch_data:
        N = args.N
        data_list = prepare_data_list(original_complex_graph, N)
        randomize_position(data_list, False, score_model_args.tr_sigma_max)
        
        complex_info.append({
            'original_complex_graph': original_complex_graph,
            'write_dir': write_dir,
            'complex_name': complex_name,
            'start_idx': len(all_data_list),
            'count': len(data_list)
        })
        
        all_data_list.extend(data_list)
    
    print(f'GPU batch processing {len(batch_data)} complexes with {len(all_data_list)} total samples', args.batch_size, confidence_model)
    
    # 批量GPU采样
    start_time = time.time()
    all_data_list, confidence_all, _ = sampling(
        data_list=all_data_list,
        model=model,
        args=score_model_args,
        batch_size=args.batch_size,
        no_final_step_noise=args.no_final_step_noise,
        inference_steps=args.inference_steps,
        actual_steps=(
            args.actual_steps
            if args.actual_steps is not None
            else args.inference_steps
        ),
        visualization_list=None,
        confidence_model=confidence_model,
    )
    gpu_time = time.time() - start_time
    print(f"Batch GPU sampling time: {gpu_time:.2f} seconds")
    
    # 分离结果
    results = []
    for info in complex_info:
        start_idx = info['start_idx']
        end_idx = start_idx + info['count']
        
        # 提取对应的数据
        complex_data_list = all_data_list[start_idx:end_idx]
        complex_confidence = confidence_all[start_idx:end_idx] if confidence_all is not None else None
        
        # 计算预测位置
        predict_pos = np.asarray([
            complex_graph["pep_a"].pos.cpu().numpy()
            + info['original_complex_graph'].original_center.cpu().numpy()
            for complex_graph in complex_data_list
        ])
        
        results.append({
            'original_complex_graph': info['original_complex_graph'],
            'write_dir': info['write_dir'],
            'complex_name': info['complex_name'],
            'predict_pos': predict_pos,
            'confidence': complex_confidence
        })
    
    return results

def main_optimized(args):
    """优化版本的主函数"""
    # 输入参数配置
    load_config(args)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(f"{args.model_dir}/model_parameters.yml") as f:
        score_model_args = Namespace(**yaml.full_load(f))

    # 准备数据
    inference_dataset = prepare_data(args, score_model_args)
    inference_loader = DataListLoader(
        dataset=inference_dataset, batch_size=1, shuffle=False
    )

    # 加载模型
    model = load_model(score_model_args, f"{args.model_dir}/{args.ckpt}", device)

    # 加载置信度模型
    confidence_model = None
    if args.scoring_function == "confidence":
        with open(f"{args.confidence_model_dir}/model_parameters.yml") as f:
            confidence_args = Namespace(**yaml.full_load(f))

        confidence_model = get_model(
            confidence_args, no_parallel=True, confidence_mode=True
        )
        state_dict = torch.load(
            f"{args.confidence_model_dir}/{args.confidence_ckpt}",
            map_location=torch.device("cpu"),
        )
        confidence_model.load_state_dict(state_dict["model"], strict=True)
        confidence_model = confidence_model.to(device)
        confidence_model.eval()

    # 收集有效的复合物数据
    valid_complexes = []
    skipped = 0
    
    for idx, original_complex_graph in enumerate(inference_loader):
        if not original_complex_graph[0].success:
            skipped += 1
            print(f"Skipping {inference_dataset.complex_names[idx]} - dataset preparation failed")
            continue
            
        valid_complexes.append({
            'idx': idx,
            'original_complex_graph': original_complex_graph[0],
            'write_dir': f"{args.output_dir}/{inference_dataset.complex_names[idx]}",
            'complex_name': inference_dataset.complex_names[idx]
        })

    print(f"Processing {len(valid_complexes)} valid complexes (skipped {skipped})")

    # 根据参数决定批处理大小
    gpu_batch_size = getattr(args, 'gpu_batch_size', 2)  # 默认每批处理2个复合物
    
    # 使用线程池进行异步CPU处理
    cpu_futures = []
    total_start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=2) as executor:  # 1个用于CPU处理，1个备用
        # 批量处理GPU任务
        for i in range(0, len(valid_complexes), gpu_batch_size):
            batch = valid_complexes[i:i+gpu_batch_size]
            batch_data = [(c['original_complex_graph'], c['write_dir'], c['complex_name']) for c in batch]
            
            print(f"\nProcessing GPU batch {i//gpu_batch_size + 1}/{(len(valid_complexes)-1)//gpu_batch_size + 1}")
            
            # GPU批处理
            gpu_results = process_batch_gpu(model, confidence_model, score_model_args, args, batch_data)
            
            # 立即提交CPU任务到线程池
            for result in gpu_results:
                future = executor.submit(
                    save_predictions_async,
                    result['write_dir'],
                    result['predict_pos'],
                    result['original_complex_graph'],
                    args,
                    result['confidence']
                )
                cpu_futures.append((future, result['complex_name']))
                print(f"Submitted CPU task for {result['complex_name']}")
        
        # 等待所有CPU任务完成
        print("\nWaiting for CPU scoring tasks to complete...")
        completed = 0
        for future, complex_name in cpu_futures:
            try:
                start_time = time.time()
                result = future.result()
                cpu_time = time.time() - start_time
                completed += 1
                print(f"Completed CPU scoring for {complex_name} ({completed}/{len(cpu_futures)}) - Time: {cpu_time:.2f}s")
            except Exception as e:
                print(f"CPU scoring failed for {complex_name}: {e}")

    total_time = time.time() - total_start_time
    print(f"\nOptimized processing completed in {total_time:.2f} seconds")
    print(f"Average time per complex: {total_time/len(valid_complexes):.2f} seconds")
    print(f"Results are in {args.output_dir}")

def main_sequential(args):
    """原始串行版本的主函数"""
    # 输入参数配置
    load_config(args)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(f"{args.model_dir}/model_parameters.yml") as f:
        score_model_args = Namespace(**yaml.full_load(f))

    inference_dataset = prepare_data(args, score_model_args)
    inference_loader = DataListLoader(
        dataset=inference_dataset, batch_size=1, shuffle=False
    )

    model = load_model(score_model_args, f"{args.model_dir}/{args.ckpt}", device)

    # load confidence model
    confidence_model = None
    if args.scoring_function == "confidence":
        with open(f"{args.confidence_model_dir}/model_parameters.yml") as f:
            confidence_args = Namespace(**yaml.full_load(f))

        confidence_model = get_model(
            confidence_args, no_parallel=True, confidence_mode=True
        )
        state_dict = torch.load(
            f"{args.confidence_model_dir}/{args.confidence_ckpt}",
            map_location=torch.device("cpu"),
        )
        confidence_model.load_state_dict(state_dict["model"], strict=True)
        confidence_model = confidence_model.to(device)
        confidence_model.eval()

    failures, skipped = 0, 0
    print("Size of test dataset: ", len(inference_dataset))

    for idx, original_complex_graph in tqdm(enumerate(inference_loader)):
        if not original_complex_graph[0].success:
            skipped += 1
            print(
                f"HAPPENING | The inference dataset did not contain {inference_dataset.complex_names[idx]} for {inference_dataset.peptide_descriptions[idx]} and {inference_dataset.protein_descriptions[idx]}. We are skipping this complex."
            )
            continue
        try:
            # GPU处理
            predict_pos, confidence, _ = process_complex_gpu_only(
                model, confidence_model, score_model_args, args, original_complex_graph[0]
            )
            
            # CPU处理
            start_time = time.time()
            re_order = save_predictions_async(
                f"{args.output_dir}/{inference_dataset.complex_names[idx]}",
                predict_pos,
                original_complex_graph[0],
                args,
                confidence
            )
            cpu_time = time.time() - start_time
            print(f"CPU scoring time: {cpu_time:.2f} seconds")
            
        except Exception as e:
            print("Failed on", original_complex_graph[0]["name"], e)
            failures += 1

    print(f"Failed for {failures} complexes")
    print(f"Skipped {skipped} complexes")
    print(f"Results are in {args.output_dir}")

def main_optimized_with_data(args, inference_dataset, inference_loader, model, confidence_model, score_model_args):
    """优化版本的主函数，使用预准备的数据"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 收集有效的复合物数据
    valid_complexes = []
    skipped = 0
    
    for idx, original_complex_graph in enumerate(inference_loader):
        if not original_complex_graph[0].success:
            skipped += 1
            print(f"Skipping {inference_dataset.complex_names[idx]} - dataset preparation failed")
            continue
            
        valid_complexes.append({
            'idx': idx,
            'original_complex_graph': original_complex_graph[0],
            'write_dir': f"{args.output_dir}/{inference_dataset.complex_names[idx]}",
            'complex_name': inference_dataset.complex_names[idx]
        })

    print(f"Processing {len(valid_complexes)} valid complexes (skipped {skipped})")

    # 根据参数决定批处理大小
    gpu_batch_size = getattr(args, 'gpu_batch_size', 2)  # 默认每批处理2个复合物
    
    # 使用线程池进行异步CPU处理
    cpu_futures = []
    total_start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=2) as executor:  # 1个用于CPU处理，1个备用
        # 批量处理GPU任务
        for i in range(0, len(valid_complexes), gpu_batch_size):
            batch = valid_complexes[i:i+gpu_batch_size]
            batch_data = [(c['original_complex_graph'], c['write_dir'], c['complex_name']) for c in batch]
            
            print(f"\nProcessing GPU batch {i//gpu_batch_size + 1}/{(len(valid_complexes)-1)//gpu_batch_size + 1}")
            
            # GPU批处理
            gpu_results = process_batch_gpu(model, confidence_model, score_model_args, args, batch_data)
            
            # 立即提交CPU任务到线程池
            for result in gpu_results:
                future = executor.submit(
                    save_predictions_async,
                    result['write_dir'],
                    result['predict_pos'],
                    result['original_complex_graph'],
                    args,
                    result['confidence']
                )
                cpu_futures.append((future, result['complex_name']))
                print(f"Submitted CPU task for {result['complex_name']}")
        
        # 等待所有CPU任务完成
        print("\nWaiting for CPU scoring tasks to complete...")
        completed = 0
        for future, complex_name in cpu_futures:
            try:
                start_time = time.time()
                result = future.result()
                cpu_time = time.time() - start_time
                completed += 1
                print(f"Completed CPU scoring for {complex_name} ({completed}/{len(cpu_futures)}) - Time: {cpu_time:.2f}s")
            except Exception as e:
                print(f"CPU scoring failed for {complex_name}: {e}")

    total_time = time.time() - total_start_time
    print(f"\nOptimized processing completed in {total_time:.2f} seconds")
    print(f"Average time per complex: {total_time/len(valid_complexes):.2f} seconds")
    print(f"Results are in {args.output_dir}")

def main_sequential_with_data(args, inference_dataset, inference_loader, model, confidence_model, score_model_args):
    """串行版本的主函数，使用预准备的数据"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    failures, skipped = 0, 0
    print("Size of test dataset: ", len(inference_dataset))

    for idx, original_complex_graph in tqdm(enumerate(inference_loader)):
        if not original_complex_graph[0].success:
            skipped += 1
            print(
                f"HAPPENING | The inference dataset did not contain {inference_dataset.complex_names[idx]} for {inference_dataset.peptide_descriptions[idx]} and {inference_dataset.protein_descriptions[idx]}. We are skipping this complex."
            )
            continue
        try:
            # GPU处理
            predict_pos, confidence, _ = process_complex_gpu_only(
                model, confidence_model, score_model_args, args, original_complex_graph[0]
            )
            
            # CPU处理
            start_time = time.time()
            re_order = save_predictions_async(
                f"{args.output_dir}/{inference_dataset.complex_names[idx]}",
                predict_pos,
                original_complex_graph[0],
                args,
                confidence
            )
            cpu_time = time.time() - start_time
            print(f"CPU scoring time: {cpu_time:.2f} seconds")
            
        except Exception as e:
            print("Failed on", original_complex_graph[0]["name"], e)
            failures += 1

    print(f"Failed for {failures} complexes")
    print(f"Skipped {skipped} complexes")
    print(f"Results are in {args.output_dir}")

def main(args):
    """主入口函数"""
    if hasattr(args, 'disable_parallel_processing') and args.disable_parallel_processing:
        print("Using sequential processing")
        main_sequential(args)
    else:
        print("Using optimized batch processing")
        main_optimized(args)

if __name__ == "__main__":
    _args = get_parser().parse_args()
    main(_args) 