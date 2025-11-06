#!/usr/bin/env python3
##########################################################################
# File Name: eval.py
# Author: AI Assistant  
# Created Time: Wed 17 Sep 2025
# Description: Unit test script for RAPiDock - evaluate on a fixed set of 
#              21 protein complexes for consistent benchmarking
##########################################################################

import os
import sys
import pandas as pd
import glob
import shutil
import argparse
import yaml
from pathlib import Path
import subprocess
import time
from datetime import datetime

# Add current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .inference_optimized import main as inference_main
from .utils.inference_parsing import get_parser
from .rmsd_bound_pyrosetta import rmsd as calculate_rmsd

# Fixed list of 21 protein complexes for unit testing
UNIT_TEST_COMPLEXES = [
    '7lfd', '7dng', '7rly', '7qv5', '7q45', '7mb9', '7kpu', '7k7r', '7xhg', 
    '7dsb', '7ds6', '7kei', '7pvt', '7qp4', '8gom', '7doh', '7oxg', '7xat', 
    '7q42', '7zy4', '7bbg'
]

def get_all_complexes(testdata_dir="testdataset/RefPepDB-RecentSet"):
    """获取所有可用的复合物列表"""
    complex_dirs = []
    if os.path.exists(testdata_dir):
        for item in os.listdir(testdata_dir):
            complex_path = os.path.join(testdata_dir, item)
            if os.path.isdir(complex_path):
                # 检查是否包含必要的文件
                peptide_file = os.path.join(complex_path, f"{item}_peptide.pdb")
                protein_file = os.path.join(complex_path, f"{item}_protein_pocket.pdb")
                sequence_file = os.path.join(complex_path, f"{item}_peptide_sequence")
                
                if all(os.path.exists(f) for f in [peptide_file, protein_file, sequence_file]):
                    complex_dirs.append(item)
                else:
                    print(f"Warning: {item} is missing required files, skipping...")
    
    return sorted(complex_dirs)

def get_unit_test_complexes(all_complexes, testdata_dir="testdataset/RefPepDB-RecentSet"):
    """获取固定的单元测试复合物列表，验证它们是否存在"""
    available_complexes = []
    missing_complexes = []
    
    for complex_name in UNIT_TEST_COMPLEXES:
        if complex_name in all_complexes:
            available_complexes.append(complex_name)
        else:
            # 检查是否在data目录中（如7bbg）
            data_complex_dir = os.path.join("data", complex_name)
            if os.path.exists(data_complex_dir):
                peptide_file = os.path.join(data_complex_dir, f"{complex_name}_peptide.pdb")
                protein_file = os.path.join(data_complex_dir, f"{complex_name}_protein_pocket.pdb")
                if os.path.exists(peptide_file) and os.path.exists(protein_file):
                    available_complexes.append(complex_name)
                    continue
            
            missing_complexes.append(complex_name)
    
    print(f"Unit test complexes - Available: {len(available_complexes)}, Missing: {len(missing_complexes)}")
    if missing_complexes:
        print(f"Missing complexes: {', '.join(missing_complexes)}")
    
    print(f"Testing {len(available_complexes)} complexes: {', '.join(available_complexes)}")
    return available_complexes

def create_evaluation_csv(selected_complexes, testdata_dir, output_csv):
    """创建用于推理的CSV文件"""
    data = []
    
    for complex_name in selected_complexes:
        # 首先检查testdata_dir
        complex_dir = os.path.join(testdata_dir, complex_name)
        sequence_file = os.path.join(complex_dir, f"{complex_name}_peptide_sequence")
        protein_pdb = os.path.join(complex_dir, f"{complex_name}_protein_pocket.pdb")
        
        # 如果在testdata_dir中不存在，检查data目录
        if not os.path.exists(sequence_file):
            complex_dir = os.path.join("data", complex_name)
            sequence_file = os.path.join(complex_dir, f"{complex_name}_peptide_sequence")
            protein_pdb = os.path.join(complex_dir, f"{complex_name}_protein_pocket.pdb")
        
        # 读取肽序列
        if os.path.exists(sequence_file):
            with open(sequence_file, 'r') as f:
                peptide_sequence = f.read().strip()
        else:
            print(f"Warning: Sequence file not found for {complex_name}, skipping...")
            continue
        
        data.append({
            'complex_name': complex_name,
            'protein_description': protein_pdb,
            'peptide_description': peptide_sequence
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Created evaluation CSV: {output_csv}")
    return output_csv

def prepare_ground_truth_files(selected_complexes, testdata_dir, output_dir):
    """准备ground truth文件用于RMSD计算"""
    ground_truth_info = {}
    
    for complex_name in selected_complexes:
        complex_output_dir = os.path.join(output_dir, complex_name)
        os.makedirs(complex_output_dir, exist_ok=True)
        
        # 首先检查testdata_dir
        source_protein = os.path.join(testdata_dir, complex_name, f"{complex_name}_protein_pocket.pdb")
        source_peptide = os.path.join(testdata_dir, complex_name, f"{complex_name}_peptide.pdb")
        
        # 如果在testdata_dir中不存在，检查data目录
        if not os.path.exists(source_protein):
            source_protein = os.path.join("data", complex_name, f"{complex_name}_protein_pocket.pdb")
            source_peptide = os.path.join("data", complex_name, f"{complex_name}_peptide.pdb")
        
        dest_protein = os.path.join(complex_output_dir, f"{complex_name}_protein_raw.pdb")
        dest_peptide = os.path.join(complex_output_dir, f"{complex_name}_peptide_raw.pdb")
        
        if os.path.exists(source_protein) and os.path.exists(source_peptide):
            shutil.copy2(source_protein, dest_protein)
            shutil.copy2(source_peptide, dest_peptide)
            
            ground_truth_info[complex_name] = {
                'protein_file': dest_protein,
                'peptide_file': dest_peptide
            }
            
            print(f"Prepared ground truth files for {complex_name}")
        else:
            print(f"Warning: Ground truth files not found for {complex_name}")
    
    return ground_truth_info

def run_inference(csv_file, output_dir, config_file="default_inference_args.yaml"):
    """运行推理"""
    print(f"\n{'='*60}")
    print(f"Running inference...")
    print(f"{'='*60}")
    
    # 构建推理参数
    parser = get_parser()
    args = parser.parse_args([])  # 创建空的参数对象
    
    # 设置基本参数
    args.protein_peptide_csv = csv_file
    args.output_dir = output_dir
    args.config = None
    
    # 如果配置文件存在，加载配置
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(args, key, value)
        print(f"Loaded configuration from {config_file}")
    else:
        # 使用默认值
        args.N = 10
        args.model_dir = "train_models/CGTensorProductEquivariantModel"
        args.ckpt = "rapidock_local.pt"
        args.scoring_function = "ref2015"
        args.batch_size = 40
        args.gpu_batch_size = 1  # 减少批处理大小以避免索引问题
        args.no_final_step_noise = True
        args.inference_steps = 16
        args.actual_steps = 16
        args.cpu = 16
        args.disable_parallel_processing = True  # 使用串行处理避免批处理索引问题
        print("Using default configuration")
    
    # 检查必要的模型文件
    model_path = os.path.join(args.model_dir, args.ckpt)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPU batch size: {args.gpu_batch_size}")
    print(f"Number of samples per complex: {args.N}")
    
    # 运行推理
    start_time = time.time()
    try:
        inference_main(args)
        inference_time = time.time() - start_time
        print(f"\nInference completed successfully in {inference_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def calculate_rmsd_for_all(selected_complexes, output_dir):
    """为所有复合物计算RMSD"""
    print(f"\n{'='*60}")
    print(f"Calculating RMSD for all complexes...")
    print(f"{'='*60}")
    
    rmsd_results = {}
    failed_complexes = []
    
    for complex_name in selected_complexes:
        complex_output_dir = os.path.join(output_dir, complex_name)
        
        if not os.path.exists(complex_output_dir):
            print(f"Warning: Output directory not found for {complex_name}")
            failed_complexes.append(complex_name)
            continue
        
        # 检查是否有预测结果文件
        prediction_files = glob.glob(os.path.join(complex_output_dir, "rank*_ref2015.pdb"))
        if not prediction_files:
            print(f"Warning: No prediction files found for {complex_name}")
            failed_complexes.append(complex_name)
            continue
        
        print(f"\nCalculating RMSD for {complex_name}...")
        try:
            # 调用RMSD计算函数
            calculate_rmsd(complex_output_dir)
            
            # 读取RMSD结果
            rmsd_file = os.path.join(complex_output_dir, "rmsd_pyrosetta.csv")
            if os.path.exists(rmsd_file):
                df = pd.read_csv(rmsd_file)
                rmsd_results[complex_name] = df
                print(f"Successfully calculated RMSD for {complex_name}")
                
                # 打印最佳结果
                if len(df) > 0:
                    best_row = df.iloc[0]  # 第一行通常是rank1，最佳结果
                    print(f"  Best result - DockQ: {best_row.get('dockq', 'N/A')}, "
                          f"CAPRI: {best_row.get('capri_class', 'N/A')}")
            else:
                print(f"Warning: RMSD file not found for {complex_name}")
                failed_complexes.append(complex_name)
                
        except Exception as e:
            print(f"Error calculating RMSD for {complex_name}: {e}")
            failed_complexes.append(complex_name)
    
    return rmsd_results, failed_complexes

def summarize_results(rmsd_results, output_dir):
    """汇总评估结果"""
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS SUMMARY")
    print(f"{'='*60}")
    
    if not rmsd_results:
        print("No valid results to summarize.")
        return
    
    # 收集所有结果
    all_results = []
    capri_counts = {'High': 0, 'Medium': 0, 'Acceptable': 0, 'Incorrect': 0}
    
    for complex_name, df in rmsd_results.items():
        if len(df) > 0:
            # 获取最佳结果（rank1）
            best_result = df.iloc[0]
            all_results.append({
                'complex_name': complex_name,
                'rank': best_result.get('id', 'N/A'),
                'dockq': best_result.get('dockq', None),
                'fnat': best_result.get('fnat', None),
                'irms': best_result.get('irms', None),
                'lrms': best_result.get('lrms', None),
                'capri_class': best_result.get('capri_class', 'Incorrect')
            })
            
            # 统计CAPRI分类
            capri_class = best_result.get('capri_class', 'Incorrect')
            if capri_class in capri_counts:
                capri_counts[capri_class] += 1
    
    # 创建汇总DataFrame
    summary_df = pd.DataFrame(all_results)
    
    # 保存详细结果
    summary_file = os.path.join(output_dir, "evaluation_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # 打印统计信息
    print(f"Total complexes evaluated: {len(all_results)}")
    print(f"\nCAPRI Classification Distribution:")
    total_complexes = len(all_results)
    for capri_class, count in capri_counts.items():
        percentage = (count / total_complexes * 100) if total_complexes > 0 else 0
        print(f"  {capri_class}: {count} ({percentage:.1f}%)")
    
    # 计算平均DockQ
    valid_dockq = [float(r['dockq']) for r in all_results if r['dockq'] is not None and r['dockq'] != 'None']
    if valid_dockq:
        avg_dockq = sum(valid_dockq) / len(valid_dockq)
        print(f"\nAverage DockQ: {avg_dockq:.3f}")
        print(f"Best DockQ: {max(valid_dockq):.3f}")
        print(f"Worst DockQ: {min(valid_dockq):.3f}")
    
    # 成功率统计
    success_rate = (capri_counts['High'] + capri_counts['Medium'] + capri_counts['Acceptable']) / total_complexes * 100
    high_quality_rate = capri_counts['High'] / total_complexes * 100
    
    print(f"\nSuccess Rate (Acceptable or better): {success_rate:.1f}%")
    print(f"High Quality Rate: {high_quality_rate:.1f}%")
    
    print(f"\nDetailed results saved to: {summary_file}")
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description="RAPiDock Unit Test - Evaluate on fixed set of 21 protein complexes")
    parser.add_argument("--testdata_dir", type=str, default="testdataset/RefPepDB-RecentSet",
                       help="Directory containing test complexes")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--config", type=str, default="default_inference_args.yaml",
                       help="Configuration file for inference parameters")
    parser.add_argument("--skip_inference", action="store_true",
                       help="Skip inference step and only calculate RMSD on existing results")
    parser.add_argument("--list_complexes", action="store_true",
                       help="List the fixed set of unit test complexes and exit")
    
    args = parser.parse_args()
    
    # 如果只是列出复合物，直接返回
    if args.list_complexes:
        print("Fixed unit test complexes (21 total):")
        for i, complex_name in enumerate(UNIT_TEST_COMPLEXES, 1):
            print(f"{i:2d}. {complex_name}")
        return 0
    
    # 设置输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"unit_test_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"RAPiDock Unit Test Script")
    print(f"{'='*60}")
    print(f"Test data directory: {args.testdata_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Configuration file: {args.config}")
    print(f"Fixed test complexes: {len(UNIT_TEST_COMPLEXES)}")
    
    # 1. 获取所有可用的复合物
    print(f"\nStep 1: Scanning available complexes...")
    all_complexes = get_all_complexes(args.testdata_dir)
    print(f"Found {len(all_complexes)} valid complexes in test dataset")
    
    # 2. 获取固定的单元测试复合物
    print(f"\nStep 2: Validating unit test complexes...")
    selected_complexes = get_unit_test_complexes(all_complexes, args.testdata_dir)
    
    if len(selected_complexes) == 0:
        print("Error: No valid unit test complexes found!")
        return 1
    
    # 3. 准备ground truth文件
    print(f"\nStep 3: Preparing ground truth files...")
    ground_truth_info = prepare_ground_truth_files(selected_complexes, args.testdata_dir, args.output_dir)
    
    # 4. 创建推理用的CSV文件
    print(f"\nStep 4: Creating evaluation CSV...")
    eval_csv = os.path.join(args.output_dir, "unit_test_complexes.csv")
    create_evaluation_csv(selected_complexes, args.testdata_dir, eval_csv)
    
    # 5. 运行推理（如果未跳过）
    if not args.skip_inference:
        print(f"\nStep 5: Running inference...")
        inference_success = run_inference(eval_csv, args.output_dir, args.config)
        if not inference_success:
            print("Error: Inference failed!")
            return 1
    else:
        print(f"\nStep 5: Skipping inference (using existing results)")
    
    # 6. 计算RMSD
    print(f"\nStep 6: Calculating RMSD...")
    rmsd_results, failed_complexes = calculate_rmsd_for_all(selected_complexes, args.output_dir)
    
    if failed_complexes:
        print(f"\nWarning: RMSD calculation failed for {len(failed_complexes)} complexes:")
        for complex_name in failed_complexes:
            print(f"  - {complex_name}")
    
    # 7. 汇总结果
    print(f"\nStep 7: Summarizing results...")
    summary_df = summarize_results(rmsd_results, args.output_dir)
    
    # 8. 单元测试结果评估
    print(f"\n{'='*60}")
    print(f"UNIT TEST RESULTS")
    print(f"{'='*60}")
    
    if len(rmsd_results) > 0:
        # 计算通过率（Acceptable或更好）
        total_tested = len(rmsd_results)
        capri_counts = {'High': 0, 'Medium': 0, 'Acceptable': 0, 'Incorrect': 0}
        
        for complex_name, df in rmsd_results.items():
            if len(df) > 0:
                capri_class = df.iloc[0].get('capri_class', 'Incorrect')
                if capri_class in capri_counts:
                    capri_counts[capri_class] += 1
        
        success_count = capri_counts['High'] + capri_counts['Medium'] + capri_counts['Acceptable']
        success_rate = success_count / total_tested * 100
        high_quality_rate = capri_counts['High'] / total_tested * 100
        
        print(f"Unit Test Summary:")
        print(f"  Total complexes tested: {total_tested}/{len(UNIT_TEST_COMPLEXES)}")
        print(f"  Success rate (≥Acceptable): {success_rate:.1f}% ({success_count}/{total_tested})")
        print(f"  High quality rate: {high_quality_rate:.1f}% ({capri_counts['High']}/{total_tested})")
        
        # 判断单元测试是否通过（可以根据需要调整阈值）
        PASS_THRESHOLD = 50.0  # 50%的成功率作为通过标准
        test_passed = success_rate >= PASS_THRESHOLD
        
        print(f"\nUnit Test Result: {'PASS' if test_passed else 'FAIL'}")
        if not test_passed:
            print(f"  (Success rate {success_rate:.1f}% < threshold {PASS_THRESHOLD}%)")
            
        return 0 if test_passed else 1
    else:
        print("Unit Test Result: FAIL (No valid results)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
