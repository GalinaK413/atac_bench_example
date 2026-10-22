# -*- coding: utf-8 -*-

import numpy as np
import torch
from tqdm import tqdm
import json
import pyBigWig
from alphagenome.data import genome
from alphagenome.models import dna_client
import pandas as pd
import os
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# ===== CONFIGURATION =====

csv_file = 'samples_all.csv'
output_dir = ''
api_key = ''
batching = False
batch_size = 500
bin_size = 1
eval_window_size = 1000
jsd_pseudocount = 1e-16

# ===== FUNCTIONS =====

def compute_log_total_count_correlation(pred_sum, truth_sum):
    """Compute Pearson r on log-transformed total counts per region."""
    pred_log = np.log1p(pred_sum)
    truth_log = np.log1p(truth_sum)
    
    if np.std(pred_log) > 0 and np.std(truth_log) > 0:
        corr, _ = pearsonr(pred_log, truth_log)
        return corr
    else:
        return np.nan


def compute_jensen_shannon_distance(pred_values, truth_values, pseudocount=1e-16):
    """Compute Jensen-Shannon distance between predicted and ground truth signal distributions."""
    pred = pred_values + pseudocount
    truth = truth_values + pseudocount
    
    pred_prob = pred / np.sum(pred)
    truth_prob = truth / np.sum(truth)
    
    js_distance = jensenshannon(pred_prob, truth_prob)
    
    return js_distance if not np.isnan(js_distance) else np.nan


def compute_binned_correlation(pred_values, truth_values, bin_size):
    """Compute Pearson correlation at specified binning resolution."""
    if bin_size == 1:
        if np.std(pred_values) > 0 and np.std(truth_values) > 0:
            return np.corrcoef(pred_values, truth_values)[0, 1]
        else:
            return np.nan
    
    effective_len = (len(pred_values) // bin_size) * bin_size
    
    if effective_len < bin_size:
        return np.nan
    
    pred_binned = pred_values[:effective_len].reshape(-1, bin_size).sum(axis=1)
    truth_binned = truth_values[:effective_len].reshape(-1, bin_size).sum(axis=1)
    
    if np.std(pred_binned) > 0 and np.std(truth_binned) > 0:
        return np.corrcoef(pred_binned, truth_binned)[0, 1]
    else:
        return np.nan


def get_evaluation_window(interval, window_size=1000):
    """Get a window of specified size centered at the peak (midpoint of interval)."""
    center = (interval.start + interval.end) // 2
    window_start = center - window_size // 2
    window_end = window_start + window_size
    
    return genome.Interval(
        chromosome=interval.chromosome,
        start=window_start,
        end=window_end
    )


def predict_with_reverse_complement(model, pred_interval, eval_interval, biosample, data_type):
    """
    Run prediction on both forward and reverse complement strands and average.
    
    Args:
        model: AlphaGenome model
        pred_interval: Full prediction interval (resized to model input length)
        eval_interval: Evaluation window interval
        biosample: Biosample ontology term
        data_type: Data type for prediction
    
    Returns:
        Averaged prediction values for the evaluation window
    """
    # ===== FORWARD STRAND PREDICTION =====
    interval_forward = genome.Interval(
        chromosome=pred_interval.chromosome,
        start=pred_interval.start,
        end=pred_interval.end,
        strand='+'
    )
    
    pred_forward = model.predict_interval(
        interval=interval_forward,
        requested_outputs=[dna_client.OutputType[str(data_type)]],
        ontology_terms=[biosample],
    )
    
    pred_forward_eval = getattr(pred_forward, data_type.lower()).slice_by_interval(
        eval_interval, match_resolution=True
    )
    pred_values_forward = np.asarray(pred_forward_eval.values).squeeze()
    
    # ===== REVERSE STRAND PREDICTION =====
    interval_reverse = genome.Interval(
        chromosome=pred_interval.chromosome,
        start=pred_interval.start,
        end=pred_interval.end,
        strand='-'
    )
    
    pred_reverse = model.predict_interval(
        interval=interval_reverse,
        requested_outputs=[dna_client.OutputType[str(data_type)]],
        ontology_terms=[biosample],
    )
    
    pred_reverse_eval = getattr(pred_reverse, data_type.lower()).slice_by_interval(
        eval_interval, match_resolution=True
    )
    pred_values_reverse = np.asarray(pred_reverse_eval.values).squeeze()
    
    # Average forward and reverse complement predictions
    pred_values_avg = (pred_values_forward + pred_values_reverse) / 2.0
    
    return pred_values_avg


def evaluate_sample(sample_name, biosample, bw_path, bed, data_type, 
                   model, seq_length, batch_size, bin_size, output_dir, 
                   eval_window_size=1000, jsd_pseudocount=1e-16):
    """Evaluate a single sample and return metrics."""
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {sample_name}")
    print(f"{'='*60}")
    
    regions_df = pd.read_table(bed, names=['chr', 'start', 'end', 'name', 'value', 'strand', 
                                           'value1', 'value2', 'value3', 'summit'])
    truth_bw = pyBigWig.open(bw_path)
    
    if batching and len(regions_df) > batch_size:
        batch_df = regions_df.sample(n=batch_size, random_state=42)
        batch_df.reset_index(inplace=True, drop=True)
    else:
        batch_df = regions_df
    
    all_pred_values = []
    all_truth_values = []
    per_sample_sum_preds = []
    per_sample_sum_targs = []
    per_sample_js_distances = []
    total_pred_sum = 0.0
    total_target_sum = 0.0
    failed_regions = 0
    
    print(f"Processing {len(batch_df)} regions...")
    print(f"Evaluation window: {eval_window_size}bp centered at peak")
    print(f"Using reverse-complement averaging (forward + reverse strands)")
    print(f"JS pseudocount: {jsd_pseudocount}")
    
    with tqdm(total=len(batch_df), desc=f"  {sample_name}") as pbar:
        for row in range(len(batch_df)):
            try:
                orig_interval = genome.Interval(
                    chromosome=batch_df.at[row, 'chr'],
                    start=batch_df.at[row, 'start'],
                    end=batch_df.at[row, 'end']
                )
                
                eval_interval = get_evaluation_window(orig_interval, eval_window_size)
                pred_interval = eval_interval.resize(seq_length)
                
                pred_values = predict_with_reverse_complement(
                    model=model,
                    pred_interval=pred_interval,
                    eval_interval=eval_interval,
                    biosample=biosample,
                    data_type=data_type
                )
                
                truth_values = np.nan_to_num(truth_bw.values(
                    eval_interval.chromosome,
                    eval_interval.start,
                    eval_interval.end,
                    numpy=True
                ))
                
                all_pred_values.extend(pred_values.tolist())
                all_truth_values.extend(truth_values.tolist())
                
                pred_sum = np.sum(pred_values)
                truth_sum = np.sum(truth_values)
                per_sample_sum_preds.append(pred_sum)
                per_sample_sum_targs.append(truth_sum)
                
                if np.sum(pred_values) > 0 and np.sum(truth_values) > 0:
                    jsd = compute_jensen_shannon_distance(pred_values, truth_values, jsd_pseudocount)
                    per_sample_js_distances.append(jsd)
                else:
                    per_sample_js_distances.append(np.nan)
                
                total_pred_sum += float(pred_sum)
                total_target_sum += float(truth_sum)
                
            except Exception as e:
                print(f"\n  Error processing region {row}: {e}")
                failed_regions += 1
                # Fill with NaN for failed regions (will be excluded from global correlation)
                all_pred_values.extend([np.nan] * eval_window_size)
                all_truth_values.extend([np.nan] * eval_window_size)
                per_sample_sum_preds.append(0)
                per_sample_sum_targs.append(0)
                per_sample_js_distances.append(np.nan)
            
            pbar.update(1)
    
    truth_bw.close()
    
    if failed_regions > 0:
        print(f"\n  Warning: {failed_regions} regions failed processing")
    
    # ===== COMPUTE GLOBAL PEARSON CORRELATION =====
    # Remove any NaN values (from failed regions)
    pred_array = np.array(all_pred_values)
    truth_array = np.array(all_truth_values)
    
    valid_mask = ~(np.isnan(pred_array) | np.isnan(truth_array))
    pred_clean = pred_array[valid_mask]
    truth_clean = truth_array[valid_mask]
    
    if len(pred_clean) > 1 and np.std(pred_clean) > 0 and np.std(truth_clean) > 0:
        global_pearson_r, global_pearson_pvalue = pearsonr(pred_clean, truth_clean)
    else:
        global_pearson_r = np.nan
        global_pearson_pvalue = np.nan
    
    if bin_size > 1 and len(pred_clean) >= bin_size:
        global_pearson_r_binned = compute_binned_correlation(pred_clean, truth_clean, bin_size)
    else:
        global_pearson_r_binned = global_pearson_r
    
    if len(per_sample_sum_preds) > 1 and len(per_sample_sum_targs) > 1:
        log_count_corr = compute_log_total_count_correlation(
            np.array(per_sample_sum_preds),
            np.array(per_sample_sum_targs)
        )
    else:
        log_count_corr = np.nan
    
    valid_jsd = [j for j in per_sample_js_distances if not np.isnan(j)]
    
    results = {
        "sample_name": sample_name,
        "biosample": biosample,
        "data_type": data_type,
        "n_regions": len(batch_df),
        "n_failed_regions": failed_regions,
        "n_valid_regions": len(batch_df) - failed_regions,
        "n_valid_bases": len(pred_clean),
        "eval_window_size": eval_window_size,
        "jsd_pseudocount": jsd_pseudocount,
        "reverse_complement": True,
        f"global_pearson_r_{bin_size}bp": float(global_pearson_r_binned) if not np.isnan(global_pearson_r_binned) else np.nan,
        f"global_pearson_r_1bp": float(global_pearson_r) if not np.isnan(global_pearson_r) else np.nan,
        f"global_pearson_pvalue_1bp": float(global_pearson_pvalue) if not np.isnan(global_pearson_pvalue) else np.nan,
        "log_total_count_pearson_r": float(log_count_corr) if not np.isnan(log_count_corr) else np.nan,
        "mean_js_distance": float(np.mean(valid_jsd)) if valid_jsd else np.nan,
        "median_js_distance": float(np.median(valid_jsd)) if valid_jsd else np.nan,
        "total_pred_sum": float(total_pred_sum),
        "total_truth_sum": float(total_target_sum),
        "ratio_pred_truth": float(total_pred_sum / total_target_sum) if total_target_sum > 0 else np.nan,
        "total_bases_evaluated": len(pred_clean),
        "per_sample_js_distances": per_sample_js_distances,
        "per_sample_pred_sums": per_sample_sum_preds,
        "per_sample_truth_sums": per_sample_sum_targs
    }
    
    # Save individual sample results
    sample_output_dir = os.path.join(output_dir, sample_name.replace(' ', '_'))
    os.makedirs(sample_output_dir, exist_ok=True)
    
    with open(os.path.join(sample_output_dir, f"{sample_name}_results.json"), "w") as f:
        results_serializable = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                results_serializable[k] = v.tolist()
            elif isinstance(v, (np.float32, np.float64)):
                results_serializable[k] = float(v)
            elif isinstance(v, list):
                results_serializable[k] = [float(x) if isinstance(x, (np.floating,)) else x for x in v]
            else:
                results_serializable[k] = v
        json.dump(results_serializable, f, indent=2)
    
    # Print summary to terminal
    print(f"\n  Results for {sample_name}:")
    print(f"  Valid regions: {results['n_valid_regions']}/{results['n_regions']}")
    print(f"  Total bases evaluated: {results['total_bases_evaluated']:,}")
    print(f"  Global Pearson r (1bp): {results['global_pearson_r_1bp']:.4f}" if not np.isnan(results['global_pearson_r_1bp']) else "  Global Pearson r (1bp): N/A")
    print(f"  Global Pearson r ({bin_size}bp): {results[f'global_pearson_r_{bin_size}bp']:.4f}" if not np.isnan(results[f'global_pearson_r_{bin_size}bp']) else f"  Global Pearson r ({bin_size}bp): N/A")
    print(f"  Log total count Pearson r: {results['log_total_count_pearson_r']:.4f}" if not np.isnan(results['log_total_count_pearson_r']) else "  Log total count Pearson r: N/A")
    print(f"  Mean JS distance: {results['mean_js_distance']:.4f}" if not np.isnan(results['mean_js_distance']) else "  Mean JS distance: N/A")
    
    return results


def plot_comparison_barplots(all_results, output_dir, bin_size):
    """Create bar plots comparing metrics across samples"""
    
    samples = [r['sample_name'] for r in all_results]
    pearson_r = [r[f'global_pearson_r_{bin_size}bp'] if not np.isnan(r[f'global_pearson_r_{bin_size}bp']) else 0 for r in all_results]
    log_count_r = [r['log_total_count_pearson_r'] if not np.isnan(r['log_total_count_pearson_r']) else 0 for r in all_results]
    js_distance = [r['mean_js_distance'] if not np.isnan(r['mean_js_distance']) else 0 for r in all_results]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    bar_color = '#4682B4'
    
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14
    })
    
    # Plot 1: Global Pearson r (concatenated across all regions)
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(samples)), pearson_r, color=bar_color, edgecolor='black', linewidth=1.2)
    ax1.set_xticks(range(len(samples)))
    ax1.set_xticklabels(samples, rotation=45, ha='right', fontsize=14)
    ax1.set_ylabel(f'Global Pearson r ({bin_size}bp)', fontsize=16, fontweight='bold')
    ax1.set_title(f'Global Pearson Correlation\n({bin_size}bp bins, all bases concatenated)', fontsize=18, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars1, pearson_r):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # Plot 2: Log total count Pearson r
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(samples)), log_count_r, color=bar_color, edgecolor='black', linewidth=1.2)
    ax2.set_xticks(range(len(samples)))
    ax2.set_xticklabels(samples, rotation=45, ha='right', fontsize=14)
    ax2.set_ylabel('Pearson r', fontsize=16, fontweight='bold')
    ax2.set_title('Log Total Count\nPearson Correlation', fontsize=18, fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars2, log_count_r):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # Plot 3: Jensen-Shannon distance
    ax3 = axes[2]
    bars3 = ax3.bar(range(len(samples)), js_distance, color=bar_color, edgecolor='black', linewidth=1.2)
    ax3.set_xticks(range(len(samples)))
    ax3.set_xticklabels(samples, rotation=45, ha='right', fontsize=14)
    ax3.set_ylabel('JS Distance', fontsize=16, fontweight='bold')
    ax3.set_title('Mean Jensen-Shannon\nDistance', fontsize=18, fontweight='bold')
    ax3.set_ylim(0, 1.05)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars3, js_distance):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    plt.suptitle('AlphaGenome Benchmarking Results\n(Forward + Reverse Complement Averaging, 1000bp Windows)', 
                 fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout(pad=3.0)
    
    # Save plot
    plot_path = os.path.join(output_dir, f'metrics_comparison_{bin_size}bp.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nBar plot saved to: {plot_path}")
    
    
# ===== MAIN EXECUTION =====

def main():
    """Main function to process all samples from CSV."""
    
    print(f"Loading sample information from: {csv_file}")
    samples_df = pd.read_csv(csv_file)
    
    print(f"Found {len(samples_df)} samples to process:")
    for _, row in samples_df.iterrows():
        print(f"  - {row['sample_name']} ({row['data_type']})")
    
    print("\nInitializing AlphaGenome model...")
    fold_to_use = dna_client.ModelVersion.FOLD_0
    seq_length = dna_client.SEQUENCE_LENGTH_1MB
    model = dna_client.create(api_key, model_version=fold_to_use)
    
    print(f"Model initialized with sequence length: {seq_length}")
    print(f"Reverse complement averaging: ENABLED")
    
    all_results = []
    
    for idx, row in samples_df.iterrows():
        sample_name = row['sample_name']
        biosample = row['biosample']
        bw_path = row['bw_path']
        bed = row['bed']
        data_type = row['data_type']
        
        if not os.path.exists(bw_path):
            print(f"  WARNING: BigWig file not found: {bw_path}")
            continue
        if not os.path.exists(bed):
            print(f"  WARNING: BED file not found: {bed}")
            continue
        
        results = evaluate_sample(
            sample_name=sample_name,
            biosample=biosample,
            bw_path=bw_path,
            bed=bed,
            data_type=data_type,
            model=model,
            seq_length=seq_length,
            batch_size=batch_size,
            bin_size=bin_size,
            output_dir=output_dir,
            eval_window_size=eval_window_size,
            jsd_pseudocount=jsd_pseudocount
        )
        
        all_results.append(results)
    
    # ===== SAVE AGGREGATE RESULTS =====
    print(f"\n{'='*60}")
    print("Saving aggregate results...")
    
    # Create summary DataFrame
    summary_data = []
    for r in all_results:
        summary_data.append({
            'sample_name': r['sample_name'],
            'data_type': r['data_type'],
            'biosample': r['biosample'],
            'n_regions': r['n_regions'],
            'n_valid_regions': r['n_valid_regions'],
            'n_valid_bases': r['n_valid_bases'],
            'global_pearson_r_1bp': r['global_pearson_r_1bp'],
            f'global_pearson_r_{bin_size}bp': r[f'global_pearson_r_{bin_size}bp'],
            'log_total_count_pearson_r': r['log_total_count_pearson_r'],
            'mean_js_distance': r['mean_js_distance'],
            'median_js_distance': r['median_js_distance'],
            'ratio_pred_truth': r['ratio_pred_truth']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(output_dir, 'all_samples_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary CSV saved to: {summary_csv}")
    
    # Save all results as JSON
    with open(os.path.join(output_dir, 'all_samples_results.json'), 'w') as f:
        clean_results = []
        for r in all_results:
            clean_r = {}
            for k, v in r.items():
                if k in ['per_sample_js_distances', 'per_sample_pred_sums', 'per_sample_truth_sums']:
                    continue
                elif isinstance(v, (np.floating,)):
                    clean_r[k] = float(v)
                else:
                    clean_r[k] = v
            clean_results.append(clean_r)
        json.dump(clean_results, f, indent=2)
    
    print(f"Aggregate JSON saved to: {os.path.join(output_dir, 'all_samples_results.json')}")
    
    # ===== CREATE BAR PLOTS =====
    print(f"\n{'='*60}")
    print("Creating comparison bar plots...")
    plot_comparison_barplots(all_results, output_dir, bin_size)
    
    # ===== PRINT FINAL SUMMARY =====
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"Method: Global concatenated Pearson r + Reverse complement averaging + {eval_window_size}bp windows")
    print(f"{'='*60}")
    print(f"\n{'Sample':<25} {'Global r':<12} {'Log Count r':<12} {'JS Dist':<12} {'Ratio P/T':<12}")
    print("-" * 73)
    for r in all_results:
        name = r['sample_name'][:24]
        pearson = f"{r[f'global_pearson_r_{bin_size}bp']:.4f}" if not np.isnan(r[f'global_pearson_r_{bin_size}bp']) else "N/A"
        log_r = f"{r['log_total_count_pearson_r']:.4f}" if not np.isnan(r['log_total_count_pearson_r']) else "N/A"
        js = f"{r['mean_js_distance']:.4f}" if not np.isnan(r['mean_js_distance']) else "N/A"
        ratio = f"{r['ratio_pred_truth']:.2f}" if not np.isnan(r['ratio_pred_truth']) else "N/A"
        print(f"{name:<25} {pearson:<12} {log_r:<12} {js:<12} {ratio:<12}")
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()