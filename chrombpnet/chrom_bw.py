# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pyBigWig
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import chrombpnet.training.utils.losses as losses
import argparse
import pysam
from collections import defaultdict

def load_model_wrapper(model_h5):
    """Load the chrombpnet model"""
    custom_objects = {"tf": tf, "multinomial_nll": losses.multinomial_nll}
    get_custom_objects().update(custom_objects)
    model = load_model(model_h5, compile=False)
    print("Model loaded successfully")
    return model

def reverse_complement(seq):
    """Reverse complement a DNA sequence"""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq))

def one_hot_encode(sequence):
    """One-hot encode a DNA sequence"""
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 
               'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    return np.array([mapping.get(base, [0, 0, 0, 0]) for base in sequence])

def softmax(x, temp=1):
    """Apply softmax to convert logits to probabilities"""
    norm_x = x - np.mean(x, axis=1, keepdims=True)
    return np.exp(temp * norm_x) / np.sum(np.exp(temp * norm_x), axis=1, keepdims=True)

def predict_region(model, sequence, outputlen):
    """Predict profile for a single sequence"""
    seq_onehot = one_hot_encode(sequence)
    seq_onehot = np.expand_dims(seq_onehot, axis=0)  # Add batch dimension

    profile_logits, counts = model.predict(seq_onehot, verbose=0)
    
    profile_probs = softmax(profile_logits)
    
    profile = profile_probs * np.exp(counts)
    
    return profile.flatten()

def predict_averaged_profile(model, genome, chrom, start, end, inputlen, outputlen):
    """Predict and average forward and reverse complement profiles"""

    fwd_seq = genome.fetch(chrom, start, end).upper()
    rev_seq = reverse_complement(fwd_seq)
    
    # Predict on forward strand
    fwd_profile = predict_region(model, fwd_seq, outputlen)
    
    # Predict on reverse complement
    rev_profile = predict_region(model, rev_seq, outputlen)
    
    # Reverse the reverse complement profile
    rev_profile_reversed = rev_profile[::-1]
    
    # Average the two profiles
    avg_profile = (fwd_profile + rev_profile_reversed) / 2.0
    
    return avg_profile

def write_bigwig(profiles_list, output_bw, chrom_sizes):
    """Write profiles to bigwig - assumes entries are already properly sorted"""
    with pyBigWig.open(output_bw, "w") as bw:
        used_chroms = sorted(set(entry['chrom'] for entry in profiles_list))
        header = [(chrom, chrom_sizes[chrom]) for chrom in used_chroms if chrom in chrom_sizes]
        
        if len(header) == 0:
            raise ValueError("No valid chromosomes found in data that match chrom_sizes file")
        
        bw.addHeader(header)
        
        for chrom in used_chroms:
            chrom_entries = [e for e in profiles_list if e['chrom'] == chrom]
            
            if len(chrom_entries) == 0:
                continue
            
            starts = [e['start'] for e in chrom_entries]
            ends = [e['end'] for e in chrom_entries]
            values = [e['value'] for e in chrom_entries]
            
            # Final validation
            if len(starts) > 1:
                for i in range(len(starts)-1):
                    if starts[i] >= starts[i+1]:
                        print(f"WARNING: Order violation in {chrom} at index {i}: "
                              f"start={starts[i]}, next start={starts[i+1]}")
            
            print(f"Writing {len(starts)} entries for chromosome {chrom}")
            
            bw.addEntries(
                [chrom] * len(starts),
                starts,
                ends,
                values
            )
    
    print(f"Bigwig written to {output_bw}")

def main():
    parser = argparse.ArgumentParser(description='Predict averaged two-strand profiles')
    parser.add_argument('--model', required=True, help='Path to model .h5 file')
    parser.add_argument('--genome', required=True, help='Path to genome fasta file')
    parser.add_argument('--regions', required=True, help='Path to 10-column BED file')
    parser.add_argument('--output_prefix', required=True, help='Output prefix for bigwig')
    parser.add_argument('--inputlen', type=int, required=True, help='Model input length')
    parser.add_argument('--outputlen', type=int, required=True, help='Model output length')
    parser.add_argument('--chrom_sizes', required=True, help='Path to chromosome sizes file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (not used currently)')
    
    args = parser.parse_args()
    
    print("Loading model...")
    model = load_model_wrapper(args.model)
    
    print("Loading genome...")
    genome = pysam.FastaFile(args.genome)
    
    print("Loading regions...")
    bed_columns = ['chr', 'start', 'end', 'name', 'score', 'strand', 
                   'signalValue', 'pValue', 'qValue', 'summit']
    regions = pd.read_csv(args.regions, sep='\t', names=bed_columns)
    print(f"Loaded {len(regions)} regions")
    
    chrom_sizes = {}
    with open(args.chrom_sizes) as f:
        for line in f:
            chrom, size = line.strip().split()
            chrom_sizes[chrom] = int(size)
    
    # Process regions and collect profiles in a dictionary for averaging overlaps
    profile_accumulator = defaultdict(lambda: defaultdict(list))
    
    print(f"Processing {len(regions)} regions...")
    skipped_count = 0
    error_count = 0
    
    for idx, region in regions.iterrows():
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(regions)} regions")
        
        chrom = region['chr']
        start = int(region['start'])
        end = int(region['end'])
        summit_offset = int(region['summit'])
        
        # Skip chromosomes not in chrom_sizes
        if chrom not in chrom_sizes:
            skipped_count += 1
            continue
        
        # Calculate absolute summit position
        summit = start + summit_offset
        
        # Calculate the genomic window for input
        window_start = summit - args.inputlen // 2
        window_end = summit + args.inputlen // 2
        
        # Skip if window extends beyond chromosome boundaries
        if window_start < 0 or window_end > chrom_sizes[chrom]:
            skipped_count += 1
            continue
        
        try:
            # Get averaged profile centered at summit
            profile = predict_averaged_profile(
                model, genome, chrom, window_start, window_end, 
                args.inputlen, args.outputlen
            )
            
            # Calculate genomic coordinates for each position in the profile
            profile_start = summit - args.outputlen // 2
            
            # Add each position's value to the accumulator
            for i, val in enumerate(profile):
                pos = profile_start + i
                if 0 <= pos < chrom_sizes[chrom]:
                    profile_accumulator[chrom][pos].append(float(val))
                    
        except Exception as e:
            error_count += 1
            if error_count <= 10:  # Only print first 10 errors
                print(f"Error processing region {chrom}:{start}-{end} (summit={summit}): {e}")
            continue
    
    print(f"Finished processing regions.")
    print(f"Skipped: {skipped_count} regions (out of bounds or invalid chromosome)")
    print(f"Errors: {error_count} regions")
    print(f"Successfully processed: {len(regions) - skipped_count - error_count} regions")
    
    # Now create sorted, non-overlapping entries by averaging overlapping positions
    print("Averaging overlapping predictions...")
    profiles_list = []
    
    for chrom in sorted(profile_accumulator.keys()):
        positions = sorted(profile_accumulator[chrom].keys())
        
        for pos in positions:
            values = profile_accumulator[chrom][pos]
            avg_value = np.mean(values)
            
            profiles_list.append({
                'chrom': chrom,
                'start': pos,
                'end': pos + 1,
                'value': float(avg_value)
            })
    
    print(f"Collected {len(profiles_list)} unique positions after averaging overlaps")
    
    if len(profiles_list) == 0:
        print("ERROR: No valid entries collected. Check your input regions and chromosome sizes.")
        genome.close()
        return

    profiles_list.sort(key=lambda x: (x['chrom'], x['start']))
    
    output_bw = args.output_prefix + "_averaged.bw"
    print(f"Writing bigwig to {output_bw}...")
    write_bigwig(profiles_list, output_bw, chrom_sizes)
    
    genome.close()
    print("Done!")

if __name__ == "__main__":
    main()