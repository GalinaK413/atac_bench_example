# atac_bench_example
an example for Alpha Genome and Chrombpnet ATAC-seq benchmarking

Alpha Genome

use alpha.py

samples_all.csv should contain paths to ground truth bw (1) and to regions bed (2)
1) ground truth bw can be found on gencode - example https://www.encodeproject.org/files/ENCFF650PHA/
you can get accessions for all benchmarks here https://www.synapse.org/Synapse:syn59449898/wiki/628018
look for "observed signal profile" in gencode summary annotation
2) creation of regions bed is described in google collab - https://colab.research.google.com/drive/1qjR2W1LpRbrb0zmGescaiUGe7Ft1k0kN?usp=sharing
peaks file is on gencode (look for "training and test regions", testtest peaks)
use chrombpnet fold 0 and alphagenome fold 1 for comparison with GENA_LM

Chrombpnet

First use chrom_bw.py to create predicted bw, then use chrom_metrics.py to calculate metrics

ground truth bw and regions bed are the same as in Alpha Genome. You also need hg38 genome files and .h5 chrombpnet model file (see gencode, look for 'models'). depending on chrombpnet version, you may need to reformat h5 files, use https://github.com/kundajelab/chrombpnet/blob/master/chrombpnet/helpers/postprocessing/reformat_chrombpnet_h5.py
