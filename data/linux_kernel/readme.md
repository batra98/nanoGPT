# Linux Kernel Dataset

This dataset contains Linux kernel source code (C and header files) used to study the effect of training data size on language model performance.

## Dataset Creation

The dataset is created by:
1. Cloning the Linux kernel repository (shallow clone, ~3-4 GB)
2. Collecting all `.c` and `.h` files
3. Concatenating them into a single corpus
4. Creating 5 subsets of varying sizes

## Subsets

By default, subsets are stored in `/nobackup/gaurav/kernel_code/` (configurable via `--output_dir`):

| Name  | Size (chars) | Train/Val Split | Default Path |
|-------|--------------|-----------------|--------------|
| 100k  | 100,000      | 90% / 10%       | `/nobackup/gaurav/kernel_code/100k/`   |
| 500k  | 500,000      | 90% / 10%       | `/nobackup/gaurav/kernel_code/500k/`   |
| 1m    | 1,000,000    | 90% / 10%       | `/nobackup/gaurav/kernel_code/1m/`     |
| 5m    | 5,000,000    | 90% / 10%       | `/nobackup/gaurav/kernel_code/5m/`     |
| 10m   | 10,000,000   | 90% / 10%       | `/nobackup/gaurav/kernel_code/10m/`    |

Each subset directory contains:
- `train.bin`: Training data (binary encoded)
- `val.bin`: Validation data (binary encoded)
- `meta.pkl`: Character vocabulary and metadata

## Characteristics

**Content**: Linux kernel source code written in C
- Function definitions, struct declarations
- Preprocessor directives (#include, #define, etc.)
- Comments (both `/* */` and `//` style)
- Kernel-specific coding style and conventions

**Character Distribution**:
- High frequency: `{}[]();,*` (code syntax)
- Common words: `static`, `void`, `struct`, `return`, `int`, `const`
- Lower vowel frequency compared to natural language
- Many underscores in identifiers (snake_case)

**Expected Differences from Shakespeare**:
- More symbols, less natural language flow
- Stricter syntax rules (must be valid C)
- More technical vocabulary
- Different perplexity baseline due to code structure

## Usage

### Prepare Dataset
```bash
python data/linux_kernel/prepare.py --output_dir /nobackup/gaurav/kernel_code
```

**Options:**
- `--output_dir PATH`: Where to save datasets (default: `/nobackup/gaurav/kernel_code`)
- `--repo_dir PATH`: Where to clone kernel repo (default: `linux_kernel_repo`)
- `--cache_file PATH`: Corpus cache location (default: `linux_kernel_corpus_cache.txt`)

This will:
1. Clone Linux kernel (or use existing clone)
2. Create all 5 subsets with train/val splits in specified output directory
3. Generate vocabulary files
4. Cache corpus for faster re-runs

### Train Model
```bash
python run_data_scaling_experiment.py --num_gpus 8 --data_dir /nobackup/gaurav/kernel_code
```

This trains models on all 5 subsets and evaluates metrics.

## Experiment Goal

**Research Question**: How much training data is needed to achieve reasonable performance on code generation?

**Hypothesis**: 
- Small datasets (100k) will struggle with syntax and vocabulary
- Medium datasets (500k-1M) will learn basic patterns
- Large datasets (5M-10M) will approach plateau in performance

**Metrics Tracked**:
- Validation loss and perplexity
- N-gram overlap with training data (1, 2, 3-grams)
- KL divergence of character distributions
- Self-BLEU (diversity)
- Distinct-n ratios (uniqueness)
- Shannon entropy (randomness)

## Notes

- Linux kernel is constantly updated; this uses a snapshot (shallow clone)
- Source files are concatenated in arbitrary order
- Some files may contain non-UTF-8 characters (handled with error='ignore')
- Vocabulary size will vary slightly between subsets due to character coverage

