# bayesian-inference-engine

# Bayesian Network Inference Engine

This is a command-line tool for performing inference on Bayesian Networks using:
- Exact Inference
- Rejection Sampling
- Gibbs Sampling

## Features

- Parse Bayesian networks from a text file
- Query variables given evidence using different inference methods
- Supports interactive and non-interactive modes

## Usage

### Load a network:
```bash
load example_network.txt


## run:
xquery GrassWet | Rain=yes
rquery GrassWet | Rain=yes
gquery GrassWet | Rain=yes

##Example Network File Format
3
Rain yes no
Sprinkler yes no
GrassWet yes no
3
Sprinkler | Rain:
0.5 0.5
0.9 0.1
GrassWet | Sprinkler Rain:
0.99 0.01
0.8 0.2
0.9 0.1
0.0 1.0
Rain:
0.2 0.8
