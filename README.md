# MIAT: Maneuver-Intention-Aware Transformer for Spatio-Temporal Trajectory Prediction

This repository contains the codebase of **MIAT**.

**Paper:** [MIAT: Maneuver-Intention-Aware Transformer for Spatio-Temporal Trajectory Prediction](https://arxiv.org/abs/2504.05059)  
**Authors:** Chandra Raskoti, Iftekharul Islam, Xuan Wang, Weizi Li  
**Institution:** University of Tennessee, Knoxville; George Mason University

## Abstract

Accurate vehicle trajectory prediction is critical
for safe and efficient autonomous driving, especially in mixed
traffic environments when both human-driven and autonomous
vehicles co-exist. However, uncertainties introduced by inherent
driving behaviors—such as acceleration, deceleration, and left
and right maneuvers—pose significant challenges for reliable
trajectory prediction. We introduce a Maneuver-IntentionAware Transformer (MIAT) architecture, which integrates a
maneuver intention awareness control mechanism with spatiotemporal interaction modeling to enhance long-horizon trajectory predictions. We systematically investigate the impact
of varying awareness of maneuver intention on both shortand long-horizon trajectory predictions. Evaluated on the
real-world NGSIM dataset and benchmarked against various
transformer- and LSTM-based methods, our approach achieves
an improvement of up to 4.7% in short-horizon predictions and
a 1.6% in long-horizon predictions compared to other intentionaware benchmark methods. Moreover, by leveraging intention
awareness control mechanism, MIAT realizes an 11.1% performance boost in long-horizon predictions, with a modest drop in
short-horizon performance. The source code and datasets are
available at https://github.com/cpraskoti/MIAT.

Key features:
- **Unified Architecture:** Integrates maneuver intention awareness directly into the Transformer learning objective.
- **Tunable Control Mechanism:** Allows explicit control over the trade-off between minimizing trajectory prediction error and correctly classifying the underlying maneuver.
- **Performance:** Achieves significant improvements in long-horizon predictions (up to 11.1%) on the real-world NGSIM dataset compared to intention-aware benchmarks.

## Project Structure

```
MIAT/
├── config.yaml          # Configuration file for experiments
├── requirements.txt     # Python dependencies
├── src/                 # Source code
│   ├── config.py        # Configuration loader
│   ├── data_loader.py   # Dataset classes (NGSIM, HighD)
│   ├── model.py         # MIAT model architecture (Encoder, Decoder, Generator)
│   ├── train.py         # Training script
│   └── evaluate.py      # Evaluation script
├── data/                # Data directory
│   ├── processed_data/  # Processed .mat files for training
│   └── NGSIM_raw_data/  # Raw NGSIM data
└── docs/                # Documentation and paper
```

## Installation

1. Clone the repository.
```
git clone https://github.com/cpraskoti/MIAT.git
```
2. Create a virtual environment
```
python -m venv venv
```
Activate environemt for Unix
```
  source venv/bin/activate
```

Activate environemt: for windows
```
  venv\Scripts\activate
```
3. Install dependencies:

```bash
pip install -r requirements.txt

```

## Data Preparation
Download processed dataset from [GOOGLE DRIVE](https://drive.google.com/file/d/1pmcs3InSJpoAjQLLgdeV4-_SesUKa-F0/view?usp=sharing)
OR
[BAIDU](https://pan.baidu.com/s/1ur34Au8h3b3WZFM5a2RWOw) (4p44)


Save these processed `.mat` files containing in `/data/processed_data/` folder.
- **Train Set:** `data/processed_data/TrainSet.mat`
- **Validation Set:** `data/processed_data/ValSet.mat`
- **Test Set:** `data/processed_data/TestSet.mat`

See [src/Data_description.md](src/Data_description.md) for detailed information on the data dataset.
See [src/data_processing.md](src/data_processing.md) for detailed information on how data was processed.



## Configuration

The project uses `config.yaml` for experimental settings. You can modify this file or override parameters via command line arguments.

**Key Parameters:**
- `experiment.name`: Name of the experiment (used for logging).
- `model.encoder_size`: Dimension of the transformer model.
- `model.use_maneuvers`: Whether to use maneuver classification.
- `training.batch_size`: Batch size.
- `data.dataset`: "ngsim"

## Usage

### Training


To train the model with default configuration:

```bash
python3 src/train.py
```

To override configuration parameters:

```bash
python3 src/train.py --exp_name my_experiment --gpu_id 0
```

Training logs and checkpoints will be saved in `checkpoint/`.



### Evaluation

To evaluate the best trained model:

```bash
python3 src/evaluate.py
```

This will compute RMSE or NLL metrics on the test set.


## Extensions & Future Work

This codebase provides a solid foundation for further research in trajectory prediction. Here are some areas for potential improvement and extension:

1.  **Multi-modal Distribution:** Extend the model to generate multi-modal trajectory distributions explicitly (e.g., using Gaussian Mixture Models or CVAE) to capture a wider range of plausible future paths.
2.  **Dataset Scaling:** Validate the robustness of MIAT on larger, more complex datasets like Argoverse, NuScenes, or Waymo Open Motion Dataset.
3.  **Graph Neural Networks:** Replace the spatial interaction mechanism with more advanced Graph Neural Networks (GNNs) or Graph Attention Networks (GATs) to better model agent-to-agent interactions.
4.  **Ablation Studies:** Experiment with different weighting factors for the intention awareness control mechanism in `config.yaml` (`scale_cross_entropy_loss`) to further analyze the trade-off between maneuver classification and trajectory accuracy.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{raskoti2025miat,
  title={MIAT: Maneuver-Intention-Aware Transformer for Spatio-Temporal Trajectory Prediction},
  author={Raskoti, Chandra and Islam, Iftekharul and Wang, Xuan and Li, Weizi},
  journal={arXiv preprint arXiv:2504.05059},
  year={2025}
}
```
