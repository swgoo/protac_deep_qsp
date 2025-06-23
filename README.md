# PROTAC Deep QSP Model

## Abstract
This study presents an integrated computational modeling framework combining deep learning and Quantitative Systems Pharmacology (QSP) to predict the efficacy of PROTAC (PROteolysis Targeting Chimera) molecules. PROTACs have emerged as promising therapeutics for targeted protein degradation (TPD), offering significant advantages in addressing proteins that traditional small-molecule inhibitors cannot target. However, experimental evaluation of PROTAC efficacy is hindered by extensive variability in molecular configurations, necessitating efficient computational prediction methods.

The proposed model integrates binding affinity predictions from DeepCalici, a convolutional neural network-based deep learning model, with a mechanistic QSP Hook model to estimate key pharmacodynamic parameters, notably half-maximal degradation concentration($DC_{50}$) and maximal degradation($D_{max}$). This study utilized curated experimental data from PROTAC-DB, including experimentally validated $DC_{50}$ and $D_{max}$ values. The dissociation constants ($K_d$) between PROTAC molecules and their protein targets (POI) or E3 ligases were predicted using DeepCalici and, then incorporated into the Hook model.

To enhance the prediction accuracy, a supplementary deep neural network adjusted the hook model parameters based on chemical and biochemical features. The integrated modeling approach achieved a strong predictive performance for $DC_{50}$, demonstrating its practical value in prioritizing effective PROTAC candidates. However, the predictions for $D_{max}$ were less accurate, likely reflecting the variability in the experimental conditions not captured in the current dataset. 

This study highlights the critical importance of comprehensive structural data for accurate modeling of PROTAC efficacy and suggests future improvements using standardized experimental data. Such integrative modeling approaches promise to accelerate the discovery and optimization of PROTAC therapeutics.

## Requirements
To run this project, you will need the following installed on your system:
- **NVIDIA GPU RTX 3060 6GB**
- **Docker** ([Installation Guide](https://docs.docker.com/engine/install/))
- **NVIDIA Container Toolkit** ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

This code has been tested using the `pytorchlightning/pytorch_lightning:2.5.0-py3.12-torch2.5-cuda12.1.1` Docker image. All necessary dependencies are included in this image.

## Usage

Follow these steps to train the model and generate predictions.

**1. Clone the repository:**
```bash
git clone https://github.com/swgoo/protac_deep_qsp.git
cd protac_deep_qsp
```

**2. Run the training script via Docker:**
The following command will start a Docker container, mount the project directory, train the model, and save the predictions.
```bash
docker run --rm -v .:/workdir -w /workdir --gpus all \
  pytorchlightning/pytorch_lightning:2.5.0-py3.12-torch2.5-cuda12.1.1 \
  python model.py
```

### Expected Output
After the script finishes, the following files will be generated in the `data/` directory:
- `data/model.pth`: The trained model weights.
- `data/train_prediction.csv`: Predictions for the training set.
- `data/test_prediction.csv`: Predictions for the test set.

## Reproducing Paper Results
The script is configured to reproduce the results presented in our manuscript. The random seed is fixed within `model.py` to ensure deterministic outcomes. By running the `Usage` command, you can generate the prediction files (`test_prediction.csv`) used for our analysis. The results can then be compared with the figures and tables in our paper.

## Project Structure
```
.
├── data/
│   ├── config.yaml         # Model configuration file
│   ├── train.csv           # Training dataset
│   └── test.csv            # Test dataset
├── model.py                # Main script for training and prediction
├── prepare.py              # (For reference only) Script for dataset generation
```

### Configuration (`data/config.yaml`)
This file contains all settings for the model, including feature definitions, target variable bounds, and training hyperparameters.
Below is an example of how the `data/config.yaml` file should be structured:

```yaml
feature:
    - xlogp3 # xlogP3 value of the PROTAC molecule
    - hydrogen_bond_acceptor_count # Number of hydrogen bond acceptors in the PROTAC molecule
    - hydrogen_bond_donor_count # Number of hydrogen bond donors in the PROTAC molecule
    - rotatable_bond_count # Number of rotatable bonds in the PROTAC molecule
    - topological_polar_surface_area # Topological polar surface area of the PROTAC molecule
    - molecular_weight # Molecular weight of the PROTAC molecule
    - linker_xlogp3 # xlogP3 value of the linker part of the PROTAC molecule
    - linker_hydrogen_bond_acceptor_count # Number of hydrogen bond acceptors in the linker part of the PROTAC molecule
    - linker_hydrogen_bond_donor_count # Number of hydrogen bond donors in the linker part of the PROTAC molecule
    - linker_rotatable_bond_count # Number of rotatable bonds in the linker part of the PROTAC molecule
    - linker_topological_polar_surface_area # Topological polar surface area of the linker part of the PROTAC molecule
    - linker_molecular_weight # Molecular weight of the linker part of the PROTAC molecule
    - warhead_xlogp3 # xlogP3 value of the warhead part of the PROTAC molecule
    - warhead_hydrogen_bond_acceptor_count # Number of hydrogen bond acceptors in the warhead part of the PROTAC molecule
    - warhead_hydrogen_bond_donor_count # Number of hydrogen bond donors in the warhead part of the PROTAC molecule
    - warhead_rotatable_bond_count # Number of rotatable bonds in the warhead part of the PROTAC molecule
    - warhead_topological_polar_surface_area # Topological polar surface area of the warhead part of the PROTAC molecule
    - warhead_molecular_weight # Molecular weight of the warhead part of the PROTAC molecule
    - warhead_binding_energy # Binding energy of the warhead part of the PROTAC molecule between the target protein and the warhead
    - e3_ligand_xlogp3 # xlogP3 value of the E3 ligase ligand
    - e3_ligand_hydrogen_bond_acceptor_count # Number of hydrogen bond acceptors in the E3 ligase ligand
    - e3_ligand_hydrogen_bond_donor_count # Number of hydrogen bond donors in the E3 ligase ligand
    - e3_ligand_rotatable_bond_count # Number of rotatable bonds in the E3 ligase ligand
    - e3_ligand_topological_polar_surface_area # Topological polar surface area of the E3 ligase ligand
    - e3_ligand_molecular_weight # Molecular weight of the E3 ligase ligand
    - e3_ligand_binding_energy # Binding energy of the E3 ligase ligand between the target protein and the E3 ligase
    - kde # Affinity of the PROTAC molecule to the E3 ligase
    - kdp # Affinity of the PROTAC molecule to the target protein
    - kdegp # Kinetic parameter for the degradation of the target protein itself
target: # upper value and lower value of the target variable
    dmax:
        - 0.1
        - 1
    dc50:
        - 0
        - 10000
argument: # upper and lower bounds of the coefficients of the arguments to be corrected
    kde:
        - 0.1
        - 10
    kdp:
        - 0.1
        - 10
    kdegp:
        - 0.1
        - 10
    e0:
        - 0.01
        - 100
    alpha:
        - 0.01
        - 100
    kcat:
        - 0.1
        - 5000
batch_size: 1000
num_epochs: 1000000
lr: 0.005 # learning rate
train_dataset_path: data/train.csv
test_dataset_path: data/test.csv
test_prediction_path: data/test_prediction.csv
train_prediction_path: data/train_prediction.csv
log_interval: 100
model_save_path: data/model.pth
```

### Datasets (`train.csv` and `test.csv`)
These files contain the features and target values as specified in `config.yaml`. To predict the properties of new PROTAC molecules, add their feature information to `test.csv` (target values can be 0.).

### Prediction Output (`train_prediction.csv` and `test_prediction.csv`)
These files contain the actual and predicted values for `dmax` and `dc50`. The columns are:
- `dmax_predict`: Predicted value of `dmax`
- `dc50_predict`: Predicted value of `dc50`
- `dmax`: Actual value of `dmax`
- `dc50`: Actual value of `dc50`

### Dataset Preparation (`prepare.py`)
**Note:** You do not need to run `prepare.py`. The required datasets (`train.csv`, `test.csv`) are already provided.

This script is included for transparency to show how the dataset for this study was constructed from [PROTAC-DB 3.0](https://protac-db.com/) and [ProteomicsDB](https://www.proteomicsdb.org/). Running this script may require specific environment setups or API access and is intended for methodological reference only.

```bibtex
@article{protacdb,
  title={PROTAC-DB 3.0: an updated database of PROTACs with extended pharmacokinetic parameters},
  author={Ge, Jingxuan and Li, Shimeng and Weng, Gaoqi and Wang, Huating and Fang, Meijing and Sun, Huiyong and Deng, Yafeng and Hsieh, Chang-Yu and Li, Dan and Hou, Tingjun},
  journal={Nucleic Acids Research},
  volume={53},
  number={D1},
  pages={D1510--D1515},
  year={2025},
  publisher={Oxford University Press}
}
@article{proteomicsdb,
  title={ProteomicsDB: a multi-omics and multi-organism resource for life science research},
  author={Samaras, Patroklos and Schmidt, Tobias and Frejno, Martin and Gessulat, Siegfried and Reinecke, Maria and Jarzab, Anna and Zecha, Jana and Mergner, Julia and Giansanti, Piero and Ehrlich, Hans-Christian and others},
  journal={Nucleic acids research},
  volume={48},
  number={D1},
  pages={D1153--D1163},
  year={2020},
  publisher={Oxford University Press}
}
```
<!-- 
## Citation
If you use this code or our model in your research, please cite our paper:
```bibtex

``` -->