# XiHe-BiasCo

## 1. Project Overview

XiHe-BiasCo is a multivariable knowledge-transfer deep-learning framework developed to correct systematic biases in operational numerical global ocean forecasting systems (GOFS).

This repository provides the released code and configuration files associated with the XiHe bias-pretraining and CKPT-based evaluation workflow used in the XiHe-BiasCo study. The released code contains separate modules for the surface/shallow-ocean and deep-ocean components, together with data-loading, normalization, evaluation, and utility functions.

The complete long-term XiHe forecast archive, operational PSY4 datasets, trained model checkpoints, and full experimental outputs are not stored directly in this repository. Representative PSY4 fields before and after bias correction are provided separately to document the data organization used in the study.

---

## 2. Repository Structure

The released code is organized as follows:

```text
XiHe-BiasCo/
├── XiHe_1day_bais_preTraining.sh
│
├── configs/
│   ├── global_forecast_1to22_gpunode57_Xihe1D_bais.yaml
│   └── global_forecast_23to33_gpunode52_Xihe1D_bais.yaml
│
├── src/
│   ├── climax/
│   │   ├── __init__.py
│   │   ├── attention.py
│   │   ├── train.py
│   │   │
│   │   ├── global_forecast/
│   │   │   ├── __init__.py
│   │   │   ├── datamodule_1to22_refactor.py
│   │   │   ├── datamodule_23to33_refactor.py
│   │   │   ├── module_1to22.py
│   │   │   ├── module_23to33.py
│   │   │   ├── test_1to22.py
│   │   │   └── test_23to33.py
│   │   │
│   │   ├── pretrain/
│   │   │   ├── __init__.py
│   │   │   ├── datamodule_refactor.py
│   │   │   └── dataset_refactor.py
│   │   │
│   │   └── utils/
│   │       ├── data_utils.py
│   │       ├── lr_scheduler.py
│   │       ├── metrics.py
│   │       └── pos_embed.py
│   │
│   └── configs/
│
└── auxiliary_data/
    ├── normalize_mean_50.npz
    ├── normalize_std_50.npz
    ├── lat.npz
    ├── lon.npz
    ├── mask_surface.npy
    └── mask_deep.npy
```

The `1to22` modules correspond to the surface and upper-ocean component of the workflow, whereas the `23to33` modules correspond to the deeper-ocean component.

The source files in this repository reflect the research code used for the experiments described in the associated manuscript. Some file paths therefore retain the directory structure of the original computing environment and must be replaced with local paths before execution, as described in Section 7.

---

## 3. Environment

The released implementation is based on Python, PyTorch, and PyTorch Lightning and was developed for a Linux-based GPU computing environment.

Users should prepare a compatible Python environment containing the dependencies imported by the released source files, including PyTorch, PyTorch Lightning, NumPy, TorchData, and related packages.

GPU IDs, the number of devices, distributed-training settings, and worker numbers in `XiHe_1day_bais_preTraining.sh` reflect the original computing environment and should be adjusted according to the local hardware configuration.

---

## 4. Auxiliary Files

Several auxiliary files are required by the data-processing and evaluation workflow.

The following files are supplied separately from the source code:

```text
normalize_mean_50.npz
normalize_std_50.npz
lat.npz
lon.npz
mask_surface.npy
mask_deep.npy
```

### 4.1 Normalization parameters

`normalize_mean_50.npz` and `normalize_std_50.npz` contain the normalization statistics used for the ocean variables.

The data modules load the normalization information from the directory specified by `data.root_dir` in the YAML configuration or command-line arguments.

### 4.2 Grid coordinates

The global model grid contains 2041 latitude points and 4320 longitude points:

```python
lat = np.linspace(-80, 90, 2041)
lon = np.linspace(-180, 179.9167, 4320)
```

The corresponding coordinate information is also provided in:

```text
lat.npz
lon.npz
```

### 4.3 Ocean masks

`mask_surface.npy` and `mask_deep.npy` provide masks associated with the surface/shallow-ocean and deep-ocean fields, respectively.

The released evaluation code explicitly uses the surface mask when writing corrected surface fields. Users should ensure that the mask paths in the source code are consistent with their local directory structure.

---

## 5. Representative Sample Data

Because the full-resolution global PSY4 input and corrected-output datasets are large, representative one-day examples are provided separately rather than stored directly in this GitHub repository.

The example dataset contains:

```text
mra5_20220116_surface_before.npy
mra5_20220116_surface_after.npy
mra5_20220116_deep_before.npy
mra5_20220116_deep_after.npy
```

Sample data are available at:

**Download:** [SAMPLE_DATA_URL]

**Access code:** [ACCESS_CODE, if required]

The example files are provided to document the data dimensions, variable ordering, and before/after correction format used in the XiHe-BiasCo workflow.

---

## 6. Data Format

All example fields use the same horizontal grid:

```text
Latitude:  2041 points, from -80° to 90°
Longitude: 4320 points, from -180° to 179.9167°
```

The last two dimensions of all NPY arrays therefore correspond to:

```text
(latitude, longitude) = (2041, 4320)
```

### 6.1 Surface/shallow-ocean data before correction

The PSY4 surface/shallow-ocean field before bias correction has shape:

```text
(1, 52, 2041, 4320)
```

where the dimensions correspond to:

```text
(sample/time, variable channel, latitude, longitude)
```

The 52 channels contain four surface variables and temperature, salinity, and horizontal currents at 12 ocean depths.

The first four channels are:

| Channel | Variable                                 |
| ------- | ---------------------------------------- |
| 1       | Sea surface height (`zos`)               |
| 2       | Surface wind, zonal component (`u`)      |
| 3       | Surface wind, meridional component (`v`) |
| 4       | Sea surface temperature (`sst`)          |

The remaining 48 channels contain four ocean variables at 12 depths:

```text
T = seawater temperature
S = seawater salinity
U = zonal ocean current velocity
V = meridional ocean current velocity
```

The 12 depths are:

```text
0.49, 2.65, 5.08, 7.93, 11.41, 15.81,
21.60, 29.44, 40.34, 55.76, 77.85, 92.32 m
```

For each depth, variables are ordered as:

```text
T, S, U, V
```

Therefore:

```text
Channels 5-8   : T, S, U, V at 0.49 m
Channels 9-12  : T, S, U, V at 2.65 m
Channels 13-16 : T, S, U, V at 5.08 m
Channels 17-20 : T, S, U, V at 7.93 m
Channels 21-24 : T, S, U, V at 11.41 m
Channels 25-28 : T, S, U, V at 15.81 m
Channels 29-32 : T, S, U, V at 21.60 m
Channels 33-36 : T, S, U, V at 29.44 m
Channels 37-40 : T, S, U, V at 40.34 m
Channels 41-44 : T, S, U, V at 55.76 m
Channels 45-48 : T, S, U, V at 77.85 m
Channels 49-52 : T, S, U, V at 92.32 m
```

### 6.2 Surface/shallow-ocean data after correction

The corresponding corrected surface/shallow-ocean field has shape:

```text
(1, 50, 2041, 4320)
```

The first two channels contain:

| Channel | Variable                        |
| ------- | ------------------------------- |
| 1       | Sea surface height (`zos`)      |
| 2       | Sea surface temperature (`sst`) |

Channels 3-50 contain `T`, `S`, `U`, and `V` at the same 12 depths:

```text
Channels 3-6   : T, S, U, V at 0.49 m
Channels 7-10  : T, S, U, V at 2.65 m
Channels 11-14 : T, S, U, V at 5.08 m
Channels 15-18 : T, S, U, V at 7.93 m
Channels 19-22 : T, S, U, V at 11.41 m
Channels 23-26 : T, S, U, V at 15.81 m
Channels 27-30 : T, S, U, V at 21.60 m
Channels 31-34 : T, S, U, V at 29.44 m
Channels 35-38 : T, S, U, V at 40.34 m
Channels 39-42 : T, S, U, V at 55.76 m
Channels 43-46 : T, S, U, V at 77.85 m
Channels 47-50 : T, S, U, V at 92.32 m
```

The surface wind components are input variables and are not included in the 50-channel corrected output.

### 6.3 Deep-ocean data before correction

The PSY4 deep-ocean field before bias correction has shape:

```text
(1, 48, 2041, 4320)
```

The first four channels follow the same surface-variable organization:

| Channel | Variable                                 |
| ------- | ---------------------------------------- |
| 1       | Sea surface height (`zos`)               |
| 2       | Surface wind, zonal component (`u`)      |
| 3       | Surface wind, meridional component (`v`) |
| 4       | Sea surface temperature (`sst`)          |

Channels 5-48 contain `T`, `S`, `U`, and `V` at 11 deep-ocean levels.

The 11 depths are:

```text
109.73, 130.67, 155.85, 186.13, 222.48, 266.04,
318.31, 380.21, 453.94, 541.09, 643.57 m
```

The variables are ordered as `T, S, U, V` at each depth:

```text
Channels 5-8   : T, S, U, V at 109.73 m
Channels 9-12  : T, S, U, V at 130.67 m
Channels 13-16 : T, S, U, V at 155.85 m
Channels 17-20 : T, S, U, V at 186.13 m
Channels 21-24 : T, S, U, V at 222.48 m
Channels 25-28 : T, S, U, V at 266.04 m
Channels 29-32 : T, S, U, V at 318.31 m
Channels 33-36 : T, S, U, V at 380.21 m
Channels 37-40 : T, S, U, V at 453.94 m
Channels 41-44 : T, S, U, V at 541.09 m
Channels 45-48 : T, S, U, V at 643.57 m
```

### 6.4 Deep-ocean data after correction

The corrected deep-ocean field has shape:

```text
(1, 44, 2041, 4320)
```

The 44 channels contain `T`, `S`, `U`, and `V` at the 11 deep-ocean levels:

```text
Channels 1-4   : T, S, U, V at 109.73 m
Channels 5-8   : T, S, U, V at 130.67 m
Channels 9-12  : T, S, U, V at 155.85 m
Channels 13-16 : T, S, U, V at 186.13 m
Channels 17-20 : T, S, U, V at 222.48 m
Channels 21-24 : T, S, U, V at 266.04 m
Channels 25-28 : T, S, U, V at 318.31 m
Channels 29-32 : T, S, U, V at 380.21 m
Channels 33-36 : T, S, U, V at 453.94 m
Channels 37-40 : T, S, U, V at 541.09 m
Channels 41-44 : T, S, U, V at 643.57 m
```

Sea surface height, SST, and surface wind components are not included in the 44-channel corrected deep-ocean output.

---

## 7. Local Path Configuration

The released source code retains several absolute paths from the original computing environment.

These paths must be replaced with paths appropriate for the user's local environment before running the code.

### 7.1 Surface/shallow-ocean data paths

Edit:

```text
src/climax/global_forecast/datamodule_1to22_refactor.py
```

Replace the original absolute paths used for:

* training input data;
* training reference/label data;
* validation input data;
* validation reference/label data;
* test input data;
* test reference/label data.

The original code contains paths beginning with directories such as:

```text
/public/home/...
```

These paths refer to the original computing environment and are not part of the released repository.

### 7.2 Deep-ocean data paths

Edit:

```text
src/climax/global_forecast/datamodule_23to33_refactor.py
```

Replace the corresponding absolute paths for the deep-ocean training, validation, test, and reference datasets.

### 7.3 Normalization and grid-data path

The YAML configuration files contain a `data.root_dir` parameter.

For example:

```yaml
data:
  root_dir: /datadrive/datasets/5.625deg_equally_np/
```

Replace this path with the local directory containing the required normalization and grid files.

For example:

```yaml
data:
  root_dir: /your/local/path/XiHe-BiasCo/auxiliary_data/
```

The directory should contain the auxiliary files required by the corresponding data-loading workflow.

### 7.4 Model checkpoint paths

The released test scripts contain checkpoint paths from the original computing environment.

Edit:

```text
src/climax/global_forecast/test_1to22.py
src/climax/global_forecast/test_23to33.py
```

and replace the value of `ckpt` with the corresponding local CKPT file.

For example:

```python
ckpt = "/your/local/path/to/model.ckpt"
```

The trained checkpoints are not stored directly in this GitHub repository.

**Checkpoint availability:** [CHECKPOINT_AVAILABILITY_STATEMENT]

### 7.5 Surface mask and corrected-output path

Edit:

```text
src/climax/utils/metrics.py
```

The evaluation routine contains an absolute path to:

```text
mask_surface.npy
```

Replace this path with the local location of `mask_surface.npy`.

The same routine also contains an absolute output directory used by `np.save()` to save corrected surface fields. Replace this path with the desired local output directory and make sure that the directory exists and is writable.

---

## 8. Running the Released CKPT-Based Evaluation Code

The repository contains the original one-day shell script:

```text
XiHe_1day_bais_preTraining.sh
```

The commands in this script reflect the original HPC environment. Before execution, users should update:

* `CUDA_VISIBLE_DEVICES`;
* the number of GPU devices;
* distributed-training/evaluation settings;
* `data.root_dir`;
* data paths in the corresponding data modules;
* checkpoint paths in the test scripts;
* the surface-mask path;
* the corrected-output directory.

### 8.1 Surface/shallow-ocean component

The surface/shallow-ocean evaluation entry point is:

```bash
python src/climax/global_forecast/test_1to22.py \
    --config configs/global_forecast_1to22_gpunode57_Xihe1D_bais.yaml \
    --data.root_dir=/your/local/path/XiHe-BiasCo/auxiliary_data/ \
    --data.predict_range=0 \
    --data.batch_size=1
```

Additional trainer/device arguments can be specified according to the local computing environment.

### 8.2 Deep-ocean component

The released deep-ocean configuration file is:

```text
configs/global_forecast_23to33_gpunode52_Xihe1D_bais.yaml
```

The corresponding evaluation entry point is:

```bash
python src/climax/global_forecast/test_23to33.py \
    --config configs/global_forecast_23to33_gpunode52_Xihe1D_bais.yaml \
    --data.root_dir=/your/local/path/XiHe-BiasCo/auxiliary_data/ \
    --data.predict_range=0 \
    --data.batch_size=1
```

Additional GPU and distributed settings should be adjusted to the local environment.

> **Note:** The deep-ocean command retained in the original `XiHe_1day_bais_preTraining.sh` refers to a configuration filename from the original development environment that is not included in the released package. For the released one-day code, use the `global_forecast_23to33_gpunode52_Xihe1D_bais.yaml` configuration included in `configs/`.

---

## 9. Output

During evaluation, the model predictions are converted back from normalized values to their physical-variable representation by the evaluation workflow.

The corrected surface output is saved as a NumPy array by the output routine in:

```text
src/climax/utils/metrics.py
```

The output directory must be changed from the original absolute path to a valid local directory before execution.

Representative corrected fields are provided in the external sample-data package:

```text
mra5_20220116_surface_after.npy
mra5_20220116_deep_after.npy
```

These example files document the data organization used in the XiHe-BiasCo study and are not intended to replace the complete evaluation dataset used to generate the results reported in the manuscript.

---

## 10. Data and Model Availability

The complete 25-year XiHe forecast archive, operational PSY4 forecast fields, observational/reference datasets, trained checkpoints, and full corrected forecast archive are not duplicated in this GitHub repository.

Representative one-day PSY4 fields before and after correction are provided separately:

**Sample data:** [SAMPLE_DATA_URL]

**Access code:** [ACCESS_CODE, if required]

The availability of the trained CKPT files should be described consistently with the Code Availability statement of the associated manuscript:

**[CHECKPOINT_AVAILABILITY_STATEMENT]**

Information on the observational and operational datasets used for scientific evaluation is provided in the Data Availability statement of the associated manuscript.

---

## 11. Notes

Before running the released code, please make sure that:

* all `/public/home/...` and other original absolute paths have been replaced with valid local paths;
* `data.root_dir` points to the local auxiliary-data directory;
* the required normalization files are available;
* latitude and longitude information is available for interpretation of the global output fields;
* the required ocean-mask files are available;
* the surface and deep CKPT paths have been updated;
* the corrected-output directory exists and is writable;
* GPU IDs and device numbers match the local hardware;
* the input NPY arrays follow the dimensions and variable ordering described in Section 6.

The released code reflects the research implementation used in the study and therefore requires local path configuration before execution.

---

## 12. Citation

If you use the XiHe-BiasCo code or sample data in your research, please cite the associated manuscript:

**Physically Coherent Bias Correction of Operational Numerical Global Ocean Forecasting via Multivariable Knowledge Transfer**

[Citation details will be added after publication.]

---

## 13. License

[LICENSE INFORMATION TO BE CONFIRMED]

---

## 14. Contact

For questions regarding the released code, auxiliary files, or representative sample data, please contact the corresponding authors of the associated manuscript.
