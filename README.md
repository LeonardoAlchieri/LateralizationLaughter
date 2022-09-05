[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyPI license](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/LeonardoAlchieri/LateralizationLaughter/blob/main/LICENCE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/LeonardoAlchieri/LateralizationLaughter/graphs/commit-activity)

# LateralizationLaughter

Code for the 7th International Workshop on Mental Health and Well-being: Sensing and Intervention @ UBICOMP 2022 paper: **Lateralization Analysis in Physiological Data from Wearable Sensors for Laughter Detection**.

The code can be divided into 3 main parts:
1. Pre-processing
2. Statistical Analysis
3. Machine Learning Task

To install the used libraries, just run `pip install -r requirements.txt`. Keep in mind that some libraries used are not published on pip, but they are directly GitHub branches. For more information on them, feel free to contact me.

## Pre-processing

For the first, everything is run using some custom scripts (`src/run/pre_processing`). Each of them is controlled by a specific configuration file, where some simple parameters can be changed (by default the ones used for our work are present).
While the order of scripts is not important, to run them using the order we performed you can simply do:
    ```bash
        sh src/run/pre_processing_run_all_preprocessing.sh
    ```
We suggest to use this, since other configurations would require to change a little bit the configuration files.

## Statistical Analysis

To run the statistical analysis, and replicate the plots shown in the paper, you can just do:
    ```bash
        python src/run/statistical_analysis/run_statistical_feature.py
    ```
The script will use a configuration file (`src/run/statistical_analysis/config_statistical_feature.yml`), which by default has the same values used by us. I would suggest to only change the path to the data.

*WARNING*: the script will fail if it cannot find a `logs` folder and a `visualizations` folder in the root of the repo. While I plan to automate this easily, for the moment please just create them.

## Machine Learning Task




@2022, Leonardo Alchieri, Nouran Abdalazim, Lidia Alecci, Shkurta Gashi, Elena Di Lascio, Silvia Santini
 
Contact: leonardo.alchieri@usi.ch


