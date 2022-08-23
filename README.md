[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://github.com/LeonardoAlchieri/LateralizationLaughter/blob/main/LICENSE)
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



