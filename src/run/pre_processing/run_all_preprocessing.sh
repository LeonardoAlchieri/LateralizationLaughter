# Script used to run pre-processing python scripts all together
echo "Experiment info preloading"
python src/run/run_experimentinfo_preloading.py
echo "Penguin Corretion"
python src/run/run_penguins_segments.py
echo "Data preloading"
python src/run/run_preloading.py
echo "EDA filtering"
python src/run/run_eda_filtering.py
echo "BVP filtering"
python src/run/run_bvp_filtering.py
echo "ACC filtering"
python src/run/run_acc_filtering.py
echo "Min Max normalization"
python src/run/run_min_max_norm.py
echo "Laughter segmentation"
python src/run/run_segmentation.py
