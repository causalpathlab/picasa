

echo "$1" 

rm "figures/$1/results/picasa_model.log"
rm "figures/$1/results/picasa_common.model"
rm "figures/$1/results/picasa.h5ad"
rm "figures/$1/results/picasa_common_umap.png"

echo "run"
python  "figures/$1/1_picasa_run.py"
 

echo "analysis"
python  "figures/$1/2_picasa_analysis.py" 