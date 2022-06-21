full_path=$(realpath $0)
dir_path=$(dirname $full_path)
root_path=$(dirname $dir_path)
data_path=$(dirname $root_path)/data/df2_processed.csv
echo $data_path

# install dependencies
pip install -r $root_path/requirements/requirements.txt

# train model
# the data path, model root path, figure root path all should change accordingly
python $root_path/train/train.py -d $data_path -mr $root_path/train/ -fr $root_path/train/

# python deploy/deploy.py
cd $root_path/deploy
uvicorn deploy:app --reload --port 8001