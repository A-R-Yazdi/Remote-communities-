
# install dependencies
pip install -r requirements/requirements.txt

# train model
# the data path, model root path, figure root path all should change accordingly
python train/train.py -d ../data/df2_processed.csv -mr train/ -fr train/

# python deploy/deploy.py
cd deploy
uvicorn deploy:app --reload --port 8001