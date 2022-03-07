## **Overview**

**Remote Communities Power Consumption** project aims to predict future power consumption with limited historical data. 

### Data preparation
The data preparation phase (sampling) allows creating sub-images that will be used for either training, validation or testing.
The first phase of the process is to determine sub-images (samples) to be used for training, validation and, optionally, test.
Images to be used must be of the geotiff type.
Sample locations in each image must be stored in a GeoPackage.

[comment]: <> (> Note: A data analysis module can be found [here]&#40;./utils/data_analysis.py&#41; and the documentation in [`docs/README.md`]&#40;./docs/README.md&#41;. Useful for balancing training data.)

### Training, along with validation and testing
The training phase is where the neural network learn to use the data prepared in the previous phase to make all the predictions.
The crux of the learning process is the training phase.  

### Inference
The inference phase allows the use of a trained model to predict on new input data.

## **Requirement**
This project comprises a set of commands to be run at a shell command prompt.  Examples used here are for a bash shell in an Ubuntu GNU/Linux environment.

- [Python 3.9](https://www.python.org/downloads/release/python-390/), see the full list of dependencies in [environment.yml](environment.yml)
- [miniconda](https://docs.conda.io/en/latest/miniconda.html) (highly recommended)

## **Installation**
Those steps are for your workstation on Ubuntu 18.04 using miniconda.
Set and activate your python environment with the following commands:  
```shell
conda env create -f environment.yml
conda activate remote_com_env
```
## **Running the notebooks**
This is an example of how to run the notebooks:

1. Clone this github repo.
```shell
git clone https://github.com/A-R-Yazdi/Remote-communities-.git
cd Remote-communities-
```

2. 

3. Run the wanted notebooks.
```shell
jupyter notebook
```

## **Folder Structure**
We suggest a high level structure to organize the images and the code.
```
├── environment.yml
├── notebooks
│   ├── XGBoost_Ali_March1.ipynb
│   ├── XGBoost.ipynb
│   └── XGBoost-LingJun.ipynb
└── README.md
```

## Scaling

To scale our notebook to include all the locations, we use `papermill` package. 

Papermill allows us to parametrize and execute notebooks. We need to set the `dff` parameter to `df1, df2...`

To do this, first add a `parameters` tag tothe cell in the notebook. In this case, we need to add the tag to `thedf` parameter. 
- Menu bar -> View -> Cell toolbar -> Tags
- Enter parameters in the textbook on the top of the cell
- Click add tag

Next we create `runner.ipynb` to run `XGBoost-LingJun_df1.ipynb` with different parameters. 

