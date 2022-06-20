# # bash script takes one fh as input param 
# curl -X POST localhost:8001/predict?fh=$1

# can change -fh to optional param with a default
python model_monitoring/score.py -fh 23 > model_monitoring/score.txt

score=$(cat model_monitoring/score.txt)

# if statement doesn't have the correct syntax, 
if [$score -eq "{'17542':0.6735439099177059}"]; then 
    echo "Retrain model..."
    bash run.sh
else
    echo "Model does not need update"
fi 

