# # bash script takes one fh as input param 
# curl -X POST localhost:8001/predict?fh=$1

# echo $0 
full_path=$(realpath $0)
# echo $full_path
dir_path=$(dirname $full_path)
# echo $dir_path
# echo $(dirname $(realpath $0))


# it's important that we used absolute path here with $dir_path, this means we can use relative path in the monitor.py file
# and it won't matter where we run this script.
python $dir_path/monitor.py
