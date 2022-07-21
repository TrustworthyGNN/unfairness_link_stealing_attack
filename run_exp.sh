# shellcheck disable=SC2006
py_path=`which python`
run() {
    dataset=$1
    $py_path train.py --dataset "$dataset" --model gcn
    $py_path train.py --dataset "$dataset" --model mlp
    $py_path partial_graph_generation.py --dataset "$dataset"
    $py_path attack.py --dataset "$dataset"
}

# shellcheck disable=SC2046
# shellcheck disable=SC2006
#echo $epoch
run "$1"