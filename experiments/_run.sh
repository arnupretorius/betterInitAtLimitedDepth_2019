#!/bin/bash
#chmod +x run_experiments.sh
#./run_experiments.sh $1 $2 $3 $4 $5

while IFS='' read -r hyperparams || [[ -n "$hyperparams" ]]; do
	while IFS='' read -r experiment || [[ -n "$experiment" ]]; do
		python experiment.py "${hyperparams[@]}" "${experiment[@]}" "$3" "$4" "$5" "$1" "$2" "$6"
	done < "$2"
done < "$1"
