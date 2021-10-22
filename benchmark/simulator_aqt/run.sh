#!/bin/bash

c="jupyter nbconvert --execute --to html --output-dir=results/ --ExecutePreprocessor.timeout=-1  "
FILES="notebooks/*.ipynb"
for f in $FILES
do

        temp="$c"
        temp+="$f"
        tmux new -d "$temp"
	echo "$temp"
done
