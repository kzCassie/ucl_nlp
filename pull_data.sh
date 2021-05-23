#!/bin/bash

echo "download original Conala json dataset"
data_file="conala-corpus-v1.1.zip"
wget -cP data/ http://www.phontron.com/download/${data_file}
unzip data/${data_file} -d data/
rm data/${data_file}

for dataset in conala;
do
	mkdir -p saved_models/${dataset}
	mkdir -p logs/${dataset}
	mkdir -p decodes/${dataset}
done

echo "Done!"
