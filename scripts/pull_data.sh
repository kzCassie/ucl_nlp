#!/bin/bash
#set -x

echo "download original Conala json dataset"
data_file="conala-corpus-v1.1.zip"
wget -cP data/ http://www.phontron.com/download/${data_file}
unzip data/${data_file} -d data/ && mv data/conala-corpus
rm data/${data_file}

echo "download preprocessed Conala zip from GoogleDrive"
mined_size="100000"
fileid="1ePpFGiRaSHH0CzuOzG0CP7M1dfzp-y5O"
proc_data_file="${mined_size}.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${proc_data_file}
rm ./cookie
mv ${proc_data_file} data/
unzip data/${proc_data_file} -d data/conala
rm data/${proc_data_file}

for dataset in conala;
do
	mkdir -p saved_models/${dataset}
	mkdir -p logs/${dataset}
	mkdir -p decodes/${dataset}
done

echo "Done!"


