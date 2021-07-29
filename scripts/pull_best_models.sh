#!/bin/bash
#set -x

echo "download trained_best_models from GoogleDrive"
filename='best_pretrained_models.zip'
fileid='1NRBrjK46UmRsdCJeNiX0u4TJIeE5JKdR'
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm ./cookie

# Unzip
unzip -q ${filename}
rm ${filename}

echo "Done!"


