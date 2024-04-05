curl https://dl.fbaipublicfiles.com/LAMA/data.zip --output data.zip
unzip data.zip
cp -r ./data ./remasked_data
find remasked_data/. -type f -name '*.jsonl' -print0 | xargs -0 sed -i "" -e "s/ \[MASK\]/\[MASK\]/g"
