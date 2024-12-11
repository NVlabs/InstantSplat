# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

model=$1
outfile="stereoflow_models/${model}"
if [[ ! -f $outfile ]]
then
	mkdir -p stereoflow_models/;
	wget https://download.europe.naverlabs.com/ComputerVision/CroCo/StereoFlow_models/$1 -P stereoflow_models/;
else
	echo "Model ${model} already downloaded in ${outfile}."
fi