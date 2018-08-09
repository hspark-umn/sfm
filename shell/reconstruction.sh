EXE_FOLDER="/home/hspark/sfm_github"


cd image
cp $EXE_FOLDER/calib/calib_fisheye_gopro5_960.txt calib_fisheye.txt
$EXE_FOLDER/shell/BatchSIFT++.sh
cd ..
$EXE_FOLDER/Matching_sift/build/Matching_sift
$EXE_FOLDER/Aggregation_sift/build/Aggregation_sift
$EXE_FOLDER/InitialFrameSelection/build/InitialFrameSelection
$EXE_FOLDER/SfM/build/SfM


