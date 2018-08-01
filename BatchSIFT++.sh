
rm extraction_time.log
start=`date +%s`

SIFT=sift++

IN_IMAGE_DIR="."

# make list file for video frame file names
# ls *.bmp > undist_video.list

for IM in `ls -1 $IMAGE_DIR | egrep ".bmp$"`
do
	$SIFT $IM ${IM%bmp}key &
	NPROC=$(($NPROC+1))
	if [ "$NPROC" -ge 5 ]; then
		wait
		NPROC=0
	fi 
done
wait
end=`date +%s`
diff=`expr $end - $start`
echo "Elapsed time = $diff second" >> extraction_time.log

ls *.key > filelist.list
