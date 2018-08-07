
SIFT=sift++

IN_IMAGE_DIR="."

for IM in `ls -1 $IMAGE_DIR | egrep ".bmp$"`
do
	$SIFT $IM ${IM%bmp}key &
done
wait

ls *.key > filelist.list
