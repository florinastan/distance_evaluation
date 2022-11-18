if [ ! -d "$samples_resized" ]; 
then
  mkdir samples_resized
fi

for file in *
do 	
	if [ "$file" == "resize.sh"  ]
	then
		echo "Skiping $file file..."
	else 
		filename="${file%.*}"
		ffmpeg -i $file -vf scale=-1:720 -c:v libx264 -crf 0 -preset veryslow -c:a copy samples_resized/$filename"_resized".mp4
	fi
done

