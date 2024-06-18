conda env export --no-builds --from-history | grep -v "prefix" > environment.yml

conda env create -f environment.yml

rsync -av /Users/yusuf/Desktop/ESC50  ybrima@beam.cv.uos.de:/net/projects/scratch/winter/valid_until_31_July_2024/ybrima/Gym

rsync -v /Users/yusuf/Desktop/ESC50/environment.yml  ybrima@beam.cv.uos.de:/net/projects/scratch/winter/valid_until_31_July_2024/ybrima/Gym/ESC50/

rsync -av /Users/yusuf/Desktop/ESC50  ybrima@beam.cv.uos.de:/net/projects/scratch/winter/valid_until_31_July_2024/ybrima/Gym

rsync -av  ybrima@beam.cv.uos.de:/net/projects/scratch/winter/valid_until_31_July_2024/ybrima/Gym/ESC50/ /Users/yusuf/Desktop/ESC50