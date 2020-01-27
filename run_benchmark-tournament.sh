echo "5000 niches, fixed-size tournament"
ROOT=`pwd`
for T in 1 5 10 50 100 500 1000 5000; do 
    for DIM in `seq 2 15`; do
       	cd $ROOT
	DIR=data/$DIM/tournament-$T/${$}/
	mkdir -p $DIR
	cd $DIR
	cp $ROOT/centroids_1000_2.dat .
	cp $ROOT/centroids_5000_2.dat .
	python3 ../../../../multitask_arm.py tournament $T $DIM 
    done
done

