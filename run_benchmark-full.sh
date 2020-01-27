echo "1000 niches, bandit"
ROOT=`pwd`
for DIM in `seq 2 15`; do
	cd $ROOT
	DIR=data/$DIM/full/${$}/
	mkdir -p $DIR
	cd $DIR
	cp $ROOT/centroids_1000_2.dat .
	cp $ROOT/centroids_5000_2.dat .
	python3 ../../../../multitask_arm.py full 5 $DIM 
done

