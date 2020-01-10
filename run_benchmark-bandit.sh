echo "1000 niches, bandit"
ROOT=`pwd`
for DIM in `seq 2 15`; do
       	cd $ROOT
	DIR=data/$DIM/bandit/${$}/
	mkdir -p $DIR
	cd $DIR
	cp $ROOT/centroids_1000_2.dat .
        python3 ../../../../multitask_arm.py bandit_niche 5 $DIM 
done

