echo "1000 niches, fixed-size tournament"
ROOT=`pwd`
for T in 1 3 5 10 50 100 500 1000; do 
    for DIM in `seq 2 15`; do
       	cd $ROOT
	DIR=data/$DIM/tournament-$T/${$}/
	mkdir -p $DIR
	cd $DIR
	cp $ROOT/centroids_1000_2.dat .
        python3 ../../../../multitask_arm.py bandit_niche $T $DIM 
    done
done

