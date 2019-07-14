ROOT=`pwd`
for DIM in 2 15; do
    for T in 1 3 5 10 50 100 500 1000; do
        echo $T
       	cd $ROOT
	DIR=data/$DIM/tournament$T/${$}/
	mkdir -p $DIR
	cd $DIR
	cp $ROOT/centroids_5000_2.dat .
        python3 ../../../../map_elites.py neighbors_tournament $T $DIM 
    done
done
