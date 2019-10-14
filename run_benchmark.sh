ROOT=`pwd`
for DIM in `seq 2 15`; do
#    for T in 1 3 5 10 50 100 500 1000; do
#        echo $T
       	cd $ROOT
	DIR=data/$DIM/tournament-bandit/${$}/
	mkdir -p $DIR
	cd $DIR
	cp $ROOT/centroids_5000_2.dat .
        python3 ../../../../map_elites.py tournament_gp 5 $DIM 
 #   done
done
