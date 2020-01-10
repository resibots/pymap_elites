ROOT=`pwd`
for DIM in `seq 2 15`; do
       	cd $ROOT
	DIR=data/$DIM/sampling/${$}/
	mkdir -p $DIR
	cd $DIR
	cp $ROOT/centroids_1000_2.dat .
        python3 ../../../../benchmark_sampling.py centroids_1000_2.dat $DIM 
done
