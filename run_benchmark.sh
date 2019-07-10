for DIM in `seq 2 15`; do
    for T in 1 3 5 10 50 100 500 1000; do
        echo $T
        python3 ./map_elites.py neighbors_tournament $T $DIM
        mkdir -p data/$DIM/tournament$T/
        mv archive*.dat data/$DIM/tournament$T/
    done
done
