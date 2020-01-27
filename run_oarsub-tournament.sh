for i in `seq 1 30`; do
	oarsub -d `pwd` -l /nodes=1/core=24,walltime=96:00 `pwd`/run_benchmark-tournament.sh; 
done
