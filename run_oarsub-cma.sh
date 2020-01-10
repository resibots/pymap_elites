for i in `seq 1 30`; do
	oarsub -d `pwd` -l /nodes=1/core=32,walltime=23:00 `pwd`/run_benchmark-cma.sh; 
done
