#!/bin/bash
# Remaining degrade7 finals (no half_clock); fs7c_i0/i1 already exist.
C7="MTAH_DEG=7 MTAH_DROP=kickclock"
for i in 2 3 4; do
  qsub -N fs7c_i$i -v CMD="$C7 python seq_set.py --final --seeds 0 1 2 3 4 5 6 7 --steps 8000 --aux 1 --init $i --tag _seta8nkc_i${i}_d7" hpc/run.pbs
done
qsub -N fs7c_nk -v CMD="$C7 python seq_set.py --final --seeds 0 1 2 3 --steps 6000 --tag _setnkc_d7" hpc/run.pbs
qsub -N fl7c -v CMD="bash hpc/lgbm8_fin_d7c.sh" hpc/run.pbs
