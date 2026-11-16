#!/bin/bash
# degrade7 (d6 + ball presence recalibrated per tempo), no half_clock. CV paired both ways with d6:
# s67c = d6-trained on d7 eval (vs s77c), s76c = d7-trained on d6 eval (vs s66c_i0 0.675). Then finals.
F=$(qsub -N feats_d7 -v CMD="bash hpc/feats_d7.sh" hpc/run.pbs); echo $F
DEP="-W depend=afterok:$F"
S8="--seeds 0 1 2 3 4 5 6 7 --steps 8000 --aux 1 --tta"
C7="MTAH_DEG=7 MTAH_DROP=kickclock"; X67="MTAH_DEG=6 MTAH_EVAL_DEG=7 MTAH_DROP=kickclock"; X76="MTAH_DEG=7 MTAH_EVAL_DEG=6 MTAH_DROP=kickclock"
qsub $DEP -N s77c_i0 -v CMD="$C7 python seq_set.py --cv $S8 --init 0 --tag _seta8nkc_i0tta_d7" hpc/run.pbs
qsub $DEP -N s67c_i0 -v CMD="$X67 python seq_set.py --cv $S8 --init 0 --tag _seta8nkc_i0tta_d6e7" hpc/run.pbs
qsub $DEP -N s76c_i0 -v CMD="$X76 python seq_set.py --cv $S8 --init 0 --tag _seta8nkc_i0tta_d7e6" hpc/run.pbs
qsub $DEP -N s77c_i1 -v CMD="$C7 python seq_set.py --cv $S8 --init 1 --tag _seta8nkc_i1tta_d7" hpc/run.pbs
qsub $DEP -N s67c_i1 -v CMD="$X67 python seq_set.py --cv $S8 --init 1 --tag _seta8nkc_i1tta_d6e7" hpc/run.pbs
for i in 0 1 2 3 4; do
  qsub $DEP -N fs7c_i$i -v CMD="$C7 python seq_set.py --final --seeds 0 1 2 3 4 5 6 7 --steps 8000 --aux 1 --init $i --tag _seta8nkc_i${i}_d7" hpc/run.pbs
done
qsub $DEP -N fs7c_nk -v CMD="$C7 python seq_set.py --final --seeds 0 1 2 3 --steps 6000 --tag _setnkc_d7" hpc/run.pbs
qsub $DEP -N fl7c -v CMD="bash hpc/lgbm8_fin_d7c.sh" hpc/run.pbs
