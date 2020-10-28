#!/bin/bash
qsub -V -N zipping -o out/zip.out -e out/zip.err -l nodes=1:ppn=1,mem=1gb,vmem=2gb -q S /srv01/technion/danielab/zip_files.sh /storage/ph_daniel/danielab/ECMC_simulation_results2.0