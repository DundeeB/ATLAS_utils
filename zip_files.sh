#!/bin/bash
for d in /storage/ph_daniel/danielab/ECMC_simulation_results2.0/*; do
    cd $d
    zip all * -m
    cd ..
done