#!/bin/bash
for d in $1*; do
    cd $d
    zip all * -m
    cd ..
done