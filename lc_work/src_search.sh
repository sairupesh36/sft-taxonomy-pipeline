#!/bin/bash
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy
NEEDLE="Saturday after the 3rd Friday" PYTHONPATH=pylibs python3 lc_work/find_source5.py sft_to_reasoning final Reasoning NL generated_data Generated_Dataset Safety Tool_Use data Software_Engineering > lc_work/src_search.out 2>&1
echo FINISHED >> lc_work/src_search.out
