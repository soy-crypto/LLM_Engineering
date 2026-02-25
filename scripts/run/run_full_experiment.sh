#!/bin/bash
./scripts/bootstrap/bootstrap_hf.sh
./scripts/bootstrap/bootstrap_vllm.sh

# TRT must run inside container
./scripts/bootstrap/bootstrap_trtllm.sh
./scripts/run/run_all_backends.sh