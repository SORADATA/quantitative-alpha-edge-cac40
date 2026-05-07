#!/bin/bash
BRANCH=$(git rev-parse --abbrev-ref HEAD)
export HF_DATA_BRANCH=$BRANCH
echo "Using HF branch: $BRANCH"
python -m streamlit run app.py