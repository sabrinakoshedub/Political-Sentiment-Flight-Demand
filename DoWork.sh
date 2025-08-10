#!/bin/bash

set -e
set -o pipefail

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python packages..."
pip install --only-binary=:all: numpy pandas matplotlib seaborn statsmodels ipywidgets scikit-learn

echo "Running Python scripts in Code/..."
for script in Code/*.py; do
    echo "Executing $script..."
    if ! python "$script"; then
        echo "Failed to run $script"
    fi
done

echo "Compiling LaTeX..."
TEX_MAIN="Files/main.tex"
if [ -f "$TEX_MAIN" ]; then
    cd Files
    pdflatex main.tex || { echo "First LaTeX pass failed"; exit 1; }
    bibtex main || { echo "BibTeX failed"; exit 1; }
    pdflatex main.tex || { echo "Second LaTeX pass failed"; exit 1; }
    pdflatex main.tex || { echo "Third LaTeX pass failed"; exit 1; }
    mv main.pdf ../SKColaco_Final.pdf
    cd ..
    echo "PDF built and saved to SKColaco_Final.pdf"

else
    echo "LaTeX file not found at $TEX_MAIN. Skipping PDF compilation."
fi




echo "Done!"

