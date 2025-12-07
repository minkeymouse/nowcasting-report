#!/bin/bash
# Compile LaTeX report and save all build artifacts to compiled/ directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPILED_DIR="${SCRIPT_DIR}/compiled"

# Create compiled directory if it doesn't exist
mkdir -p "${COMPILED_DIR}"

# Change to report directory
cd "${SCRIPT_DIR}"

echo "Compiling LaTeX report..."
echo "Build artifacts will be saved to: ${COMPILED_DIR}"

# Compile with output directory set to compiled/
pdflatex -interaction=nonstopmode -output-directory="${COMPILED_DIR}" main.tex > "${COMPILED_DIR}/compile.log" 2>&1 || {
    echo "First pdflatex pass failed. Check ${COMPILED_DIR}/compile.log"
    exit 1
}

# Run bibtex (needs to be in the compiled directory but needs access to references.bib)
cd "${COMPILED_DIR}"
# Copy references.bib to compiled directory for bibtex
cp "${SCRIPT_DIR}/references.bib" "${COMPILED_DIR}/references.bib" 2>/dev/null || true
bibtex main > bibtex.log 2>&1 || {
    echo "BibTeX failed. Check ${COMPILED_DIR}/bibtex.log"
    exit 1
}
cd "${SCRIPT_DIR}"

# Run pdflatex again to resolve references
pdflatex -interaction=nonstopmode -output-directory="${COMPILED_DIR}" main.tex >> "${COMPILED_DIR}/compile.log" 2>&1 || {
    echo "Second pdflatex pass failed. Check ${COMPILED_DIR}/compile.log"
    exit 1
}

# Run pdflatex one more time to ensure all references are resolved
pdflatex -interaction=nonstopmode -output-directory="${COMPILED_DIR}" main.tex >> "${COMPILED_DIR}/compile.log" 2>&1 || {
    echo "Third pdflatex pass failed. Check ${COMPILED_DIR}/compile.log"
    exit 1
}

# Copy PDF back to main directory for easy access (optional)
if [ -f "${COMPILED_DIR}/main.pdf" ]; then
    cp "${COMPILED_DIR}/main.pdf" "${SCRIPT_DIR}/main.pdf"
    echo "✓ Compilation successful!"
    echo "  PDF: ${COMPILED_DIR}/main.pdf (also copied to ${SCRIPT_DIR}/main.pdf)"
    echo "  All build artifacts saved in: ${COMPILED_DIR}"
else
    echo "✗ Compilation failed: PDF not found"
    exit 1
fi

