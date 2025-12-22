#!/bin/bash

# Script to add source attribution headers to Python files in the source directory
# Usage: bash add_source_headers.sh

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SOURCE_DIR="perturblab/model/scgpt/source"
HEADER_SCGPT="# modified from https://github.com/bowang-lab/scGPT"
HEADER_TORCHTEXT="# modified from torchtext.vocab (https://github.com/pytorch/text)"

echo -e "${BLUE}Adding source attribution headers to Python files...${NC}"
echo "Target directory: $SOURCE_DIR"
echo ""

# Counter for modified files
modified_count=0
skipped_count=0

# Function to add header to a file
add_header() {
    local file=$1
    local header=$2
    
    # Check if file already has the header
    if head -n 1 "$file" 2>/dev/null | grep -q "# modified from"; then
        echo -e "${YELLOW}[SKIP]${NC} $file (already has header)"
        return 1
    fi
    
    # Create temporary file with header
    {
        echo "$header"
        cat "$file"
    } > "${file}.tmp"
    
    # Replace original file
    mv "${file}.tmp" "$file"
    
    echo -e "${GREEN}[ADDED]${NC} $file"
    return 0
}

# Find all Python files and process them
while IFS= read -r file; do
    # Get filename without path
    filename=$(basename "$file")
    
    # Special case for torch_vocab.py
    if [[ "$filename" == "torch_vocab.py" ]]; then
        if add_header "$file" "$HEADER_TORCHTEXT"; then
            ((modified_count++))
        else
            ((skipped_count++))
        fi
    else
        if add_header "$file" "$HEADER_SCGPT"; then
            ((modified_count++))
        else
            ((skipped_count++))
        fi
    fi
done < <(find "$SOURCE_DIR" -name "*.py" -type f)

echo ""
echo "========================================"
echo -e "${GREEN}Modified files: $modified_count${NC}"
echo -e "${YELLOW}Skipped files:  $skipped_count${NC}"
echo "========================================"
echo -e "${BLUE}Done!${NC}"
