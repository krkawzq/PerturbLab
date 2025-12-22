#!/bin/bash

# Script to add source attribution headers to Python files in scfoundation source directory
# Usage: bash add_scfoundation_headers.sh

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SOURCE_DIR="perturblab/model/scfoundation/source"
HEADER_SCFOUNDATION="# modified from https://github.com/BiomedSciAI/scFoundation"

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
    
    # Check if file already has any attribution header
    if head -n 1 "$file" 2>/dev/null | grep -q "# modified from\|# Copyright"; then
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
    if add_header "$file" "$HEADER_SCFOUNDATION"; then
        ((modified_count++))
    else
        ((skipped_count++))
    fi
done < <(find "$SOURCE_DIR" -name "*.py" -type f)

echo ""
echo "========================================"
echo -e "${GREEN}Modified files: $modified_count${NC}"
echo -e "${YELLOW}Skipped files:  $skipped_count${NC}"
echo "========================================"
echo -e "${BLUE}Done!${NC}"

