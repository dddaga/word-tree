#!/bin/bash
# Batch caveman compress for learnings/ and .claude/skills/
# Usage: bash scripts/caveman_batch_compress.sh

COMPRESS_DIR="/Users/indra/.claude/plugins/cache/caveman/caveman/84cc3c14fa1e/skills/compress"
ROOT="/Volumes/T9/IndraAstra/dhiraj/neuro_graph"

total=0
success=0
skip=0
fail=0

compress_file() {
    local f="$1"
    total=$((total + 1))

    # Skip tiny files (<3 lines)
    lines=$(wc -l < "$f")
    if [ "$lines" -lt 3 ]; then
        echo "SKIP ($lines lines): $f"
        skip=$((skip + 1))
        return
    fi

    echo "[$total] Compressing: ${f#$ROOT/}"
    cd "$COMPRESS_DIR" && python3 -m scripts "$f" 2>&1 | tail -1
    if [ $? -eq 0 ]; then
        success=$((success + 1))
    else
        echo "FAIL: $f"
        fail=$((fail + 1))
    fi
}

# Learnings
find "$ROOT/learnings" -name "*.md" -not -name "*.original.md" -not -name "._*" -type f | sort | while read f; do
    compress_file "$f"
done

# Skills
find "$ROOT/.claude/skills" -name "*.md" -not -name "*.original.md" -not -name "._*" -type f | sort | while read f; do
    compress_file "$f"
done

echo ""
echo "=== DONE ==="
echo "Total: $total  Success: $success  Skip: $skip  Fail: $fail"
