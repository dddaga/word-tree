# UV Python Environment Setup

## Issue: ExFAT Filesystem Limitation

The T9 drive uses **ExFAT**, which doesn't support:
- Symlinks (required for Python virtual environments)
- Some filesystem operations UV needs for caching

**Solution**: Keep all UV-related files on your local drive (APFS/HFS+).

## Setup

### 1. Virtual Environment Created

✅ Virtual environment already created at: `~/.virtualenvs/word-tree`  
✅ Packages installed: `torch`, `qdrant-client`, `numpy`

### 2. Configure Environment (IMPORTANT!)

The `.uvenv` file sets `TMPDIR` to `~/tmp` which is **required** for UV to work from the ExFAT drive.

**Add to your `~/.zshrc`:**
```bash
cat /Volumes/T9/work/word-tree/.uvenv >> ~/.zshrc
source ~/.zshrc
```

### 3. Activate Environment

```bash
source ~/.virtualenvs/word-tree/bin/activate
```

Or source `.uvenv` for the alias:
```bash
source .uvenv
activate-word-tree
```

### 4. Install Packages

```bash
# Activate first
source ~/.virtualenvs/word-tree/bin/activate

# Load UV configuration (sets TMPDIR)
source .uvenv

# Install with uv (faster)
uv pip install torch qdrant-client numpy

# Or use regular pip
pip install -r requirements.txt
```

## UV Configuration

- **Cache**: `~/.cache/uv` (default, on local drive)
- **Tools**: `~/.local/share/uv/tools` (default)
- **Python installs**: `~/.local/share/uv/python` (default)
- **Virtual env**: `~/.virtualenvs/word-tree`
- **TMPDIR**: `~/tmp` ⚠️ **REQUIRED** - UV needs temp files on local drive

## Alternative: Project-local .venv

If you prefer to keep the venv in the project:

```bash
cd /Volumes/T9/work/word-tree
uv venv .venv
source .venv/bin/activate
```

⚠️ **Note**: The `.venv` directory must be on a local drive, not on T9.
Consider symlinking the project to your home directory if needed.

## Usage with UV

```bash
# Create venv
uv venv ~/.virtualenvs/word-tree

# Install packages
uv pip install package-name

# Run Python
uv run python script.py

# Sync dependencies (if using pyproject.toml)
uv pip sync
```

## Troubleshooting

If you see "Operation not supported (os error 45)":
1. ✅ **Ensure `TMPDIR` is set**: `echo $TMPDIR` should show `/Users/dhiraj/tmp`
2. **Load the configuration**: `source .uvenv` before using UV
3. Ensure you're not setting `UV_CACHE_DIR` to a path on T9
4. Ensure the virtual environment is on your local drive
5. Check: `diskutil info /Volumes/T9 | grep "File System"` (should show ExFAT)

**Quick fix:**
```bash
source .uvenv  # This sets TMPDIR and other env vars
uv pip install <package>
```

