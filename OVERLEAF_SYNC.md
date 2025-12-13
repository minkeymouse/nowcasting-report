# Overleaf Sync Instructions

## Issue
Compilation errors persist in Overleaf, and updated contents are not visible.

## Root Cause
The fix has been committed and pushed to GitHub, but Overleaf needs to be manually synced with the remote repository.

## Solution

### Step 1: Sync Overleaf with GitHub
1. In Overleaf, go to **Menu** → **Git** (or click the Git icon in the top bar)
2. Click **Pull from GitHub** (or **Sync from GitHub**)
3. Wait for the sync to complete

### Step 2: Verify the Fix
After syncing, check that `preamble.tex` contains:
```latex
\usepackage{threeparttable} % for tablenotes environment
```
This should be on line 19, after `\usepackage{longtable}`.

### Step 3: Recompile in Overleaf
1. Click **Recompile** in Overleaf
2. The compilation should now succeed without `tablenotes` errors

## What Was Fixed
- **Commit**: `c697225` - "Fix LaTeX compilation: Add threeparttable package for tablenotes environment"
- **File**: `preamble.tex` (line 19)
- **Change**: Added `\usepackage{threeparttable}` package

## Verification
The fix is confirmed in the remote repository:
- Remote commit: `b2e0be380fd218ecd781a3b17341d24b27ed0ff0`
- Local compilation: ✅ Successful
- Remote status: ✅ Up to date

## If Sync Doesn't Work
If Overleaf Git sync doesn't work:
1. Check Overleaf project settings → Git integration is enabled
2. Try **Menu** → **Download** → **Source** to get the latest files
3. Manually upload the updated `preamble.tex` file

## Files That May Still Cause Errors
If you still see `tablenotes` errors after syncing, check if you have files that use `\begin{tablenotes}`:
- These files should be in `contents/tables/` directory (if they exist)
- The error messages mentioned: `multivariate_groups.tex` and `forecast_method_results.tex`
- These files don't exist in the current repository structure, so the errors may be from a different project or old files

