#!/bin/bash
# push_and_tag.sh — Commits, pushes, and tags a LIVE-ready release

set -e  # Exit on any error

REPO_URL="${1:-https://github.com/BLACKSAND24/my-project.git}"
TAG_VERSION="${2:-v1.0.0}"
TAG_MESSAGE="${3:-live-ready: EE stable, LIVE mode activated}"

echo "=========================================="
echo "Financial Organism: LIVE Release"
echo "=========================================="
echo "Repo URL    : $REPO_URL"
echo "Tag         : $TAG_VERSION"
echo "Message     : $TAG_MESSAGE"
echo "=========================================="

# Ensure we're in the right directory
if [ ! -f "config.py" ]; then
    echo "ERROR: config.py not found. Run this script from the repo root."
    exit 1
fi

# 1. Check if remote exists
if ! git remote | grep -q origin; then
    echo "Adding remote..."
    git remote add origin "$REPO_URL"
else
    echo "Remote already exists:"
    git remote -v
fi

# 2. Stage changes
echo "Staging changes..."
git add -A

# 3. Commit (if needed)
if git diff --cached --quiet; then
    echo "No changes to commit."
else
    git commit -m "feat: switch to LIVE mode

- Updated config.py: MODE=LIVE
- Added EE metrics logging (ee_metrics.csv)
- Integrated analytics.report() in main loop
- One-line monitoring summary enabled
- Ready for production deployment"
fi

# 4. Push to origin
echo "Pushing to $REPO_URL..."
git push -u origin HEAD

# 5. Create and push tag
echo "Creating tag $TAG_VERSION..."
git tag -a "$TAG_VERSION" -m "$TAG_MESSAGE"
git push origin "$TAG_VERSION"

echo "=========================================="
echo "✅ Release complete!"
echo "Repo URL: $REPO_URL"
echo "Tag     : $TAG_VERSION"
echo "=========================================="
