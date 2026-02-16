@echo off
REM push_and_tag.cmd — Windows batch script for commit, push, and tag

setlocal enabledelayedexpansion

set REPO_URL=%1
if "%REPO_URL%"=="" (
    set REPO_URL=https://github.com/BLACKSAND24/my-project.git
)

set TAG_VERSION=%2
if "%TAG_VERSION%"=="" (
    set TAG_VERSION=v1.0.0
)

set TAG_MESSAGE=%3
if "%TAG_MESSAGE%"=="" (
    set TAG_MESSAGE=live-ready: EE stable, LIVE mode activated
)

echo.
echo ==========================================
echo Financial Organism: LIVE Release
echo ==========================================
echo Repo URL    : !REPO_URL!
echo Tag         : !TAG_VERSION!
echo Message     : !TAG_MESSAGE!
echo ==========================================
echo.

REM Check if config.py exists
if not exist "config.py" (
    echo ERROR: config.py not found. Run this script from the repo root.
    exit /b 1
)

REM 1. Check if remote exists
git remote | findstr /C:"origin" >nul
if errorlevel 1 (
    echo Adding remote...
    call git remote add origin "!REPO_URL!"
) else (
    echo Remote already exists:
    call git remote -v
)

REM 2. Stage changes
echo Staging changes...
call git add -A

REM 3. Commit (if needed)
git diff --cached --quiet
if errorlevel 1 (
    echo Committing changes...
    call git commit -m "feat: switch to LIVE mode - Updated config.py: MODE=LIVE - Added EE metrics logging (ee_metrics.csv) - Integrated analytics.report() in main loop - One-line monitoring summary enabled - Ready for production deployment"
) else (
    echo No changes to commit.
)

REM 4. Push to origin
echo Pushing to !REPO_URL!...
call git push -u origin HEAD

REM 5. Create and push tag
echo Creating tag !TAG_VERSION!...
call git tag -a "!TAG_VERSION!" -m "!TAG_MESSAGE!"
call git push origin "!TAG_VERSION!"

echo.
echo ==========================================
echo ✓ Release complete!
echo Repo URL: !REPO_URL!
echo Tag     : !TAG_VERSION!
echo ==========================================
echo.

endlocal
