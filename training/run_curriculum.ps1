# run_curriculum.ps1 — run the 2-stage disturbance curriculum end to end.
#
# Each stage fine-tunes from the previous stage's BEST checkpoint at a gentle
# learning rate (3e-5) so it adapts to the harder disturbance without wrecking
# the policy. Stages ramp force + duration + damping (see disturbance_stages):
#   stage2a : 4-8 N,  1-10 frames, damp 0.5  (from stage1/best)
#   stage2b : 6-12 N, 1-20 frames, damp 1.0  (from stage2a/best)
#
# (A 3rd stage was dropped — 8-15 N needs damping ~2.0, stiffer than real air
# drag, so it would teach a sim-to-real crutch. See config.yaml.)
#
# The fixed-seed eval (clean best selection) applies to all stages. A stage only
# runs if the previous one succeeded, so a crash never feeds a broken checkpoint
# into the next stage.
#
# Usage (from the training/ directory):
#   .\run_curriculum.ps1                 # 400k steps per stage (default)
#   .\run_curriculum.ps1 -Steps 1000000  # full 1M steps per stage
#   .\run_curriculum.ps1 -Python ..\.venv\Scripts\python.exe   # explicit interpreter

param(
    [int]$Steps = 400000,
    [string]$Python = "python",
    [double]$Lr = 3e-5
)

# Resolve to this script's own directory so it works regardless of CWD.
Set-Location -Path $PSScriptRoot

$ErrorActionPreference = "Stop"

# (stage name, disturbance-curriculum stage number, init-from checkpoint)
$chain = @(
    @{ name = "stage2a"; dist = 1; init = "models/stage1/best.zip"  },
    @{ name = "stage2b"; dist = 2; init = "models/stage2a/best.zip" }
)

foreach ($s in $chain) {
    if (-not (Test-Path $s.init)) {
        Write-Host "ABORT: init checkpoint '$($s.init)' missing — previous stage did not produce a best.zip." -ForegroundColor Red
        exit 1
    }

    Write-Host "`n=== curriculum $($s.name) [dist-stage $($s.dist), $Steps steps, lr $Lr] ===" -ForegroundColor Cyan
    & $Python train.py --stage $s.name --dist-stage $s.dist --init-from $s.init --lr $Lr --timesteps $Steps

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ABORT: $($s.name) exited with code $LASTEXITCODE — not continuing the chain." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "`n=== curriculum complete. Final policy: models/stage2b/best.zip ===" -ForegroundColor Green
Write-Host "Watch it:  $Python eval.py stage2b"
