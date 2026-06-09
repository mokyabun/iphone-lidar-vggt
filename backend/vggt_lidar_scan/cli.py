from __future__ import annotations

from pathlib import Path

import typer

from .reconstruct import reconstruct_scan

app = typer.Typer(help="Reconstruct iPhone/iPad LiDAR scan packages.")


@app.command()
def main(
    package: Path = typer.Argument(..., help="Path to ScanPackage.zip or an unpacked ScanPackage directory."),
    output: Path = typer.Option(Path("runs/latest"), "--output", "-o", help="Output directory."),
    max_frames: int = typer.Option(48, help="Maximum frames/keyframes to process."),
    stride: int = typer.Option(4, help="Depth pixel stride for point-cloud baseline."),
    confidence_minimum: int = typer.Option(1, help="Minimum ARKit confidence value to keep."),
    run_vggt: bool = typer.Option(False, "--run-vggt", help="Run optional VGGT stage."),
) -> None:
    metrics = reconstruct_scan(
        package,
        output,
        max_frames=max_frames,
        stride=stride,
        confidence_minimum=confidence_minimum,
        run_vggt_stage=run_vggt,
    )
    typer.echo(metrics.model_dump_json(indent=2))


if __name__ == "__main__":
    app()

