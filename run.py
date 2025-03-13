#!/usr/bin/env python3

import asyncio
import base64
import os
import sys
from collections import defaultdict, deque
from datetime import datetime
from typing import DefaultDict, Dict, List, NamedTuple

import numpy as np

try:
    from rich import box
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
    )
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("Installing required packages...")
    os.system("pip install rich")
    from rich import box
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
    )
    from rich.table import Table
    from rich.text import Text

# Initialize Rich console
console = Console()

# Keep track of console output
console_lines = deque(maxlen=10)  # Keep last 10 lines


class JobConfig(NamedTuple):
    nqubits: int
    measure: bool
    seed: int


class JobResult(NamedTuple):
    config: JobConfig
    job_id: str
    status: str
    duration: float
    exec_id: str
    start_time: datetime


# Grid search parameters
NQUBITS_VALUES = [4, 6, 8, 10, 12]
MEASURE_VALUES = [True, False]
# Generate 15 random seeds between 1000 and 9999
SEED_VALUES = sorted(np.random.randint(1000, 10000, size=15))


def create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )


def create_console_panel() -> Panel:
    # Join the console lines with newlines and wrap in a panel
    content = Group(*[Text(line, style="bright_black") for line in console_lines])
    return Panel(
        content,
        title="Live Console Output",
        title_align="left",
        border_style="blue",
        box=box.ROUNDED,
        height=12,
    )


def create_qubit_progress(
    completed_jobs: List[JobResult], in_progress_jobs: Dict[JobConfig, str]
) -> Table:
    # Create a compact table showing progress for each number of qubits
    table = Table(
        box=None,  # Remove box for more compact display
        show_header=False,
        show_edge=False,
        pad_edge=False,
        padding=(0, 1),
    )

    # Add one column for each qubit value
    for nq in NQUBITS_VALUES:
        table.add_column(justify="center", style="cyan", width=8)

    # Count completed jobs per nqubits
    completed_per_nq: DefaultDict[int, int] = defaultdict(int)
    for job in completed_jobs:
        completed_per_nq[job.config.nqubits] += 1

    # Count in-progress jobs per nqubits
    running_per_nq: DefaultDict[int, int] = defaultdict(int)
    for config in in_progress_jobs:
        running_per_nq[config.nqubits] += 1

    # Calculate total jobs per nqubits
    total_per_nq = len(MEASURE_VALUES) * len(SEED_VALUES)

    # Create status for each qubit count
    statuses = []
    for nq in NQUBITS_VALUES:
        completed = completed_per_nq[nq]
        running = running_per_nq[nq]
        total = total_per_nq

        if completed == total:
            status = f"[green]{nq}q ✓[/green]"
        elif running > 0:
            status = f"[yellow]{nq}q {completed}/{total}[/yellow]"
        else:
            status = f"[dim]{nq}q {completed}/{total}[/dim]"
        statuses.append(status)

    table.add_row(*statuses)
    return table


def create_status_table(
    completed_jobs: List[JobResult], in_progress_jobs: Dict[JobConfig, str]
) -> Table:
    # Create the results table
    table = Table(
        box=box.ROUNDED,
        header_style="bold magenta",
        border_style="blue",
        padding=(0, 1),
    )

    # Add columns for each qubit value plus other info
    for nq in NQUBITS_VALUES:
        table.add_column(f"{nq}q", justify="center", style="cyan", width=12)
    table.add_column("Measure", justify="center", style="cyan", width=8)
    table.add_column("Seed", justify="center", style="cyan", width=6)
    table.add_column("Current Output", justify="left", style="cyan", width=50)

    # Create a mapping of completed and running jobs for easy lookup
    completed_map = {
        (j.config.nqubits, j.config.measure, j.config.seed): j for j in completed_jobs
    }
    running_map = {
        (c.nqubits, c.measure, c.seed): output for c, output in in_progress_jobs.items()
    }
    start_times = {
        (c.nqubits, c.measure, c.seed): datetime.now() for c in in_progress_jobs.keys()
    }

    # Add rows for all combinations
    for measure in MEASURE_VALUES:
        for seed in SEED_VALUES:
            row = []
            current_output = ""

            # Add status for each qubit count
            for nq in NQUBITS_VALUES:
                key = (nq, measure, seed)
                if key in completed_map:
                    job = completed_map[key]
                    duration = f"{job.duration:.1f}s"
                    row.append(f"[green]{duration} ✓[/green]")
                elif key in running_map:
                    duration = (datetime.now() - start_times[key]).total_seconds()
                    row.append(f"[yellow]{duration:.1f}s ⟳[/yellow]")
                    if not current_output:
                        output = running_map[key]
                        current_output = (
                            output.split("TIME")[-1].strip()[-50:]
                            if "TIME" in output
                            else "..."
                        )
                else:
                    row.append("...")

            # Add measure and seed
            row.append(str(measure))
            row.append(str(seed))

            # Add current output
            row.append(current_output if current_output else "...")

            table.add_row(*row)

    return table


async def read_stream(stream, job_outputs, config):
    while True:
        line = await stream.readline()
        if not line:
            break
        line_str = line.decode().strip()
        if line_str:
            console_lines.append(f"{config.nqubits}:{config.seed} > {line_str}")
            job_outputs[config] = line_str


async def run_job(config: JobConfig, job_outputs: Dict[JobConfig, str]) -> JobResult:
    start_time = datetime.now()

    # Encode script
    with open("simon.py", "rb") as f:
        script_content = base64.b64encode(f.read()).decode()

    # Construct and run command
    cmd = f"""bacalhau job run simon.yaml --id-only \
        --template-vars=script="{script_content}" \
        --template-vars=nq="{config.nqubits}" \
        --template-vars=measure="{str(config.measure).lower()}" \
        --template-vars=seed="{config.seed}" \
        --api-host="{os.getenv("BACALHAU_API_HOST")}" """

    # Add command to console output
    console_lines.append(
        f"[bold blue]Executing job: nq={config.nqubits}, measure={config.measure}, seed={config.seed}[/bold blue]"
    )

    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    # Start reading streams
    stdout_task = asyncio.create_task(read_stream(proc.stdout, job_outputs, config))
    stderr_task = asyncio.create_task(read_stream(proc.stderr, job_outputs, config))

    # Wait for process to complete
    await proc.wait()
    await stdout_task
    await stderr_task

    # Parse output for job ID and exec ID
    job_id = ""
    exec_id = ""
    status = "Completed"

    output = job_outputs.get(config, "")
    for line in output.split("\n"):
        if "Job ID:" in line:
            job_id = line.split("Job ID:")[-1].strip()
        elif "EXEC. ID" in line and "Execution" in line and "Completed" in line:
            exec_id = line.split()[1].strip()

    duration = (datetime.now() - start_time).total_seconds()
    return JobResult(config, job_id, status, duration, exec_id, start_time)


async def main():
    if not os.getenv("BACALHAU_API_HOST"):
        console.print("[red]Error: BACALHAU_API_HOST is not set[/red]")
        sys.exit(1)

    # Create all job configurations
    configs = [
        JobConfig(nq, measure, seed)
        for nq in NQUBITS_VALUES
        for measure in MEASURE_VALUES
        for seed in SEED_VALUES
    ]

    # Randomize the order of jobs
    np.random.shuffle(configs)

    total_jobs = len(configs)
    completed_jobs: List[JobResult] = []
    in_progress_jobs: Dict[JobConfig, str] = {}  # Config -> current output
    max_concurrent_jobs = 5  # Limit concurrent jobs

    # Create layout
    layout = Layout()

    # Split main layout
    layout.split(
        Layout(name="progress", size=3),
        Layout(name="table"),
        Layout(name="console", size=12),
    )

    # Create progress bar
    progress = create_progress()
    overall_task = progress.add_task(f"Running {total_jobs} jobs...", total=total_jobs)

    with Live(layout, refresh_per_second=1) as live:
        while configs or in_progress_jobs:
            # Start new jobs up to max_concurrent_jobs
            while configs and len(in_progress_jobs) < max_concurrent_jobs:
                config = configs.pop(0)
                in_progress_jobs[config] = ""
                task = asyncio.create_task(run_job(config, in_progress_jobs))
                task.config = config  # Store config for reference

            # Wait for any job to complete
            if in_progress_jobs:
                done, pending = await asyncio.wait(
                    [t for t in asyncio.all_tasks() if hasattr(t, "config")],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    result = await task
                    completed_jobs.append(result)
                    del in_progress_jobs[task.config]
                    progress.advance(overall_task)

            # Update the live display
            layout["progress"].update(Panel(progress))
            layout["table"].update(
                create_status_table(completed_jobs, in_progress_jobs)
            )
            layout["console"].update(create_console_panel())

    # Display final summary
    console.print("\n")
    console.print(
        Panel.fit(
            f"Grid Search Complete - {len(completed_jobs)} jobs finished successfully",
            title="Summary",
            border_style="green",
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Job execution interrupted by user[/yellow]")
        sys.exit(1)
