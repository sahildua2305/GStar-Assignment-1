import subprocess
import modal

app = modal.App("gstar-assignment-1")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch",
        "triton",
    )
    .workdir("/workspace")
    .add_local_dir(
        ".",
        remote_path="/workspace",
        ignore=[".git", ".venv", "__pycache__"],
    )
)


@app.function(
    image=image,
    gpu="L4",
    timeout=30 * 60,
)
def run_grader(problem: int, optional: bool = False):
    grader = "autograder_optional.py" if optional else "autograder.py"

    cmd = ["python", grader, f"--p{problem}"]

    print("$", " ".join(cmd))

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(
            f"Autograder exited with code {result.returncode}"
        )


@app.local_entrypoint()
def main(problem: int = 1, optional: bool = False):
    run_grader.remote(problem, optional)
