# Profiling Utils

To modify the specific SLURM resources types to benchmark, adjust the launcher script `launch_benchmark.py` as needed. Modify `profiling/configs/lora-benchmark.yaml` to adjust parameters such as batch size and token width.

On the Vector cluster, run the following to launch the benchmarks:

```bash
$ mkdir data/
$ python3 launch_benchmark.py

# The launcher script will print a list of
# SLURM commands it plans to run. Press ENTER
# to accept and automatically invoke the commands.
```

After the SLURM jobs complete, profiler output can be found under `data/benchmark`. Invoke the following the to generate a Markdown summary of the results:

```bash
$ python3 profiling/parse_benchmark.py --folder data/benchmark
```
