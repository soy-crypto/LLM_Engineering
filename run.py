import subprocess

print("Running full experiment pipeline")
subprocess.run(["./scripts/run/run_full_experiment.sh"], check=True)

print("Aggregating results")
subprocess.run(["python", "analysis/aggregate.py"], check=True)

print("Plotting results")
subprocess.run(["python", "analysis/plot_results.py"], check=True)

print("Done.")