import subprocess
import argparse
import sys
import time

def run_step(script_name, n_jobs, description):
    print(f"\n{'='*60}")
    print(f"PIPELINE STEP: {description}")
    print(f"Running {script_name} with n_jobs={n_jobs}...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # We use subprocess to run the script as if it were called from the command line.
    # This ensures a clean memory state for each heavy step.
    cmd = [sys.executable, script_name, "--n_jobs", str(n_jobs)]
    
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n>>> SUCCESS: {script_name} completed in {elapsed:.2f} seconds.\n")
    except subprocess.CalledProcessError as e:
        print(f"\n>>> ERROR: {script_name} failed with exit code {e.returncode}.")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="TD2C Benchmark Pipeline Runner")
    parser.add_argument("--n_jobs", type=int, default=50, help="Number of parallel jobs/threads to use.")
    parser.add_argument("--skip_data", action="store_true", help="Skip data generation (Step 01).")
    parser.add_argument("--skip_descriptors", action="store_true", help="Skip descriptor computation (Step 02).")
    
    args = parser.parse_args()
    
    # 1. Data Generation
    if not args.skip_data:
        run_step("01.py", args.n_jobs, "Generating Synthetic Data")
    
    # 2. Descriptor Computation (Heavy)
    if not args.skip_descriptors:
        run_step("02.py", args.n_jobs, "Computing Descriptors")
    
    # 3. Threshold Finding
    run_step("03.py", args.n_jobs, "Finding Robust Threshold")
    
    # 4. Main Benchmark (Heavy)
    run_step("04.py", args.n_jobs, "Running Benchmarks (TD2C vs Competitors)")
    
    # 5. Analysis Scripts (Usually fast, but we pass n_jobs anyway)
    run_step("05.py", args.n_jobs, "Analyzing Test Dataset Metrics")
    run_step("06.py", args.n_jobs, "Analyzing Real Dataset Metrics")
    run_step("07.py", args.n_jobs, "Generating CD Diagrams")
    
    # 6. Feature Importance
    run_step("09.py", args.n_jobs, "Calculating Feature Importance")

    # Note: 08.py (Time Benchmarking) is usually run standalone as it has its own logic 
    # regarding job scaling, but you can uncomment below to include it.
    # run_step("08.py", args.n_jobs, "Time/Scaling Benchmark")

if __name__ == "__main__":
    main()