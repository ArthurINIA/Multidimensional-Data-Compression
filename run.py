import subprocess
import os

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Run dict_learning_method.py
print("Running dictionary learning method...")
result = subprocess.run(['python3', 'scikit_dict_learning_method.py'], capture_output=True, text=True)

# Check if dict_learning_method.py ran successfully
if result.returncode != 0:
    print("Error running scikit_dict_learning_method.py")
    print(result.stderr)
else:
    print("Dictionary learning completed successfully.")
    print(result.stdout)
    
    # Run PSNR_check.py
    print("Running statistics check...")
    result = subprocess.run(['python3', 'check_stats.py'], capture_output=True, text=True)
    
    # Check if PSNR_check.py ran successfully
    if result.returncode != 0:
        print("Error running check_stats.py")
        print(result.stderr)
    else:
        print("Statistics check completed successfully.")
        print(result.stdout)