"""read in ground"""

file_path = "/Users/dr/research/mdm-samples/results/humaneval_results/ground_truth/1.py"

try:
    with open(file_path, "r") as file:
        file_content = file.read()
    print("File content as a string:")
    print(file_content)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
