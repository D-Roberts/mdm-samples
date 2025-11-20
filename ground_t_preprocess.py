def get_groundt_string(i, file_path_ground):
    print(f"for case {i} *** ")
    try:
        with open(file_path_ground, "r") as file:
            file_content_ground = file.read()

    except FileNotFoundError:
        print(f"Error: The file '{file_path_ground}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return file_content_ground


for i in range(1, 21):
    file_path_ground = (
        f"/Users/dr/research/mdm-samples/results/humaneval_results/ground_truth/{i}.py"
    )
    # file_path_ground = (
    #     f"/home/ubuntu/mdm-samples/results/humaneval_results/ground_truth/{i}.py"
    # )
    file_content_ground = get_groundt_string(i, file_path_ground)
    # print("File content as a string:")
    print(file_content_ground)
