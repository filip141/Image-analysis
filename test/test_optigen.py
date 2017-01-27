import os
import json
import subprocess

content_folder = "test/content"
crossover_test_list = [0.25, 0.5, 0.7]
mutation_test_list = [0.008, 0.005, 0.01]
population_test_list = [150, 200, 350]


def start_test():
    # Emulate pytest parametrize but pythes wont work with subprocess easily
    for mutation in mutation_test_list:
        for crossover in crossover_test_list:
            for population in population_test_list:
                test_optigen_best(crossover, mutation, population)


def test_optigen_best(crossover, mutation, population):
    # Main folder path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.split(dir_path)[0]

    # Check DUT file
    dut_location = os.path.join(dir_path, "scripts", "optigen.py")
    assert os.path.isfile(dut_location), "File not found in directory.Path: {}".format(dut_location)

    # Run script
    content_path = os.path.join(dir_path, content_folder)
    files_in_dir = os.listdir(content_path)
    pid_dict = {}
    for test_file in files_in_dir:
        test_file_path = os.path.join(content_path, test_file)
        script_list = ['python', dut_location, "--input", test_file_path, "--show", "--save", "--population",
                       str(population), "--crossover", str(crossover), "--mutation", str(mutation)]
        command_text = " ".join(script_list)
        optigen_process = subprocess.Popen(command_text, shell=True, stdout=subprocess.PIPE)
        optigen_pid = optigen_process.pid
        pid_dict[test_file] = {"PID": optigen_pid, "crossover": crossover, "mutation": mutation,
                               "population": population}
        optigen_process.wait()
        img_stdout = optigen_process.stdout.read()
        max_chromo_first = float(img_stdout.split("\n")[2].split(":")[1])
        max_chromo_last = float(img_stdout.split("\n")[-5].split(":")[1])
        assert max_chromo_last >= max_chromo_first, "Last Chromosome should be better than first."
    print "Saving JSON Mapping to file"
    json_path = os.path.join(dir_path, "data", "figures", "mapping.json")
    with open(json_path, 'w') as fp:
        json.dump(pid_dict, fp)

if __name__ == '__main__':
    start_test()
