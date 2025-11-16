import joblib
import os 

# script to compare if two pkls are identical

def compare_pkl(data1, data2):
    if type(data1) != type(data2):
        print(f"Different types in {file1} and {file2}")
        return False
    
    if type(data1) == list:
        if len(data1) != len(data2):
            print(f"Different list lengths in {file1} and {file2}")
            return False
        for i in range(len(data1)):
            return compare_pkl(data1[i], data2[i])  
        print(f"{file1} and {file2} are identical")
        return True

    if data1.keys() != data2.keys():
        print(f"Different keys in {file1} and {file2}")
        return False

    for key in data1.keys():
        if isinstance(data1[key], dict):
            if data1[key].keys() != data2[key].keys():
                print(f"Different sub-keys in key '{key}' of {file1} and {file2}")
                return False
            for sub_key in data1[key].keys():
                if not (data1[key][sub_key] == data2[key][sub_key]).all():
                    breakpoint()
                    print(f"Different values in sub-key '{sub_key}' of key '{key}'")
                    return False
                else:
                    print(f"{sub_key} are identical")
        else:
            if not (data1[key] == data2[key]).all():
                value_diff = data1[key] - data2[key]
                valud_diff_abs = abs(value_diff)
                relative_error = valud_diff_abs / (abs(data1[key]) + 1e-8)
                if relative_error.max() < 1e-3: # 1/1000 error tolerance
                    continue
                else:
                    print(f"Different values in key '{key}' beyond tolerance")
                    breakpoint()
                    return False
            else:
                continue

    print(f"{file1} and {file2} are identical")
    return True


source_path = 'cmu_stills'

files = ['CMU - Still - 01', 'CMU - Still - 02', 'CMU - Still - 03']
tag1 = '-orig1'
tag2 = '-orig'

for file in files:
    file1 = os.path.join(source_path, file + tag1 + '.pkl')
    file2 = os.path.join(source_path, file + tag2 + '.pkl')

    f1 = open(file1, "rb")
    f2 = open(file2, "rb")
    data1 = joblib.load(f1)
    data2 = joblib.load(f2)

    compare_pkl(data1, data2)