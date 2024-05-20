import json

def main():
    dataset = []
    with open('./mbjp.jsonl', 'r') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            dataset.append(js)

    filtered_dataset = []
    for js in dataset:
        if js['canonical_solution'] is not None:
            head = remove_part(js['prompt'])
            js['canonical_solution'] = head + js['canonical_solution']
            filtered_dataset.append(js)

    with open('./mbjp-filtered.json', 'w') as f:
        json.dump(filtered_dataset, f)


def remove_part(input_string, start_char='/**\n', end_char='*/\n'):
    # first remove import
    input_string = input_string.split('\n\n\n')[1]

    start_index = input_string.find(start_char)
    end_index = input_string.find(end_char)

    if start_index != -1 and end_index != -1 and end_index > start_index:
        removed_part = input_string[start_index:end_index + len(end_char)]
        modified_string = input_string.replace(removed_part, "")
        return modified_string
    else:
        return input_string


if __name__ == '__main__':
    main()