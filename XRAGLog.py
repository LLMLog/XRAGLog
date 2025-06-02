import json
import os
import random

from llm_common import get_embedding, gpt3_5_function_call
from scipy import stats

dataset = "Thunderbird"
dataset_dir = f"parsed_dataset/{dataset}_sample"

prompt_content = \
    "Your task is to determine if a given set of log messages contains an anomaly or not (Sorted by timestamp).\n\
    We will provide you with the following logs and ask you to determine if they contain any anomalies. \
    Additionally, we will provide the most semantically similar normal log for each log entry. Be careful to consider the contextual information of the log sequence.\n\
    Use the following format:\n \
    Logs: Given a set of log messages here. (a python list of log template id with parameter)\n \
    Similar Logs: Given a set of log messages here. (a python list of log template id with parameter)\n \
    Answer:\n yes or no (Output your thought process)\n \
    Log: %s\n, \
    Similar Logs: %s\n, \
    Answer: \n"

with open(f"parsed_dataset/{dataset}_sample/event_template.json", "r") as fr:
    event_template_map = json.load(fr)


def get_similar_log(normal_log_db, log):
    if log in normal_log_db.keys():
        return log
    else:
        curr_vec = get_embedding(log)
        max_sim = 0
        max_log = ""
        for log, item in normal_log_db.items():
            vec = item[0]
            if len(vec) == 0 or len(curr_vec) == 0:
                continue
            sim = stats.pearsonr(vec, curr_vec)[0]
            if max_sim < sim:
                max_sim = sim
                max_log = log
        return max_log


def compress_logs(logs):
    compressed_log = ""
    last_event = ''
    last_event_num = 0
    event_ids = set()
    for file_log in logs:
        tuples = file_log.split("-")
        event_id = tuples[0]
        event_ids.add(event_id)
        if last_event == event_id:
            last_event_num += 1
        else:
            if last_event == '':
                last_event = event_id
            else:
                compressed_log += last_event + "x" + str(last_event_num + 1) + ","
                last_event = event_id
    compressed_log += last_event + "x" + str(last_event_num + 1)
    return compressed_log, event_ids


if not os.path.exists("output/raw/XRAGLog"):
    os.mkdir("output/raw/XRAGLog")

# create normal log db
if not os.path.exists(f"output/raw/XRAGLog/{dataset}_db.json"):
    normal_log_db = {}
    files = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir)]
    for file in files:
        print(file)
        with open(file, "r") as fr:
            file_logs = fr.readlines()
        if 'abnormal' not in file and 'event_template' not in file:
            line, event_ids = compress_logs(file_logs)
            print(line)
            log_embedding = get_embedding(line)
            normal_log_db[line] = [log_embedding, list(event_ids)]
    num_to_remove = int(len(normal_log_db) * 0.2)
    keys_to_remove = random.sample(list(normal_log_db.keys()), num_to_remove)
    for key in keys_to_remove:
        del normal_log_db[key]
    with open(f"output/raw/XRAGLog/{dataset}_db.json", "w") as f:
        json.dump(normal_log_db, f)

files = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir)]
with open(f"output/raw/XRAGLog/{dataset}_db.json", "r") as f:
    normal_log_db = json.load(f)
    for file in files:
        if 'event_template' in file:
            continue
        print(file)
        if not os.path.exists(f"output/raw/XRAGLog/{dataset}_sample/{file.split('/')[-1]}"):
            fw = open(f"output/raw/XRAGLog/{dataset}_sample/{file.split('/')[-1]}", "w")
            fw.close()
        with open(f"output/raw/XRAGLog/{dataset}_sample/{file.split('/')[-1]}", "r") as fr1:
            try:
                json.load(fr1)
            except:
                print(file)
                if not os.path.exists(f"output/raw/XRAGLog/{dataset}_similar/{file.split('/')[-1]}"):
                    fw = open(f"output/raw/XRAGLog/{dataset}_similar/{file.split('/')[-1]}", "w")
                    fw.close()
                with open(f"output/raw/XRAGLog/{dataset}_similar/{file.split('/')[-1]}", "r") as fr2:
                    if len(fr2.readlines()) > 0:
                        with open(file, "r") as fr:
                            file_logs = fr.readlines()
                            line, event_ids = compress_logs(file_logs)
                            print(fr2.readlines())
                            similar_log = fr2.readlines()[0]
                            similar_event_ids = normal_log_db[similar_log][1]

                            used_event_template = {}
                            used_similar_event_template = {}
                            for event_id in event_ids:
                                used_event_template[event_id] = event_template_map[event_id]
                            for event_id in similar_event_ids:
                                used_similar_event_template[event_id] = event_template_map[event_id]
                            file_logs_with_template = str(line) + "\n" + str(used_event_template)
                            sim_logs_with_template = str(similar_log) + "\n" + str(used_similar_event_template)
                            answer = gpt3_5_function_call(
                                prompt_content % (file_logs_with_template, sim_logs_with_template))
                            with open(f"output/raw/XRAGLog/{dataset}_sample/{file.split('/')[-1]}", "w") as fw:
                                fw.write(answer)
                    else:
                        with open(file, "r") as fr:
                            file_logs = fr.readlines()
                            line, event_ids = compress_logs(file_logs)
                            similar_log = get_similar_log(normal_log_db, line)
                            print(similar_log)
                            with open(f"output/raw/XRAGLog/{dataset}_similar/{file.split('/')[-1]}", "w") as fw:
                                fw.writelines(similar_log)

                            used_event_template = {}
                            used_similar_event_template = {}
                            for event_id in event_ids:
                                used_event_template[event_id] = event_template_map[event_id]
                            similar_event_ids = normal_log_db[similar_log][1]
                            for event_id in similar_event_ids:
                                used_similar_event_template[event_id] = event_template_map[event_id]
                            file_logs_with_template = str(line) + "\n" + str(used_event_template)
                            sim_logs_with_template = str(similar_log) + "\n" + str(used_similar_event_template)

                            answer = gpt3_5_function_call(
                                prompt_content % (file_logs_with_template, sim_logs_with_template))
                            with open(f"output/raw/XRAGLog/{dataset}_prompt/{file.split('/')[-1]}", "w") as fw:
                                fw.write(prompt_content % (file_logs_with_template, sim_logs_with_template))
                            with open(f"output/raw/XRAGLog/{dataset}_sample/{file.split('/')[-1]}", "w") as fw:
                                fw.write(answer)
