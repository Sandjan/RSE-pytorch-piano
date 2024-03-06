import random
import os
import shutil
import pickle
import subprocess
import time
import threading

donts = set()
dos = set()

loaded_fonts = {}
soundfonts_dir = None

current_processes = {}

def start_process(process_id,midi_files,soundfonts,output,sr):
    midi_file_paths = f"./{process_id}midi_files.pkl"
    font_info_out = f"./{process_id}font_info.pkl"
    loaded_fonts = f"./{process_id}loaded_fonts.pkl"
    processed = f"./{process_id}processed_files.txt"
    with open(midi_file_paths, 'wb') as f:
        pickle.dump(midi_files, f)

    command = f"python ./synthesize_files.py --midi_file_paths {midi_file_paths} --soundfonts_dir {soundfonts} --output_dir {output} --font_info_out_path {font_info_out} --loaded_fonts_out_path {loaded_fonts} --sampling_rate {sr} --processed_file_path {processed}"
    time.sleep(20)
    current_processes[process_id] = subprocess.Popen(command,shell=True)

def restart_process(process_id,soundfonts,output,sr):
    current_processes[process_id].terminate()
    print("terminated process")
    font_info = f"./{process_id}font_info.pkl"
    current_info = "./font_info.pickle"
    with open(font_info, 'rb') as f:
        dos1,donts1 = pickle.load(f)
    with open(current_info, 'rb') as f:
        dos2,donts2 = pickle.load(f)
    with open(current_info, 'wb') as f:
        pickle.dump((dos1.union(dos2),donts1.union(donts2)), f)
    midi_file_paths = f"./{process_id}midi_files.pkl"
    with open(midi_file_paths, 'rb') as f:
        allmidis = set(pickle.load(f))
    with open(f"./{process_id}processed_files.txt", 'r') as f:
        processed_midis = set(m.strip() for m in f.readlines())
    midis = list(allmidis.difference(processed_midis))
    random.shuffle(midis)
    start_process(process_id,midis,soundfonts,output,sr)
    print("process restarted")
    time.sleep(30)

def watch_process(process_id,soundfonts,output,sr):
    lastprocessed = None
    processed = f"./{process_id}processed_files.txt"
    while True:
        time.sleep(130)
        print("CHECKING PROCESSED CHANGED")
        with open(processed, 'r') as file:
            lines = file.readlines()
            if len(lines)>0:
                lastline = lines[-1]
            else:
                print("Process freezed, restarting")
                restart_process(process_id,soundfonts,output,sr)
            if lastline==lastprocessed:
                print("Process freezed, restarting")
                restart_process(process_id,soundfonts,output,sr)
            lastprocessed = lastline

# Irgendwo ein Fehler aus unerfindlichen gründen das gleiche MIDI mehrfach synthetisiert
def create_dataset(midi_source,midi_count,output_dir,soundfonts,proc=1,sr=16000):
    midi_files = os.listdir(midi_source)
    if midi_count>len(midi_files):
        raise ValueError('Your midi source dir must contain more or equal files than midi_count')
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # if os.path.exists(output_dir+'/audio'):
        #     shutil.rmtree(output_dir+'/audio')
        # if os.path.exists(output_dir+'/room'):
        #     shutil.rmtree(output_dir+'/room')
        # if os.path.exists(output_dir+'/midi'):
        #     shutil.rmtree(output_dir+'/midi')
        # os.makedirs(output_dir+'/audio')
        # os.makedirs(output_dir+'/room')
        # os.makedirs(output_dir+'/midi')
        random.shuffle(midi_files)

        midi_paths = [os.path.join(midi_source,path) for path in midi_files[:midi_count]]

        segment = len(midi_paths)//proc

        def task(i):
            idx = i*segment
            start_process(i,midi_paths[idx:idx+segment],soundfonts,output_dir,sr)
            print("Watching",i)
            watch_process(i,soundfonts,output_dir,sr)
        threads = []
        for i in range(proc):
            thread = threading.Thread(target=task, args=(i,))
            thread.start()
            threads.append(thread)
            time.sleep(10)

        for thread in threads:
            thread.join()
        print("COMPLETED")
            

create_dataset('../MIDI/g',1400,'../Synthetic_data','./soundfonts/',1)