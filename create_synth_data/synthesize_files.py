import pretty_midi
from scipy import signal
from scipy.io.wavfile import write
from timeout_decorator import timeout
import numpy as np
import copy
import random
import os
import pickle
import fluidsynth
import threading
import time
import argparse
import pyroomacoustics as pra

fl = fluidsynth.Synth(samplerate=33000.0)

def getRandomSF2Path(soundfonts_dir):
    paths = os.listdir(soundfonts_dir)
    return random.choice(paths)

donts = set()
dos = set()

loaded_fonts = {}
soundfonts_dir = None

def synthAudio(inst,sf2_path):
        fs = 33000.0
        if not os.path.exists(sf2_path):
            raise ValueError("No soundfont file found at the supplied path "
                             "{}".format(sf2_path))

        # If the instrument has no notes, return an empty array
        if len(inst.notes) == 0:
            return np.array([])
        
        sfid = loaded_fonts[sf2_path]
        # If this is a drum instrument, use channel 9 and bank 128
        if inst.is_drum:
            channel = 9
            # Try to use the supplied program number
            res = fl.program_select(channel, sfid, 128, inst.program)
            # If the result is -1, there's no preset with this program number
            if res == -1:
                # So use preset 0
                fl.program_select(channel, sfid, 128, 0)
        # Otherwise just use channel 0
        else:
            channel = 0
            fl.program_select(channel, sfid, 0, inst.program)
        # Collect all notes in one list
        event_list = []
        for note in inst.notes:
            event_list += [[note.start, 'note on', note.pitch, note.velocity]]
            event_list += [[note.end, 'note off', note.pitch]]
        for bend in inst.pitch_bends:
            event_list += [[bend.time, 'pitch bend', bend.pitch]]
        for control_change in inst.control_changes:
            event_list += [[control_change.time, 'control change',
                            control_change.number, control_change.value]]
        # Sort the event list by time, and secondarily by whether the event
        # is a note off
        event_list.sort(key=lambda x: (x[0], x[1] != 'note off'))
        # Add some silence at the beginning according to the time of the first
        # event
        current_time = event_list[0][0]
        # Convert absolute seconds to relative samples
        next_event_times = [e[0] for e in event_list[1:]]
        for event, end in zip(event_list[:-1], next_event_times):
            event[0] = end - event[0]
        # Include 1 second of silence at the end
        event_list[-1][0] = 1.
        # Pre-allocate output array
        total_time = current_time + np.sum([e[0] for e in event_list])
        synthesized = np.zeros(int(np.ceil(fs*total_time)))
        # Iterate over all events
        for event in event_list:
            # Process events based on type
            if event[1] == 'note on':
                fl.noteon(channel, event[2], event[3])
            elif event[1] == 'note off':
                fl.noteoff(channel, event[2])
            elif event[1] == 'pitch bend':
                fl.pitch_bend(channel, event[2])
            elif event[1] == 'control change':
                fl.cc(channel, event[2], event[3])
            # Add in these samples
            current_sample = int(fs*current_time)
            end = int(fs*(current_time + event[0]))
            samples = fl.get_samples(end - current_sample)[::2]
            synthesized[current_sample:end] += samples
            # Increment the current sample
            current_time += event[0]

        return synthesized

@timeout(360)
def getAudio(instr,soundfonts_path):
    global fl
    audio = None
    combi = None
    while True:
        try:
            font = getRandomSF2Path(soundfonts_path)
            combi = (instr.program,font)
            if combi in donts:
                continue
            audio = synthAudio(instr,soundfonts_path+font)
            if audio.shape[0]>0:
                if (np.abs(audio)).var()<3.0:
                    print("ADDED DONTS")
                    donts.add(combi)
                    continue
            else:
                print("ADDED DONTS")
                donts.add(combi)
                continue
            break
        except:
            print("ADDED DONTS")
            donts.add(combi)
            continue
    if combi not in dos:
        dos.add(combi)
    return audio

def room_sound_simulation(data):
    width = random.uniform(4, 20)
    length = random.uniform(4, 14)
    height = random.uniform(2, 7)
    wall_materials = ['brickwork','rough_concrete','unpainted_concrete','rough_lime_wash','smooth_brickwork_flush_pointing','brick_wall_rough',
                    'smooth_brickwork_10mm_pointing','brick_wall_rough','ceramic_tiles','limestone_wall','reverb_chamber','wood_1.6cm','plywood_thin','wood_16mm',
                    'curtains_cotton_0.5','curtains_0.2','curtains_cotton_0.33','curtains_densely_woven','blinds_half_open','curtains_velvet',
                    'curtains_fabric','curtains_fabric_folded','studio_curtains','panel_fabric_covered_6pcf','panel_fabric_covered_8pcf',
                    'acoustical_plaster_25mm','rockwool_50mm_80kgm3','rockwool_50mm_40kgm3','mineral_wool_50mm_40kgm3','mineral_wool_50mm_70kgm3','gypsum_board',
                    'fibre_absorber_1']
    floor_materials = ['concrete_floor','marble_floor','audience_floor','stage_floor','linoleum_on_concrete','carpet_cotton','carpet_tufted_9.5mm','carpet_thin',
                    'felt_5mm','carpet_soft_10mm','carpet_hairy','carpet_rubber_5mm','cocos_fibre_roll_29mm','plywood_thin','wood_16mm',
                        'rough_concrete','ceramic_tiles','hard_surface','rough_concrete']
    ceiling_materials = ['ceiling_plasterboard','ceiling_fissured_tile','ceiling_perforated_gypsum_board','ceiling_melamine_foam','wood_1.6cm','plywood_thin','wood_16mm',
                        'rough_concrete','ceramic_tiles','hard_surface','rough_concrete']
    # Raum erstellen
    room = pra.ShoeBox([width, length, height],fs=33000,ray_tracing=True, air_absorption=True,
                    max_order=random.randint(1,20),
                    temperature=random.uniform(18, 33),
                    humidity=random.uniform(0, 80),
                    materials={
                            'east':pra.Material(random.choice(wall_materials)), 
                            'west':pra.Material(random.choice(wall_materials)), 
                            'north':pra.Material(random.choice(wall_materials)), 
                            'south':pra.Material(random.choice(wall_materials)), 
                            'ceiling':pra.Material(random.choice(ceiling_materials)), 
                            'floor':pra.Material(random.choice(floor_materials))
                    },
                    use_rand_ism = True, max_rand_disp = 0.05
                    )


    dist_from_wall = random.uniform(0.1, length-0.1)
    for i in range(data.shape[0]):
        source_location = [random.uniform(0.1, width-0.1),dist_from_wall, random.uniform(0.7, 1.5)]
        room.add_source(source_location,data[i])

    # Mikrofon hinzufügen
    middle = width/2
    mic_spc = random.uniform(0.1, 1)/2
    mic_dist_wall = random.uniform(0.1, 1)
    diff = random.uniform(-0.5,0.5)
    R = np.c_[
        [middle-mic_spc, mic_dist_wall+diff, 1.2],
        [middle+mic_spc, mic_dist_wall, 1.2]
        ]
    room.add_microphone_array(R)
    room.compute_rir()
    room.simulate()
    mic_1_audio = room.mic_array.signals[0, :]
    mic_2_audio = room.mic_array.signals[1, :]
    res = np.vstack((mic_1_audio,mic_2_audio)).astype(np.float32)
    res = res.sum(axis=0)
    res = res/np.abs(res).max()
    return res.astype(np.float32)

def synthesize_midi_file(filepath,soundfonts_path,with_drums=False):
    midi_data = pretty_midi.PrettyMIDI(filepath)
    audios = []
    no_drums = copy.deepcopy(midi_data)
    no_drums.instruments = []
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        if len(instrument.notes)>0:
            try:
                audio = getAudio(instrument,soundfonts_path)
            except:
                continue
            no_drums.instruments.append(instrument)
            mv = np.array([note.velocity for note in instrument.notes]).max()/127.0
            audio /= np.abs(audio).max()
            s = (mv/np.abs(audio).max() + 1)/2.0
            audios.append(audio*s)
        
    if with_drums:
        midi_copy = copy.deepcopy(midi_data)
        midi_copy.instruments = []
        maxv = 0
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                continue
            if len(instrument.notes)>0:
                midi_copy.instruments.append(instrument)
                mv = np.array([note.velocity for note in instrument.notes]).max()/127.0
                if mv > maxv:
                    maxv = mv
        audio = midi_copy.fluidsynth(fs=33000.0, sf2_path=getRandomSF2Path(soundfonts_path))
        s = (maxv/np.abs(audio).max()+1)/2.0
        audios.append(audio)

    data = np.zeros((len(audios),np.max([w.shape[0] for w in audios])))
    for i in range(len(audios)):
        data[i,:audios[i].shape[0]] += audios[i]
    room_sound = room_sound_simulation(data)
    data = data.sum(axis=0)
    data /= np.abs(data).max()
    data = data.astype(np.float32)
    return data,room_sound[:data.shape[0]], midi_data if with_drums else no_drums

stopped = False 

def saveFontInfo():
    while True:
        if stopped:
            break
        time.sleep(30)
        print("SAVING FONT INFO")
        with open(font_info_out, 'wb') as f:
            pickle.dump((dos,donts), f)

parser = argparse.ArgumentParser(description="midi_file_synthesizer")
parser.add_argument("--midi_file_paths", type=str, help="")
parser.add_argument("--soundfonts_dir", type=str, help="")
parser.add_argument("--output_dir", type=str, help="")
parser.add_argument("--font_info_out_path", type=str, help="")
parser.add_argument("--loaded_fonts_out_path", type=str, help="")
parser.add_argument("--processed_file_path", type=str, help="")
parser.add_argument("--sampling_rate", type=int,default=11000, help="")

args = parser.parse_args()
midi_files = None
sr = args.sampling_rate
soundfonts_dir = args.soundfonts_dir
font_info = './font_info.pickle'
font_info_out = args.font_info_out_path
output_dir = args.output_dir
processed_file_path = args.processed_file_path
with open(processed_file_path, 'w') as datei:
    datei.write('')

if os.path.exists(font_info):
    with open(font_info, 'rb') as f:
        dos,donts = pickle.load(f)

thread = threading.Thread(target=saveFontInfo)
thread.start()

loaded_fonts = {}
paths = os.listdir(soundfonts_dir)
for path in paths:
    sf2_path = soundfonts_dir + path
    loaded_fonts[sf2_path] = fl.sfload(sf2_path)

with open(args.loaded_fonts_out_path, 'wb') as f:
    pickle.dump(loaded_fonts, f)

with open(args.midi_file_paths, 'rb') as f:
    midi_files = pickle.load(f)
for midi_file in midi_files:
    try:
        audio,room_sound,midi = synthesize_midi_file(midi_file,soundfonts_dir,False)
        audio = signal.resample(audio, int(audio.shape[0] * sr / 33000))
        room_sound = signal.resample(room_sound, int(room_sound.shape[0] * sr / 33000))
        output_id = (abs(hash(midi))+abs(hash(audio.shape)))
        midi.write(os.path.join(output_dir,f'midi/{output_id}.mid'))
        write(os.path.join(output_dir,f'audio/{output_id}.wav'), sr, audio)
        write(os.path.join(output_dir,f'room/{output_id}.wav'), sr, room_sound)
        with open(processed_file_path, 'a') as datei:
            datei.write(midi_file + '\n')
    except Exception as e:
        print("ERROR, continuing",e)
stopped = True
thread.join()
with open(font_info_out, 'wb') as f:
    pickle.dump((dos,donts), f)
fl.delete()