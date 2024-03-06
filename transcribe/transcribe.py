import argparse
from tools import predictAudio,thresholding
import pretty_midi

parser = argparse.ArgumentParser(description="Piano to MIDI")
parser.add_argument("--wav_path", type=str, help="Path to input wav file")
parser.add_argument("--out_name", type=str, help="Name for output midi file")
parser.add_argument("--stride", type=int,default=165, help="a smaller stride slows down the transcription, but can lead to better results")

args = parser.parse_args()
prediction = predictAudio(args.wav_path,16384,args.stride)
pred_bin, notes = thresholding(prediction,0.5, 0.2, 0.005, 0.45, int((165/args.stride)*3),stride=args.stride)

midi_data = pretty_midi.PrettyMIDI()
inst = pretty_midi.Instrument(program=0)
for n in notes:
    inst.notes.append(pretty_midi.Note(velocity=n[0], pitch=n[1], start=n[2], end=n[3]))
midi_data.instruments.append(inst)
midi_data.write(args.out_name+'.mid')