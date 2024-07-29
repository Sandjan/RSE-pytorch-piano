# RSE-pytorch-piano
An implementation of the RSE in pytorch with customizations, trained on MusicNet, MAESTRO and synthetic data from the ADLPiano dataset. The post-processing has been significantly improved and the model seems to generalize better for unknown piano pieces

## Demo:
input for transcription:
https://www.youtube.com/watch?v=Gus4dnQuiGk  
transcribed midi result: [Result](https://soundcloud.com/j-s-221934774/chopin-fantaisie-impromptu-op-66-transcribed-midi?si=36f751363b2840fc8a2d732b610c27e6&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)

## Transcribe a wav file to midi:
install the necessary packages with conda:
```
conda env create -f transcript_env.yml
```

activate the environment:
```
conda activate rse_transcript
```
**insert your absolute path to the repository into transcribe/tools.py**  
and start the transcription process:
```
python transcribe/transcribe.py --wav_path "your_file.wav" --out_name midi_name
```
The process takes about as long as the wav file is long on a rtx3070 laptop version
