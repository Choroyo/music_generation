import pretty_midi
import soundfile as sf
import fluidsynth

def midi_to_wav(midi_path, sf2_path, output_wav):
    # Load MIDI
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    # Initialize FluidSynth with the given SoundFont
    fs = fluidsynth.Synth()
    fs.start()
    sfid = fs.sfload(sf2_path)
    fs.program_select(0, 1, 0, 0)

    # Render the MIDI to audio
    audio = midi_data.fluidsynth(fs=fs)

    # Stop FluidSynth
    fs.delete()
    # Save as WAV
    sf.write(output_wav, audio, 44100)

# Example usage
midi_to_wav("input.mid", "FluidR3_GM.sf2", "output.wav")