import subprocess

def midi_to_wav(midi_file, soundfont, output_wav):
    """Convert MIDI file to WAV using FluidSynth."""
    command = [
        r"D:\chori\Downloads\fluidsynth-2.4.4-win10-x64\bin\fluidsynth.exe",
        "-ni", soundfont,  # Load SoundFont
        midi_file,         # MIDI input
        "-F", output_wav,  # Output file
        "-r", "44100"      # Sample rate
    ]
    subprocess.run(command, check=True)

# Example usage
midi_to_wav("generated2.mid", "FluidR3_GM.sf2", "input.wav")
