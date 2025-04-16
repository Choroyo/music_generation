import pretty_midi
import pandas as pd

def csv_to_midi(csv_path, midi_output_path):
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()

    # Create an instrument (default to Acoustic Grand Piano if program 0)
    instrument = pretty_midi.Instrument(program=0)

    # Iterate through the rows and add notes
    for _, row in df.iterrows():
        note = pretty_midi.Note(
            velocity=int(row['velocity']),
            pitch=int(row['note_value']),
            start=row['start_time'],
            end=row['end_time']
        )
        instrument.notes.append(note)

    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument)

    # Write to a MIDI file
    midi.write(midi_output_path)

# Example usage
csv_to_midi("debussy_cc_1.csv", "output.mid")
