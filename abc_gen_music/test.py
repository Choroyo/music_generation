import os
import subprocess
from music21 import converter, environment, stream

# Optional: disable music21 warnings
environment.UserSettings()['warnings'] = 0

def abc_to_midi_file(input_path="output.abc", output_path="generated.mid"):
    try:
        score = converter.parse(input_path, format="abc")

        # Remove duplicate meta objects
        clean_score = stream.Score()
        for part in score.parts:
            new_part = stream.Part()
            for el in part.flat.notesAndRests:
                new_part.append(el)
            clean_score.append(new_part)

        clean_score.write('midi', fp=output_path)
        print(f"[OK] MIDI file saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"[ERROR] ABC to MIDI failed: {e}")
        return None
# === Run Both Steps ===
abc_to_midi_file("output2.abc", "generated2.mid")