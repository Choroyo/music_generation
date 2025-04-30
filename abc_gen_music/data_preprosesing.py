import subprocess
import os
import re

def clean_abc_text(raw_text):
    lines = raw_text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(X:|T:|K:|M:|L:|Q:|V:|%%|%)", line):
            continue  # Skip metadata and comment lines
        cleaned_lines.append(line)

    return " ".join(cleaned_lines)  # Flatten into a single string


def load_and_clean_abc_files(folder_path, output_file="./dataset/all_abc_data.txt"):
    all_text = ""

    for filename in os.listdir(folder_path):
        if filename.endswith(".abc"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
                cleaned = clean_abc_text(raw_text)
                all_text += cleaned.strip() + " "

    with open(output_file, "w", encoding="utf-8") as out:
        out.write(all_text)
        print(f"[OK] Saved cleaned ABC data to {output_file}")



def midi_to_abc(midi_path, abc_path):
    try:
        subprocess.run(["midi2abc", midi_path, "-o", abc_path], check=True)
        print(f"[OK] Saved ABC to {abc_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] MIDI to ABC conversion failed: {e}")

def convert_folder_midi_to_abc(midi_folder, abc_output_folder):
    os.makedirs(abc_output_folder, exist_ok=True)

    for folders in os.listdir(midi_folder):
        for files in os.listdir(os.path.join(midi_folder, folders)):
            filename = files
            if filename.lower().endswith(".mid") or filename.lower().endswith(".midi"):
                midi_path = os.path.join(midi_folder, folders, filename)
                abc_filename = os.path.splitext(filename)[0] + ".abc"
                abc_path = os.path.join(abc_output_folder, abc_filename)

                midi_to_abc(midi_path, abc_path)
        if filename.lower().endswith(".mid") or filename.lower().endswith(".midi"):
            midi_path = os.path.join(midi_folder, filename)
            abc_filename = os.path.splitext(filename)[0] + ".abc"
            abc_path = os.path.join(abc_output_folder, abc_filename)

            print(f"[INFO] Converting {midi_path} to {abc_path}")
            midi_to_abc(midi_path, abc_path)

#convert_folder_midi_to_abc("./dataset/transposed_midi", "./dataset/abc_dataset")
#load_and_clean_abc_files("./dataset/abc_dataset")
midi_to_abc("bethoven.mid", "./dataset/input2.abc")
