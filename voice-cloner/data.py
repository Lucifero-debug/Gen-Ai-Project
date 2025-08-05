import os
import sounddevice as sd
import keyboard
import time
import soundfile as sf

# Set up paths
DATASET_DIR = "my_datasets"
AUDIO_DIR = os.path.join(DATASET_DIR, "wavs")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")

os.makedirs(AUDIO_DIR, exist_ok=True)

from scipy.signal import find_peaks


# Sentences to record (100 sentences ≈ 30–40 mins of speech)
sentences = [
    "Hello! My name is Alex, and I love artificial intelligence.",
    "The sun sets behind the mountains in the evening sky.",
    "Do you know where the nearest library is?",
    "Please call me back when you get this message.",
    "Tomorrow, I'll be traveling to New York for a meeting.",
    "I prefer tea over coffee, especially in the morning.",
    "Technology has changed the way we live and work.",
    "She bought a dozen apples, bananas, and oranges.",
    "This is a test sentence for synthetic voice training.",
    "Can you hear the clarity in my pronunciation?",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming our world.",
    "He whispered a secret to his best friend.",
    "I will meet you at the park around 3 PM.",
    "It’s important to drink enough water every day.",
    "Learning something new each day is a good habit.",
    "He turned off the lights before leaving the room.",
    "There was a strange noise coming from the attic.",
    "Don’t forget to carry your ID and passport.",
    "Would you like tea, coffee, or juice?",
    "We watched a beautiful sunset at the beach.",
    "The professor gave a lecture on quantum physics.",
    "My favorite hobby is painting with watercolors.",
    "Turn right at the next intersection and go straight.",
    "It’s been raining continuously for three days.",
    "The stars were shining brightly in the night sky.",
    "She practices yoga every morning before work.",
    "Can you repeat that sentence one more time?",
    "The restaurant serves delicious Italian food.",
    "He read the book from cover to cover in one day.",
    "This chair is more comfortable than it looks.",
    "Please submit your assignment before Friday.",
    "Her voice was calm, soothing, and confident.",
    "We decided to go hiking in the mountains.",
    "That movie was both thrilling and emotional.",
    "You can leave your bag on the table.",
    "Everything happens for a reason.",
    "She sings beautifully and plays the guitar.",
    "The bakery opens at six in the morning.",
    "They moved into a new apartment last week.",
    "His jokes always make everyone laugh.",
    "I enjoy walking in the park after dinner.",
    "Science and technology evolve rapidly.",
    "He ordered a burger, fries, and a milkshake.",
    "Could you help me find my missing keys?",
    "The airplane landed safely at the airport.",
    "Our team won the championship last year.",
    "Let’s go out and enjoy the fresh air.",
    "I have a meeting scheduled for tomorrow.",
    "The cat slept peacefully on the windowsill.",
    "She wore a red dress to the party.",
    "Time flies when you're having fun, doesn’t it?",
    "There’s a package waiting at your doorstep.",
    "He fixed the broken chair with some glue.",
    "Please bring an umbrella, it might rain.",
    "I’m learning to play the violin this year.",
    "The library is quiet and perfect for studying.",
    "He baked a chocolate cake for her birthday.",
    "Close the window before it gets too cold.",
    "She likes to dance in the rain.",
    "The exam starts exactly at ten o’clock.",
    "This phone has an amazing camera.",
    "We watched the fireworks on New Year's Eve.",
    "That was the best performance of the night.",
    "What’s your favorite movie of all time?",
    "The engine roared as the car accelerated.",
    "My grandparents live in a small village.",
    "He smiled warmly and waved goodbye.",
    "The storm knocked out the power lines.",
    "The baby giggled at the funny sound.",
    "I can’t believe it’s already December.",
    "Always read the instructions carefully.",
    "This soup needs a little more salt.",
    "Let me know if you need any help.",
    "She kept all her letters in a box.",
    "I visited the museum last weekend.",
    "Please be quiet during the presentation.",
    "The road ahead is closed for construction.",
    "He ran as fast as he could.",
    "The moonlight shimmered on the lake.",
    "They adopted a puppy from the shelter.",
    "Let’s bake cookies together this afternoon.",
    "You forgot to turn off the stove.",
    "The flowers bloomed beautifully this spring.",
    "It’s dangerous to text while driving.",
    "The baby bird chirped loudly for food.",
    "Your presentation was really impressive.",
    "Do you remember the first time we met?",
    "She wore a bright smile on her face.",
    "They danced under the starry sky.",
    "This is exactly what I was looking for.",
    "He carefully painted the wall blue.",
    "We went grocery shopping this morning.",
    "I’m excited about our upcoming vacation.",
    "He folded the letter and put it away.",
    "Let’s try that new restaurant downtown.",
    "Please turn down the volume.",
    "The water was cold but refreshing.",
    "She dreamed of flying above the clouds.",
    "Everyone clapped after the performance.",
    "I love the smell of fresh coffee.",
    "They sat around the campfire, telling stories.",
]

# Audio settings
DURATION = 5  # seconds per sentence (can increase for longer sentences)
SAMPLE_RATE = 22050

import soundfile as sf
import numpy as np
import librosa
import noisereduce as nr

def clean_audio(input_path, output_path):
    reduction_strength = 0.8
    gain_db = 6
    audio, sr = librosa.load(input_path, sr=SAMPLE_RATE)

    # Use first 0.5s as noise clip
    noise_clip = audio[:int(0.5 * sr)]

    # Reduce fan/background noise
    cleaned = nr.reduce_noise(y=audio, y_noise=noise_clip, sr=sr, stationary=True,prop_decrease=reduction_strength)

    # Normalize
    max_val = np.max(np.abs(cleaned))
    if max_val > 0:
        cleaned = cleaned / max_val * 0.99
    
    cleaned *= 10 ** (gain_db / 20)  # +6dB gain

    # Soft limit (prevent clipping)
    cleaned = np.clip(cleaned, -0.99, 0.99)

    sf.write(output_path,cleaned, sr)


def record_audio(sr):
    import queue
    audio_queue = queue.Queue()
    recorded_frames = []

    def callback(indata, frames, time, status):
        audio_queue.put(indata.copy())

    print("⏺ Press 'r' to start recording and 's' to stop.")
    
    while not keyboard.is_pressed('r'):
        time.sleep(0.1)

    print("🎙 Recording...")
    with sd.InputStream(samplerate=sr, channels=1, callback=callback):
        while not keyboard.is_pressed('s'):
            while not audio_queue.empty():
                recorded_frames.append(audio_queue.get().flatten())
            time.sleep(0.01)

    print("🛑 Stopped.")
    audio = np.concatenate(recorded_frames)
    return audio


print("🎤 Starting voice dataset recording...")

with open(METADATA_PATH, "w") as meta_file:
    for i, sentence in enumerate(sentences, 1):
        print(f"\n👉 Sentence #{i}:\n\"{sentence}\"")
        input("📌 Press Enter when ready. Then press 'r' to start and 's' to stop...")

        raw_audio = record_audio(SAMPLE_RATE)
        raw_audio = raw_audio * 7  # Amplify by 1.5x

        # File paths
        raw_path = os.path.join(AUDIO_DIR, f"temp_sample{i:03d}.wav")
        cleaned_path = os.path.join(AUDIO_DIR, f"sample{i:03d}.wav")

        # Save raw → clean → delete raw
        sf.write(raw_path, raw_audio, SAMPLE_RATE)
        clean_audio(raw_path, cleaned_path)
        os.remove(raw_path)

        print(f"✅ Cleaned file saved: {cleaned_path}")
        meta_file.write(f"sample{i:03d}.wav|{sentence}\n")

print(f"\n✅ All recordings cleaned and saved to: {DATASET_DIR}")