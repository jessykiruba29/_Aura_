import cv2
import speech_recognition as sr
import pyttsx3
from ultralytics import YOLO
import time
import os
import pytesseract
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

# ============================================================
# 0. CONFIGURATION
# ============================================================
# For Windows, uncomment and set your Tesseract path:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ============================================================
# 1. LOAD MODELS
# ============================================================
print("📦 Loading models...")

print("   Loading BLIP...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

print("   Loading YOLO...")
yolo_model = YOLO("yolov8n.pt")

print("   Loading Summarizer...")
summarizer_model_name = "sshleifer/distilbart-cnn-12-6"
try:
    sum_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name)
    summarizer_loaded = True
except Exception as e:
    print(f"⚠️ Could not load summarizer: {e}")
    summarizer_loaded = False

print("✅ Models loaded!")

# ============================================================
# 2. TEXT-TO-SPEECH (Re-init engine every call to avoid Windows bug)
# ============================================================
def speak(text):
    """Speak text aloud and WAIT until done. Re-creates engine each time."""
    print(f"🤖 Aura: {text}")
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 165)
        tts_engine.setProperty('volume', 0.95)
        tts_engine.say(text)
        tts_engine.runAndWait()
        tts_engine.stop()
        del tts_engine
        time.sleep(0.2)
    except Exception as e:
        print(f"⚠️ Speech error: {e}")

# ============================================================
# 3. SPEECH-TO-TEXT (Simple blocking listen, like main2.py)
# ============================================================
def listen_for_command():
    """Listen for a voice command. Blocks until speech is heard."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print(f"📝 Heard: {command}")
        return command.lower()
    except sr.UnknownValueError:
        print("❓ Could not understand audio")
        return ""
    except sr.RequestError:
        print("❌ Speech recognition service error")
        return ""

# ============================================================
# 4. BLIP SCENE DESCRIPTION
# ============================================================
def describe_scene(frame):
    """Use BLIP to describe what's in the frame."""
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        inputs = blip_processor(pil_image, return_tensors="pt")
        out = blip_model.generate(**inputs, max_length=50)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"❌ BLIP error: {e}")
        return None

# ============================================================
# 5. OCR & SUMMARIZATION
# ============================================================
def extract_text(frame):
    """Extract text from the current frame using Tesseract."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(gray)
        text = pytesseract.image_to_string(pil_image)
        return text.strip()
    except Exception as e:
        print(f"❌ OCR error: {e}")
        return ""

def get_summary(text):
    """Summarize text using BART model."""
    try:
        if not summarizer_loaded:
            return "Summarization is not available."
        inputs = sum_tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = sum_model.generate(inputs["input_ids"], num_beams=4, max_length=100, early_stopping=True)
        return sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"❌ Summarization error: {e}")
        return "I had trouble summarizing."

# ============================================================
# 6. YOLO OBJECT DETECTION
# ============================================================
def find_object_with_yolo(frame, target):
    """Use YOLO to find a specific object in the frame."""
    try:
        results = yolo_model(frame, verbose=False)
        best_match = None
        best_confidence = 0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                label = yolo_model.names[class_id]
                confidence = float(box.conf[0])

                # Draw all detections
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                if (target.lower() in label.lower() or label.lower() in target.lower()):
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = {'label': label, 'bbox': (x1, y1, x2, y2), 'confidence': confidence}

        if best_match and best_confidence > 0.4:
            x1, y1, x2, y2 = best_match['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"TARGET: {best_match['label']}", (x1, y1-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return True, best_match

        return False, None
    except Exception as e:
        print(f"❌ YOLO error: {e}")
        return False, None

# ============================================================
# 7. MODE HANDLERS
# ============================================================
def handle_detect(cap):
    """Object detection mode with voice guidance."""
    speak("What should I find?")
    command = listen_for_command()

    if not command:
        speak("I didn't catch that.")
        return

    # Extract the object name
    target = command.split()[-1]  # Take last word as target
    speak(f"Searching for {target}. I will guide you.")
    print(f"🔍 SEARCHING FOR: {target}")

    search_start = time.time()
    search_duration = 60  # 60 seconds to search
    last_instruction = 0
    cooldown = 2.5

    while time.time() - search_start < search_duration:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))

        # Display search info
        remaining = int(search_duration - (time.time() - search_start))
        cv2.putText(frame, f"SEARCHING: {target} | {remaining}s left", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        found, obj_data = find_object_with_yolo(frame, target)

        if found and obj_data and time.time() - last_instruction > cooldown:
            x1, y1, x2, y2 = obj_data['bbox']
            frame_center = frame.shape[1] / 2
            obj_center = (x1 + x2) / 2

            if obj_center < frame_center - 80:
                speak(f"The {target} is on your left. Move left.")
            elif obj_center > frame_center + 80:
                speak(f"The {target} is on your right. Move right.")
            else:
                speak(f"The {target} is straight ahead! Reach out now.")
                speak("Great, you found it!")
                cv2.imshow("Aura Vision", frame)
                cv2.waitKey(1)
                return

            last_instruction = time.time()

        elif not found and time.time() - last_instruction > cooldown * 2:
            speak("I can't see it yet. Keep moving slowly.")
            last_instruction = time.time()

        cv2.imshow("Aura Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            speak("Search cancelled.")
            return

    speak(f"Sorry, I couldn't find the {target}. Time is up.")

def handle_describe(frame):
    """Describe the current scene using BLIP."""
    speak("Let me look around...")
    caption = describe_scene(frame)
    if caption:
        speak(f"I can see {caption}")
    else:
        speak("Sorry, I couldn't describe what I see.")

def handle_read(frame):
    """Read text from the current frame using OCR."""
    speak("Let me check for text...")
    text = extract_text(frame)

    if text:
        print(f"📄 Extracted: {text[:100]}...")
        speak("I found some text.")
        speak("Should I read it or summarize it? Say read or summarize.")

        command = listen_for_command()

        if "read" in command:
            speak("Reading aloud now.")
            speak(text)
        elif "summarize" in command or "summarise" in command:
            speak("Let me summarize that for you.")
            summary = get_summary(text)
            speak(f"Here is the summary: {summary}")
        else:
            speak("I'll just read it for you.")
            speak(text)
    else:
        speak("I couldn't find any text in the image.")

# ============================================================
# 8. MAIN LOOP
# ============================================================
def main():
    # Camera setup
    cap = cv2.VideoCapture("http://192.168.68.110:4747/video")

    if not cap.isOpened():
        print("⚠️ DroidCam not found, trying webcam...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ No camera found!")
        speak("No camera detected. Please check your connection.")
        return

    speak("Aura is online. Say Aura whenever you need me.")

    while True:
        # Show camera feed
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, (640, 480))
        cv2.putText(frame, "STANDBY - Say 'Aura' to activate", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Aura Vision", frame)

        # Non-blocking key check (keeps camera alive)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            speak("Goodbye!")
            break

        # Listen for wake word (blocking - this IS the main loop)
        print("👂 Listening for 'Aura'...")
        command = listen_for_command()

        if not command:
            continue

        # Check for wake word
        wake_words = ["aura", "hey aura", "ara", "ora", "aurora",
                      "hello aura", "aaura", "auro"]

        if any(w in command for w in wake_words):
            # ---- OCULUS ACTIVATED ----
            speak("Yes Sir! How can I help?")
            speak("You can say. Detect. Describe. Read. or Goodbye.")

            choice = listen_for_command()

            if not choice:
                speak("I didn't hear anything. Say Aura again when ready.")
                continue

            if "detect" in choice or "find" in choice or "search" in choice:
                handle_detect(cap)

            elif "describe" in choice or "scene" in choice or "see" in choice or "look" in choice:
                # Grab a fresh frame for description
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (640, 480))
                    handle_describe(frame)
                else:
                    speak("I couldn't capture an image.")

            elif "read" in choice or "text" in choice:
                # Grab a fresh frame for OCR
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (640, 480))
                    handle_read(frame)
                else:
                    speak("I couldn't capture an image.")

            elif "goodbye" in choice or "bye" in choice or "exit" in choice or "quit" in choice:
                speak("Goodbye Sir! Turning off now.")
                break

            else:
                speak("Sorry, I didn't understand. Say Aura again when you need me.")

        elif "goodbye" in command or "bye" in command:
            speak("Goodbye Sir!")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Goodbye!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")