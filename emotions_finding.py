import json

# Load the JSON file
with open("aligned_data_with_traits.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Ask for an emotion to search
search_emotion = input("Enter the emotion to search for: ").strip()

# Search and print results
for idx, entry in enumerate(data, start=1):
    if entry.get("emotion") == search_emotion:
        print(f"\n--- Match {idx} ---")
        print(f"Transcript: {entry['transcript']}")
        print(f"Response Time: {entry['response_time']}")
        print(f"Body Language: {entry['body_language']}")
        print(f"Speech Attributes: {entry['speech_attributes']}")
        print(f"Traits: {entry['traits']}")
    else:
        pass  # ignore if not matching

print("\nDone searching.")
