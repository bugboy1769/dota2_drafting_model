import requests
import json
import os
import sys

def get_match_data(match_id):
    """
    Fetches match data from the OpenDota API.
    """
    url = f"https://api.opendota.com/api/matches/{match_id}"
    print(f"Fetching data from: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        return None

def save_to_json(data, filename):
    """
    Saves data to a JSON file in the same folder as the script.
    """
    script_folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_folder, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Match data saved to {file_path}")

if __name__ == "__main__":
    match_id = 8461476910
    
    match_data = get_match_data(match_id)
    if match_data:
        filename = f"dota_match_{match_id}.json"
        save_to_json(match_data, filename)