import csv
import ollama
import sys

# --- Configuration ---
# IMPORTANT: Make sure this path points to your actual CSV file.
CSV_FILE = r'C:\Desktop\CSV-LLM\data.csv' 
MODEL_NAME = 'llama3.2:1b'
OUTPUT_FILE = 'header_descriptions.txt' # The file where results will be saved

def get_csv_headers(filepath):
    """Reads the first line of a CSV file to get the headers."""
    try:
        with open(filepath, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            # Handle potential empty file
            headers = next(reader, None)
            return headers
    except FileNotFoundError:
        print(f"❌ Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred while reading the file: {e}")
        return None

def generate_description(header, model):
    """Generates a short description for a CSV header using an offline LLM."""
    prompt = (
        f"Provide a very short, one-line description for the CSV header '{header}'. "
        "Describe its likely meaning in a business or data context. "
        "Do not add any introductory text like 'This header likely represents...'."
    )
    
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.2} # Lower temperature for more deterministic output
        )
        description = response['message']['content'].strip()
        return description
    except Exception as e:
        return f"Error generating description: {e}"

def main():
    """Main function to run the script."""
    # Check if the Ollama service is running and the model is available
    try:
        ollama.show(MODEL_NAME)
    except Exception:
        print(f"❌ Error: Could not connect to Ollama or find model '{MODEL_NAME}'.")
        print("   Please ensure Ollama is running and you have pulled the model:")
        print(f"   'ollama pull {MODEL_NAME}'")
        sys.exit(1)

    # Read the headers from the specified CSV file
    headers = get_csv_headers(CSV_FILE)
    
    if headers:
        print(f"\n⚙️  Generating descriptions for headers from '{CSV_FILE}'...")
        
        try:
            # Open the output file to write the results
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write("--- Header Descriptions ---\n")
                
                # Process each header
                for header in headers:
                    # Show progress in the console
                    print(f"'{header}' → ", end="", flush=True)
                    description = generate_description(header, MODEL_NAME)
                    print(description) # Print to console
                    
                    # Write the result to the output file
                    f.write(f"'{header}' → {description}\n")
                
                f.write("---------------------------\n")

            print(f"\n✅ Success! Output saved to '{OUTPUT_FILE}'")

        except IOError as e:
            print(f"\n❌ Error: Could not write to the output file '{OUTPUT_FILE}'. Reason: {e}")

if __name__ == "__main__":
    main()
