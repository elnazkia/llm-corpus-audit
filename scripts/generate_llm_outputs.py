"""
This script generates a small set of LLM outputs for initial testing and demo purposes.
It uses the Hugging Face transformers library with a pre-trained model.
"""

import sys
import os
import json
from datetime import datetime
from transformers import pipeline, set_seed

# Adjust the Python path to include the 'src' directory if needed for future utils
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configuration
MODEL_NAME = "gpt2"  # Using a smaller model for quick testing
NUM_COMPLETIONS_PER_PROMPT = 2
MAX_LENGTH_COMPLETION = 75  # Max length for each generated completion
OUTPUT_DIR = "data/raw/llm_outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "initial_llm_outputs.json")

# Set a seed for reproducibility
set_seed(42)

def generate_text(text_generator, prompt_text):
    """Generates text using the provided pipeline and prompt."""
    try:
        generated_sequences = text_generator(
            prompt_text,
            max_length=MAX_LENGTH_COMPLETION,
            num_return_sequences=NUM_COMPLETIONS_PER_PROMPT,
            pad_token_id=text_generator.model.config.eos_token_id # Suppress padding warning
        )
        return [seq['generated_text'] for seq in generated_sequences]
    except Exception as e:
        print(f"Error during text generation for prompt '{prompt_text[:30]}...': {e}")
        return []

def main():
    """Main function to generate and save LLM outputs."""
    print(f"Starting LLM output generation with model: {MODEL_NAME}")

    # Ensure output directory exists (it should have been created by a previous step)
    if not os.path.exists(OUTPUT_DIR):
        print(f"Output directory {OUTPUT_DIR} not found. Please create it first.")
        # os.makedirs(OUTPUT_DIR) # Or create it here if preferred
        return

    # Initialize the text generation pipeline
    try:
        text_generator = pipeline("text-generation", model=MODEL_NAME)
    except Exception as e:
        print(f"Error initializing text generation pipeline for model '{MODEL_NAME}': {e}")
        print("Please ensure the model is available and you have an internet connection if it needs to be downloaded.")
        return

    # Define simple prompt templates
    prompt_templates = [
        {
            "id": "news_lead_1",
            "text": "Scientists today announced a breakthrough discovery regarding",
            "register": "news_lead",
            "demographic_var": "general"
        },
        {
            "id": "email_opener_1",
            "text": "Dear valued customer, we are writing to inform you about",
            "register": "email",
            "demographic_var": "general"
        },
        {
            "id": "short_story_1",
            "text": "In a land far away, there lived a curious creature who one day found a mysterious object. It was",
            "register": "short_story",
            "demographic_var": "neutral"
        }
    ]

    all_generated_outputs = []
    timestamp = datetime.now().isoformat()

    print(f"\nGenerating {NUM_COMPLETIONS_PER_PROMPT} completions for each of the {len(prompt_templates)} prompts...")

    for i, template in enumerate(prompt_templates):
        prompt_text = template["text"]
        print(f"  Processing prompt {i+1}/{len(prompt_templates)}: \"{prompt_text[:40]}...\"")
        
        generated_texts = generate_text(text_generator, prompt_text)
        
        for gen_text in generated_texts:
            output_record = {
                "prompt_id": template["id"],
                "prompt_text": prompt_text,
                "register": template["register"],
                "demographic_var": template["demographic_var"],
                "model_name": MODEL_NAME,
                "generated_text": gen_text, # Full generated text including prompt
                "completion_only": gen_text[len(prompt_text):].strip(), # Attempt to get only the completion
                "timestamp": timestamp
            }
            all_generated_outputs.append(output_record)
    
    print(f"\nTotal outputs generated: {len(all_generated_outputs)}")

    # Save the outputs to a JSON file
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_generated_outputs, f, indent=4)
        print(f"Successfully saved {len(all_generated_outputs)} outputs to {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error saving outputs to {OUTPUT_FILE}: {e}")
        return

    # Print a sample of the generated data
    if all_generated_outputs:
        print("\nSample of generated data (first record):")
        print(json.dumps(all_generated_outputs[0], indent=4))
    
    print("\nLLM output generation finished.")

if __name__ == "__main__":
    main() 