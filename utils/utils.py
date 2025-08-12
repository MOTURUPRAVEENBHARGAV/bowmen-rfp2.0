import yaml

def load_prompts():
    """
    Loads all prompts from the prompts.yml file.
    """
    try:
        with open("Agents/prompts/prompts.yml", 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("ERROR: prompts/prompts.yml file not found. Please ensure it exists.")
        return {}
    except Exception as e:
        print(f"An error occurred while loading prompts: {e}")
        return {}