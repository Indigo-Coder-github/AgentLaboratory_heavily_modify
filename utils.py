import os
import re
import shutil


class Util:
    @staticmethod
    def remove_figures() -> None:
        """Remove a directory if it exists."""
        for _file in os.listdir("."):
            if "Figure_" in _file and ".png" in _file:
                os.remove(_file)

    @staticmethod
    def remove_directory(dir_path) -> None:
        """Remove a directory if it exists."""
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Directory {dir_path} removed successfully.")
            except Exception as e:
                print(f"Error removing directory {dir_path}: {e}")
        else:
            print(f"Directory {dir_path} does not exist or is not a directory.")

    @staticmethod
    def save_to_file(location, filename, data) -> None:
        """Utility function to save data as plain text."""
        filepath = os.path.join(location, filename)
        try:
            with open(filepath, 'w') as f:
                f.write(data)  # Write the raw string instead of using json.dump
            print(f"Data successfully saved to {filepath}")
        except Exception as e:
            print(f"Error saving file {filename}: {e}")

    @staticmethod
    def extract_prompt(text, word) -> str:
        code_block_pattern = rf"```{word}(.*?)```"
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
        extracted_code: str = "\n".join(code_blocks).strip()
        return extracted_code

remove_figures: None = Util.remove_figures()
remove_directory: None = Util.remove_directory()
save_to_file: None = Util.save_to_file()
extract_prompt: str = Util.extract_prompt()