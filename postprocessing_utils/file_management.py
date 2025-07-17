import tkinter as tk 
from tkinter import filedialog

@DeprecationWarning
def select_file(): 
    """Select a specific file from the window and return
    it's absolute path as a variable"""

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(title="Select a media file", 
                    filetypes=[("MP4 files", ".mp4"), ("MOV files", ".mov")])
    
    return file_path


def select_file(*args):
    """
    Open a file dialog to select a file with a specific extension.

    Args:
        *extensions: One or more file extensions as strings (e.g., ".mp4", ".mov").
        title (str): Title for the file selection dialog.

    Returns:
        str: Absolute file path selected by the user.
    """
    root = tk.Tk()
    root.withdraw()

    # Default -> .mp4 para no joder el codigo que ya tengo. 
    if not args:
        args = (".mp4",)

    # Ensure all extensions start with a dot
    args = tuple(f".{ext.lstrip('.')}" for ext in args)

    # Create filetypes dynamically
    filetypes = [(f"{ext.upper()} files", ext) for ext in args]

    file_path = filedialog.askopenfilename(title="Select a file", filetypes=filetypes)
    
    return file_path
