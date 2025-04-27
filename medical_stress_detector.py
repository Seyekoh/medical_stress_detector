"""
medical_stress_detector.py

This module contains the main function to run the medical stress detector application.

It initializes the application, loads the model, and starts the GUI for user interaction.
It also includes a function to handle the prediction of stress levels based on user input.
It is designed to be run as a standalone script.
"""

__author__ = "James Bridges"
__version__ = "1.0.0"

import tkinter as tk
from tkinter import ttk, messagebox
import csv
from datetime import datetime
from pathlib import Path
import pickle
import numpy as np

# Load the pre-trained model from a file
with open('stress_detection_model.pkl', 'rb') as model_file:
    stress_model = pickle.load(model_file)
with open('stress_scaler_model.pkl', 'rb') as scaler_file:
    input_scaler = pickle.load(scaler_file)

# Mapping from numeric prediction (0-4) to human-readable labels.
stress_label_map = {
    0: "None",
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Extreme"
}

# Expected requirements for each field
field_requirements = {
    "Snoring Range": "Expected range: 0-100",
    "Respiration Rate": "Expected range: 0-120",
    "Body Temperature (°F)": "Expected range: 50-115",
    "Limb Movement": "Expected range: 0-100",
    "Blood Oxygen": "Expected range: 30-100",
    "Eye Movement": "Expected range: 0-200",
    "Hours of Sleep": "Expected range: 0-24",
    "Heart Rate": "Expected range: 0-250"
}

# Keep track of open popups
open_popups = []

def validate_input(field_name, value_str, min_val=None, max_val=None, allow_null=False, soft_min=None, soft_max=None):
    """
    Validates the input for each field in the GUI based on range and nullability.

    :param field_name: Label for error messages
    :param value_str: Raw string input
    :param min_val: Minimum allowed value (inclusive)
    :param max_val: Maximum allowed value (inclusive)
    :param allow_null: Whether empty input is acceptable
    :param soft_max: If value exceeds this, warn but do not block submission

    :return: float or None
    :raises: ValueError if input is invalid
    """
    if not value_str:
        if allow_null:
            return None
        else:
            raise ValueError(f"{field_name} cannot be empty.")

    try:
        value = float(value_str)
    except ValueError:
        raise ValueError(f"{field_name} must be a number.")

    if value < 0:
        raise ValueError(f"{field_name} cannot be negative.")

    if min_val is not None and value < min_val:
        raise ValueError(f"{field_name} must be at least {min_val}.")
    if max_val is not None and value > max_val:
        raise ValueError(f"{field_name} must be at most {max_val}.")

    if soft_min is not None and value < soft_min:
        show_warning_popup("Warning", f"{field_name} is below the expected minimum of {soft_min}. This could indicate a medical emergency.")
    if soft_max is not None and value > soft_max:
        show_warning_popup("Warning", f"{field_name} is above the expected maximum of {soft_max}. This could indicate a medical emergency.")

    return value

def show_warning_popup(title, message):
    """
    Displays a warning popup with a specified title and message.
    The popup is positioned relative to the main application window.

    :param title: The title of the popup window
    :param message: The message to display in the popup window
    """
    popup = tk.Toplevel(root)
    open_popups.append(popup)
    popup.title(title)

    # Wait for the root window to be drawn before calculating position
    root.update_idletasks()

    base_x = root.winfo_x()
    base_y = root.winfo_y()

    popup.geometry(f"300x100+{base_x - 8}+{base_y - 150}")

    # Hide momentarily to avoid window manager focus jump
    popup.withdraw()

    label = ttk.Label(popup, text=message, wraplength=260)
    label.pack(pady=10, padx=10)

    button = ttk.Button(popup, text="OK", command=lambda: close_warning(popup))
    button.pack(pady=(0,10))

    # Show the popup after setting its position
    popup.deiconify()

def close_warning(popup):
    """
    Closes the warning popup

    :param popup: The popup window to be closed
    """
    if popup in open_popups:
        open_popups.remove(popup)
    popup.destroy()

def show_error_popup(title, message):
    """
    Displays an error popup with a specified title and message.
    It currently creates a simple popup window with the provided title and message.

    :param title: The title of the popup window
    :param message: The message to display in the popup window
    """
    popup = tk.Toplevel(root)
    open_popups.append(popup)
    popup.title(title)

    # Wait for the root window to be drawn before calculating position
    popup.update_idletasks()

    base_x = root.winfo_x()
    base_y = root.winfo_y()

    popup.geometry(f"350x250+{base_x - 375}+{base_y + 50}")
    label = ttk.Label(popup, text=message, wraplength=360)
    label.pack(pady=10, padx=10)

    button = ttk.Button(popup, text="OK", command=popup.destroy)
    button.pack(pady=(0,10))

def close_all_popups():
    """
    Closes all open popups.
    """
    for popup in open_popups:
        popup.destroy()
    open_popups.clear()

def model_prediction(input_features):
    """
    Predicts the stress level using the loaded model based on input features.

    :param input_features: List of input features for the model

    :return: Numeric prediction (0-4)
    """
    input_array = np.array(input_features).reshape(1, -1)
    scaled_input = input_scaler.transform(input_array)
    prediction = stress_model.predict(scaled_input)
    return int(prediction[0])


def log_result(input_values, result_label):
    """
    Logs the date, time, input values, and prediction result to a CSV file.

    :param input_values: The list of input values provided by the user.
    :param result_label: The label of the prediction result (e.g., "Low", "Medium", etc.).
    """
    log_file = "stress_predictions_log.csv"
    timestamp = datetime.now().strftime("%d-%m-%y %H:%M:%S")
    headers = ["DateTime"] + [field[0] for field in input_fields] + ["Prediction"]
    row = [timestamp] + input_values + [result_label]
    try:
        write_header = not Path(log_file).exists()
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(headers)
            writer.writerow(row)
    except Exception as e:
        messagebox.showerror("Logging Error", f"Could not save the result to log: {str(e)}")

def submit_data():
    """
    Collects user input from the GUI, arranges it, and sends it to the
    model for prediction. The result is then displayed to the user.
    """
    close_all_popups()

    try:
        invalid_fields = []
        validated_inputs = []

        # Collect and validate input from the GUI
        for field_name, var in input_fields:
            entry_widget = input_entries[field_name]
            entry_widget.configure(background="white")
            try:
                if field_name == "Snoring Range":
                    value = validate_input(field_name, var.get(), min_val=0, max_val=100)
                elif field_name == "Respiration Rate":
                    value = validate_input(field_name, var.get(), min_val=0, max_val=120, soft_min=10, soft_max=60)
                elif field_name == "Body Temperature (°F)":
                    value = validate_input(field_name, var.get(), min_val=50, max_val=115, allow_null=True, soft_min=95, soft_max=106)
                elif field_name == "Limb Movement":
                    value = validate_input(field_name, var.get(), min_val=0, max_val=100, allow_null=True, soft_max=30)
                elif field_name == "Blood Oxygen":
                    value = validate_input(field_name, var.get(), min_val=30, max_val=100, allow_null=True, soft_min=90)
                elif field_name == "Eye Movement":
                    value = validate_input(field_name, var.get(), min_val=0, max_val=200, allow_null=True, soft_min=5, soft_max=120)
                elif field_name == "Hours of Sleep":
                    value = validate_input(field_name, var.get(), min_val=0, max_val=24, allow_null=True, soft_min=4, soft_max=12)
                elif field_name == "Heart Rate":
                    value = validate_input(field_name, var.get(), min_val=0, max_val=250, allow_null=True, soft_min=40, soft_max=180)
                else:
                    raise ValueError(f"Unexpected field: {field_name}")

                validated_inputs.append(value)
            except ValueError as e:
                # Highlight the entry field in light red if validation fails
                entry_widget.configure(background="#ffcccc")
                invalid_fields.append(field_name)

        # If any fields are invalid, focus on the first invalid field and raise an error
        if invalid_fields:
            input_entries[invalid_fields[0]].focus()
            error_msg = "Invalid input for:\n"
            for field in invalid_fields:
                requirement = field_requirements.get(field, "No specific requirement")
                error_msg += f"   {field}:  {requirement}\n"
            raise ValueError(error_msg)

        # Call the model prediction function
        prediction_numeric = model_prediction(validated_inputs)
        prediction_text = stress_label_map.get(prediction_numeric, "Unknown")

        # Display the prediction result to the user
        result_var.set(f"Stress Level: {prediction_text} (Code: {prediction_numeric})")

        # Log the result to a CSV file
        log_result(validated_inputs, prediction_text)

    except ValueError as e:
        show_error_popup("Input Error", f"Please enter a valid value for all fields.\n\n{str(e)}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

def clear_form():
    """
    Clears all input fields in the GUI to allow for new data entry.
    """
    for _, var in input_fields:
        var.set("")
    result_var.set("")
    for entry in input_entries.values():
        entry.configure(background="white")

    # Focus on the first input field after clearing
    list(input_entries.values())[0].focus()

# Create the main application window
root = tk.Tk()
root.title("Stress Detector")

# Center the window on the screen
window_width = 290
window_height = 350

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))

root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Create a frame to hold all the widgets
mainframe = ttk.Frame(root, padding="10")
mainframe.grid(row=0, column=0, sticky="nwes")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Define input variables for the GUI
snoring_range_var = tk.StringVar()
respiration_rate_var = tk.StringVar()
body_temperature_var = tk.StringVar()
limb_movement_var = tk.StringVar()
blood_oxygen_var = tk.StringVar()
eye_movement_var = tk.StringVar()
hours_of_sleep_var = tk.StringVar()
heart_rate_var = tk.StringVar()
result_var = tk.StringVar()

# Create a list of input fields and their corresponding StringVars
input_fields = [
    ("Snoring Range", snoring_range_var),
    ("Respiration Rate", respiration_rate_var),
    ("Body Temperature (°F)", body_temperature_var),
    ("Limb Movement", limb_movement_var),
    ("Blood Oxygen", blood_oxygen_var),
    ("Eye Movement", eye_movement_var),
    ("Hours of Sleep", hours_of_sleep_var),
    ("Heart Rate", heart_rate_var)
]

# Create a dictionary to hold the input entries for easy access
input_entries = {}

# Create and place input fields and labels in the GUI
row = 0
for label, var in input_fields:
    ttk.Label(mainframe, text=f"{label}:").grid(row=row, column=0, sticky=tk.W)
    entry = tk.Entry(mainframe, textvariable=var)
    entry.grid(column=1, row=row)
    input_entries[label] = entry
    row += 1

# Create buttons for submitting data and clearing the form
ttk.Button(mainframe, text="Submit", command=submit_data).grid(column=0, row = row, pady=10)
ttk.Button(mainframe, text="Clear", command=clear_form).grid(column=1, row=row, pady=10)
row += 1

# Display for the result of the prediction
ttk.Label(mainframe, textvariable=result_var, font=("Helvetica", 14)).grid(column=0, row=row, columnspan=2, pady=10)

# Add padding around each widget for cleaner layout
for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

# Set focus on the first input field
list(input_entries.values())[0].focus()

# Start the application
root.mainloop()
