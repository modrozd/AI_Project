from DataHandler.PatientDatabase import PatientDatabase
import os


# Get proper path of resource
dir = os.path.realpath(os.path.dirname(__file__))  # Directory of your Python file
file_path = os.path.join(dir, "resources", "arrhythmia.csv")  # Create the path of the file


patient_db = PatientDatabase(file_path)



