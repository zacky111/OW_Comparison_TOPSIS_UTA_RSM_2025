from src.data_import import data_import
from src.gui import DecisionAidApp

import customtkinter as ctk
import pandas as pd
from tkinter import filedialog, messagebox
from tkinter import ttk


#import danych
#data_import("2025-11-03T09-11_export.csv")

#uruchomienie gui
if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = DecisionAidApp()
    app.mainloop()

