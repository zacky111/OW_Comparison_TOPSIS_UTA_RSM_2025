# gui.py
import customtkinter as ctk
import pandas as pd
from tkinter import filedialog, messagebox
from tkinter import ttk

## models
from src.alg.topsis import calculate_topsis_score
from src.alg.rsm import compute_rsm_scores
from src.alg.uta import compute_uta_results
from src.alg.spcs import compute_spcs_scores

class DecisionAidApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Decision Aid Tool")
        self.geometry("1000x650")

        self.method_index = 0
        self.df = None

        # --- Górny pasek kontrolny ---
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(fill="x", padx=10, pady=10)

        self.load_btn = ctk.CTkButton(top_frame, text="Load CSV", command=self.load_data)
        self.load_btn.pack(side="left", padx=5)

        self.method_var = ctk.StringVar(value="TOPSIS")
        self.method_menu = ctk.CTkOptionMenu(top_frame, variable=self.method_var,
                                             values=["TOPSIS", "RSM", "SP-CS", "UTA"])
        self.method_menu.pack(side="left", padx=5)

        self.generate_btn = ctk.CTkButton(top_frame, text="Generate Ranking",
                                          state="disabled", command=self.generate_ranking)
        self.generate_btn.pack(side="left", padx=5)

        # --- Tabele ---
        table_frame = ctk.CTkFrame(self)
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # lewa: dane wejściowe (DataFrame)
        left_frame = ctk.CTkFrame(table_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=5)
        ttk.Label(left_frame, text="Input data (rows = alternatives)").pack()
        self.data_table = ttk.Treeview(left_frame, show="headings")
        self.data_table.pack(fill="both", expand=True)

        # prawa: wyniki
        right_frame = ctk.CTkFrame(table_frame)
        right_frame.pack(side="left", fill="both", expand=True, padx=5)
        ttk.Label(right_frame, text="Ranking results").pack()
        self.result_table = ttk.Treeview(right_frame, show="headings")
        self.result_table.pack(fill="both", expand=True)

        # stylowanie
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b")
        style.map("Treeview", background=[("selected", "#1f538d")])

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path, index_col=0)
            self.df = df.copy()
            self.populate_table(self.data_table, df)
            self.generate_btn.configure(state="normal")
            messagebox.showinfo("Success", "Data loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def populate_table(self, table, dataframe):
        # wyczyść starą zawartość (itemy), a nie kolumny
        for it in table.get_children():
            table.delete(it)

        # ustaw nagłówki
        table["columns"] = list(dataframe.columns)
        for col in dataframe.columns:
            table.heading(col, text=str(col))
            table.column(col, width=120, anchor="center")

        # wstaw dane
        for idx, row in dataframe.iterrows():
            vals = [row[c] for c in dataframe.columns]
            table.insert("", "end", values=vals)

    def generate_ranking(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Load data first!")
            return

        method = self.method_var.get()
        data_values = self.df.values.tolist()
        if method == "TOPSIS":
            result = calculate_topsis_score(data_values)
        elif method == "RSM":
            result = compute_rsm_scores(data_values)
        elif method == "SP-CS":
            result = compute_spcs_scores(data_values)
        elif method == "UTA":
            result = compute_uta_results(data_values)
        else:
            result = []

        # prepare DataFrame for results
        df_result = pd.DataFrame(result, columns=["PointIndex", "Score"])
        # sort by index for nicer view
        df_result = df_result.sort_values(by="PointIndex").reset_index(drop=True)
        self.populate_table(self.result_table, df_result)
