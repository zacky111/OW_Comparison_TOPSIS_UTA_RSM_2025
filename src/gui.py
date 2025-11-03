import customtkinter as ctk
import pandas as pd
from tkinter import filedialog, messagebox
from tkinter import ttk

## models
from src.alg.topsis import calculate_topsis_score
from src.alg.rsm import compute_rsm_scores
from src.alg.uta import compute_uta_results


class DecisionAidApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Decision Aid Tool")
        self.geometry("900x600")

        self.method_index = 0
        self.data = None

        # --- Górny pasek kontrolny ---
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(fill="x", padx=10, pady=10)

        self.load_btn = ctk.CTkButton(top_frame, text="Load CSV", command=self.load_data)
        self.load_btn.pack(side="left", padx=5)

        self.method_var = ctk.StringVar(value="TOPSIS")
        self.method_menu = ctk.CTkOptionMenu(top_frame, variable=self.method_var,
                                             values=["TOPSIS", "RSM", "UTA"])
        self.method_menu.pack(side="left", padx=5)

        self.generate_btn = ctk.CTkButton(top_frame, text="Generate Ranking",
                                          state="disabled", command=self.generate_ranking)
        self.generate_btn.pack(side="left", padx=5)

        # --- Tabele ---
        table_frame = ctk.CTkFrame(self)
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # tabela z danymi wejściowymi
        self.data_table = ttk.Treeview(table_frame, show="headings")
        self.data_table.pack(side="left", fill="both", expand=True, padx=5)

        # tabela z wynikami
        self.result_table = ttk.Treeview(table_frame, show="headings")
        self.result_table.pack(side="left", fill="both", expand=True, padx=5)

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
            self.data = df.values.tolist()
            self.populate_table(self.data_table, df)
            self.generate_btn.configure(state="normal")
            messagebox.showinfo("Success", "Data loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def populate_table(self, table, dataframe):
        # wyczyść starą zawartość
        for col in table.get_children():
            table.delete(col)

        # ustaw nagłówki
        table["columns"] = list(dataframe.columns)
        for col in dataframe.columns:
            table.heading(col, text=col)
            table.column(col, width=120)

        # wstaw dane
        for _, row in dataframe.iterrows():
            table.insert("", "end", values=list(row))

    def generate_ranking(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Load data first!")
            return

        method = self.method_var.get()
        match method:
            case "TOPSIS":
                result = calculate_topsis_score(self.data)
            case "RSM":
                result = compute_rsm_scores(self.data)
            case "UTA":
                result = compute_uta_results(self.data)
            case _:
                result = []

        df_result = pd.DataFrame(result, columns=["Point", "Score"])
        self.populate_table(self.result_table, df_result)


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = DecisionAidApp()
    app.mainloop()
