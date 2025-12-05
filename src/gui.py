import customtkinter as ctk
import pandas as pd
from tkinter import filedialog, messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


## models
from src.alg.topsis import calculate_topsis_score
from src.alg.rsm import compute_rsm_scores
from src.alg.uta import compute_uta_results
from src.alg.spcs import compute_spcs_scores


class DecisionAidApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Decision Aid Tool")
        self.geometry("1200x800")

        self.df = None

        # ================= TOP BAR =================
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=10, pady=10)

        self.load_btn = ctk.CTkButton(top, text="Load CSV", command=self.load_data)
        self.load_btn.pack(side="left", padx=5)

        self.method_var = ctk.StringVar(value="TOPSIS")
        self.method_menu = ctk.CTkOptionMenu(
            top, variable=self.method_var,
            values=["TOPSIS", "RSM", "SP-CS", "UTA-DIS"]
        )
        self.method_menu.pack(side="left", padx=5)

        self.generate_btn = ctk.CTkButton(
            top, text="Generate Ranking",
            state="disabled", command=self.generate_ranking
        )
        self.generate_btn.pack(side="left", padx=5)
        
        self.plot_btn = ctk.CTkButton(
            top, text="Show 3D Plot",
            state="disabled", command=self.show_3d_plot
        )
        self.plot_btn.pack(side="left", padx=5)

        # ---- SORTING SWITCH ----
        ctk.CTkLabel(top, text="Sort by:").pack(side="left", padx=10)

        self.sort_var = ctk.StringVar(value="Index")
        self.sort_menu = ctk.CTkOptionMenu(
            top,
            variable=self.sort_var,
            values=["Index", "Score"]
        )
        self.sort_menu.pack(side="left", padx=5)

        # ================= SPLIT VERTICAL LAYOUT =================
        main = ctk.CTkFrame(self)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # ---------- TOP: INPUT DATA ----------
        top_frame = ctk.CTkFrame(main)
        top_frame.pack(fill="both", expand=True, padx=5, pady=5)

        ctk.CTkLabel(top_frame, text="Input Data").pack()

        # Data table + scrollbars
        self.data_table = ttk.Treeview(top_frame, show="headings")
        data_scroll_y = ttk.Scrollbar(top_frame, orient="vertical", command=self.data_table.yview)
        data_scroll_x = ttk.Scrollbar(top_frame, orient="horizontal", command=self.data_table.xview)
        self.data_table.configure(yscrollcommand=data_scroll_y.set, xscrollcommand=data_scroll_x.set)

        self.data_table.pack(fill="both", expand=True)
        data_scroll_y.pack(side="right", fill="y")
        data_scroll_x.pack(side="bottom", fill="x")

        # ---------- BOTTOM: RANKING ----------
        bottom_frame = ctk.CTkFrame(main)
        bottom_frame.pack(fill="both", expand=True, padx=5, pady=5)

        ctk.CTkLabel(bottom_frame, text="Ranking Results").pack()

        # Ranking table + scrollbars
        self.result_table = ttk.Treeview(bottom_frame, show="headings")
        result_scroll_y = ttk.Scrollbar(bottom_frame, orient="vertical", command=self.result_table.yview)
        result_scroll_x = ttk.Scrollbar(bottom_frame, orient="horizontal", command=self.result_table.xview)
        self.result_table.configure(yscrollcommand=result_scroll_y.set, xscrollcommand=result_scroll_x.set)

        self.result_table.pack(fill="both", expand=True)
        result_scroll_y.pack(side="right", fill="y")
        result_scroll_x.pack(side="bottom", fill="x")

        # Table styling
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Treeview",
            background="#2b2b2b",
            foreground="white",
            fieldbackground="#2b2b2b",
            rowheight=25
        )
        style.map("Treeview", background=[("selected", "#1f538d")])

    # ============== LOAD DATA ==============
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

    # ============== POPULATE ANY TABLE ==============
    def populate_table(self, table, dataframe):
        table.delete(*table.get_children())  # clear rows
        table["columns"] = list(dataframe.columns)

        for col in dataframe.columns:
            table.heading(col, text=str(col))
            table.column(col, width=120, anchor="center")

        for _, row in dataframe.iterrows():
            table.insert("", "end", values=[row[c] for c in dataframe.columns])

    # ============== RANK GENERATION ==============
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
        elif method == "UTA-DIS":
            result = compute_uta_results(data_values)
        else:
            result = []

        df_result = pd.DataFrame(result, columns=["PointIndex", "Score"])

        param_cols = self.df.columns
        for i, col in enumerate(param_cols):
            df_result[col] = self.df.iloc[df_result["PointIndex"].values][col].values

        # ===== SORT BASED ON GUI SELECTION =====
        sort_mode = self.sort_var.get()

        if sort_mode == "Score":
            df_result = df_result.sort_values(by="Score", ascending=False)
        else:
            df_result = df_result.sort_values(by="PointIndex", ascending=True)

        df_result = df_result.reset_index(drop=True)

        self.populate_table(self.result_table, df_result)
        self.plot_btn.configure(state="normal")

    def show_3d_plot(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Load data first!")
            return

        if len(self.df.columns) < 3:
            messagebox.showwarning("Warning", "Dataset must contain at least 3 columns for 3D plot.")
            return

        x = self.df.iloc[:, 0].astype(float).values
        y = self.df.iloc[:, 1].astype(float).values
        z = self.df.iloc[:, 2].astype(float).values

        score_values = None
        try:
            rows = self.result_table.get_children()
            if rows:
                score_values = []
                for item in rows:
                    row = self.result_table.item(item)["values"]
                    score_values.append(float(row[1]))  # KONWERSJA do float
        except Exception as e:
            print("Error reading scores:", e)
            score_values = None


        plot_window = ctk.CTkToplevel(self)
        plot_window.title("3D Plot")
        plot_window.geometry("800x600")

        fig = Figure(figsize=(7, 6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        if score_values is not None and len(score_values) == len(x):
            sc = ax.scatter(x, y, z, c=score_values, cmap="viridis", s=60)
            fig.colorbar(sc, ax=ax, shrink=0.6)
        else:

            sc = ax.scatter(x, y, z, s=60)

        ax.set_xlabel(self.df.columns[0])
        ax.set_ylabel(self.df.columns[1])
        ax.set_zlabel(self.df.columns[2])

        ax.set_title("3D Scatter Plot of Points")

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

