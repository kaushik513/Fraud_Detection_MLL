__version__ = "1.0.0"

import json
import os
import queue
import subprocess
import sys
import threading
import traceback
import warnings

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  

import config  
from src.data_loader import load_csv, make_synthetic, prepare_frame  
from src.eda import run_eda  
from src.models import XGB_BACKEND  
from src.pipeline import run_study  

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PAD = {"padx": 6, "pady": 4}


def open_path(path):
    if not os.path.exists(path):
        messagebox.showwarning("Not found", f"{path} does not exist yet.")
        return
    if sys.platform.startswith("win"):
        os.startfile(path)  
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class ImageViewer(ttk.Frame):

    def __init__(self, master):
        super().__init__(master)
        self.paths, self._img = [], None
        left = ttk.Frame(self)
        left.pack(side="left", fill="y")
        self.listbox = tk.Listbox(left, width=34, exportselection=False)
        self.listbox.pack(side="top", fill="y", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._show)
        ttk.Button(left, text="Open image externally",
                   command=self._open_external).pack(side="top", fill="x", pady=4)

        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(right, background="white")
        vs = ttk.Scrollbar(right, orient="vertical", command=self.canvas.yview)
        hs = ttk.Scrollbar(right, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        vs.pack(side="right", fill="y")
        hs.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

    def set_paths(self, paths):
        self.paths = [p for p in paths if p and os.path.exists(p)]
        self.listbox.delete(0, "end")
        for p in self.paths:
            self.listbox.insert("end", os.path.basename(p))
        if self.paths:
            self.listbox.selection_set(0)
            self._show()

    def _current(self):
        sel = self.listbox.curselection()
        return self.paths[sel[0]] if sel else None

    def _show(self, _event=None):
        path = self._current()
        if not path:
            return
        img = tk.PhotoImage(file=path)
        width = max(self.canvas.winfo_width(), 600)
        factor = max(1, int(round(img.width() / width + 0.49))) # integer shrink to fit
        if factor > 1:
            img = img.subsample(factor, factor)
        self._img = img     
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=img)
        self.canvas.configure(scrollregion=(0, 0, img.width(), img.height()))

    def _open_external(self):
        if self._current():
            open_path(self._current())


class FraudApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Credit-Card Fraud Detection — Reproduction & HPO  (v{__version__})")
        self.geometry("1180x780")
        self.minsize(980, 640)

        self.X = self.y = self.info = None
        self.synthetic = False
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.nb = nb
        self._build_data_tab(nb)
        self._build_eda_tab(nb)
        self._build_exp_tab(nb)
        self._build_results_tab(nb)

        self.status = tk.StringVar(value=f"Ready. XGBoost backend: {XGB_BACKEND}")
        ttk.Label(self, textvariable=self.status, anchor="w", relief="sunken") \
            .pack(fill="x", side="bottom")
        self.after(100, self._poll_queue)

    def _build_data_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="1 · Data")
        box = ttk.LabelFrame(tab, text="Dataset")
        box.pack(fill="x", **PAD)

        self.src_var = tk.StringVar(value="csv")
        ttk.Radiobutton(box, text="CSV file", variable=self.src_var, value="csv") \
            .grid(row=0, column=0, sticky="w", **PAD)
        self.path_var = tk.StringVar()
        ttk.Entry(box, textvariable=self.path_var, width=70).grid(row=0, column=1, **PAD)
        ttk.Button(box, text="Browse…", command=self._browse).grid(row=0, column=2, **PAD)
        ttk.Radiobutton(box, text="Synthetic demo data (pipeline test only — not real data)",
                        variable=self.src_var, value="synthetic") \
            .grid(row=1, column=0, columnspan=3, sticky="w", **PAD)
        ttk.Label(box, text="Target column:").grid(row=2, column=0, sticky="e", **PAD)
        self.target_var = tk.StringVar(value=config.TARGET_COLUMN)
        ttk.Entry(box, textvariable=self.target_var, width=20) \
            .grid(row=2, column=1, sticky="w", **PAD)
        ttk.Button(box, text="Load data", command=self._load).grid(row=2, column=2, **PAD)

        self.data_text = ScrolledText(tab, height=26, font=("Consolas", 10))
        self.data_text.pack(fill="both", expand=True, **PAD)
        self.data_text.insert("end", "Load a CSV (e.g. the Kaggle credit-card dataset cited "
                                     "in the reference paper) or the synthetic demo.\n")

    def _browse(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All", "*.*")])
        if path:
            self.path_var.set(path)
            self.src_var.set("csv")

    def _load(self):
        try:
            if self.src_var.get() == "synthetic":
                self.X, self.y, self.info = prepare_frame(make_synthetic(), config.TARGET_COLUMN)
                self.synthetic = True
            else:
                if not self.path_var.get():
                    messagebox.showinfo("No file", "Choose a CSV file first.")
                    return
                self.X, self.y, self.info = load_csv(self.path_var.get(), self.target_var.get())
                self.synthetic = False
        except Exception as exc:
            messagebox.showerror("Could not load data", str(exc))
            return
        i = self.info
        lines = [
            ("SYNTHETIC DEMO DATA — do not report these numbers\n\n" if self.synthetic else ""),
            f"Rows: {i['rows']:,}    Features: {i['features']}\n",
            f"Fraud (1): {i['fraud']:,}    Legitimate (0): {i['legit']:,}    "
            f"Fraud rate: {i['fraud_rate']:.2%}\n",
            f"Dropped ID columns: {i['dropped_id_columns'] or 'none'}\n",
            f"Duplicates removed: {i['duplicates_removed']}    "
            f"Missing values imputed: {i['missing_values_imputed']}\n\n",
            f"Categorical (one-hot encoded): {', '.join(i['categorical_columns']) or 'none'}\n\n",
            f"Numeric (standard-scaled): {', '.join(i['numeric_columns'])}\n\n",
            "First rows:\n", self.X.head(8).to_string(), "\n",
        ]
        self.data_text.delete("1.0", "end")
        self.data_text.insert("end", "".join(lines))
        self.status.set(f"Loaded {i['rows']:,} rows ({i['fraud_rate']:.2%} fraud).")

    def _need_data(self):
        if self.X is None:
            messagebox.showinfo("No data", "Load a dataset on the Data tab first.")
            return True
        return False

    def _build_eda_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="2 · Exploratory analysis")
        bar = ttk.Frame(tab)
        bar.pack(fill="x", **PAD)
        ttk.Button(bar, text="Run EDA", command=self._run_eda).pack(side="left", **PAD)
        ttk.Button(bar, text="Open EDA folder",
                   command=lambda: open_path(os.path.join(self.out_var.get(), "eda"))) \
            .pack(side="left", **PAD)
        ttk.Label(bar, text="Class balance, distributions, categorical fraud rates, "
                            "correlations, outliers, PCA, k-means.").pack(side="left", **PAD)
        self.eda_viewer = ImageViewer(tab)
        self.eda_viewer.pack(fill="both", expand=True, **PAD)

    def _run_eda(self):
        if self._need_data() or self._busy():
            return
        out = os.path.join(self.out_var.get(), "eda")

        def job():
            paths, _ = run_eda(self.X, self.y, out, log=self._log)
            self.queue.put(("eda_done", paths))
        self._start(job, "Running EDA…")

    def _build_exp_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="3 · Experiments")
        top = ttk.Frame(tab)
        top.pack(fill="x", **PAD)

        mbox = ttk.LabelFrame(top, text="Models")
        mbox.pack(side="left", fill="y", **PAD)
        self.model_vars = {}
        ttk.Label(mbox, text="Reference paper:", font=("TkDefaultFont", 9, "bold")) \
            .pack(anchor="w")
        for m in config.PAPER_MODELS:
            self.model_vars[m] = tk.BooleanVar(value=True)
            ttk.Checkbutton(mbox, text=config.MODEL_LABELS[m], variable=self.model_vars[m]) \
                .pack(anchor="w")
        ttk.Label(mbox, text="Lab-manual extensions:", font=("TkDefaultFont", 9, "bold")) \
            .pack(anchor="w", pady=(6, 0))
        for m in config.EXTENSION_MODELS:
            self.model_vars[m] = tk.BooleanVar(value=False)
            ttk.Checkbutton(mbox, text=config.MODEL_LABELS[m], variable=self.model_vars[m]) \
                .pack(anchor="w")

        sbox = ttk.LabelFrame(top, text="Data settings")
        sbox.pack(side="left", fill="y", **PAD)
        self.samp_vars = {}
        for s in config.SAMPLINGS:
            self.samp_vars[s] = tk.BooleanVar(value=True)
            ttk.Checkbutton(sbox, text=s.upper(), variable=self.samp_vars[s]).pack(anchor="w")
        ttk.Separator(sbox).pack(fill="x", pady=6)
        ttk.Label(sbox, text="Evaluation protocol:").pack(anchor="w")
        self.protocol_var = tk.StringVar(value=config.DEFAULT_PROTOCOL)
        ttk.Radiobutton(sbox, text="strict (resample training folds only)",
                        variable=self.protocol_var, value="strict").pack(anchor="w")
        ttk.Radiobutton(sbox, text="paper (resample before CV — leaky)",
                        variable=self.protocol_var, value="paper").pack(anchor="w")

        obox = ttk.LabelFrame(top, text="Search")
        obox.pack(side="left", fill="y", **PAD)
        ttk.Label(obox, text="Random configs per model:").grid(row=0, column=0, sticky="w")
        self.niter_var = tk.IntVar(value=config.DEFAULT_N_ITER)
        ttk.Spinbox(obox, from_=0, to=100, textvariable=self.niter_var, width=6) \
            .grid(row=0, column=1, sticky="w", **PAD)
        self.sens_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(obox, text="One-at-a-time sensitivity sweeps",
                        variable=self.sens_var).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Label(obox, text="Output folder:").grid(row=2, column=0, sticky="w")
        self.out_var = tk.StringVar(value=os.path.abspath(config.RESULTS_DIR))
        ttk.Entry(obox, textvariable=self.out_var, width=34) \
            .grid(row=3, column=0, columnspan=2, sticky="w", **PAD)
        ttk.Button(obox, text="Quick preset (4 configs, no sweeps)",
                   command=self._quick_preset).grid(row=4, column=0, columnspan=2,
                                                    sticky="we", **PAD)
        ttk.Label(obox, text=f"Folds: {config.CV_FOLDS}   Hold-out: "
                             f"{int(config.HOLDOUT_FRACTION * 100)} %   "
                             f"Seed: {config.RANDOM_STATE}").grid(row=5, column=0,
                                                                   columnspan=2, sticky="w")

        btns = ttk.Frame(top)
        btns.pack(side="left", fill="y", **PAD)
        self.run_btn = ttk.Button(btns, text="▶ Run study", command=self._run_study)
        self.run_btn.pack(fill="x", pady=4)
        ttk.Button(btns, text="■ Stop", command=self._stop).pack(fill="x", pady=4)
        ttk.Button(btns, text="Open results folder",
                   command=lambda: open_path(self.out_var.get())).pack(fill="x", pady=4)

        self.progress = ttk.Progressbar(tab, mode="determinate")
        self.progress.pack(fill="x", **PAD)
        self.log_text = ScrolledText(tab, height=20, font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, **PAD)

    def _quick_preset(self):
        self.niter_var.set(4)
        self.sens_var.set(False)

    def _run_study(self):
        if self._need_data() or self._busy():
            return
        models = [m for m, v in self.model_vars.items() if v.get()]
        samplings = [s for s, v in self.samp_vars.items() if v.get()]
        if not models or not samplings:
            messagebox.showinfo("Nothing selected", "Pick at least one model and one setting.")
            return
        try:
            n_iter = int(self.niter_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid value", "Configs per model must be a whole number.")
            return
        kwargs = dict(models=models, samplings=samplings, protocol=self.protocol_var.get(),
                      n_iter=n_iter, do_sensitivity=self.sens_var.get(),
                      out_dir=self.out_var.get(), synthetic=self.synthetic)
        self.log_text.delete("1.0", "end")
        self.progress["value"] = 0

        def job():
            final_df, _, plots = run_study(
                self.X, self.y, self.info, log=self._log,
                should_stop=self.stop_event.is_set,
                progress=lambda d, t: self.queue.put(("progress", (d, t))), **kwargs)
            self.queue.put(("study_done", (final_df, plots)))
        self._start(job, "Running study… (see log)")

    def _stop(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.status.set("Stop requested — finishing the current configuration…")

    def _build_results_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="4 · Results")
        bar = ttk.Frame(tab)
        bar.pack(fill="x", **PAD)
        ttk.Button(bar, text="Load results folder…", command=self._load_results) \
            .pack(side="left", **PAD)
        ttk.Button(bar, text="Open summary.md",
                   command=lambda: open_path(os.path.join(self.out_var.get(), "summary.md"))) \
            .pack(side="left", **PAD)
        ttk.Label(bar, text="Hold-out scores (fraud = positive class). CV F1 = 5-fold mean "
                            "on the development split; time_s = fit + predict per fold.") \
            .pack(side="left", **PAD)

        pane = ttk.PanedWindow(tab, orient="vertical")
        pane.pack(fill="both", expand=True, **PAD)
        cols = ("sampling", "model", "variant", "precision", "recall", "f1", "accuracy",
                "pr_auc", "cv_f1", "time_s")
        frame = ttk.Frame(pane)
        self.tree = ttk.Treeview(frame, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (90, 230, 80, 80, 80, 80, 80, 80, 110, 70)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
        ys = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        pane.add(frame, weight=1)
        self.res_viewer = ImageViewer(pane)
        pane.add(self.res_viewer, weight=3)

    def _fill_results(self, final_df, plots):
        self.tree.delete(*self.tree.get_children())
        for _, r in final_df.iterrows():
            self.tree.insert("", "end", values=(
                r.sampling, config.MODEL_LABELS.get(r.model, r.model), r.variant,
                f"{r.holdout_precision:.3f}", f"{r.holdout_recall:.3f}",
                f"{r.holdout_f1:.3f}", f"{r.holdout_accuracy:.3f}",
                f"{r.holdout_pr_auc:.3f}",
                f"{r.cv_f1_mean:.3f} ± {r.cv_f1_std:.3f}",
                f"{r.cv_fit_time_mean + r.get('cv_predict_time_mean', 0.0):.2f}"))
        order = sorted(plots, key=lambda p: (not os.path.basename(p).startswith("compare"),
                                             os.path.basename(p)))
        self.res_viewer.set_paths(order)

    def _load_results(self):
        folder = filedialog.askdirectory(initialdir=self.out_var.get())
        if not folder:
            return
        csv = os.path.join(folder, "final_comparison.csv")
        if not os.path.exists(csv):
            messagebox.showerror("Not a results folder", f"{csv} not found.")
            return
        plots_dir = os.path.join(folder, "plots")
        plots = [os.path.join(plots_dir, f) for f in os.listdir(plots_dir)
                 if f.endswith(".png")] if os.path.isdir(plots_dir) else []
        self.out_var.set(folder)
        self._fill_results(pd.read_csv(csv), plots)
        cfg = os.path.join(folder, "run_config.json")
        if os.path.exists(cfg):
            with open(cfg) as fh:
                c = json.load(fh)
            self.status.set(f"Loaded results: protocol={c.get('protocol')}, "
                            f"n_iter={c.get('n_iter')}, synthetic={c.get('synthetic')}")

    def _busy(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "A job is already running.")
            return True
        return False

    def _start(self, fn, status):
        self.stop_event.clear()
        self.status.set(status)
        self.run_btn.state(["disabled"])

        def wrapper():
            try:
                fn()
            except Exception:
                self.queue.put(("error", traceback.format_exc()))
            finally:
                self.queue.put(("finished", None))
        self.worker = threading.Thread(target=wrapper, daemon=True)
        self.worker.start()

    def _log(self, msg):
        self.queue.put(("log", str(msg)))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self.log_text.insert("end", payload + "\n")
                    self.log_text.see("end")
                elif kind == "progress":
                    done, total = payload
                    self.progress["maximum"] = max(total, 1)
                    self.progress["value"] = done
                elif kind == "eda_done":
                    self.eda_viewer.set_paths(payload)
                    self.status.set(f"EDA finished: {len(payload)} figures.")
                elif kind == "study_done":
                    final_df, plots = payload
                    if final_df is not None and not final_df.empty:
                        self._fill_results(final_df, plots)
                        self.nb.select(3)
                    self.status.set("Study finished — see the Results tab.")
                elif kind == "error":
                    self.log_text.insert("end", payload + "\n")
                    messagebox.showerror("Error", payload.strip().splitlines()[-1])
                    self.status.set("Error — see log.")
                elif kind == "finished":
                    self.run_btn.state(["!disabled"])
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)


if __name__ == "__main__":
    FraudApp().mainloop()
