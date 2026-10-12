"""Lightweight desktop interface backed by the same saved pipeline as the CLI."""

from pathlib import Path
from tkinter import StringVar, Tk, messagebox, ttk

from car_price_prediction.data import CATEGORICAL_FEATURES, load_listings
from car_price_prediction.model import load_model, predict_price


def launch_gui(model_path: Path, data_path: Path) -> None:
    """Open a small local interface. No training or network access occurs on startup."""
    model = load_model(model_path)
    data = load_listings(data_path)

    root = Tk()
    root.title("Historical Car Price Estimator")
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=18)
    frame.grid()

    inputs: dict[str, StringVar] = {}
    labels = [
        ("brand", "Model / مدل"),
        ("color", "Color / رنگ"),
        ("gear", "Transmission / دنده"),
        ("year", "Year / سال"),
        ("body", "Condition / وضعیت بدنه"),
        ("loc_1", "City / شهر"),
    ]
    for row, (field, label) in enumerate(labels):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
        items = sorted(str(value) for value in data[field].unique())
        variable = StringVar(value=items[0])
        ttk.Combobox(frame, textvariable=variable, values=items, state="readonly",
                     width=31).grid(row=row, column=1, pady=4)
        inputs[field] = variable

    ttk.Label(frame, text="Mileage (km) / کارکرد").grid(
        row=len(labels), column=0, sticky="w", pady=4
    )
    mileage = StringVar(value="0")
    ttk.Entry(frame, textvariable=mileage, width=34).grid(
        row=len(labels), column=1, pady=4
    )
    prediction = StringVar(value="Enter car details to estimate its price.")

    def calculate() -> None:
        try:
            values = {name: control.get() for name, control in inputs.items()}
            values["usage"] = mileage.get()
            price = predict_price(model, values)
            if price <= 0:
                prediction.set("Prediction is outside the plausible price range.")
                return
            text = f"Estimated price: {price:,.0f} toman (historical data)"
            matched = data
            for field in CATEGORICAL_FEATURES:
                matched = matched.loc[matched[field] == values[field]]
            if matched.empty:
                text += "\nCaution: no identical categorical combination in the dataset."
            prediction.set(text)
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=root)

    ttk.Button(frame, text="Estimate / محاسبه کن", command=calculate).grid(
        row=len(labels) + 1, column=0, columnspan=2, pady=(14, 10)
    )
    ttk.Label(frame, textvariable=prediction, wraplength=490).grid(
        row=len(labels) + 2, column=0, columnspan=2
    )
    ttk.Label(
        frame, text="Research demo only. The data is historical, not a live market quote.",
        wraplength=490
    ).grid(row=len(labels) + 3, column=0, columnspan=2, pady=(10, 0))
    root.mainloop()
