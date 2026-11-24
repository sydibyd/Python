# Flatmates Bill

A web app that splits a shared bill between flatmates according to how many days each one stayed in the house during the billing period, and generates a PDF report of the split.

The original object design is in [`design.txt`](design.txt).

## How the split works

```
pays = bill amount × (days in house ÷ total days of all flatmates)
```

Example: a 120.00 bill for March 2026, Saeii stayed 20 days and Mahi 25 days (45 days in total):

| Flatmate | Days | Share | Pays   |
|----------|-----:|------:|-------:|
| Saeii    |   20 | 44.4% |  53.33 |
| Mahi     |   25 | 55.6% |  66.67 |
| **Total**|   45 |  100% | 120.00 |

Amounts are rounded to the cent. Any leftover cent goes to the flatmate with the most days, so the shares always add up to the bill.

## Features

- Bill amount, currency and period
- Two or more flatmates, each with a name and days in the house
- Live report preview with each person's share and the cost per day
- PDF report download (built in the browser with [jsPDF](https://github.com/parallax/jsPDF)), with a file name you can change
- Input checks, saved inputs (browser storage), phone layout and dark mode

## Objects in the code

The classes in `index.html` follow the design:

| Design      | Code                                                  |
|-------------|-------------------------------------------------------|
| `Bill`      | `class Bill { amount, period }`                       |
| `Flatmate`  | `class Flatmate { name, days_in_house, pays(bill) }`  |
| `PdfReport` | `class PdfReport { filename, generate(flatmates, bill) }` |

## Run it

It is a single HTML file with no build step. Open `index.html` in a browser, or serve the folder:

```bash
python3 -m http.server 8000
```

then visit http://localhost:8000. An internet connection is needed for the fonts and the jsPDF library (loaded from cdnjs).
