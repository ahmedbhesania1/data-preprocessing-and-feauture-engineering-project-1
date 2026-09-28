# Data Profiler

**Data Preprocessing and Feature Engineering on Customer Churn Data**

A 5-hour data analysis project that takes a customer dataset through the full pre-machine-learning pipeline: acquiring data from four different sources, cleaning it, exploring it visually, and generating an automated profiling report. The end goal is to frame a machine learning problem: **predict whether a customer will churn based on purchase behaviour.**

---

## Table of Contents

1. [Objective](#objective)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Tech Stack](#tech-stack)
5. [Project Workflow](#project-workflow)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Data Profiling Report](#data-profiling-report)
8. [Key Insights](#key-insights)
9. [How to Run](#how-to-run)
10. [Project Structure](#project-structure)
11. [Learning Outcomes](#learning-outcomes)
12. [Author](#author)

---

## Objective

To understand, clean, transform and analyse a real-world style dataset before it is used for machine learning, with emphasis on data profiling, handling multiple data formats, and exploratory data analysis (EDA).

## Problem Statement

As a Junior Data Analyst at a consumer insights company, the task is to work with customer purchase data collected from multiple sources (CSV, JSON, SQL database and an API), perform preprocessing and profiling, and make the dataset ML-ready.

**ML framing:** supervised **binary classification**. Given a customer's profile and purchase behaviour, predict whether the customer will churn (`Yes`) or stay (`No`).

## Dataset

The main analysis uses `customer_churn_200.csv`: **200 customers, 11 columns**.

| Column | Type | Description |
|---|---|---|
| `CustomerID` | Text | Unique customer identifier (C001 to C200) |
| `Age` | Numeric | Customer age in years (18 to 65) |
| `Gender` | Categorical | Female or Male |
| `Income` | Numeric | Customer income (18,301 to 119,279) |
| `Purchases` | Numeric | Number of purchases (1 to 12) |
| `TotalSpent` | Numeric | Total amount spent (1,086 to 41,727) |
| `Visits` | Numeric | Number of visits (2 to 16) |
| `SupportCalls` | Numeric | Number of support calls (0 to 6) |
| `City` | Categorical | One of 7 cities: Hyderabad, Ahmedabad, Surat, Bengaluru, Delhi, Pune, Mumbai |
| `TenureMonths` | Numeric | Months as a customer (1 to 60) |
| `Churn` | Target | `Yes` (78 customers, 39%) or `No` (122 customers, 61%) |

**Data quality before cleaning:** no duplicate rows; missing values in `Age` (6), `Gender` (3) and `Income` (5).

## Tech Stack

| Tool | Use |
|---|---|
| Python 3 | Programming language |
| Pandas | Loading, cleaning and transforming data |
| NumPy | Tensor examples and numerical work |
| Matplotlib and Seaborn | Data visualization |
| MySQL + `mysql-connector-python` | Reading records from a SQL table |
| Requests | Fetching data from a REST API |
| YData Profiling 4.18.4 | Automated profiling report |
| Jupyter Notebook | Development environment |

## Project Workflow

All work is in [`main.ipynb`](main.ipynb), organised into five parts.

### Part A: Fundamentals and Tensors
Theory on data analysis, the data science project lifecycle and the churn problem statement, plus NumPy examples of tensors of rank 0 to 3:

| Rank | Type | NumPy shape in the notebook |
|---|---|---|
| 0D | Scalar | `()` |
| 1D | Vector | `(3,)` |
| 2D | Matrix | `(2, 3)` |
| 3D | Tensor | `(2, 2, 2)` |

### Part B: Data Acquisition

| Source | Method |
|---|---|
| CSV | `pd.read_csv("customer_churn_200.csv")` |
| JSON | `pd.read_json("customer_churn_200.json")` |
| SQL (MySQL) | `mysql.connector.connect(...)` then `pd.read_sql_query("select * from customer_data", conn)` |
| REST API | `requests.get(...)` on a Postman mock endpoint, then `pd.DataFrame(response.json()["data"])` |

The CSV file is used for the rest of the analysis.

### Part C: Data Understanding and Cleaning
- **Exploration:** `.head()`, `.info()`, `.describe().T`, `.shape`
- **Missing values:** `Age` 6, `Gender` 3, `Income` 5
- **Duplicates:** 0 found; `drop_duplicates()` still applied for practice
- **Imputation:**
  - `Age`: filled with the mean, then converted to integer
  - `Gender`: filled with the mode, then whitespace stripped and capitalised
  - `Income`: filled with the mean (70,696.35)

### Part D: Exploratory Data Analysis
- **Univariate:** histograms, KDE plots and box plots of `Age`, `Income` and `Purchases`
- **Bivariate:** `Gender` vs `Purchases`, and `Income` distribution by `Churn`
- **Multivariate:** correlation heatmap and pair plot of `Age`, `Income`, `Purchases` and `TotalSpent`

### Part E: Data Profiling
```python
from ydata_profiling import ProfileReport

profile = ProfileReport(df_csv)
profile.to_file(output_file="ProfileReport.html")
```

## Exploratory Data Analysis

### Univariate: Age, Income and Purchases

![Univariate box plots](screenshots/04_eda_univariate_boxplots.png)

All three variables are spread fairly evenly with no outliers flagged by the box plots. Age has a median of 42 (IQR 31 to 54), income a median of about 71,600 (IQR 45,700 to 96,200), and purchases a median of 6 (IQR 4 to 8).

### Bivariate: Gender vs Purchases

![Gender vs purchases](screenshots/05_eda_gender_vs_purchases.png)

Male and female customers have the same median (6) and the same interquartile range (4 to 8), so gender does not appear to influence purchase volume.

### Bivariate: Income vs Churn

![Income by churn status](screenshots/06_eda_income_vs_churn.png)

The income distributions of churned and retained customers overlap heavily. Churners are slightly more concentrated around 70,000, while retained customers show two mild peaks near 40,000 and 85,000, but there is no clear separation. Income alone is a weak predictor of churn.

### Multivariate: Correlation Heatmap

![Correlation heatmap](screenshots/07_eda_correlation_heatmap.png)

`Purchases` and `TotalSpent` are strongly correlated (r = 0.71). Every other pair is weak (absolute r below 0.15); the largest is `Visits` vs `SupportCalls` (r = -0.14).

## Data Profiling Report

The full interactive report is in [`ProfileReport.html`](ProfileReport.html) (generated in about 12.7 seconds).

### Overview

The dataset is clean after preprocessing: no missing cells and no duplicate rows.

![Profiling overview](screenshots/01_overview.png)

### Alerts

The report raised 5 alerts:

| Alert | Type |
|---|---|
| `Purchases` is highly correlated with `TotalSpent` | High correlation |
| `TotalSpent` is highly correlated with `Purchases` | High correlation |
| `CustomerID` has unique values | Unique |
| `TotalSpent` has unique values | Unique |
| `SupportCalls` has 31 (15.5%) zeros | Zeros |

![Profiling alerts](screenshots/02_alerts.png)

### Variable Detail: Income

![Income variable](screenshots/03_income_variable.png)

`Income` has 196 distinct values, no zeros or negatives, and a fairly even spread from 18,301 to 119,279. Its mean equals the value used to fill the 5 missing entries (70,696.35), which is why that value appears 5 times in the report.

## Key Insights

- **Churn is moderately imbalanced:** 39% of customers churned. Use precision, recall and F1 rather than accuracy alone when modelling.
- **Purchases and TotalSpent overlap** (r = 0.71). Keep one, or derive average spend per purchase, to limit multicollinearity.
- **CustomerID is an identifier, not a feature.** All 200 values are unique, so drop it before modelling.
- **No single variable separates churners from stayers.** Income and gender show almost no relationship with purchases or churn, so the signal probably comes from combinations such as tenure, support calls and city.
- **15.5% of customers never called support.** This group is meaningful and could become a feature.
- **Data is clean and well-behaved:** no outliers in the box plots and only 14 missing cells out of 2,200 (0.6%) before imputation.

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```
2. Install dependencies
   ```bash
   pip install pandas numpy matplotlib seaborn requests mysql-connector-python ydata-profiling jupyter
   ```
3. Update the file paths in `main.ipynb` (the notebook uses absolute Windows paths) so they point to `customer_churn_200.csv` and `customer_churn_200.json` in your copy of the repository.
4. *(Optional, for the SQL step)* Create a MySQL database named `customer_churn` with a table `customer_data`, and adjust the host, user and password in the connection cell.
5. Launch Jupyter and run all cells from top to bottom
   ```bash
   jupyter notebook main.ipynb
   ```
6. Open `ProfileReport.html` in any browser to view the profiling report.

## Project Structure

```
.
├── README.md
├── main.ipynb                     # full analysis notebook (Parts A to E)
├── customer_churn_200.csv         # main dataset
├── customer_churn_200.json        # same data in JSON format
├── ProfileReport.html             # YData Profiling report
├── Data_Profiler_Project_Documentation.pdf
└── screenshots/
    ├── 01_overview.png
    ├── 02_alerts.png
    ├── 03_income_variable.png
    ├── 04_eda_univariate_boxplots.png
    ├── 05_eda_gender_vs_purchases.png
    ├── 06_eda_income_vs_churn.png
    └── 07_eda_correlation_heatmap.png
```

## Learning Outcomes

- Loaded datasets from CSV, JSON, MySQL and a REST API
- Understood the role of tensors in machine learning
- Cleaned data by handling duplicates, missing values and data types
- Carried out univariate, bivariate and multivariate EDA with visualizations
- Generated an automated profiling report to support decisions
- Framed a machine learning problem from raw data

## Author

**Ahmed Bhesania**
