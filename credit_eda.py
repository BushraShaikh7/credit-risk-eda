import matplotlib.pyplot as plt
import pandas as pd 

# ============================================================
# CREDIT RISK EDA PROJECT
# Dataset: Give Me Some Credit (Kaggle) - 150,000 people
# Question: Who is likely to default on a loan?
# Tools: Python (this file), SQL, Power BI
# ============================================================

# Load the dataset from the GiveMeSomeCredit folder
dataframe = pd.read_csv("GiveMeSomeCredit/cs-training.csv")

# First look at the data - print the whole table
print(dataframe)

# Check how many rows and columns we have (150,000 rows, 12 columns)
print(dataframe.shape)

# Check all column names so we know what variables we are working with
print(dataframe.columns)

# Check how many missing values exist in each column
# Finding: MonthlyIncome has 29,731 missing, NumberOfDependents has 3,924 missing
print(dataframe.isnull().sum())

# ============================================================
# STAGE 1 - DATA CLEANING
# Problem: Two columns had missing values which would cause
# errors in analysis. We need to fill them before proceeding.
# ============================================================

# Chart 1 - Income Distribution (used to decide how to fill missing values)
# We plotted income first to understand its shape before filling blanks
# Filtered to <20000 to remove extreme outliers that squash the chart
# Finding: Income is RIGHT SKEWED - most people earn $3,000-5,000
# but a few earn much more, pulling the mean upward
# Conclusion: Use MEDIAN not MEAN to fill missing values
dataframe[dataframe["MonthlyIncome"]<20000]["MonthlyIncome"].hist(bins=50)
plt.title("Distribution Of Monthly Income")
plt.xlabel("MonthlyIncome")
plt.ylabel("NumberOfPeople")
plt.savefig("income_distribution.png")
plt.close()

# Fill missing MonthlyIncome values with median
# Reason: Income is right skewed so median is more representative than mean
# Mean would be pulled up by high earners and misrepresent typical income
dataframe["MonthlyIncome"] = dataframe["MonthlyIncome"].fillna(dataframe["MonthlyIncome"].median())

# Confirm no missing values remain in MonthlyIncome - should print 0
print(dataframe["MonthlyIncome"].isnull().sum())
print(dataframe["MonthlyIncome"])

# Fill missing NumberOfDependents values with median
# Same reasoning - median is safer than mean for skewed data
dataframe["NumberOfDependents"] = dataframe["NumberOfDependents"].fillna(dataframe["NumberOfDependents"].median())

# Confirm no missing values remain in NumberOfDependents - should print 0
print(dataframe["NumberOfDependents"].isnull().sum())
print(dataframe["NumberOfDependents"])

# ============================================================
# STAGE 2 - EXPLORATORY DATA ANALYSIS (EDA)
# Now that data is clean we explore patterns and ask:
# What factors are linked to defaulting on a loan?
# We split data into two groups and compare visually
# ============================================================

# Chart 2 - Default vs Non Default (Target Variable Overview)
# Shows how many people defaulted vs did not default
# 0 = did not default, 1 = defaulted
# Finding: 93% did not default, only 7% defaulted
# Problem: CLASS IMBALANCE - a lazy model could predict 
# "no default" for everyone and still be 93% accurate but useless
dataframe["SeriousDlqin2yrs"].value_counts().plot(kind="bar")
plt.title("Default vs didn't Default")
plt.xlabel("Default Status")
plt.ylabel("Number Of People")
plt.savefig("default_vs_nondefault.png")
plt.close()

# Split data into two groups for comparison in all remaining charts
# defaulted = 10,026 people who did not repay their loan
# not_defaulted = 139,974 people who repaid successfully
defaulted = dataframe[dataframe["SeriousDlqin2yrs"]==1]
not_defaulted = dataframe[dataframe["SeriousDlqin2yrs"]==0]

# Chart 3 - Age Distribution by Default Status
# Question: Does age affect likelihood of defaulting?
# We plot age histograms for both groups on the same chart
# alpha=0.5 makes bars transparent so both groups are visible when overlapping
# Finding: Defaults peak at age 40-60, younger people default less
# Conclusion: Age is a WEAK predictor - defaults spread across all ages
defaulted["age"].hist(bins=20, alpha=0.5, label="defaulted")
not_defaulted["age"].hist(bins=20, alpha=0.5, label="non_defaulted")
plt.legend()
plt.title("Age Distribution by Default Status")
plt.xlabel("Age")
plt.ylabel("Number of People")
plt.savefig("age_vs_default.png")
plt.close()

# Chart 4 - Debt Ratio vs Default
# Question: Do people with higher debt default more?
# DebtRatio = monthly debt payments divided by monthly income
# Filtered to <2 to remove extreme outliers (some have ratio of 100,000+)
# Finding: Both groups have similar debt ratio distributions
# Conclusion: Debt ratio is NOT a strong predictor of default
# This was surprising - went against our initial intuition
defaulted[defaulted["DebtRatio"]<2]["DebtRatio"].hist(bins=50, alpha=0.5, label="Defaulted")
not_defaulted[not_defaulted["DebtRatio"]<2]["DebtRatio"].hist(bins=50, alpha=0.5, label="Not Defaulted")
plt.legend()
plt.title("Debt Ratio vs Default")
plt.xlabel("DebtRatio")
plt.ylabel("NumberOfPeople")
plt.savefig("debtratio_vs_default.png")
plt.close()

# Chart 5 - Monthly Income vs Default
# Question: Do lower earners default more than higher earners?
# Filtered to <20000 to remove extreme outliers
# Finding: Both groups peak around $5,000 monthly income
# The spike at exactly $5,000 is our median fill showing up
# Conclusion: Monthly income alone is NOT a strong predictor of default
defaulted[defaulted["MonthlyIncome"]<20000]["MonthlyIncome"].hist(bins=50, alpha=0.5, label="defaulted")
not_defaulted[not_defaulted["MonthlyIncome"]<20000]["MonthlyIncome"].hist(bins=50, alpha=0.5, label="not defaulted")
plt.legend()
plt.title("Monthly Income vs Default")
plt.xlabel("MonthlyIncome")
plt.ylabel("NumberOfPeople")
plt.savefig("income_vs_default.png")
plt.close()

# Chart 6 - Late Payments vs Default
# Question: Does missing payments predict default?
# NumberOfTimes90DaysLate = how many times someone was 90+ days late
# Filtered to <10 to remove extreme outliers
# Finding: Non-defaulters are almost entirely at 0 late payments
# Defaulters start appearing as soon as late payments increase
# Conclusion: Late payments is the STRONGEST predictor of default
# Even 1-2 missed payments is a serious warning signal
# This makes business sense - someone already missing payments
# is showing they cannot or will not repay
defaulted[defaulted["NumberOfTimes90DaysLate"]<10]["NumberOfTimes90DaysLate"].hist(bins=50, alpha=0.5, label="defaulted")
not_defaulted[not_defaulted["NumberOfTimes90DaysLate"]<10]["NumberOfTimes90DaysLate"].hist(bins=50, alpha=0.5, label="not defaulted")
plt.legend()
plt.title("Late Payment vs Default")
plt.xlabel("NumberOfTimes90DaysLate")
plt.ylabel("NumberOfPeople")
plt.savefig("latepayment_vs_default.png")
plt.close()