import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


data_info = pd.read_csv("csv_files/lending_club_info.csv")
df = pd.read_csv("csv_files/lending_club_loan_two.csv")
print(data_info)


# ----- Exploratory Data Analysis -----
print(df.info)
print(df.head())

print(f"The dataframe contains {df.shape[1]} columns and {df.shape[0]} rows.")


# A count plot being separated based on the loan_status
sns.countplot(x="loan_status", data=df)
plt.show()


# A histogram of the loan_amnt column.
plt.figure(figsize=(12, 4))
sns.distplot(df['loan_amnt'], kde=False, bins=40)
plt.xlim(0, 45000)
plt.ylabel("num people")
plt.show()
plt.savefig("graphs/check.png")


sns.scatterplot(x='installment', y='loan_amnt', data=df,)
plt.show()

# A boxplot showing the relationship between the loan_status and the Loan Amount.
sns.boxplot(x='loan_status', y='loan_amnt', data=df)
plt.show()


# Calculate the summary statistics for the loan amount, grouped by the loan_status.
print(df.groupby('loan_status')['loan_amnt'].describe())

# The unique possible grades and subgrades
print(sorted(df['grade'].unique()))
print(sorted(df['sub_grade'].unique()))


# A countplot per grade
sns.countplot(x="grade", data=df, hue="loan_status")  # According to the graph, most loans in grades A, B, C, D were fully paid, and in F, G not.
plt.show()

# A count plot per subgrade
plt.figure(figsize=(12, 4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade', data=df, order=subgrade_order, palette='coolwarm')
plt.show()

# A count plot per subgrade being separated based on the loan_status
plt.figure(figsize=(12, 4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade', data=df, order=subgrade_order, palette='coolwarm', hue="loan_status")
plt.show()


# Because F and G subgrades don't get paid back that often, I'll isloate those and recreate the countplot just for those subgrades.
f_and_g = df[(df["grade"] == "G") | (df["grade"] == "F")]
plt.figure(figsize=(12, 4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade', data=f_and_g, order=subgrade_order, hue='loan_status')
plt.show()


print(df['loan_status'].unique())

# Creating a new column called 'load_repaid' which will contain a 1 if the loan status was "Fully Paid" and a 0 if it was "Charged Off".
# -- Way 1 --
df['loan_repaid'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})
print(df[['loan_repaid', 'loan_status']])


# -- Way 2 --
# def assign_new_value(row):
#     if row["loan_status"] == "Charged Off":
#         return 0
#     elif row['loan_status'] == "Fully Paid":
#         return 1
#
#
# df['loan_repaid'] = df.apply(assign_new_value, axis=1)
# print(df[['loan_repaid', 'loan_status']])


pd.set_option('display.width', None)  # display the full width of the df while printing it

df_for_correlation = df.copy()  # A copy of the df for correlation changes
string_columns = df_for_correlation.select_dtypes(include='object').columns
for col_name in string_columns:
    df_for_correlation = df_for_correlation.drop(col_name, axis=1)

print(f"\ncorrelation values:\n{df_for_correlation.corr()}")

# A heatmap showing the correlation of the numeric features
plt.figure(figsize=(12, 7))
sns.heatmap(df_for_correlation.corr(), annot=True, cmap='viridis')
plt.ylim(10, 0)
plt.title("Correlation Heatmap")
plt.show()

# A bar plot showing the correlation of the numeric features to the new loan_repaid column
plt.figure(figsize=(12, 7))
df_for_correlation.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
plt.title("Correlation of numeric features to the new loan_repaid column")
plt.show()


# ----- Data PreProcessing -----
print(f"Dataframe length: {len(df)}")


# A Series that displays the total count of missing values per column.
print(f"The total count of missing values per column:\n{df.isnull().sum()}")

# Convert this Series to be in terms of percentage of the total DataFrame
print(f"The total count of missing values per column in percentage:\n{df.isnull().sum() / len(df) * 100}")


print(f"The number of all the unique values in the col 'emp_title': {df['emp_title'].nunique()}")

print(df['emp_title'].value_counts())


# Realistically there are too many unique job titles to try to convert this to a dummy variable feature. Let's remove that emp_title column.
df = df.drop('emp_title', axis=1)


# A count plot of the emp_length feature column (Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years)
print(sorted(df['emp_length'].dropna().unique()))

emp_length_order = ['< 1 year',
                    '1 year',
                    '2 years',
                    '3 years',
                    '4 years',
                    '5 years',
                    '6 years',
                    '7 years',
                    '8 years',
                    '9 years',
                    '10+ years']

plt.figure(figsize=(12, 4))
sns.countplot(x='emp_length', data=df, order=emp_length_order)
plt.show()


# Countplot of emp_length separating Fully Paid vs Charged Off
plt.figure(figsize=(12, 4))
sns.countplot(x="emp_length", data=df, order=emp_length_order, hue="loan_status")
plt.show()


# This still doesn't really inform us if there is a strong relationship between employment length
# and being charged off, what we want is the percentage of charge offs per category. Essentially informing us
# what percent of people per employment category didn't pay back their loan.
emp_charged_off = df[df['loan_status'] == "Charged Off"].groupby("emp_length").count()['loan_status']
emp_fully_paid = df[df['loan_status'] == "Fully Paid"].groupby("emp_length").count()['loan_status']
percentage_per_category_series = emp_charged_off / emp_fully_paid
print(f"\npercentage_per_category_series:\n{percentage_per_category_series}\n")

# Visualize it with a bar plot
plt.figure(figsize=(10, 8))
percentage_per_category_series.plot(kind="bar")
# plt.title("") # Something about percentage
plt.show()

# Charge off rates are extremely similar across all employment lengths, so I'll drop the emp_length column
df = df.drop('emp_length', axis=1)

# Revisit the DataFrame to see what feature columns still have missing data.
print(f"Check if there are columns with missing data:\n{df.isnull().sum()}")

# Reviewing the title column vs the purpose column. Is this repeated information?
print(f"Purpose col:\n{df['purpose'].head(10)}\n")
print(f"Title col:\n{df['title'].head(10)}\n")

# We can see that the title column is simply a string description of the purpose column, so let's drop the title column
df = df.drop('title', axis=1)


print(f"mort_acc value counts:\n{df['mort_acc'].value_counts()}\n")

# There are many ways we could deal with this missing data. We could attempt to build a simple model to fill it in,
# such as a linear model, we could just fill it in based on the mean of the other columns, or you could even bin
# the columns into categories and then set NaN as its own category. Let's review the other columsn to see which most
# highly correlates to mort_acc

corr_by_mort_acc = df_for_correlation.corr()['mort_acc'].sort_values().drop('mort_acc')
print(f"Correlation with the mort_acc column:\n{corr_by_mort_acc}\n")
#  The total_acc feature correlates with the mort_acc. Let's try this fillna() approach.

print(f"Mean of mort_acc column per total_acc:\n{df_for_correlation.groupby('total_acc').mean()['mort_acc']}")  # Group the dataframe by the total_acc and calculate the mean value for the mort_acc per total_acc entry.

# Let's fill in the missing mort_acc values based on their total_acc value. If the mort_acc is missing, then we will fill in that missing value with the mean value corresponding to its total_acc value from the Series we created above.
total_acc_avg = df_for_correlation.groupby('total_acc').mean()['mort_acc']

print(total_acc_avg[2.0])


def fill_mort_acc(total_acc,mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.

    total_acc_avg here should be a Series or dictionary containing the mapping of the
    groupby averages of mort_acc per total_acc values.
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


df_for_correlation['mort_acc'] = df_for_correlation.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

print(f"\nCheck if there are null values before using df.dropna():\n{df.isnull().sum()}\n")
df = df.dropna()
print(f"Check if there are null values after using df.dropna():\n{df.isnull().sum()}\n")


# ----- Categorical Variables and Dummy Variables -----
print(f"'term' column value_counts:\n{df['term'].value_counts()}")

# Or just use .map()
df['term'] = df['term'].apply(lambda term: int(term[:3]))  # Convert str type to int in the term column

# We already know grade is part of sub_grade, so just drop the grade feature.
df = df.drop('grade', axis=1)

# Convert the subgrade into dummy variables. Then concatenate these new columns to the original dataframe. Remember to drop the original subgrade column and to add drop_first=True to your get_dummies call.
subgrade_dummies = pd.get_dummies(df['sub_grade'], drop_first=True)

df = pd.concat([df.drop('sub_grade', axis=1), subgrade_dummies], axis=1)

# Converting these columns: ['verification_status', 'application_type','initial_list_status','purpose'] into dummy variables and concatenate them with the original dataframe. Then drop the original columns.
dummies = pd.get_dummies(df[['verification_status', 'application_type', 'initial_list_status', 'purpose']], drop_first=True)
df = df.drop(['verification_status', 'application_type', 'initial_list_status', 'purpose'], axis=1)
df = pd.concat([df, dummies], axis=1)


# Review the value_counts for the home_ownership column.
print(f"home_ownership value counts:\n{df['home_ownership'].value_counts()}")


# Convert these to dummy variables, but replace NONE and ANY with OTHER, so that we end up with just 4 categories, MORTGAGE, RENT, OWN, OTHER. Then concatenate them with the original dataframe. Then drop the original columns.
df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
dummies = pd.get_dummies(df['home_ownership'], drop_first=True)
df = df.drop('home_ownership', axis=1)
df = pd.concat([df, dummies], axis=1)

# Let's feature engineer a zip code column from the address in the data set. Create a column called 'zip_code' that extracts the zip code from the address column.
df['zip_code'] = df['address'].apply(lambda address: address[-5:])

# Now make this zip_code column into dummy variables using pandas. Concatenate the result and drop the original zip_code column along with dropping the address column.
dummies = pd.get_dummies(df['zip_code'], drop_first=True)
df = df.drop(['zip_code', 'address'], axis=1)
df = pd.concat([df, dummies], axis=1)

# drop issue_date feature because it is not necessary for our training.
df = df.drop('issue_d', axis=1)

# earliest_cr_line appears to be a historical time stamp feature. Extract the year from this feature using a .apply function, then convert it to a numeric feature. Set this new data to a feature column called 'earliest_cr_year'.Then drop the earliest_cr_line feature.
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))
df = df.drop('earliest_cr_line', axis=1)

# Drop the load_status column we created earlier, since its a duplicate of the loan_repaid column. We'll use the loan_repaid column since its already in 0s and 1s.
df = df.drop('loan_status', axis=1)


print(f"\nAll the columns after all the changes:\n{df.columns}")  # Print all the columns after all the changes.

string_columns_remaining = df.select_dtypes(['object']).columns
print(f"\nstring columns remaining: {string_columns_remaining}")

# Train test split
X = df.drop('loan_repaid', axis=1)  # input columns
y = df['loan_repaid'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Normalizing the Data
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Building the model
model = Sequential()

# input layer
model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Fit the model
model.fit(x=X_train,
          y=y_train,
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test),
          )


overall_model_data = pd.DataFrame(model.history.history)
# overall_model_data.to_csv(os.path.join(MODEL_PATH, HISTORY_FILE), index=False)
print(f"\noverall_model_data:\n {overall_model_data}")

overall_model_data[["accuracy", "val_accuracy"]].plot()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

overall_model_data[["loss", "val_loss"]].plot()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


print(f"\nModel Evaluation [val_loss, val_accuracy]: {model.evaluate(X_test, y_test, verbose=0)}")

predictions = model.predict(X_test)
predictions = predictions > 0.5  # if the model is more than 50% sure in his prediction, than give it to one class, otherwise, give it to the second class.


print(f"\nclassification_report:\n {classification_report(y_true=y_test, y_pred=predictions)}")

print(f"confusion_matrix:\n {confusion_matrix(y_true=y_test, y_pred=predictions)}\n")


print(f"-<> person status: {df.iloc[0]}")

# Test on new data: Given the customer below, would you offer this person a loan?
random_ind = random.randint(0, len(df))

new_customer = df.drop('loan_repaid', axis=1).iloc[random_ind]
new_customer = new_customer.astype(int)
new_customer = np.array(new_customer)
new_customer = new_customer.reshape(1, -1)
print(f"new_customer details:\n{new_customer}")

new_pred = model.predict(new_customer)
print(f"This person prediction status: {new_pred[0][0]}")

print(f"This person real status: {df.iloc[random_ind]['loan_repaid']}")
