import pandas as pd
from lifelines import CoxPHFitter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('./hotel_booking.csv')

data['reservation_status_date'] = pd.to_datetime(data['reservation_status_date'], errors='coerce')
data['arrival_date'] = pd.to_datetime(
    data['arrival_date_year'].astype(str) +'-' +
    data['arrival_date_month'].astype(str) + '-'+
    data['arrival_date_day_of_month'].astype(str),
    format='%Y-%B-%d', errors='coerce'
)

data['booking_date'] = data['arrival_date'] - pd.to_timedelta(data['lead_time'], unit='d')


data['days_between_booking_and_cancellation'] = (
    data['reservation_status_date'] - data['booking_date']
).dt.days

data.to_csv('hotel_bookings_modified.csv')
canceled_bookings = data[data['is_canceled'] == 1]

# Select relevant features for the model
features = canceled_bookings[['lead_time', 'arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']]
features = pd.get_dummies(features, columns=['arrival_date_month'], drop_first=True)
target = canceled_bookings['days_between_booking_and_cancellation']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print("Random Forest Regressor Score:", model.score(X_test, y_test))
# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

"""COX'S PROPORTIONAL HAZARDS"""


canceled_bookings['event'] = canceled_bookings['is_canceled']  # Use the is_canceled column as the event indicator
survival_data = canceled_bookings[['days_between_booking_and_cancellation', 'event', 'lead_time', 'arrival_date_year', 'arrival_date_month']]
survival_data = pd.get_dummies(survival_data, columns=['arrival_date_month'], drop_first=True)

# Initialize and fit the Cox model
cph = CoxPHFitter()
cph.fit(survival_data, duration_col='days_between_booking_and_cancellation', event_col='event')

# Display summary of results
cph.print_summary()

# Plot the survival curves for interpretation
cph.plot()