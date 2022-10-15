import numpy as np

# Mean Squared Error function
def mean_squared_error(y_true, y_pred, squared=True):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    error = np.mean((y_true - y_pred) ** 2)
    if squared:  # if squared is True, return the Mean Squared Error (MSE)
        return error
    return np.sqrt(
        error
    )  # if squared is False, return the Root Mean Squared Error (RMSE)


# Mean Absolute Error function
def mean_absolute_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.abs(y_true - y_pred))


# R2 Score function
def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # calculating the residuals and total sum of squares
    SSres = np.sum((y_true - y_pred) ** 2)
    SStot = np.sum((y_true - np.mean(y_true)) ** 2)

    # applying the formula
    r2_score = 1 - np.divide(
        SSres, SStot, out=np.zeros_like(SSres, dtype=type(SStot)), where=SStot != 0
    )  # using np.divide to avoid ZeroDivisionError

    return r2_score
