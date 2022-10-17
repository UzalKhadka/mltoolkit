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


# Confusion Matrix Function
def confusion_matrix(y_true, y_pred, labels=None):
    assert len(y_true) == len(y_pred), "Length of y_true and y_pred must be equal"

    if labels is None or len(labels) == 0:
        labels = np.unique(
            y_true + y_pred
        )  # get all unique labels from both y_true and y_pred
    else:
        labels = np.unique(labels)

    len_labels = len(labels)  # number of unique labels
    # create a confusion matrix
    conf_matrix = np.zeros((len_labels, len_labels), dtype=int)

    # fill the confusion matrix
    for i in range(len(y_true)):  # for each y_true sample
        # iterating over all labels to find the index of y_true sample in labels
        for j in range(len_labels):
            if y_true[i] == labels[j]:  # check for match
                # iterating over all labels to find the index of y_pred sample in labels
                for k in range(len_labels):
                    # check for match and now for y_pred sample
                    if y_pred[i] == labels[k]:
                        # increment the value of confusion matrix at the index of y_true and y_pred sample
                        conf_matrix[j][k] += 1
                        break  # for optimization
                break  # for optimization

    return conf_matrix
