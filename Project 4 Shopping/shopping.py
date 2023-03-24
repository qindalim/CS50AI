import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []

    month_dict = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5, \
        'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}

    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        for i in reader:
            row = []
            row.append(int(i["Administrative"]))
            row.append(float(i["Administrative_Duration"]))
            row.append(int(i["Informational"]))
            row.append(float(i["Informational_Duration"]))
            row.append(int(i["ProductRelated"]))
            row.append(float(i["ProductRelated_Duration"]))
            row.append(float(i["BounceRates"]))
            row.append(float(i["ExitRates"]))
            row.append(float(i["PageValues"]))
            row.append(float(i["SpecialDay"]))
            row.append(int(month_dict[i["Month"]]))
            row.append(int(i["OperatingSystems"]))
            row.append(int(i["Browser"]))
            row.append(int(i["Region"]))
            row.append(int(i["TrafficType"]))
            row.append(int(i["VisitorType"] == 'Returning_Visitor'))
            row.append(int(i["Weekend"] == "TRUE"))

            evidence.append(row)
            labels.append(int(i["Revenue"] == "TRUE"))

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    negative = 0
    positive = 0
    true_positive = 0
    true_negative = 0

    for i in range(len(labels)):
        if labels[i] == 0:
            negative += 1
            if labels[i] == predictions[i]:
                true_negative += 1
        else:
            positive += 1
            if labels[i] == predictions[i]:
                true_positive += 1

    sensitivity = true_positive / positive
    specificity = true_negative / negative 
    
    return (sensitivity, specificity)



if __name__ == "__main__":
    main()
