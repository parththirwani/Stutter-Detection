# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy*100)
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n",conf_matrix)