import csv
import random
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import utils

def prediction_accuracy(predictions, true_labels):

    if len(predictions) != len(true_labels):
        print("Predictions array not same length as True_labels array. Something went wrong here.")
        return

    correct_predictions = 0
    for i in range(0, len(predictions)):
        if int(predictions[i]) == int(true_labels[[i]]):
            correct_predictions += 1

    acc = correct_predictions / len(predictions)
    return acc


def downsample_for_balanced_acc(predictions, true_labels, random_seed):
    """
    :param predictions: tensor with predictins made by the model, shape = (x, )
    :param true_labels: tensor with the true labels, shape = (x, 1)
    :param random_seed: seed used for random downsampling
    :return: predictions tensor (shape = (x, )) and true_labels tensor (shape = (x, 1)), both beeing downsampled to
                the numbers of occurences of the rarest class and sorted by class label
    """

    random.seed(a=random_seed)

    unique, counts = np.unique(true_labels, return_counts=True)
    min_occurrence = np.amin(counts)  # descripes how often the rarest class is occuring
    print("Occurrence of rarest class: " + str(min_occurrence))

    new_preds = np.array([])
    new_true_labels = np.array([])

    # for every class:
    for i in range(0, 10):  # index represents the class label, so 10 = amount of classes
        current_preds = []  # must be a normal list for random.sample() which is used later

        # loop through true_labels
        for j in range(0, len(true_labels)):

            # if true_labels index == current class -> collect prediction(index)
            if i == true_labels[[j]]:
                current_preds.append(int(predictions[j]))

        # only considere min_occurrence of the predictions
        current_preds_sampled = random.sample(current_preds, min_occurrence)
        new_preds = np.append(new_preds, current_preds_sampled)

        # since predictions are now sorted by class, we can easily add the current class index as true_label
        current_true_labels = [i] * min_occurrence
        new_true_labels = np.append(new_true_labels, current_true_labels)

    new_true_labels = new_true_labels.reshape(-1, 1)

    prediction_tensor = tf.constant(new_preds)
    true_tensor = tf.constant(new_true_labels)

    return prediction_tensor, true_tensor


def Show_Confusion_Matrix(class_names, model, test_acc, test_images, test_labels, fold=1):
    predictions = model.predict(test_images)
    predictions = tf.argmax(predictions, axis=-1)

    write_predictions_to_csv(predictions, test_labels, fold)

    plot_confusion_matrix(class_names, test_acc, predictions, test_labels)
    return


def Show_Cross_Confusion_Matrix(class_names, test_acc, predictions, true_labels):

    plot_confusion_matrix(class_names, test_acc, predictions, true_labels)
    return


def plot_confusion_matrix(class_names, test_acc, predictions, true_labels):

    balanced_pred, balanced_true = downsample_for_balanced_acc(predictions, true_labels, random_seed=42069)
    balanced_acc = prediction_accuracy(balanced_pred, balanced_true)

    print("----")
    print("Accuracy: " + str(round(test_acc, 3)))
    print("Balanced Accuracy: " + str(round(balanced_acc, 3)))
    print("----")

    cm = tf.math.confusion_matrix(true_labels, predictions)
    cm_balanced = tf.math.confusion_matrix(balanced_true, balanced_pred)

    cm = cm / cm.numpy().sum(axis=1)[:, tf.newaxis]
    cm_balanced = cm_balanced / cm_balanced.numpy().sum(axis=1)[:, tf.newaxis]

    cm = np.transpose(cm)
    cm_balanced = np.transpose(cm_balanced)

    # Plot settings for confusion matrix with standard accuracy
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names, fmt='.2f', cmap="binary",
                vmin=0, vmax=1, linewidths=0.005, linecolor='#B7B7B7', cbar_kws={'label': 'Genauigkeit'})
    # plt.colorbar().set_label("Ausgewogene Genauigkeit", size=11)
    plt.ylabel("Vorhersage", fontsize=11)
    plt.xlabel("Echte Klasse", fontsize=11)
    fig.suptitle(("Genauigkeit = " + str(round(test_acc, 3))), fontsize=13)
    plt.subplots_adjust(left=0.27, bottom=0.352, right=0.905, top=0.92, wspace=0.2, hspace=0.2)
    fig.savefig("results/ConfusionMatrix_Normal.png", bbox_inches='tight')
    # plt.show()

    # Same again but for balanced accuracy
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm_balanced, annot=True, xticklabels=class_names, yticklabels=class_names, fmt='.2f', cmap="binary",
                vmin=0, vmax=1, linewidths=0.005, linecolor='#B7B7B7', cbar_kws={'label': 'Ausgewogene Genauigkeit'})
    plt.ylabel("Vorhersage", fontsize=11)
    plt.xlabel("Echte Klasse", fontsize=11)
    fig.suptitle(("Ausgewogene Genauigkeit = " + str(round(balanced_acc, 3))), fontsize=13)
    plt.subplots_adjust(left=0.27, bottom=0.352, right=0.905, top=0.92, wspace=0.2, hspace=0.2)
    fig.savefig("results/ConfusionMatrix_Balanced.png", bbox_inches='tight')
    # plt.show()

    return


def evaluate_epochs(histories, fold=1):
    all_loss_arr, all_acc_arr = np.array(histories[0].history['loss']), np.array(histories[0].history['accuracy'])
    all_val_loss_arr, all_val_acc_arr = np.array(histories[0].history['val_loss']), np.array(histories[0].history['val_accuracy'])
    counter = 0

    for history in histories:
        if counter > 0:
            loss_arr, acc_arr = np.array(history.history['loss']), np.array(history.history['accuracy'])
            val_loss_arr, val_acc_arr = np.array(history.history['val_loss']), np.array(history.history['val_accuracy'])

            all_loss_arr = np.add(all_loss_arr, loss_arr)
            all_acc_arr = np.add(all_acc_arr, acc_arr)
            all_val_loss_arr = np.add(all_val_loss_arr, val_loss_arr)
            all_val_acc_arr = np.add(all_val_acc_arr, val_acc_arr)
        counter += 1

    all_loss_arr, all_acc_arr = all_loss_arr / counter, all_acc_arr / counter
    all_val_loss_arr, all_val_acc_arr = all_val_loss_arr / counter, all_val_acc_arr / counter

    all_loss_arr, all_acc_arr = np.around(all_loss_arr, 3), np.around(all_acc_arr, 3)
    all_val_loss_arr, all_val_acc_arr = np.around(all_val_loss_arr, 3), np.around(all_val_acc_arr, 3)

    print("best mean-loss value: " + str(min(all_loss_arr)))
    print("best mean-loss epoch: " + str(all_loss_arr.argmin()+1))
    print("best mean-acc value: " + str(max(all_acc_arr)))
    print("best mean-acc epoch: " + str(all_acc_arr.argmax()+1))
    print("------")
    print("best mean-val-loss value: " + str(min(all_val_loss_arr)))
    print("best mean-val-loss epoch: " + str(all_val_loss_arr.argmin() + 1))
    print("best mean-val-acc value: " + str(max(all_val_acc_arr)))
    print("best mean-val-acc epoch: " + str(all_val_acc_arr.argmax() + 1))

    write_results_to_csv(all_loss_arr, all_acc_arr, all_val_loss_arr, all_val_acc_arr, fold)
    plot_accuracy_over_epochs(all_acc_arr, all_val_acc_arr)
    plot_loss_over_epochs(all_loss_arr, all_val_loss_arr)

    return


def EvaluteCrossValidation(histories, accs, losses, predictions_arr, test_labels_arr, class_names):
    total_acc, total_loss = 0, 0
    counter = 0
    for acc in accs:
        total_acc += acc
        counter += 1
    counter = 0
    for loss in losses:
        total_loss += loss
        counter += 1

    total_acc, total_loss = total_acc / counter, total_loss / counter

    print("---------------------------------------------------")
    print("Cross-Validation Results:\n")
    print("Average Accuracy: " + str(round(total_acc, 2)))
    print("Average Loss: " + str(round(total_loss, 2)))
    print("-----\nEpoch Results:")
    evaluate_epochs(histories)

    Show_Cross_Confusion_Matrix(class_names, total_acc, predictions_arr, test_labels_arr)

    return


def write_results_to_csv(loss_arr, acc_arr, val_loss_arr, val_acc_arr, fold):
    result_csv = "results/fold_results.csv"

    resultCSV = open(result_csv, 'w+', encoding='UTF8', newline='')
    result_writer = csv.writer(resultCSV)

    csv_header = ["Fold", "Epoch", "Accuracy", "Val-Accuracy", "Loss", "Val-Loss"]
    result_writer.writerow(csv_header)

    for i in range(loss_arr.size):
        values = fold, i+1, acc_arr[i], val_acc_arr[i], loss_arr[i], val_loss_arr[i]
        result_writer.writerow(values)

    return

def write_predictions_to_csv(predictions, true_lables, fold):
    predictions_csv = "results/fold_predictions.csv"
    pred_arr = predictions.numpy()

    predictionsCSV = open(predictions_csv, 'w+', encoding='UTF8', newline='')
    predictions_writer = csv.writer(predictionsCSV)

    csv_header = ["Fold", "Prediction", "True_Lable"]
    predictions_writer.writerow(csv_header)

    for i in range(len(predictions)):
        values = fold, int(pred_arr[i]), int(true_lables[i])
        predictions_writer.writerow(values)

    return


def plot_accuracy_over_epochs(acc_arr, val_acc_arr, std_acc=None, std_val_acc=None):
    epochs = range(1, len(acc_arr) + 1)
    fig = plt.figure(figsize=(10, 7))

    plt.plot(epochs, acc_arr, color='blue', label='Trainingsgenauigkeit')
    plt.plot(epochs, val_acc_arr, color='#FFA359', label='Testgenauigkeit')  # Hexcode for orange

    if std_acc is not None and std_val_acc is not None:
        acc_error = std_acc
        val_acc_error = std_val_acc
        plt.errorbar(epochs, acc_arr, yerr=acc_error, fmt='none', elinewidth=1.0, color='blue')
        plt.errorbar(epochs, val_acc_arr, yerr=val_acc_error, fmt='none', elinewidth=1.0, color='#FFA359')

    plt.suptitle('Trainings- und Testgenauigkeit', fontsize=13)
    plt.xlabel('Epoche', fontsize=11)
    plt.ylabel('Genauigkeit', fontsize=11)
    plt.subplots_adjust(left=0.11, bottom=0.11, right=0.93, top=0.92, wspace=0.2, hspace=0.2)
    plt.ylim([0, 1])
    plt.xlim(0, len(acc_arr)+1)
    plt.legend()

    fig.savefig("results/AccuracyPlot.png", bbox_inches='tight')
    # plt.show()
    return


def plot_loss_over_epochs(loss_arr, val_loss_arr, std_loss=None, std_val_loss=None):
    epochs = range(1, len(loss_arr) + 1)
    fig = plt.figure(figsize=(10, 7))

    plt.plot(epochs, loss_arr, color='blue', label='Trainingsverlust')
    plt.plot(epochs, val_loss_arr, color='#FFA359', label='Testverlust')  # Hexcode for orange

    if std_loss is not None and std_val_loss is not None:
        loss_error = std_loss
        val_loss_error = std_val_loss
        plt.errorbar(epochs, loss_arr, yerr=loss_error, fmt='none', elinewidth=1.0, color='blue')
        plt.errorbar(epochs, val_loss_arr, yerr=val_loss_error, fmt='none', elinewidth=1.0, color='#FFA359')

    plt.suptitle('Trainings- und Testverlust', fontsize=13)
    plt.subplots_adjust(left=0.11, bottom=0.11, right=0.93, top=0.92, wspace=0.2, hspace=0.2)
    plt.xlabel('Epoche', fontsize=11)
    plt.ylabel('Verlust', fontsize=11)
    plt.xlim(0, len(loss_arr) + 1)
    plt.legend()

    fig.savefig("results/LossPlot.png", bbox_inches='tight')
    # plt.show()
    return


def ManualCrossVal_Eval(class_names, results_csv, predictions_csv, folds):
    csv_results = utils.ReadResultsFromCSV(results_csv, folds)

    acc_arr = csv_results[0]
    val_acc_arr = csv_results[1]
    loss_arr = csv_results[2]
    val_loss_arr = csv_results[3]
    std_acc = csv_results[4]
    std_val_acc = csv_results[5]
    std_loss = csv_results[6]
    std_val_loss = csv_results[7]

    test_acc = val_acc_arr[val_acc_arr.size - 1]

    # plot the results
    plot_accuracy_over_epochs(acc_arr, val_acc_arr, std_acc, std_val_acc)
    plot_loss_over_epochs(loss_arr, val_loss_arr, std_loss, std_val_loss)

    pred_tensor, true_lables = utils.ReadPredictionsFromCSV(predictions_csv)

    # plot confusion matrix
    # plot_confusion_matrix(class_names, test_acc, pred_tensor, true_lables)

    return
