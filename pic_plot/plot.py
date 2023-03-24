from matplotlib import pyplot as plt
import numpy as np

#plot a bar graph

def plot_bar_graph(x, y, x_label, y_label, title):
    plt.bar(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

#plot a line graph

def plot_line_graph(x, y, x_label, y_label, title):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()



# data1 = [60.01, 62.14, 60.51, 61.42, 61.49]
# data2 = [65.71, 69.33, 64.33, 62.41, 65.22]
# data3 = [67.13, 70.02, 66.67, 67.34, 70.11]
# data4 = [90.14, 80.84, 87.26, 81.44, 89.05]
# # plot data1, data2, data3, data4 in one graph
# x = np.arange(5)
# plt.bar(x, data1, width=0.2, color='b', align='center')
# plt.bar(x+0.2, data2, width=0.2, color='g', align='center')
# plt.bar(x+0.4, data3, width=0.2, color='r', align='center')
# plt.bar(x+0.6, data4, width=0.2, color='y', align='center')
#
# # show name of each bar
# plt.xticks(x+0.3, ('linear-svm', 'AB', 'KNN', 'DT', 'RF'))
#
# plt.xlabel('Method')
# plt.ylabel('Accuracy')
# plt.title('Accuracy of different image preprocessing methods')
# plt.legend(['No preprocessing', 'Crop only', 'Contrast enhancement only', 'Crop and contrast enhancement'])
# # save the graph
# plt.savefig('plot.png', dpi=300)
#
# plt.show()


# 0.01, 0.05, 0.1, 0.2, 1
data1 = [80.32, 85.45, 88.58, 88.34, 90.89]
data2 = [67.53, 75.32, 78.58, 80.56, 86.75]

# plot data1 and data2 in one line graph
x = np.arange(5)
plt.bar(x, data2, width=0.3, color='b', align='center')
plt.bar(x+0.3, data1, width=0.3, color='y', align='center')

plt.xticks(x+0.3, ('0.01', '0.05', '0.1', '0.2', '1'))
plt.legend(['FCL', 'ResNet18'])
plt.xlabel('labeled data proportion')
plt.ylabel('Accuracy')
plt.title('Accuracy of different proportion of labeled data')
plt.savefig('plot2.png', dpi=300)
plt.show()
