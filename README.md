# multi_label_wine_quality_classification:bar_chart:

This is a project where I practiced training various different multi-label wine quality classifiers with one vs. all method.

The workflow includes EDA (exploratory analysis, data visualization), data preprocessing (feature selection with chi-square test, oversampling minority classes with synthetic data, feature scaling), and trained data on different classification ML models (logistic regression, linear supported vector machine (SVM), kernel SVM, and K-NN)

**Feel free to click into the .ipynb notebook for detailed analysis.**


## EDA

The dataset is extremely skewed with minority class (i.e. wine quality) like '3' and '8' share less than 1% of the total population. We can see this by plotting a histogram on 'quality' column. 
![quality_count](https://user-images.githubusercontent.com/56566212/103471654-5d1b8200-4d48-11eb-819b-b04e8a6fd0be.png)

A clearer visualization of the correlations between features by plotting out a heatmap:
![corr_heat](https://user-images.githubusercontent.com/56566212/103471877-91dd0880-4d4b-11eb-9e69-e867528b231e.png)

Further visualize the relations between features and wine quality. Notice features like "pH", "chlorides", "residual sugar" almost have no impact on classifying the quality of the wine.
![feature_bar](https://user-images.githubusercontent.com/56566212/103471883-9dc8ca80-4d4b-11eb-9361-922268523d58.png)

## Preprocessing
* Feature selection using chi-square test
* Drop irrelevant features
* Split dataset
* Apply SMOTE to oversample minority classes data by generating synthetic training data using K-NN. Note we do not oversample testing data.
* Feature scaling

## Result

Because of the skewed nature of the dataset. Use F1-score as the performance metric. By applying synthetic minority oversampling technique, KNN model has a notable increase in its weighted F1-score avg from 0.52 to 0.67. The accuracy also went from 51% to 65%. The other models like logistic regression, linear SVM, and kernel SVM did not perform better as expected.

### Logistic Regression
![log](https://user-images.githubusercontent.com/56566212/103495351-25701100-4e00-11eb-823c-f9929e8286f4.png)

### Linear SVM & Kernel SVM
![svm](https://user-images.githubusercontent.com/56566212/103495369-36b91d80-4e00-11eb-8617-d77d1cca36df.png)

### K-NN (Rapid Prototype)
![knn](https://user-images.githubusercontent.com/56566212/103495373-3caefe80-4e00-11eb-9a1e-d55a8a272745.png)

### K-NN (Final)
![knn2](https://user-images.githubusercontent.com/56566212/103495431-5ea88100-4e00-11eb-8b9f-71ecb084bcf2.png)
