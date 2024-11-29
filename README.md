# Formation_Tester_Prediction

## Overview

This repository contains Python scripts for developing prototype machine learning models aimed at testing the feasibility of predicting the success rate of downhole sample collection. The workflow includes data cleaning, feature engineering, model training, and evaluation with performance visualization.

## Features

1. **Data Cleaning and Preparation:**
   - Standardizes input data from CSV and Excel files.
   - Handles missing values, calculates new features, and prepares datasets for analysis.
2. **Feature Engineering:**
   - Computes logarithmic transformations (log_RT).
   - Maps categorical labels to numerical classes.
   - Creates features for machine learning models.
3. **Machine Learning:**
   - Implements multiple classifiers (e.g., logistic regression, decision trees, gradient boosting).
   - Separates datasets into training and testing sets for model validation.
4. **Visualization:**
   - Produce pair plots and scatter plots of features.
   - Visualize classification performance.
   - Generate probability plots for analyzing feature correlations.
5. **Excel Integration:**
   - Exports cleaned and processed data to Excel.
   - Generates formulas for classifier coefficients for use in Excel.
6. **Cross-Area Performance Evaluation:**
   - Compares model performance across different geological zones to ensure generalizability.

## File Structure

- `data_loader`: Module for loading and saving data files.
- `one_cut`: Tools for cut-off analysis and performance evaluation.
- `visualization`: Functions for creating performance comparison plots.
- `classifiers`: Collection of machine learning classifiers.
- `feature_xclass`: Utilities for feature and class handling.
- `prob_plot`: Functions for generating probability plots.

## Workflow

1. **Data Cleaning**
   - Loads well data from CSV or Excel files.
   - Handles missing values, standardizes columns, and creates derived features:
     ```python
     merged_table = pd.read_csv(data_path('merged_table_3.csv'))
     test_with_reschart = pd.read_csv(data_path('test_with_reschart.csv'))
     reduced_merge = merged_table[['WellID', 'Category', 'GR', 'RT', 'RHOB', 'NPHI']]
     reduced_merge['log_RT'] = np.log(reduced_merge.RT)
     ```
   - Exports cleaned data to an Excel file:
     ```python
     all_data.to_excel(data_path('all_data.xlsx'), index=False)
     ```

2. **Model Training and Evaluation**
   - Train-Test Splitting:
     ```python
     well_analysis = pd.read_excel(f_name, sheet_name="well_analysis_copy")
     test_marker_dict = {row.WellID: row.group_train == 'x' for row in well_analysis.itertuples()}
     ```
   - Training Models:
     ```python
     classifiers = create_all_classifier()
     for clf_name, clf in classifiers.items():
         clf.fit(train_features, train_classes)
     ```
   - Performance Visualization:
     ```python
     visualize_classifier_performance_comparison(classifiers, test_sample)
     ```

3. **Cross-Area Evaluation**
   - Tests model performance across geological zones:
     ```python
     for train_area in areas_sample.keys():
         for test_area in areas_sample.keys():
             perf_summary = perf.get_perf_summary(classifiers, sample.test_sample)
     ```
   - Compares results using heatmaps and bar charts:
     ```python
     plot_bar_comparison(perf_summary, benchmark)
     ```

4. **Visualization**
   - Pairwise Feature Relationships:
     ```python
     sns.PairGrid(all_data_df[['GR', 'log_RT', 'DEN', 'NPHI', 'xclass']])
     ```
   - Probability Plots:
     ```python
     prob_plot(data=all_feature_xclass, x_features='GR', y_features='DEN')
     ```

## Dependencies

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` for machine learning
- `wtn_mdt` for custom data processing and visualization

## Outputs

- Cleaned data in `all_data.xlsx`.
- Model performance plots (`tree.pdf`, `prob_plot_gr_den.png`).
- Cross-area evaluation summaries in `cross_area_test.pdf`.

## Future Improvements

- Incorporate reservoir information from nearby wells to introduce a weighting mechanism for predictions.
