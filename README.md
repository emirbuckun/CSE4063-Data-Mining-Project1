# CSE4063 Data Mining Project#1

## How to Run

1. Ensure you have all the required dependencies installed. You can install them using:

   ```sh
   pip install -r requirements.txt
   ```

2. Run the main script:

   ```sh
   python main.py
   ```

3. Follow the on-screen instructions to proceed through the model training and evaluation steps.

### Step 1: Dataset Selection and Approval (Due: As soon as possible) ✅

- **Research:** Explore Kaggle, UCI Machine Learning Repository, or other reliable sources to find a dataset with interesting classification tasks and enough complexity for a thorough analysis.
- **Dataset Details:** Make sure it has sufficient rows and columns for training and testing but is manageable within the time constraints.
- **Approval:** Email your instructor (agaoglum@gmail.com) to get approval for the dataset.
- **Approved Dataset:** [Dataset#5](https://archive.ics.uci.edu/dataset/911/recipe+reviews+and+user+feedback+dataset)

### Step 2: Python Environment Setup ✅

- **Install Python:** Ensure everyone has Python installed (Anaconda or Jupyter Notebooks could be helpful).
- **Libraries:** Install necessary libraries (e.g., pandas, numpy, scikit-learn, matplotlib, seaborn, tensorflow or keras for neural networks).
- **Collaborative Platform:** Set up a shared platform (e.g., GitHub) for collaboration and version control.
- **Environment Details:**
  - Python Version: Python 3.13.0
  - Pip Version: pip 24.3.1
  - [Dataset#5](https://archive.ics.uci.edu/dataset/911/recipe+reviews+and+user+feedback+dataset) imported!

### Step 3: Data Preprocessing

- **Data Cleaning:** Handle any missing values, irrelevant columns, or noisy data.
- **Data Transformation:** Normalize or scale data if necessary, especially for algorithms like SVM.
- **Partitioning:** Decide on a partition strategy. Typically, an 80-20 split is common.
- **Cross-Validation and Ensembles:** For at least one classifier, use k-fold cross-validation. Implement bagging and boosting ensembles for comparison.

### Step 4: Model Construction and Training

**Algorithm Implementations:** Implement each classifier (e.g., Decision Tree with gain ratio and gini index, Naïve Bayes, Neural Networks, and SVM).

- Experiment with tuning hyperparameters for neural networks.

**Model Evaluation and Performance Metrics:**

- Evaluate each model using metrics such as accuracy, precision, recall, F1 score, and confusion matrices.
- Compare results from all models.

### Step 5: Presentation Preparation

**Presentation Slides:** Outline the following sections in your presentation:

- Problem Definition
- Dataset Information
- Preprocessing Steps
- Classifier Implementation
- Model Evaluation and Results Comparison
- Conclusion

**Presentation Roles:** Divide the presentation tasks among your group to ensure equal participation.

### Step 6: Demonstration Preparation

- **Demo Code:** Prepare a script that shows the key parts of your code in a logical order.
- **Time Allocation:** Practice to ensure the presentation (25 mins) and demonstration (15 mins) fit within the allocated time.

### Step 7: Final Submission (Due: Before Dec 30, 2024)

- **Document Preparation:** Make sure all project files, presentation, and "we_swear.txt" are ready.
- **File Submission:** Zip all files into the specified format (GrRepStudentNumber_P1.zip) and submit via Google Classroom.
