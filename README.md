**Breast Cancer Prediction using Stacking Ensemble Learning**
**Overview**
This project focuses on building a highly accurate breast cancer prediction model using ensemble techniques. What started as a simple soft voting model gradually evolved into a custom stacking ensemble with multiple Random Forests as base learners and an SVM as a meta-learner, resulting in a robust and generalizable classifier.




**Project Journey**: From Simplicity to Strategy
**Step 1: Soft Voting Ensemble (Baseline Model)
**What I Did:
•	Started by combining basic classifiers (Random Forest, SVM, Logistic Regression) using Soft Voting.
What I Learned:
•	Soft voting helps by averaging predicted probabilities.
•	It works best when base models are diverse and well-tuned.
•	However, I observed moderate performance and limited adaptability to non-linearities.

**Step 2: Traditional Stacking with Mixed Models**
What I Did:
•	Switched to a StackingClassifier using:
o	Base models: Random Forest, SVM, XGBoost
o	Meta-learner: Logistic Regression
What I Learned:
•	Performance improved — especially with hyperparameter tuning.
•	Logistic Regression as a meta-model had limited flexibility to capture non-linear relationships between base model predictions.
•	Feature scaling and preprocessing became critical for consistent performance.


**Step 3: Tackling Class Imbalance with SMOTE**
What I Did:
•	Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance classes in the training set.
What I Learned:
•	Balancing classes had a huge impact on recall for the minority class (Malignant).
•	Helped prevent the model from being biased toward the majority class.

**Step 4: Feature Selection and Cleanup**
What I Did:
•	Dropped highly correlated or less informative features based on domain knowledge and feature importance.
What I Learned:
•	Reducing noise improved both training speed and model performance.
•	Helped SVM (which is sensitive to feature scales and distributions) to perform more reliably.

**Step 5: Optimizing the Stacking Model**
What I Did:
•	Used GridSearchCV to fine-tune hyperparameters of base models and the meta-learner.
•	Got 97% accuracy, but training data performance was very high, raising concerns about overfitting.
What I Learned:
•	High accuracy doesn’t always mean better generalization.
•	Overfitting is real when you don’t validate carefully or when your meta-learner is too simple.


**Step 6: Final Custom Stacking (Multiple Random Forests + SVM)**
**Final Approach:**
•	Base Models: N Random Forests, each trained using different K-Folds.
•	Meta Learner: Support Vector Machine (SVM).
•	Evaluation Metrics: Accuracy, F1 Score, Precision, Recall (on both Train & Test).
Why This Approach?
•	Using the same algorithm (Random Forest) across folds ensures consistency but introduces variation via data partitioning.
•	SVM as a meta-learner offers flexibility to capture non-linear patterns from the base models’ outputs.
•	This setup reduced overfitting, improved F1 scores, and generalized well on unseen data.
What I Learned:
•	Diversity can come not just from different models, but also from how data is split and how models are trained.
•	SVM is a powerful meta-learner when base predictions aren't linearly separable.
•	Final model achieved:
o	Training Accuracy: 100%
o	Testing Accuracy: ~94%
o	F1 Score: Balanced and High across both classes

