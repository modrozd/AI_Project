import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import NuSVC


class PatientDatabase:

    def __init__(self, data_path):
        self.all_patient_data = pd.read_csv(data_path, skipinitialspace=True)
        self.personal_info_column_names = ['age', 'sex', 'height', 'weight']

        # Dataframe containing age,sex etc. info
        self.personal_info_data = self.all_patient_data[self.personal_info_column_names]

        # Dataframe containing all ECG results + binary classification
        self.all_cardio_data = self.all_patient_data.copy().drop(self.personal_info_column_names, axis='columns')
        self.all_cardio_data.loc[self.all_cardio_data['class'] > 1, 'class'] = 0
        print(self.all_cardio_data['class'].value_counts().to_string())
        self.X = self.all_cardio_data.iloc[:, 0:len(self.all_cardio_data.columns) - 1].values

        # Zastąpienie nanów wartosciami usrednionymi
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.X = pd.DataFrame(imp.fit_transform(self.X),
                              columns=self.all_cardio_data.iloc[:, 0:len(self.all_cardio_data.columns) - 1].columns)

        self.y = self.all_cardio_data.iloc[:, len(self.all_cardio_data.columns) - 1].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,

                                                                                random_state=0)

        # Zredukowanie danych X do najważniejszych kolumn, wartoscią poniżej można sie bawić
        self.importance_cutoff = 0.00001
        best_attributes = self.get_best_attributes()
        self.X = self.X[best_attributes]
        self.X_train = self.X_train[best_attributes]
        self.X_test = self.X_test[best_attributes]

        self.classify_using_gbc()
        self.classify_using_rf()

    def get_best_attributes(self):
        print("Calculating most meaningful attributes out of :", len(self.X.columns))
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(self.X_train, self.y_train)
        feature_imp = pd.Series(classifier.feature_importances_, index=self.X.columns).sort_values(ascending=False)

        best_attributes = feature_imp.loc[feature_imp > self.importance_cutoff].index
        print("Reduced ", len(self.X.columns), " attributes to ", len(best_attributes))
        print("Printing best attributes :")
        print(best_attributes)
        return best_attributes

    def classify_using_gbc(self):
        print("\nStarting GBC classification")
        gb_clf = GradientBoostingClassifier(n_estimators=100)
        gb_clf.fit(self.X_train, self.y_train)
        print("\n GBC classification finished with score :")
        print(gb_clf.score(self.X_test, self.y_test) * 100, "%")

    def classify_using_rf(self):
        print("\nStarting Random Forest classification")
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(self.X_train, self.y_train)
        print("\n Random Forest classification finished with score :")
        print(rf.score(self.X_test, self.y_test) * 100, "%")
