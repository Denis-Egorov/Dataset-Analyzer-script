import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class DatasetAnalyzer:
    def __init__(self, path_file):
        self.path_file = path_file
        self.df = None
        self.target = None
        self.problem_type = None
        self.numeric_cols = None
        self.categorical_cols = None

    # Загрузка данных
    def load_data(self):
        self.df = pd.read_csv(self.path_file)
        print(f"Данные загружены. Размер: {self.df.shape}")

    # EDA
    def explore_data(self):
        print("\n1. Первые 5 строк:")
        print(self.df.head())

        print("\n2. Информация о данных:")
        print(self.df.info())

        print("\n3. Описательная статистика:")
        print(self.df.describe())

        print("\n4. Пропущенные значения:")
        print(self.df.isnull().sum())

    # Определение типа задачи
    def identify_target(self, target_col=None):
        if target_col:
            self.target = target_col
        else:
            self.target = self.df.columns[-1]
            print(f"\n Целевая переменная автоматически выбрана: {self.target}")
        
        unique_values = self.df[self.target].nunique()
        if unique_values < 10:
            self.problem_type = 'classification'
        else:
            self.problem_type = 'regression'
        print(f"Тип задачи: {self.problem_type}")

    def identify_feature_types(self):
        # Числовые признаки
        self.numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.to_list()
        self.numeric_cols = [col for col in self.numeric_cols if col != self.target]

        # Категориальные признаки
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.categorical_cols = [col for col in self.categorical_cols if col != self.target]

        print(f"Числовые признаки: {self.numeric_cols}")
        print(f"Категориальные признаки: {self.categorical_cols}")

    def handle_missing_values(self):
        original_size = len(self.df)
        df_clone = self.df.copy()

        # Заполнение числовых признаков
        for col in self.numeric_cols:
            if df_clone[col].isnull().sum() > 0:
                median_value = df_clone[col].median()
                df_clone[col].fillna(median_value, inplace=True)
                print(f"Заполнено {df_clone[col].isnull().sum()} пропусков в {col} медианой: {median_value}")

        # Заполение категориальных признаков
        for col in self.categorical_cols:
            if df_clone[col].isnull().sum() > 0:
                mode_value = df_clone[col].mode()[0]
                df_clone[col].fillna(mode_value, inplace=True)
                print(f"Заполнено {df_clone[col].isnull().sum()} пропусков в {col} модой: {mode_value}")

        # Проверка
        rows_removed = original_size - len(df_clone)
        if rows_removed > 0:
            print(f"Удалено строк в общей сложности: {rows_removed}")
    
        self.df = df_clone
        print(f"Размер данных после обработки пропусков: {self.df.shape}")  

        if len(self.df) != len(self.df.dropna()):
            print("Внимание: В данных всё ещё есть пропуски!")
            self.df = self.df.dropna()
            print(f"Окончательный размер после dropna(): {self.df.shape}")        

    # Визуализация данных
    def visualize_data(self):
        plt.figure(figsize=(10, 6))
        if self.problem_type == 'classification':
            self.df[self.target].value_counts().plot(kind='bar')
            plt.title('Распределение целевой переменной')
        else:
            self.df[self.target].hist(bins=30)
            plt.title('Гистограмма целевой переменной')
        plt.show()
        
        # Матрица корреляций (только для числовых данных)
        numeric_df = self.df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            plt.figure(figsize=(12, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
            plt.title('Матрица корреляций')
            plt.show()

    def train_models(self):
        # Разделение данных
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        transformers = []
        if self.numeric_cols:
        # Создание конвейеров
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('numeric', numeric_transformer, self.numeric_cols))

        if self.categorical_cols:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            transformers.append(('categorical', categorical_transformer, self.categorical_cols))

        if not transformers:
            print("Нет признаков для обработки")
            return
        
        preprocessor = ColumnTransformer(transformers=transformers)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Обучение модели в зависимости от типа задачи
        if self.problem_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_processed, y_train)
            y_pred = model.predict(X_test_processed)
            score = accuracy_score(y_test, y_pred)
            print(f"Точность модели: {score:.4f}")
            
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_processed, y_train)
            y_pred = model.predict(X_test_processed)    
            score = r2_score(y_test, y_pred)
            print(f"R2 score модели: {score:.4f}")
        
        # Получение имён признаков после всех преобразований
        try:
            # Современный способ (sklearn >= 0.20)
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            # Для старых версий sklearn
            feature_names = preprocessor.get_feature_names()

        # Важность признаков
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        
        print("\nТоп-10 важных признаков:")
        print(feature_importance.head(10))
        
        # Визуализация важности признаков
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
        plt.title('Важность признаков')
        plt.show()

    # Запуск анализа
    def run_analysis(self, target_column=None):
        self.load_data()
        self.explore_data()
        self.identify_target(target_column)
        self.identify_feature_types()
        self.handle_missing_values()
        self.visualize_data()
        self.train_models()

if __name__ == "__main__":
    #analyzer = DatasetAnalyzer('iris.csv')
    #analyzer.run_analysis('species')

    analyzer = DatasetAnalyzer('housing.csv')
    analyzer.run_analysis('median_house_value')