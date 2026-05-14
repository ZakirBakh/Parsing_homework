import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#1 Общее понимание датасета
df = pd.read_csv('turbo_cars.csv')
# print(df.columns)
# print(df.head())
# print(df.shape) 
# print(df.describe())
# print(df.dtypes)
# print(df.isnull().sum())

#2 Объединение схожих колонок


# ОБЪЕДИНЕНИЕ КОЛОНОК ПРОБЕГА

# Очищаем от текста " км" и приводим к числам
df['specs_Пробег'] = df['specs_Пробег'].astype(str).str.replace(' км', '', case=False)
df['specs_Пробег'] = pd.to_numeric(df['specs_Пробег'], errors='coerce')
df['specs_Пробег ( км )'] = pd.to_numeric(df['specs_Пробег ( км )'], errors='coerce')

# Приоритет отдаем 'specs_Пробег', если пусто — берем из 'specs_Пробег ( км )'
df['Пробег_Итоговый'] = df['specs_Пробег'].fillna(df['specs_Пробег ( км )'])



# ОБЪЕДИНЕНИЕ КОЛОНОК ГОДА ВЫПУСКА

# Приводим к числовому типу обе колонки
df['year_from_catalog'] = pd.to_numeric(df['year_from_catalog'], errors='coerce')
df['specs_Год выпуска'] = pd.to_numeric(df['specs_Год выпуска'], errors='coerce')

# Объединяем данные по году
df['Год_Итоговый'] = df['specs_Год выпуска'].fillna(df['year_from_catalog'])


# 3. ОБЪЕДИНЕНИЕ КОЛОНОК КОРОБКИ ПЕРЕДАЧ

# Очищаем строки от лишних пробелов по краям
df['specs_Коробка'] = df['specs_Коробка'].astype(str).str.strip()
df['specs_Коробка передач'] = df['specs_Коробка передач'].astype(str).str.strip()

# Заменяем строковые "nan" (если появились при astype) на честные NaN
df['specs_Коробка'] = df['specs_Коробка'].replace(['nan', 'None', ''], pd.NA)
df['specs_Коробка передач'] = df['specs_Коробка передач'].replace(['nan', 'None', ''], pd.NA)

# Объединяем текстовые данные
df['Коробка_Итоговая'] = df['specs_Коробка'].fillna(df['specs_Коробка передач'])



# УДАЛЕНИЕ СТАРЫХ И ЛИШНИХ КОЛОНОК

# Список колонок, которые мы объединили
columns_to_drop = [
    'specs_Пробег', 'specs_Пробег ( км )', 
    'specs_Год выпуска', 'year_from_catalog', 
    'specs_Коробка', 'specs_Коробка передач'
]

# Удаляем их из датасета
df = df.drop(columns=columns_to_drop)



# ПРОВЕРКА РЕЗУЛЬТАТА

print("Новые объединенные колонки:")
print(df[['Пробег_Итоговый', 'Год_Итоговый', 'Коробка_Итоговая']].head(10))
print("\nОставшиеся колонки в датасете:", list(df.columns))

# Очистка данных
# Показывает первые 10 строк, где в колонке 'Коробка_Итоговая' стоит пропуск
missing_millage = df[df['Пробег_Итоговый'].isna()]
print(missing_millage.head(1)) #увидеть строку пропуска
print(df['Пробег_Итоговый'].median()) #медиана пробега

df = df.drop(columns='millage_km') #дублируется с колонкой 'Пробег_Итоговый' и при том имеет 3 пропуска
df['Пробег_Итоговый'] = df['Пробег_Итоговый'].fillna(df['Пробег_Итоговый'].median()) # 1 пропуск Пробега заполняю на медиану
columns_to_drop = ['specs_VIN', 'specs_Гос номер', 'specs_Комплектация', 'specs_Мощность', 
                   'specs_Обмен', 'specs_Объем двигателя', 'specs_Прочее', 'specs_Рассрочка', 'specs_Тип топлива']
df = df.drop(columns=columns_to_drop)

missing_motor = df[df['specs_Двигатель'].isna()]
print(missing_motor.head(3))
df.loc[24, 'specs_Двигатель'] = '1,6 л / гибрид' #ввод вручную согласно объявлению
df.loc[25, 'specs_Двигатель'] = '1,7 л / гибрид' #ввод вручную согласно объявлению
df.loc[200, 'specs_Двигатель'] = '4.0 л / бензин' #ввод вручную согласно объявлению

# Замена все колонок с пропусками на моду
columns = [
    'specs_Кузов', 'specs_Наличие', 'specs_Привод', 'specs_Регион, город', 
    'specs_Руль', 'specs_Состояние', 'specs_Таможня', 'specs_Учёт', 'specs_Цвет'
]
df[columns] = df[columns].fillna(df[columns].mode().iloc[0])

pd.set_option('display.float_format', '{:.2f}'.format) #приведение чисел в читабельный вид и с двумя знаками после запятой



print('=============================================')
print(df.isnull().sum())
print(df.shape) # (480, 20)
print(df.duplicated().sum()) # нет дубликатов
print(df.describe())

#3. Одномерный анализ

# Анализ цен
print(df['price'].mean()) #2 446 658.33
print(df['price'].median()) #1 835 000.0
plt.figure(figsize=(10, 4))
sns.histplot(data=df, x=df['price'] / 1_000_000, kde=True)
plt.title('Распределение цен') # в основмном цены до 2 млн сом, мин цена - 87 000, макс - 21 848 000 сом
plt.xlabel('Цена (в млн сом)') 
plt.ylabel('Количество')
plt.show()

sns.boxplot(x=df['price'] / 1_000_000)
plt.title('Поиск выбросов в ценах')
plt.xlabel('Цена (в млн сом)') 
plt.show()

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
print(f"Нижняя граница: {lower_bound}, Верхняя граница: {upper_bound}") #Нижняя граница: -1764625.0, Верхняя граница: 5970375.0
print(f"Количество выбросов: {len(outliers)}") # 30

# # Анализ года выпуска машин
print(df['Год_Итоговый'].mode()) #2021
print(df['Год_Итоговый'].median()) #2020
plt.figure(figsize=(10, 4))
sns.histplot(df['Год_Итоговый'].dropna(), kde=True)
plt.title('Распределение годов выпуска машин') # в основмном 2018-2021 годов, самый старый - 1981, макс - 2026
plt.show()

sns.boxplot(x=df['Год_Итоговый'])
plt.title('Поиск выбросов годах выпуска машин')
plt.show()

Q1 = df['Год_Итоговый'].quantile(0.25)
Q3 = df['Год_Итоговый'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Год_Итоговый'] < lower_bound) | (df['Год_Итоговый'] > upper_bound)]
print(f"Нижняя граница: {lower_bound}, Верхняя граница: {upper_bound}") #Нижняя граница: 2012, Верхняя граница: 2028
print(f"Количество выбросов: {len(outliers)}") # 81

# Распределение по расположению руля
df['specs_Руль'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'pink'])
plt.title('Распределение по расположению руля')
plt.ylabel('')
plt.show() #95,4% слева, 4,6% справа

# Распределение по кузовам
count_categories = df['specs_Кузов'].nunique()
print(f"Количество уникальных типов кузова: {count_categories}")
print(df['specs_Кузов'].unique())


# 1. Считаем относительные доли каждого кузова (от 0.0 до 1.0)
frequencies = df['specs_Кузов'].value_counts(normalize=True)

# 2. пороговое значение 3%
threshold = 0.03

# # 3. Разделяем на частые и редкие категории
frequent_mask = frequencies >= threshold
frequent_bodies = df['specs_Кузов'].value_counts()[frequent_mask]
rare_count = df['specs_Кузов'].value_counts()[~frequent_mask].sum()

# # 4. Формируем итоговые данные для графика
plot_data = frequent_bodies.copy()
if rare_count > 0:
    plot_data['Другие'] = rare_count

# # 5. Строим круговую диаграмму
plt.figure(figsize=(10, 8))
plt.pie(
    plot_data, 
    labels=plot_data.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=sns.color_palette('pastel')
)

plt.title('Распределение автомобилей по типам кузова', fontsize=14)
plt.axis('equal')
plt.show()

# 4. Анализ факторов, влияющие на цену машин

# Создаем матрицу графиков 2х2
fig, axes = plt.subplots(2, 2, figsize=(16, 12)) 

# # 1. График: Руль (Позиция 0,0)
sns.barplot(x='specs_Руль', y='price', data=df, ax=axes[0,0], palette='magma')
axes[0,0].set_title('Зависимость цены от расположения руля')
axes[0,0].set_ylabel('Средняя цена')

print("Средняя цена по типу руля:")
print(df.groupby('specs_Руль')['price'].mean()) 
print("-" * 40)


# # 2. График: Растаможенность (Позиция 0,1)
sns.barplot(x='specs_Таможня', y='price', data=df, ax=axes[0,1], palette='viridis')
axes[0,1].set_title('Зависимость цены от таможни')
axes[0,1].set_ylabel('Средняя цена')



# # 3. График: Год выпуска (Позиция 1,0)
# # Используем рассеяние (scatterplot) или линию тренда для двух числовых переменных
sns.scatterplot(data=df, x='Год_Итоговый', y='price', ax=axes[1,0], alpha=0.6, color='blue')
axes[1,0].set_title('Влияние года выпуска на стоимость авто')
axes[1,0].set_ylabel('Цена')


# # 4. График: Состояние авто (Позиция 1,1 - ТЕПЕРЬ НА СВОЕМ МЕСТЕ)
sns.barplot(x='specs_Состояние', y='price', data=df, ax=axes[1,1], palette='plasma')
axes[1,1].set_title('Зависимость цены от состояния авто')
axes[1,1].set_ylabel('Средняя цена')

print("Средняя цена по состоянию автомобиля:")
print(df.groupby('specs_Состояние')['price'].mean()) 
# идеальное   2960668.34
# новое       3589941.18
# хорошее     1985583.33
print("-" * 40)

plt.tight_layout()
plt.show()

#5. 🔥 Корреляции

# 1. Задаем списки числовых и категориальных признаков
num_cols = ["Пробег_Итоговый", "Год_Итоговый"]
cat_cols = [
    "specs_Двигатель",
    "specs_Кузов",
    "specs_Наличие",
    "specs_Привод",
    "specs_Регион, город",
    "specs_Руль",
    "specs_Состояние",
    "specs_Таможня",
    "specs_Учёт",
    "specs_Цвет",
    "Коробка_Итоговая",
]
all_features = ["price"] + num_cols + cat_cols

# 2. Создаем временный датафрейм для корреляционного анализа
corr_df = df[all_features].copy()

# 3. Кодируем категориальные признаки в числа
for col in cat_cols:
    corr_df[col] = corr_df[col].astype("category").cat.codes

# 4. Считаем матрицу корреляции (по Спирмену, так как категории ранговые)
corr_matrix = corr_df.corr(method="spearman")

# 5. Выделяем только влияние факторов на цену (столбец 'price')
price_corr = corr_matrix[["price"]].sort_values(by="price", ascending=False)

# Проверка растановки кодирования
cat_cols = ['specs_Привод', 'specs_Кузов', 'specs_Наличие', 
            'specs_Цвет', 'Коробка_Итоговая', 'specs_Двигатель', 'specs_Таможня', 'specs_Руль']

for col in cat_cols:
    print(f"\n--- Карта кодирования для {col} ---")
    categories = df[col].astype('category').cat.categories
    for code, name in enumerate(categories):
        print(f"  {code} -> {name}")



# 6. Визуализация тепловой карты (Heatmap)
plt.figure(figsize=(8, 10))
sns.heatmap(
    price_corr,
    annot=True,  # Вывод коэффициентов корреляции на график
    cmap="coolwarm",  # Сине-красная палитра (синий - обратная связь, красный - прямая)
    fmt=".2f",  # Округление до 2 знаков
    vmin=-1,
    vmax=1,  # Границы шкалы от -1 до 1
    linewidths=0.5,
)

plt.title("Корреляция факторов с ценой автомобиля (price)", fontsize=14)
plt.tight_layout() # подгонка размеров надписи
plt.show()







