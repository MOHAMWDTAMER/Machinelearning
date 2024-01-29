import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
cars=pd.read_csv('E:/machine learning/titanic classification/CarPrice_Assignment.csv')
companyname=cars['CarName'].apply(lambda x: x.split(' ')[0])
cars.insert(3, 'companyname',companyname )
cars.drop(['CarName'],axis=1,inplace=True)
cars.companyname=cars.companyname.str.lower()
def replace_name(a,b):
    cars.companyname.replace(a,b,inplace=True)

replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')
    #print(cars.loc[cars.duplicated()])
    #print('#################')
    #print(cars.companyname.unique()) 
plt.figure(figsize=(20,8))
plt.subplot(1,1,1)
plt.title('Car Price Distribution Plot',size=30)
sns.distplot(cars.price)
print(cars.price.describe(percentiles=[0.25,0.5,0.75,1]))
plt.figure(figsize=(25,6))
plt.subplot(1,3,1)
plt1=cars.companyname.value_counts().plot(kind='bar')
plt.title('Companies Histogram')
plt1.set(xlabel='carcompany',ylabel='frecquency')
plt.subplot(1,3,2)
plt1=cars.fueltype.value_counts().plot(kind='bar')
plt.title('fuel type Histogram')
plt1.set(xlabel='fuel type',ylabel='frecquency of fuel type')
plt.subplot(1,3,3)
plt1=cars.carbody.value_counts().plot(kind='bar')
plt.title(' carbody type Histogram')
plt1.set(xlabel='carbody type',ylabel='frecquency of carbody type')
plt.show()
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.title('symboling')
sns.countplot(cars.symboling,palette=("cubehelix"))
plt.subplot(1,2,2)
plt.title('symboling vs price')
sns.boxplot(x=cars.symboling,y=cars.price,palette=('cubehelix'))
plt.show()
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.title('Engine Type Histogram')
sns.countplot(cars.enginetype,palette='cubehelix')
plt.subplot(1,2,2)
plt.title('Engine Type vs price Histogram')
sns.boxenplot(x=cars.enginetype,y=cars.price,palette='cubehelix')
plt.show()
plt.figure(figsize=(20,8))
df=pd.DataFrame(cars.groupby(['enginetype'])['price'].mean().sort_values(ascending=False))
plt1=df.plot(kind='bar')
plt.title('Engine Type vs Average Price')
plt.show()
df=pd.DataFrame(cars.groupby(['companyname'])['price'].mean().sort_values(ascending=False))
plt1=df.plot(kind='bar')
plt.title('companyname vs Average Price')
plt.show()
df=pd.DataFrame(cars.groupby(['fueltype'])['price'].mean().sort_values(ascending=False))
plt1=df.plot(kind='bar')
plt.title('fueltype vs Average Price')
plt.show()
df=pd.DataFrame(cars.groupby(['carbody'])['price'].mean().sort_values(ascending=False))
plt1=df.plot(kind='bar')
plt.title('carbody vs Average Price')
plt.show()
plt.figure(figsize=(20,8))
plt.subplot(1,2,1 )
plt.title('Door Number Histogram')
sns.countplot(cars.doornumber,palette=("plasma"))
plt.show()
plt.subplot(1,2,2)
plt.title('Door Number vs price Histogram')
sns.boxplot(x=cars.doornumber, y=cars.price,palette=("plasma"))
plt.show()
plt.figure(figsize=(15,8))
plt.subplot(1,2,1 )
plt.title('Aspiration Histogram')
sns.countplot(cars.aspiration,palette=("plasma"))
plt.show()
plt.subplot(1,2,2)
plt.title('Aspiration vs price Histogram')
sns.boxplot(x=cars.aspiration, y=cars.price,palette=("plasma"))
plt.show()
def plot_count(x,fig):
    plt.subplot(4,2,fig)
    plt.title(x+'Histogram')
    sns.countplot(cars[x],palette=('magma'))
    plt.subplot(4,2,(fig+1))
    plt.title(x+'Histogram')
    sns.boxplot(x=cars[x],y=cars.price,palette=('magma'))
plt.figure(figsize=(20,8))
plot_count('enginelocation',1)
plot_count('cylindernumber',3)
plot_count('fuelsystem',5)
plot_count('drivewheel',7)
plt.tight_layout()

def scatter(x,fig):
    plt.subplot(4,2,fig)
    plt.scatter(cars[x],cars.price)
    plt.title(x+'vs price')
    plt.xlabel(x)
    plt.ylabel('price')
plt.figure(figsize=(10,20)) 
scatter('carlength', 1)
scatter('carwidth', 2)
scatter('carheight', 3)
scatter('curbweight', 4)

plt.tight_layout()
def pp(x,y,z):
    sns.pairplot(cars,x_vars=[x,y,z],y_vars=['price'],kind='scatter',aspect=1,size=4)
    plt.show()
plt.figure(figsize=(20,8))
pp('enginesize', 'boreratio', 'stroke')
pp('compressionratio', 'horsepower', 'peakrpm')
pp('wheelbase', 'citympg', 'highwaympg')

cars['fueleconomy'] = (0.55 * cars['citympg']) + (0.45 * cars['highwaympg'])

cars_lr = cars[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase',
                  'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 
                    'fueleconomy', 'carlength','carwidth']]
def dummies(x,df):
    temp=pd.get_dummies(df[x],drop_first=True)
    df=pd.concat([temp,df],axis=1)
    df.drop([x],axis=1,inplace=True)
    return df
    
cars_lr = dummies('fueltype',cars_lr)
cars_lr = dummies('aspiration',cars_lr)
cars_lr = dummies('carbody',cars_lr)
cars_lr = dummies('drivewheel',cars_lr)
cars_lr = dummies('enginetype',cars_lr)
cars_lr = dummies('cylindernumber',cars_lr)


print(cars_lr.head())

df_train,df_test=train_test_split(cars_lr,test_size=0.7,shuffle=True,random_state=100)
scaler=MinMaxScaler()
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower'
            ,'fueleconomy','carlength','carwidth','price']
df_train[num_vars]=scaler.fit_transform(df_train[num_vars])
print(df_train.head())
plt.figure(figsize=(20,5))
sns.heatmap(df_train.corr(),annot=True,cmap='YlGnBu')
plt.show()
y_train=df_train.pop('price')
x_train=df_train

lm = LinearRegression()
lm.fit(x_train,y_train)
rfe = RFE(lm)
rfe = rfe.fit(x_train, y_train)

x_train_rfe=x_train[x_train.columns[rfe.support_]]
lm.fit(x_train_rfe,y_train)
print (lm.score(x_train_rfe,y_train))
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']
df_test[num_vars] = scaler.fit_transform(df_test[num_vars])

y_test = df_test.pop('price')
x_test = df_test

lm.fit(x_test,y_test)
y_pred=lm.predict(x_test)

print(r2_score(y_test, y_pred))

plt.figure(figsize=(10,10))
plt.scatter(y_test,y_pred)
plt.title('y_test vs y_pred',size=30)
plt.xlabel('y_test', fontsize=18)                       
plt.ylabel('y_pred', fontsize=16)   



















    
            
       
   
    



    

    
    



    
    




































    