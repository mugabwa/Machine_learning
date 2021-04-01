import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load data
# csv_headers = pandas.read_csv('diabetes.csv', index_col=1, nrows=0).columns.tolist()
csv_headers = pandas.read_csv('diabetes.csv',skiprows=0,nrows=0).columns.to_list()
csv_file = pandas.read_csv('diabetes.csv',header=0,usecols=csv_headers)
csv_file.head()
features = ['Pregnancies','Glucose','BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
data = csv_file[features]
label = csv_file.Outcome
train_d,test_d,train_l,test_l = train_test_split(data,label,test_size=0.2,random_state=1)
clf = DecisionTreeClassifier(criterion="entropy",max_depth=3) #create DT object
cli = clf.fit(train_d,train_l)
pred_l = clf.predict(test_d)
print("Accuracy: ",metrics.accuracy_score(test_l,pred_l))


