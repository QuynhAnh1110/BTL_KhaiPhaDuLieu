from tkinter import *
from tkinter import messagebox
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys

from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
sys.stdout.reconfigure(encoding='utf-8')
from math import sqrt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

data = pd.read_csv('data_customers_xuly.csv', index_col = None)

#Thống kê mô tả các trường của bộ dữ liệu
data.describe()

#Chọn các thuộc tính là dữ liệu số và lưu vào Dataframe
numeric_columns = data.select_dtypes(include=[np.number])

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaler = scaler.fit_transform(numeric_columns)

X = pd.DataFrame(X_scaler, columns=numeric_columns.columns)

#Lấy tên các cột và giá trị của DataFrame X
columns = X.columns
data_value = X.values

#print('Data:\n',X)
#print('Trung bình các cột:\n',X.mean()) #GT trung bình của từng cột

#Khởi tạo k tâm cụm
k = 4

#Lấy n mẫu ngẫu nhiên từ X
Centroids = (X.sample(n=k))
#print("- Tâm cụm khởi tạo: \n",Centroids)

#Tính khoảng cách giữa 2 điểm DL: Tức là tính bình phương hiệu của từng đặc trưng của từng điểm dữ liệu
def distance(row_c, row_x):
    d = sqrt((row_c['Sales']-row_x['Sales'])**2 + 
                (row_c['Quantity']-row_x['Quantity'])**2 + 
                (row_c['Discount']-row_x['Discount'])**2 + 
                (row_c['Order ID']-row_x['Order ID'])**2 + 
                (row_c['Profit']-row_x['Profit'])**2      
            )
    return d

#K-means
diff=1  #số lần thay đổi tâm cụm
j=0
loop = 1
loop_max = 100 #Đặt số lần lặp max

while(diff > 1e-5 and loop <= loop_max): #nếu số lần thay đổi khác 0 thì sẽ thực hiện lặp để tìm ra tâm cụm mới
    #print("\n\n~~~~~~~~Lần lặp: ",loop)
    i=1
    #Tính toán khoảng cách từ mỗi điểm DL đến các Centroid
    for index1, row_c in Centroids.iterrows(): #1 row_c là 1 series: chứa thông tin của 1 dòng dữ liệu
        ED=[]
        for index2, row_x in X.iterrows():
            d=distance(row_c,row_x)
            ED.append(d)
        X["d(C"+str(i)+")"]=ED
        i=i+1 #tăng chỉ số tên cụm

    C=[]
    #Tìm kiếm cụm gần nhất cho điểm DL hiện tại
    for index, row in X.iterrows():
        min_dist=row["d(C1)"] #K/c từ điểm DL htai đến tâm cụm 1
        pos=1
        for i in range(k):
            if row["d(C"+str(i+1)+")"]<min_dist:
                min_dist = row["d(C"+str(i+1)+")"]
                pos=i+1
        C.append(str(pos))

    X["Cum"]=C
    #print("\n+ Dữ liệu X: \n",X)

    #Nhóm các tâm cụm có index giống nhau để tính trung bình cộng
    Centroids_new = X.groupby(["Cum"]).mean()[['Sales',"Quantity","Discount","Order ID","Profit"]]

    #print("\n+ Tâm cụm mới: \n",Centroids_new)
    #print("\n+ Tâm cụm cũ: \n",Centroids)

    #ĐK dừng:Lấy tâm cụm mới - tâm cụm cũ rồi tính tổng, nếu tổng =0 thì không có sự khác nhau => dừng thuật toán và ngược lại
    #Tính số lần thay đổi
    diff = (Centroids_new["Sales"] - Centroids["Sales"]).abs().sum() + \
       (Centroids_new["Quantity"] - Centroids["Quantity"]).abs().sum() + \
       (Centroids_new["Discount"] - Centroids["Discount"]).abs().sum() + \
       (Centroids_new["Order ID"] - Centroids["Order ID"]).abs().sum() + (Centroids_new["Profit"] - Centroids["Profit"]).abs().sum()
    
    #print("\n+ So sánh sự khác nhau: diff =",diff)
    if j==0:
        diff=1
        j=j+1

    #Sao chép giá trị của tâm cụm mới vào biến tâm cụm cũ để chuẩn bị cho lần lặp tiếp theo
    Centroids = Centroids_new.copy()
    loop+=1
    
#print("\nDữ liệu X cuối cùng:\n", X)
#print("\nTâm cụm cuối cùng:\n",Centroids_new)

# Thống kê tổng số mẫu trong mỗi cụm qua nhãn K-means
cluster_counts = X["Cum"].value_counts()
print('\n- Tổng số mẫu trong mỗi cụm K-means:\n', cluster_counts)

#Độ phù hợp
#print("\n- Mức độ phù hợp silhouette_score = ", silhouette_score(X, X["Cum"]))

#Form
form = Tk()          
form.title("Phân cụm khách hàng của trung tâm thương mại") 
form.geometry("750x500")

lable_dudoan = Label(form, text = "Dự đoán phân cụm khách hàng", font=("Arial", 20), fg = "blue").grid(row=0, column=1, columnspan=3, padx=20, pady=10, sticky="ew")

group1 = LabelFrame(form, text="Nhập thông tin để dự đoán", font=("Tahoma", 12))
group1.grid(row=1, column=1, padx=50, pady=30)
group2 = LabelFrame(form, bd=0)
group2.grid(row=1, column=2)
group3 = LabelFrame(group2, text="Đánh giá mô hình được chọn:", font=("Tahoma", 12))
group3.grid(row=1, column=1, pady=20, sticky="w")

lable_Sales = Label(group1, text = " Sales:", font=("Tahoma", 12)).grid(row = 1, column = 1, pady = 10,sticky="e")
textbox_Sales = Entry(group1)
textbox_Sales.grid(row = 1, column = 2, padx = 20)

lable_Quantity = Label(group1, text = "Quantity:", font=("Tahoma", 12)).grid(row = 2, column = 1, pady = 10,sticky="e")
textbox_Quantity = Entry(group1)
textbox_Quantity.grid(row = 2, column = 2)

lable_Discount = Label(group1, text = "Discount:", font=("Tahoma", 12)).grid(row = 3, column = 1,pady = 10,sticky="e")
textbox_Discount = Entry(group1)
textbox_Discount.grid(row = 3, column = 2)

lable_OrderID = Label(group1, text = "Order ID:", font=("Tahoma", 12)).grid(row = 4, column = 1,pady = 10,sticky="e")
textbox_OrderID = Entry(group1)
textbox_OrderID.grid(row = 4, column = 2)

lable_Profit = Label(group1, text = "Profit:", font=("Tahoma", 12)).grid(row = 5, column = 1, pady = 10,sticky="e")
textbox_Profit = Entry(group1)
textbox_Profit.grid(row = 5, column = 2)

lable_ketqua = Label(group2, text = "Kết quả", font=("Arial italic", 10)).grid(row = 3, column = 1, pady = 10)

#Đánh giá độ đo
lb_kmean = Label(group3)
lb_kmean.grid(row=0, column=1, padx = 35, pady = 20)
lb_kmean.configure(text="Số cụm được chọn: " + str(k)+'\n'
                +"\nĐộ phù hợp silhouette_score = "+ str(silhouette_score(X, X["Cum"])))

lb_num = Label(group3)
lb_num.grid(row=2, column=1, padx = 35, pady = 20)
lb_num.configure(text="Tâm của các cụm: \n" + str(Centroids_new))

#Hàm dự đoán
def dudoan():
        Sale = textbox_Sales.get()
        quantity = textbox_Quantity.get()
        discount = textbox_Discount.get()
        orderid = textbox_OrderID.get()
        profit = textbox_Profit.get()
        if((Sale == '') or (quantity == '') or (discount == '') or (orderid == '') or (profit == '')):
            messagebox.showinfo("Thông báo", "Bạn cần nhập đẩy đủ thông tin!")
        else:
            x_dudoan = pd.DataFrame({
            'Sales': [float(Sale)],
            'Quantity': [float(quantity)],
            'Discount': [float(discount)],
            'Order ID': [float(orderid)],
            'Profit': [float(profit)]
        })

        min_d = float('inf')
        C = 1
        for index1, row in Centroids_new.iterrows():
            d = distance(row, x_dudoan.iloc[0])  # Truy cập hàng đầu tiên của DataFrame mới
            if d < min_d:
                min_d = d
                C = index1
        
        lb_pred.configure(text= "Cụm: " + str(C))

button_1 = Button(group2, text = 'Kết quả dự đoán', font=("Arial Bold", 9), fg = "black", bg = "green", command = dudoan)
button_1.grid(row = 2, column = 1)
lb_pred = Label(group2, text="...", font=("Arial Bold", 9), fg = "white", bg = "SlateGray4")
lb_pred.grid(row=4, column=1)    

form.mainloop() #chạy giao diện chính chờ người dùng tương tác

#Lấy các giá trị x và y để vẽ hừn
x = X['Sales'].values
y = X['Profit'].values

colors = X['Cum'].astype(int).values  # Chuyển đổi nhãn cụm thành kiểu số

# Lấy tâm cụm
# centroids = X.groupby("Cum").mean()[['Pregnancies', "Glucose", "BloodPressure", "SkinThickness", "Insulin", "free_sulfur_dioxide", "DiabetesPedigreeFunction", "Age"]]

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
scatter = plt.scatter(x, y, c=colors, marker='o', alpha=0.6, edgecolors='k', cmap='viridis')

# Vẽ tâm cụm
for i, centroid in Centroids_new.iterrows():
    plt.scatter(centroid['Profit'], centroid['Sales'], c='red', marker='X', s=200, label=f'Centroid {i}')

# Thêm nhãn cho trục
plt.xlabel('Sales', fontsize=16)
plt.ylabel('Profit', fontsize=16)
# plt.('BloodPressure', fontsize=16)

plt.title("Customer Clustering Chart", fontsize=18)

# Thêm legend để phân biệt các tâm cụm
plt.legend(loc='upper right')

# Hiển thị biểu đồ
plt.show()