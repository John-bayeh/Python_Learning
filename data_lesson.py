import pandas as pd
data={
    "name":["Yohannes","Sara","Mike","Hana"],
    "salary":[12000,11000,10000,9000],
    "dep":["Engineering","hr","Engineering","hr"]
}
df=pd.DataFrame(data)
print(df)
print("info is \n",df.info())
print("Average salary is \n",df["salary"].mean())
print("engineering students \n",df[df["dep"]=="Engineering"])
print("Group by dep...... \n",df.groupby("dep")["salary"].mean())
df["bonus"]=df["salary"]*0.10
print("bonus is \n",df)
df.to_csv("employees.csv",index=False)
print("saved")
