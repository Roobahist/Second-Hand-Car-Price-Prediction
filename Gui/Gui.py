from tkinter import *
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
from keras import Input, layers
from sklearn.preprocessing import StandardScaler

def show(x):                  
    x = str(x)                
    lis = []                  
    for i in range(len(x)):   
        lis.append(x[i])      
    time = len(x)//3          
    z = 0                     
    for i in range(1,time+1): 
        lis.insert(-3*i-z,',')
        z += 1                
    answer = ''               
    for i in range(len(lis)): 
        answer += str(lis[i]) 
    if answer[0] == ',':      
        answer = answer[1:]   
    return answer

def rounded(x):
    flag = 0
    x = str(int(x))
    first = x[:-6]
    second = x[-6:]
    lower = str(second[0])+'00000'
    if second[0]=='9':
        flag = 1
        upper = '1000000'
    else:
        upper = str(int(second[0])+1)+'00000'
    upper_res = int(upper)-int(second)
    lower_res = int(second)-int(lower)
    answer = 0
    
    if second == '000000':
        answer = int(x)
    elif lower_res>upper_res:
        if flag==0:
            answer = int(first + upper)
        else:
            answer = int(first[:-1] + str(int(first[-1])+1) + '000000')
    elif lower_res<=upper_res:
        answer = int(first + lower)
    return answer

data = pd.read_csv('Preprocessing\cleaned_data.csv').drop('Unnamed: 0',axis=1)

reg_data = pd.read_csv('Preprocessing\\reg_encoded.csv').drop('Unnamed: 0',axis=1)

neu_data_unscaled = pd.read_csv('Preprocessing\\neu_encoded.csv').drop('Unnamed: 0',axis=1)

neu_data = pd.read_csv('Preprocessing\\neu_encoded.csv').drop('Unnamed: 0',axis=1)
scaler = StandardScaler()
scaler.fit(neu_data[['brand','color','gear','year','body','loc','usage']])
neu_data[['brand','color','gear','year','body','loc','usage']] = scaler.transform(neu_data[['brand','color','gear','year','body','loc','usage']])


root = Tk()
root.title('Car Price Estimation AI')
root.iconphoto(False, PhotoImage(file=r'C:\Users\Matility\Biria\Car_predict\Gui\icon.png'))
lb1 = Label(root,text='Car Price Estimation AI',font=('helvetica', 10, 'bold'))
lb1.grid(column=1,row=0)


Brand = StringVar()
Unique_Brands = list(data['brand'].unique())
Brand.set('مدل')
BrandDrop = OptionMenu(root , Brand , *Unique_Brands)
BrandDrop.grid(column=2,row=1)
BrandDrop.config(width=17)
Brandmenu = BrandDrop.children['menu']
pre_brand = 'مدل'


Color = StringVar()
Color.set('رنگ')
Colors = ['-']
ColorDrop = OptionMenu(root , Color, Colors)
ColorDrop.grid(column=2,row=2)
ColorDrop.config(width=17)
Colormenu = ColorDrop.children['menu']
pre_color = 'رنگ'


Gear = StringVar()
Gear.set('نوع دنده')
Gears = ['-']
GearDrop = OptionMenu(root,Gear,Gears)
GearDrop.grid(column=2,row=3)
GearDrop.config(width=17)
Gearmenu = GearDrop.children['menu']
pre_gear = 'نوع دنده'


Year = StringVar()
Year.set('سال تولید')
Years = ['-']
YearDrop = OptionMenu(root,Year,Years)
YearDrop.grid(column=2,row=4)
YearDrop.config(width=17)
Yearmenu = YearDrop.children['menu']
pre_year = 'سال تولید'


Body = StringVar()
Body.set('وضعیت بدنه')
Bodys = ['-']
BodyDrop = OptionMenu(root,Body,Bodys)
BodyDrop.grid(column=2,row=5)
BodyDrop.config(width=17)
Bodymenu = BodyDrop.children['menu']
pre_body = 'وضعیت بدنه'


Location = StringVar()
Location.set('شهر')
Locations = ['-']
LocationDrop = OptionMenu(root,Location,Locations)
LocationDrop.grid(column=2,row=6)
LocationDrop.config(width=17)
Locationmenu = LocationDrop.children['menu']
pre_loc = 'شهر'


lb1 = Label(root,text=': کارکرد')
lb1.grid(column=2,row=7,sticky=E)
Usage = Entry(root)
Usage.insert(END,0)
Usage.config(width=16)
Usage.grid(column=2,row=7,sticky=W,padx=3)


def reg_predict(index,usage):
    global data
    regression_coefs = [2330632.2634422774,
    -7.737668700021388,
    1421934.1927928533,
    383046.367207516,
    200475031.18491313,
    170222365.37992382,
    65587867.19285374,
    60642583.220031455,
    26444936.29834664,
    32715033.676310375,
    26956806.233949464,
    22182376.793796204,
    6719121.865523484,
    -413709.83517966466,
    1100547.0303668664,
    4161315.5176374894,
    3476854.333065696,
    2502159.49266354,
    2045466.673679664,
    2902013.687429093,
    -1647759.4643836413,
    2525310.1804141924,
    2437029.2511415593,
    4602541.684475984,
    4834529.076954365,
    28535418.89185507,
    41193516.855579466,
    19882240.575216986,
    31237081.80623305,
    -1847014.0192355663,
    4885581.396636419,
    -2525717.509216681,
    -2621220.4179262817,
    -4729269.712862801,
    -1831491.3174292147,
    -4208099.198341727,
    -3711312.6751918867,
    -4637697.261958543,
    244424.19131977856,
    1915813.738891825,
    -8201092.009241417,
    -797715.1643892322,
    -1509029.884935841,
    462905.8491782248,
    -1275672.7781560495,
    -3943948.885610722,
    -922728.4896782059,
    -2725682.7237656955,
    -1026611.8925222065,
    -2151187.3873078823,
    -3992437.578359805,
    -3824447.4666376524,
    -2354369.974744007,
    -1779574.7620106637,
    -2936928.904191956,
    -1607034.0590744372,
    -2752844.203483887,
    -5318785.109765299,
    -810222.9778369665,
    -2522031.904695548,
    490819.5650136359,
    -1000512.3685277328,
    383257.51781433076,
    -581313.6935981661,
    -722681.2671154141,
    -384343.1061157435,
    1101298.0555718318,
    -299860.2146006301]

    reg_intercept = -3203804625.403244

    reg_encoded_input = list(reg_data.iloc[index,:].drop('price').values)
    reg_encoded_input[1] = int(usage)
    multiplied = [reg_encoded_input[i]*regression_coefs[i] for i in range (len(regression_coefs))]
    prediction = int(sum(multiplied) + reg_intercept)
    return prediction

def neu_predict(index,usage):
    network = Sequential()
    network.add(Dense(300,input_shape=(7,),activation='relu',kernel_initializer='normal'))
    network.add(Dense(550,activation='relu',kernel_initializer='normal'))
    network.add(Dense(550,activation='relu',kernel_initializer='normal'))
    network.add(Dense(550,activation='relu',kernel_initializer='normal'))
    network.add(Dense(550,activation='relu',kernel_initializer='normal'))
    network.add(Dense(550,activation='relu',kernel_initializer='normal'))
    network.add(Dense(550,activation='relu',kernel_initializer='normal'))
    network.add(Dense(550,activation='relu',kernel_initializer='normal'))
    network.add(Dense(550,activation='relu',kernel_initializer='normal'))
    network.add(Dense(550,activation='relu',kernel_initializer='normal'))
    network.add(Dense(300,activation='relu',kernel_initializer='normal'))
    network.add(Dense(1))
    network.compile(loss='mae',optimizer='adam')
    network.load_weights(r'Gui\NN_weights\weights4.h5')
    neu_encoded_input = neu_data_unscaled.iloc[index,:].drop('price')
    neu_encoded_input['usage'] = int(usage)
    neu_encoded_input = neu_encoded_input.values.reshape(1,-1)
    neu_encoded_input = scaler.transform(neu_encoded_input)
    prediction = network.predict(neu_encoded_input).flatten()[0]
    return prediction
    

def calculate():
    x1 = Brand.get()
    x2 = Color.get()
    x3 = Gear.get()
    x4 = Year.get()
    x5 = Body.get()
    x6 = Location.get()
    x7 = Usage.get()

    f1 = data['brand'] == x1
    f2 = data['color'] == x2
    f3 = data['gear'] == x3
    f4 = data['year'] == int(x4)
    f5 = data['body'] == x5
    f6 = data['loc_1'] == x6

    same_records_index = list(data[f1&f2&f3&f4&f5&f6].index)

    reg_prediction = abs(int(reg_predict(same_records_index[0],x7)))
    neu_prediction = abs(int(neu_predict(same_records_index[0],x7)))


    same_records = data.iloc[same_records_index,[-2,-1]].reset_index(drop=True)
    same_records[['usage','price']] = same_records[['usage','price']].astype(int)
    same_records['price'] = same_records['price'].apply(lambda x:show(rounded(x)))
    l1.config(width=20,text=same_records.sample(frac=1).head().sort_values(by='price',ascending=True).reset_index(drop=True))
    l1.grid(column=0,row=3,rowspan=3)


    l2.config(width=20,text= str(show(rounded(reg_prediction)))+' تومان')
    l2.grid(column=0,row=6)

    l3.config(width=20,text= str(show(rounded(neu_prediction)))+' تومان')
    l3.grid(column=0,row=7)

    return same_records_index, reg_prediction , neu_prediction

l1 = Label(root, text = '                                               ')
l1.grid(column=0,row=3,rowspan=3)

l2 = Label(root, text= '                                               ')
l2.grid(column=0,row=6)

l3 = Label(root, text= '                                               ')
l3.grid(column=0,row=7)

l4 = Label(root, text= ': موارد مشابه')
l4.config(width=20)
l4.grid(column=1,row=3,rowspan=3,sticky=W)

l5 = Label(root, text= ': رگرسیون')
l5.config(width=20)
l5.grid(column=1,row=6,sticky=W)

l6 = Label(root, text= ': شبکه عصبی')
l6.config(width=20)
l6.grid(column=1,row=7,sticky=W)

Calculate = Button(text='محاسبه کن',bg='brown',fg='white',command=calculate)
Calculate.config(width = 20,height=2)
Calculate.grid(column=0,row=1,rowspan=2,columnspan=2)



def task():
    global pre_brand
    global pre_color
    global pre_gear
    global pre_body
    global pre_loc
    global pre_year
    
    now_brand = Brand.get()
    brandfilter = data['brand'] == now_brand
    if now_brand != pre_brand:
        pre_brand = now_brand

        l1.config(text = '                                               ')
        l2.config(text = '                                               ')
        l3.config(text = '                                               ')

        Colormenu.delete(0, "end")
        Color.set('رنگ')

        Gearmenu.delete(0, "end")
        Gear.set('نوع دنده')

        Yearmenu.delete(0, "end")
        Year.set('سال تولید')

        Bodymenu.delete(0, "end")
        Body.set('وضعیت بدنه')
        
        Locationmenu.delete(0, "end")
        Location.set('شهر')

        Usage.delete(0, "end")
        Usage.insert(END,0)
        
        for value in (list(data[brandfilter]['color'].unique())):
            Colormenu.add_command(label=value, command=lambda v=value: Color.set(v))

    now_color = Color.get()
    colorfilter = data['color'] == now_color
    if now_color != pre_color:
        pre_color = now_color
        
        l1.config(text = '                                               ')
        l2.config(text = '                                               ')
        l3.config(text = '                                               ')
        
        Gearmenu.delete(0, "end")
        Gear.set('نوع دنده')

        Yearmenu.delete(0, "end")
        Year.set('سال تولید')

        Bodymenu.delete(0, "end")
        Body.set('وضعیت بدنه')
        
        Locationmenu.delete(0, "end")
        Location.set('شهر')
        
        Usage.delete(0, "end")
        Usage.insert(END,0)

        for value in (list(data[brandfilter & colorfilter]['gear'].unique())):
            Gearmenu.add_command(label=value, command=lambda v=value: Gear.set(v))
    
    now_gear = Gear.get()
    gearfilter = data['gear'] == now_gear
    if now_gear != pre_gear:
        pre_gear = now_gear
    
        l1.config(text = '                                               ')
        l2.config(text = '                                               ')
        l3.config(text = '                                               ')
        
        Yearmenu.delete(0, "end")
        Year.set('سال تولید')

        Bodymenu.delete(0, "end")
        Body.set('وضعیت بدنه')
        
        Locationmenu.delete(0, "end")
        Location.set('شهر')
        
        Usage.delete(0, "end")
        Usage.insert(END,0)

        for value in (list(data[brandfilter & colorfilter & gearfilter]['year'].unique())):
            Yearmenu.add_command(label=value, command=lambda v=value: Year.set(v))
    
    now_year = Year.get()
    if now_year != 'سال تولید':
        yearfilter = data['year'] == int(now_year)
    else:
        yearfilter = data['year'] == now_year
    if now_year != pre_year:
        pre_year = now_year
        
        l1.config(text = '                                               ')
        l2.config(text = '                                               ')
        l3.config(text = '                                               ')
        
        Bodymenu.delete(0, "end")
        Body.set('وضعیت بدنه')
        
        Locationmenu.delete(0, "end")
        Location.set('شهر')

        Usage.delete(0, "end")
        Usage.insert(END,0)

        for value in (list(data[brandfilter & colorfilter & gearfilter & yearfilter]['body'].unique())):
            Bodymenu.add_command(label=value, command=lambda v=value: Body.set(v))

    
    now_body = Body.get()
    bodyfilter = data['body'] == now_body
    if now_body != pre_body:
        pre_body = now_body

        l1.config(text = '                                               ')
        l2.config(text = '                                               ')
        l3.config(text = '                                               ')
        
        Locationmenu.delete(0, "end")
        Location.set('شهر')

        Usage.delete(0, "end")
        Usage.insert(END,0)

        for value in (list(data[brandfilter & colorfilter & gearfilter & yearfilter & bodyfilter]['loc_1'].unique())):
    
            Locationmenu.add_command(label=value, command=lambda v=value: Location.set(v))

    root.after(200, task)  # reschedule event in 2 seconds

root.after(1, task)
root.mainloop()
