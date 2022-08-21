from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

from users.Algorithm.Algorithm import Algorithms
from .forms import UserRegistrationForm
from .models import UserRegistrationModel

algo = Algorithms()


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):

    return render(request, 'users/UserHomePage.html', {})


def Prediction(request):
    if request.method == "POST":
        from django.conf import settings
        import pandas as pd
        name = request.POST.get('name')
        Estimated_Insects_Count = request.POST.get('Estimated_Insects_Count')
        Crop_Type = request.POST.get('Crop_Type')
        Soil_Type = request.POST.get('Soil_Type')
        Pesticide_Use_Category = request.POST.get('Pesticide_Use_Category')
        Number_Doses_Week = request.POST.get('Number_Doses_Week')
        Number_Weeks_Used = request.POST.get('Number_Weeks_Used')
        Number_Weeks_Quit = request.POST.get('Number_Weeks_Quit')
        Season = request.POST.get('Season')

        path = settings.MEDIA_ROOT + "\\" + "Agriculture.csv"
        data = pd.read_csv(path, nrows=1000, delimiter=',').fillna(0)
        data = data.drop(["ID"], axis=1)

        x = data.iloc[:, 0:8]
        # print("X:", x)
        y = data.iloc[:, -1]
        # print("Y:", y)
        x = pd.get_dummies(x)

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=4)
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.fit_transform(x_test)
        # print(x_test)
        x_train = pd.DataFrame(x_train)
        from sklearn.tree import DecisionTreeClassifier
        import numpy as np

        model = DecisionTreeClassifier()
        # print('x-train:', x_train)
        test_set = [Estimated_Insects_Count, Crop_Type, Soil_Type, Pesticide_Use_Category,
                    Number_Doses_Week, Number_Weeks_Used, Number_Weeks_Quit, Season]

        test_set = pd.Series(test_set).fillna(0).tolist()
        print("d:", test_set)
        model.fit(x_train, y_train)
        # test_set = np.reshape(1, -1)

        y_pred = model.predict([test_set])
        print("y:", y_pred)

        if y_pred[0] == 0:
            msg = "bad"
            s = "use less pesticide  "
            
            return render(request, "users/Prediction.html", {"msg": msg, "s": s})

        elif y_pred[0] == 1:
            msg = "Good"
            return render(request, "users/Prediction.html", {"msg": msg})

        elif y_pred[0] == 2:
            msg = "perfect"
            return render(request, "users/Prediction.html", {"msg": msg})
        msg = "not possible"
        return render(request, "users/Prediction.html", {"msg": msg})
    else:
        return render(request, "users/Prediction.html", {})

    


def Naivebayes(request):
    dt_acc, mae, mse, r2, pf = algo.Nb()

    return render(request, "users/Naivebayes.html", {"dt_acc": dt_acc,  "mae": mae, "mse": mse, "r2": r2, "pf": pf})


def Mlp(request):
    accuracy, mse, mae, r2, pf = algo.Mlp()

    return render(request, "users/Mlp.html", {"accuracy": accuracy, "mse": mse, "mae": mae, "r2": r2, "pf": pf})


def Approch(request):
    mae, mse = algo.Approch()

    return render(request, "users/approch.html", {"mse": mse, "mae": mae})
