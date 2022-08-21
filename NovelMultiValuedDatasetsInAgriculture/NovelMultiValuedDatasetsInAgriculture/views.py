
from django.shortcuts import render
from users.forms import UserRegistrationForm


def index(request):
    return render(request, 'index.html', {})


def AdminLogin(request):
    return render(request, 'AdminLogin.html', {})


def UserLogin(request):
    return render(request, 'UserLogin.html', {})


def UserRegister(request):
    form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def Viewdata(request):

    import os
    import pandas as pd
    from django.conf import settings
    path = os.path.join(settings.MEDIA_ROOT, "Agriculture.csv")
    df = pd.read_csv(path, nrows=3000).fillna(0)
    print(df)
    df = df.to_html()
    return render(request, 'users/Viewdata.html', {'data': df})
