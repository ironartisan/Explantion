#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : cyl
# @Time : 2020/5/10 23:06 

from django.http import HttpResponse


def hello(request):
    return HttpResponse("Hello world ! ")
