#!/usr/bin/env python3

import os, pathlib
from typing import List
from dataclasses import dataclass
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
pathlib.Path.cwd()

st.set_page_config(
    page_title="Brazilian E-commerce Performance Dashboard",
    page_icon="📦",
    layout="wide")

@dataclass
class Config:
    dt = "processed_dt.csv"
    top_dt = 25
    mx_border = True

container_1, container_2 = st.columns([3,1])
metric_1, metric_2, metric_3 = container_1.columns(3)

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
mx1 = 20; mx2 = 30; mx3 = 50
if Config.mx_border == True:
    metric_1.metric("Data 1", f"{mx1} mph", border = Config.mx_border)
    metric_2.metric("Data 2", f"{mx2} mph", border = Config.mx_border)
    metric_3.metric("Data 3", f"{mx3} mph", border = Config.mx_border)

container_1.markdown("### Daily Active Users")
container_1.bar_chart(chart_data)

user_1, user_2 = container_1.columns(2)

user_1.markdown("### User Retention Rate")
user_1.bar_chart(chart_data)

user_2.markdown("### Fresh User")
user_2.bar_chart(chart_data)

# container_2.markdown("### Product Interest")

dt = pd.DataFrame(pd.read_csv('data/order_items_dataset.csv').tail(10))
container_2.table(dt)