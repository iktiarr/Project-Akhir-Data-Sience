import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import json

from data import load_data, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix