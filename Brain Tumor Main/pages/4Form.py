import streamlit as st
import pickle
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


st.title(":bookmark_tabs: Brain Score Assesment ")
st.write("---")
st.write("Fill the form below to check your Brain Score ")
st.text("0 - Never  1 - Almost Never  2 - Sometimes  3 - Fairly Often  4 - Very Often")


def ValueCount(str):
    if str == "Yes":
        return 1
    else:
        return 0

    
def assess_brain_score(answers):
    answer = ctrl.Antecedent(np.arange(0, 5, 1), 'answer')
    score = ctrl.Consequent(np.arange(0, 101, 1), 'score')

    # Define membership functions
    answer['Never'] = fuzz.trimf(answer.universe, [0, 0, 1])
    answer['Almost Never'] = fuzz.trimf(answer.universe, [0, 1, 2])
    answer['Sometimes'] = fuzz.trimf(answer.universe, [1, 2, 3])
    answer['Fairly Often'] = fuzz.trimf(answer.universe, [2, 3, 4])
    answer['Very Often'] = fuzz.trimf(answer.universe, [3, 4, 4])

    # Define membership functions for brain score
    score['Poor'] = fuzz.trimf(score.universe, [0, 0, 50])
    score['Good'] = fuzz.trimf(score.universe, [0, 50, 100])

    # Define fuzzy rules
    rule1 = ctrl.Rule(answer['Never'] | answer['Almost Never'] | answer['Sometimes'], score['Good'])
    rule2 = ctrl.Rule(answer['Fairly Often'] | answer['Very Often'], score['Poor'])

    # Create the control system
    health_score_ctrl = ctrl.ControlSystem([rule1, rule2])
    health_score = ctrl.ControlSystemSimulation(health_score_ctrl)

    # Pass user's answers to the fuzzy system
    for i, ans in enumerate(answers):
        health_score.input['answer'] = ans
        health_score.compute()

    # Compute the overall brain health score
    health_score.compute()
    overall_score = health_score.output['score']
    return overall_score


d1 = [0, 1, 2, 3, 4]
d3 = ["No", "Yes"]
val1 = st.selectbox(
    "Have you been experiencing frequent headaches, especially in the morning? ", d1)

val2 = st.selectbox(
    "Have you experienced changes in your vision, such as blurring, double vision, or loss of peripheral vision?", d1)

val3 = st.selectbox(
    "Have you experienced seizures, especially if you have never had a seizure before? ", d1)


val4 = st.selectbox(
    "Have you noticed any changes in your personality or behavior, such as irritability, depression, or a lack of motivation?  ", d1)


val5 = st.selectbox(
    "Have you experienced weakness or numbness in your arms or legs? ", d1)


val6 = st.selectbox(
    "Have you experienced any difficulty speaking or understanding speech?  ", d1)


val7 = st.selectbox(
    "Have you noticed any changes in your coordination or balance?  ", d1)


val8 = st.selectbox(
    "Have you experienced nausea or vomiting, especially if it is worsening or not improving with medication? ", d1)


val9 = st.selectbox(
    "Have you experienced unexplained weight loss or loss of appetite?  ", d1)


val10 = st.selectbox(
    "Have you experienced any other unusual symptoms, such as changes in your sense of smell or taste, or unexplained fatigue?", d1)

input_data = [val1, val2, val3, val4, val5,
              val6, val7, val8, val9, val10]

# input_data_as_numpy_array = np.asarray(input_data)

# input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

brain_score = assess_brain_score(input_data)

with st.expander("Analyze provided data"):
    st.subheader("Results:")
    if brain_score >= 50:
        st.info('Brain Score is Good :)')
    else:
        st.warning('Brain Score is Bad :( [You are recommended to go for an MRI Test]')
