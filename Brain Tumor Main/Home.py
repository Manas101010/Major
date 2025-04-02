import requests
import streamlit as st

from PIL import Image

st.set_page_config(page_title="Home Page", page_icon=":tada:", layout="wide")

st.title(":blue[Brain Tumor]")
st.write("---")
with st.container():
    col1, col2 = st.columns([3, 2])
    with col1:
        st.title("What is Brain Tumor?")
        st.write("""
        A brain tumor is a growth of cells in the brain or near it. Brain tumors can happen in the brain tissue. Brain tumors also can happen near the brain tissue. Nearby locations include nerves, the pituitary gland, the pineal gland, and the membranes that cover the surface of the brain.
        """)
    with col2:
        img1 = Image.open("image/brain_1.jpg")
        st.image(img1, width=300)


with st.container():
    col1, col2 = st.columns([4, 2])
    with col1:
        st.title("What Causes Brain Tumor?")
        st.write("""
       Brain tumors happen when cells in or near the brain get changes in their DNA. A cell's DNA holds the instructions that tell the cell what to do. The changes tell the cells to grow quickly and continue living when healthy cells would die as part of their natural life cycle. This makes a lot of extra cells in the brain. The cells can form a growth called a tumor. It's not clear what causes the DNA changes that lead to brain tumors. For many people with brain tumors, the cause is never known. Sometimes parents pass DNA changes to their children. The changes can increase the risk of having a brain tumor. These hereditary brain tumors are rare. If you have a family history of brain tumors, talk about it with your health care provider. You might consider meeting with a health care provider trained in genetics to understand whether your family history increases your risk of having a brain tumor.
        """)
    with col2:
        img1 = Image.open("image/brain_2.jpg")
        st.image(img1, width=350, caption="Causes of Brain Tumor")


with st.container():
    col1, col2 = st.columns([4, 2])
    with col1:
        st.title("Symptoms of Brain Tumor:")

        st.write("""
        1. Headache or pressure in the head that is worse in the morning.
        2. Headaches that happen more often and seem more severe.
        3. Headaches that are sometimes described as tension headaches or migraines.
        4. Nausea or vomiting.
        5. Eye problems, such as blurry vision, seeing double or losing sight on the sides of your vision.
        6. Losing feeling or movement in an arm or a leg.
        7. Trouble with balance.
        8. Speech problems.
        9. Feeling very tired.
        10. Confusion in everyday matters.
        11. Memory problems.
        12. Having trouble following simple commands.
        """)
    with col2:
        img = Image.open("image/brain_3.jpg")
        st.image(img, caption="Signs of Brain Tumor")
        img1 = Image.open("image/brain_4.jpg")
        st.image(img1, caption="")


with st.container():

    left_column, right_column = st.columns([4, 2])
    with left_column:
        st.title("Relevent statistics ")

        st.write("""
            The occurrence of brain tumours in India is steadily rising. More and more cases of brain tumours are reported each year in our country among people of varied age groups. In 2018, brain tumours was ranked as the 10th most common kind of tumour among Indians.
            
            The International Association of Cancer Registries (IARC) reported that there are over 28,000 cases of brain tumours reported in India each year and more than 24,000 people reportedly die due to brain tumours annually. A brain tumours is a serious condition and can be fatal if not detected early and treated.
            
            In India, the prevalence of brain-related tumours is 5-10 per 100,000 people. When cancer spreads from another organ in the body to the brain, it is called a 'metastatic brain tumour' where 40 per cent of all malignancies spread to the brain.
            
            Children's brain tumours are the second most frequent malignancy, accounting for around 26 per cent of all cancers in children
            """)

    with right_column:
        img = Image.open("image/brain_5.jpg")
        st.image(img, width=350, caption="")
        img1 = Image.open("image/brain_6.jpg")
        st.image(img1, width=350, caption="")

with st.container():
    st.title("World Brain Tumor Awareness Day")
    st.write(
        "World Brain Tumour Day is observed every year on June 8 to raise awareness about brain tumours and their treatment, as well as to eliminate the stigma associated with people who suffer from them."
    )
c1, c2 = st.columns([5, 5])
im = Image.open("image/brain_9.jpg")
c1.image(im, caption="")
im1 = Image.open("image/brain_7.png")

c2.image(im1, caption="")
