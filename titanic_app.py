import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import io

@st.cache_data
def convert_to_excel(df2):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df2.to_excel(writer, sheet_name="data", index=False)
    writer.close()
    return output.getvalue()


def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(\"https://cdn.wallpapersafari.com/88/75/cLUQqJ.jpg\");
            background-attachment: fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(page_title='Titanic Analysis', layout='centered')
    st.title("Analisi Dati - Titanic")
    add_bg_from_url()

    uploaded_file = st.file_uploader("Scegli un file CSV del Titanic")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep='\t')
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df.dropna(inplace=True)
        df['sex'] = df['sex'].map({'male': 0, 'female': 1})
        df['embarked'] = df['embarked'].map({'s': 0, 'c': 1, 'q': 2})
        st.dataframe(df)

        tab1, tab2, tab3 = st.tabs(['Analisi', 'Predizione Singola', 'Predizione Batch'])

        with tab1:
            st.header('Analisi Esplorativa')
            if st.button('Statistiche'):
                st.subheader('📊 Statistiche Descrittive')
                st.dataframe(df.describe())
            if st.button('Grafici'):
                fig1, ax1 = plt.subplots()
                sns.countplot(x='survived', data=df, ax=ax1)
                st.pyplot(fig1)
                fig2 = sns.pairplot(df, hue='survived')
                st.pyplot(fig2)

        with tab2:
            st.header('Input Dati Passeggero')
            pclass = st.selectbox('Classe', [1, 2, 3])
            sex_input = st.selectbox('Sesso', ['male', 'female'])
            age = st.slider('Età', 0, 100, 30)
            sibsp = st.slider('Fratelli/Coniugi a bordo', 0, 8, 0)
            parch = st.slider('Genitori/Figli a bordo', 0, 6, 0)
            fare = st.slider('Tariffa', 0.0, 600.0, 50.0)
            embarked_input = st.selectbox('Imbarco', ['S', 'C', 'Q'])

            sex = 0 if sex_input == 'male' else 1
            embarked = {'S': 0, 'C': 1, 'Q': 2}[embarked_input]
            input_data = pd.DataFrame([{
                'pclass': pclass,
                'sex': sex,
                'age': age,
                'sibsp': sibsp,
                'parch': parch,
                'fare': fare,
                'embarked': embarked
            }])

            model_file = st.file_uploader("Carica modello salvato (.pkl)", type=["pkl"])
            if model_file is not None and st.button('Predici'):
                model = joblib.load(model_file)
                pred = model.predict(input_data)[0]
                st.success('Sopravvissuto' if pred == 1 else 'Non Sopravvissuto')

        with tab3:
            st.header('Predizione su Excel')
            excel_file = st.file_uploader("Carica file Excel", type=["xlsx"])
            if excel_file is not None:
                df_pred = pd.read_excel(excel_file)
                df_pred.columns = df_pred.columns.str.strip().str.lower().str.replace(' ', '_')
                st.write("Colonne Excel:", df_pred.columns.tolist())
                st.write(df_pred.head())
                model_file_batch = st.file_uploader("Carica modello per batch", type=["pkl"], key="model_batch")
                if model_file_batch is not None and st.button('Predici Tutti'):
                    model = joblib.load(model_file_batch)
                    preds = model.predict(df_pred)
                    df_pred['prediction'] = ['Sopravvissuto' if p == 1 else 'Non Sopravvissuto' for p in preds]
                    st.dataframe(df_pred)
                    st.download_button(
                        label="📥 Scarica Risultati",
                        data=convert_to_excel(df_pred),
                        file_name="titanic_predizioni.xlsx",
                        mime="application/vnd.openxmlformats-officedocument_spreadsheetml_sheet"
                    )

if __name__ == "__main__":
    main()
